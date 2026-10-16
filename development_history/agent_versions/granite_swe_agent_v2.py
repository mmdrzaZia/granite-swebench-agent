import argparse
import json
import os
import re
import subprocess
import sys
import urllib.request
from collections import Counter

MODEL = "granite4.2:8b"
OLLAMA_URL = "http://172.31.128.1:11434/api/generate"
MAX_STEPS = 10

def run(cmd, cwd):
    p = subprocess.run(
        cmd, cwd=cwd, shell=True, text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return p.returncode, p.stdout

MAX_READS = 2
MAX_EDIT_TRIES = 6
NUM_CTX = 8192
FILE_CHARS_IN_PROMPT = 9000


def ask(prompt, schema):
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "format": schema,
        "options": {"temperature": 0, "num_ctx": NUM_CTX, "num_predict": 1200, "repeat_penalty": 1.1},
    }
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=1800) as r:
        data = json.loads(r.read().decode())
    return json.loads(data["response"])


def build_schema(phase, unread, editable):
    if phase == "read":
        return {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["read"]},
                "index": {"type": "integer", "enum": unread},
            },
            "required": ["action", "index"],
            "additionalProperties": False,
        }
    return {
        "type": "object",
        "properties": {
            "analysis": {"type": "string", "maxLength": 400},
            "action": {"type": "string", "enum": ["edit"]},
            "index": {"type": "integer", "enum": editable},
            "old_text": {"type": "string"},
            "new_text": {"type": "string"},
        },
        "required": ["analysis", "action", "index", "old_text", "new_text"],
        "additionalProperties": False,
    }


def strip_line_numbers(t):
    lines = t.split("\n")
    nonempty = [l for l in lines if l.strip()]
    if nonempty and all(re.match(r"^\d+: ", l) for l in nonempty):
        return "\n".join(re.sub(r"^\d+: ", "", l, count=1) for l in lines)
    return t


def apply_edit(repo, path, old, new):
    """Return (ok, message). Writes the file only if every check passes."""
    import ast
    full = os.path.join(repo, path)
    with open(full, encoding="utf-8", errors="replace", newline="") as f:
        text = f.read()

    old = strip_line_numbers(old).strip("\n")
    new = strip_line_numbers(new).strip("\n")

    if not old.strip():
        return False, "old_text is empty. Copy 1-6 exact lines from the file."
    if old == new:
        return False, "new_text is identical to old_text. Make an actual change."

    n = text.count(old)
    if n == 1:
        updated = text.replace(old, new, 1)
    elif n > 1:
        return False, f"old_text appears {n} times in the file. Include more surrounding lines so it is unique."
    else:
        tl = text.split("\n")
        ol = [l.rstrip() for l in old.split("\n")]
        hits = [i for i in range(len(tl) - len(ol) + 1)
                if [x.rstrip() for x in tl[i:i + len(ol)]] == ol]
        if len(hits) != 1:
            return False, ("old_text was not found in the file. Copy the lines exactly as shown, "
                           "without the 'NNN: ' line-number prefixes.")
        i = hits[0]
        updated = "\n".join(tl[:i] + new.split("\n") + tl[i + len(ol):])

    if path.endswith(".py"):
        try:
            ast.parse(updated)
        except SyntaxError as e:
            return False, f"the edit would create a Python syntax error ({e.msg}, line {e.lineno}). Fix indentation/syntax."

    with open(full, "w", encoding="utf-8", newline="") as f:
        f.write(updated)
    return True, "ok"


def build_prompt(issue, files, shown, phase, feedback):
    cand = "\n".join(
        f"[{i}] {p}" + ("   (already read)" if i in shown else "")
        for i, p in enumerate(files)
    )
    contents = ""
    for i, txt in shown.items():
        contents += f"\n### FILE [{i}] {files[i]}\n{txt[:FILE_CHARS_IN_PROMPT]}\n"
    contents = contents or "\n(nothing read yet)\n"

    if phase == "read":
        task = ("Choose ONE more candidate file to read that is most likely to contain "
                "the code that must change. Return its index.")
    else:
        task = (
            "Now fix the bug. Return:\n"
            "- analysis: 1-3 sentences: what is wrong and where.\n"
            "- index: the number of the SOURCE file (not a test) to change.\n"
            "- old_text: 1-6 consecutive lines copied EXACTLY from that file "
            "(no 'NNN: ' line-number prefixes), unique in the file.\n"
            "- new_text: the replacement for those lines.\n"
            "Make the smallest change that fixes the issue. Do not edit tests."
        )
    fb = f"\nFEEDBACK ON YOUR PREVIOUS ATTEMPT:\n{feedback}\n" if feedback else ""
    return f"""You are fixing one bug in a Python repository.

ORIGINAL ISSUE:
{issue}

CANDIDATE FILES (found by automated search):
{cand}

FILE CONTENTS READ SO FAR:{contents}
{fb}
YOUR TASK:
{task}
"""


def get_terms(issue):
    stop = {
        "the","and","for","with","from","that","this","when","both",
        "true","param","produces","query","error","description",
        "whatever","backend","missing","least","annotation","containing"
    }

    words = re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", issue)

    out = []
    for w in words:
        if w.lower() not in stop and w.lower() not in [x.lower() for x in out]:
            out.append(w)

    return out[:12]

def retrieve(repo, issue):
    from retrieval_generic import retrieve_paths, build_matches
    ranked = retrieve_paths(repo, issue)
    terms = get_terms(issue)
    return ranked, build_matches(repo, ranked, terms), terms


def read_relevant(repo, path, terms, issue):
    full = os.path.join(repo, path)

    with open(full, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    # Small files can be shown in full.
    if len(lines) <= 250:
        return "".join(lines)

    selected = set()

    # 1) Generic keyword-based excerpts.
    hit_lines = []
    for i, line in enumerate(lines):
        if any(t.lower() in line.lower() for t in terms):
            hit_lines.append(i)

    for i in hit_lines[:20]:
        for j in range(max(0, i - 12), min(len(lines), i + 13)):
            selected.add(j)

    # 2) If the original issue itself names line numbers/ranges,
    # include those regions too. This uses only the raw problem statement.
    explicit_ranges = []

    # Examples: 276-281, 276–281
    for m in re.finditer(r'\b(\d{1,6})\s*[-–]\s*(\d{1,6})\b', issue):
        a, b = int(m.group(1)), int(m.group(2))
        if a <= b and a >= 1:
            explicit_ranges.append((a, b))

    # Examples: line 280, lines 280
    for m in re.finditer(r'\blines?\s+(\d{1,6})\b', issue, flags=re.I):
        n = int(m.group(1))
        if n >= 1:
            explicit_ranges.append((n, n))

    for a, b in explicit_ranges:
        # Convert 1-based source line numbers to 0-based indexes,
        # with generic context around the referenced region.
        start = max(0, a - 1 - 25)
        stop = min(len(lines), b + 25)
        for j in range(start, stop):
            selected.add(j)

    # Fallback if nothing useful matched.
    if not selected:
        return "".join(lines[:180])

    result = []
    last = None

    for i in sorted(selected):
        if last is not None and i > last + 1:
            result.append("\n...[SNIP]...\n")
        result.append(f"{i+1}: {lines[i]}")
        last = i

    return "".join(result)[:22000]



def main():
    from retrieval_generic import _TEST_RE

    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--issue-file", required=True)
    ap.add_argument("--patch-out", required=True)
    args = ap.parse_args()

    repo = os.path.abspath(args.repo)
    with open(args.issue_file, encoding="utf-8") as f:
        issue = f.read().strip()

    files, matches, terms = retrieve(repo, issue)
    if not files:
        print("No candidate files found.")
        sys.exit(2)

    print("\nCANDIDATE FILES:")
    for i, path in enumerate(files):
        print(f"[{i}] {path}")

    shown = {}
    feedback = ""
    total_steps = MAX_READS + MAX_EDIT_TRIES

    for step in range(1, total_steps + 1):
        unread = [i for i in range(len(files)) if i not in shown]
        phase = "read" if (len(shown) < MAX_READS and unread) else "edit"
        print(f"\n===== STEP {step}/{total_steps}  phase={phase} =====")

        if phase == "read":
            schema = build_schema("read", unread, [])
        else:
            editable = [i for i in shown if not _TEST_RE.search(files[i])] or list(shown)
            schema = build_schema("edit", [], sorted(editable))

        prompt = build_prompt(issue, files, shown, phase, feedback)
        print(f"[prompt chars: {len(prompt)}]", file=sys.stderr)

        obj = ask(prompt, schema)
        print("MODEL ACTION:")
        print(json.dumps(obj, indent=2)[:3000])

        if phase == "read":
            idx = obj["index"]
            shown[idx] = read_relevant(repo, files[idx], terms, issue)
            print(f"\nREAD [{idx}] {files[idx]} ({len(shown[idx])} chars)")
            continue

        idx = obj["index"]
        ok, msg = apply_edit(repo, files[idx], obj["old_text"], obj["new_text"])
        if not ok:
            feedback = "EDIT REJECTED: " + msg
            print(feedback)
            continue

        _, final_diff = run("git diff", repo)
        if not final_diff.strip():
            feedback = "EDIT REJECTED: it produced no change."
            print(feedback)
            continue

        with open(args.patch_out, "w", encoding="utf-8") as f:
            f.write(final_diff)
        print("\nSUCCESS: EDIT APPLIED")
        print("\n===== FINAL GIT DIFF =====")
        print(final_diff)
        print("\nSaved to:", args.patch_out)
        return

    print("\nReached step limit without a valid edit.")
    sys.exit(5)


if __name__ == "__main__":
    main()
