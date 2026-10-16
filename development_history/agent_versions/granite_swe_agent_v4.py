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

MAX_READS = 5
MAX_EDIT_TRIES = 6
NUM_CTX = 8192
FILE_CHARS_IN_PROMPT = 9000


def ask(prompt, schema):
    def call_model(local_prompt, num_predict=1200):
        payload = {
            "model": MODEL,
            "prompt": local_prompt,
            "stream": False,
            "format": schema,
            "options": {
                "temperature": 0,
                "num_ctx": NUM_CTX,
                "num_predict": num_predict,
                "repeat_penalty": 1.1,
            },
        }

        req = urllib.request.Request(
            OLLAMA_URL,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )

        with urllib.request.urlopen(req, timeout=1800) as r:
            data = json.loads(r.read().decode())

        return data["response"]

    response = call_model(prompt, num_predict=1600)

    try:
        return json.loads(response)

    except json.JSONDecodeError as e:
        print(
            f"[json retry] invalid JSON from model: {e}",
            file=sys.stderr,
        )
        print(
            f"[json retry] response chars={len(response)} tail={response[-300:]!r}",
            file=sys.stderr,
        )

        retry_prompt = (
            prompt
            + "\n\nIMPORTANT: Return ONLY one complete valid JSON object matching "
              "the schema. Keep analysis extremely short. Keep old_text and new_text "
              "as small as possible while making the functional edit. "
              "Do not include markdown or commentary outside JSON."
        )

        response = call_model(retry_prompt, num_predict=2000)

        try:
            return json.loads(response)

        except json.JSONDecodeError as e2:
            print(
                f"[json retry failed] invalid JSON again: {e2}",
                file=sys.stderr,
            )
            print(
                f"[json retry failed] response chars={len(response)} "
                f"tail={response[-300:]!r}",
                file=sys.stderr,
            )
            raise


def build_schema(phase, unread, editable, all_indices=None, allow_info=True):
    if phase == "edit_strict":
        return {
            "type": "object",
            "properties": {
                "analysis": {"type": "string", "maxLength": 200},
                "action": {"type": "string", "enum": ["edit"]},
                "index": {"type": "integer", "enum": editable},
                "old_text": {"type": "string"},
                "new_text": {"type": "string"},
            },
            "required": [
                "analysis",
                "action",
                "index",
                "old_text",
                "new_text",
            ],
            "additionalProperties": False,
        }

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

    actions = ["edit"]
    if allow_info:
        if unread:
            actions.append("read")
        actions.append("inspect")

    return {
        "type": "object",
        "properties": {
            "analysis": {"type": "string", "maxLength": 200},
            "action": {"type": "string", "enum": actions},
            "index": {
                "type": "integer",
                "enum": all_indices if all_indices is not None else editable,
            },
            "query": {"type": "string", "maxLength": 100},
            "old_text": {"type": "string"},
            "new_text": {"type": "string"},
        },
        "required": ["action", "index"],
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


def inspect_symbol(repo, path, query):
    """
    Generic targeted inspection of an identifier/symbol in a repository file.
    The query is selected by the model; no task-specific names are hard-coded.
    """
    full = os.path.join(repo, path)

    try:
        with open(full, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except OSError:
        return ""

    query = query.strip()
    if not query:
        return ""

    escaped = re.escape(query)
    hits = []

    for i, line in enumerate(lines):
        score = 0

        # Prefer definitions/declarations.
        if re.search(r'^\s*(def|class)\s+' + escaped + r'\b', line):
            score = 5
        elif re.search(r'^\s*' + escaped + r'\s*=', line):
            score = 4
        elif re.search(r'\b' + escaped + r'\b', line):
            score = 2
        elif query in line:
            score = 1

        if score:
            hits.append((score, i))

    if not hits:
        return f"No exact occurrence found for symbol/query: {query!r}"

    hits.sort(key=lambda x: (-x[0], x[1]))

    centers = []
    for score, i in hits:
        if any(abs(i - prev) < 12 for prev in centers):
            continue
        centers.append(i)
        if len(centers) >= 5:
            break

    selected = set()
    for i in centers:
        for j in range(max(0, i - 10), min(len(lines), i + 21)):
            selected.add(j)

    result = []
    last = None

    for i in sorted(selected):
        if last is not None and i > last + 1:
            result.append("\n...[SNIP]...\n")
        result.append(f"{i+1}: {lines[i]}")
        last = i

    return "".join(result)[:6000]


def build_prompt(issue, files, shown, phase, feedback, allow_info=True):
    cand = "\n".join(
        f"[{i}] {p}" + ("   (already inspected/read)" if i in shown else "")
        for i, p in enumerate(files)
    )

    contents = ""
    for i, txt in shown.items():
        contents += (
            f"\n### FILE [{i}] {files[i]}\n"
            f"{txt[:FILE_CHARS_IN_PROMPT]}\n"
        )

    contents = contents or "\n(nothing read yet)\n"

    if phase == "read":
        task = (
            "Choose ONE candidate file to read first. "
            "Return action='read' and its index."
        )
    else:
        if allow_info:
            task = (
                "Decide the next best action.\n\n"
                "You may choose:\n"
                "1. action='read': read another unread candidate file.\n"
                "2. action='inspect': inspect a specific symbol, function, class, "
                "constant, variable, or identifier in a candidate file. "
                "Set 'query' to the exact symbol you want to inspect.\n"
                "3. action='edit': make the repair when you have enough evidence.\n\n"
                "Do not guess an edit if your analysis says you still need to inspect "
                "a particular symbol or piece of code. Inspect it first.\n\n"
                "For edit:\n"
                "- index must be a SOURCE file, not a test.\n"
                "- old_text must be 1-6 consecutive exact lines from the file.\n"
                "- new_text must contain the smallest functional replacement.\n"
                "- Do not edit tests."
            )
        else:
            task = (
                "You have reached the information-gathering limit. "
                "Make the best repair based on the evidence already collected.\n"
                "Return action='edit', the source-file index, old_text, and new_text. "
                "Make the smallest functional change. Do not edit tests."
            )

    fb = (
        f"\nFEEDBACK ON YOUR PREVIOUS ATTEMPT:\n{feedback}\n"
        if feedback else ""
    )

    return f"""You are fixing one bug in a Python repository.

ORIGINAL ISSUE:
{issue}

CANDIDATE FILES (found by automated search):
{cand}

FILE CONTENTS READ/INSPECTED SO FAR:
{contents}
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



def extra_issue_code_terms(issue):
    """
    Preserve short identifiers that ordinary text-term filtering may discard.
    This is generic and is derived only from the raw issue statement.
    """
    found = []

    # Backtick/code-quoted identifiers.
    for token in re.findall(r'`([^`]+)`', issue):
        token = token.strip()
        if re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', token):
            found.append(token)

    # Uppercase code-like identifiers such as MOD, PK, SQL.
    for token in re.findall(r'\b[A-Z][A-Z0-9_]{1,15}\b', issue):
        found.append(token)

    # snake_case identifiers even when short.
    for token in re.findall(r'\b[A-Za-z][A-Za-z0-9]*_[A-Za-z0-9_]+\b', issue):
        found.append(token)

    result = []
    seen = set()

    for token in found:
        key = token.lower()
        if key not in seen:
            seen.add(key)
            result.append(token)

    return result



def read_relevant(repo, path, terms, issue):
    full = os.path.join(repo, path)

    with open(full, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    # Small files can be shown in full.
    if len(lines) <= 250:
        return "".join(lines)

    selected = set()

    # Weight more code-like / specific terms more strongly.
    scored_hits = []

    normalized_terms = []
    for t in terms:
        if t not in normalized_terms:
            normalized_terms.append(t)

    for i, line in enumerate(lines):
        line_lower = line.lower()
        score = 0.0

        for term in normalized_terms:
            t = term.lower()
            if t not in line_lower:
                continue

            # More specific/code-like terms get more weight.
            weight = 1.0

            if "_" in term:
                weight += 2.0

            if any(ch.isupper() for ch in term):
                weight += 1.5

            if len(term) >= 8:
                weight += 1.0

            # Exact-ish token occurrence is stronger than substring only.
            if re.search(r'\b' + re.escape(term) + r'\b', line, flags=re.I):
                weight += 1.5

            score += weight

        if score > 0:
            scored_hits.append((score, i))

    # Prefer strong hits, but avoid taking many adjacent lines from one region.
    scored_hits.sort(key=lambda x: (-x[0], x[1]))

    chosen_centers = []
    min_distance = 18

    for score, i in scored_hits:
        if any(abs(i - prev) < min_distance for prev in chosen_centers):
            continue
        chosen_centers.append(i)
        if len(chosen_centers) >= 8:
            break

    # If strong-hit selection found too little, add distributed hits from the file.
    if len(chosen_centers) < 8 and scored_hits:
        all_hit_lines = sorted({i for _, i in scored_hits})
        if all_hit_lines:
            slots = min(8, len(all_hit_lines))
            for k in range(slots):
                pos = round(k * (len(all_hit_lines) - 1) / max(1, slots - 1))
                i = all_hit_lines[pos]
                if i not in chosen_centers:
                    chosen_centers.append(i)

    for i in chosen_centers:
        for j in range(max(0, i - 10), min(len(lines), i + 11)):
            selected.add(j)

    # Generic bounded symbol expansion.
    # Follow only symbols referenced in lines that are themselves relevant
    # to the issue terms. Keep expansion deliberately small to avoid
    # overwhelming the local model with unrelated definitions.

    relevant_selected_lines = []

    for i in sorted(selected):
        line_lower = lines[i].lower()

        if any(term.lower() in line_lower for term in normalized_terms):
            relevant_selected_lines.append(lines[i])

    relevant_text = "".join(relevant_selected_lines)

    referenced_symbols = []

    for symbol in re.findall(
        r'\b([A-Za-z_][A-Za-z0-9_]*)\s*\(',
        relevant_text,
    ):
        if symbol in {
            "if", "for", "while", "return", "print", "len", "range",
            "isinstance", "type", "super", "set", "list", "dict",
            "tuple", "str", "int", "float", "bool",
        }:
            continue

        if symbol not in referenced_symbols:
            referenced_symbols.append(symbol)

    priority_definition_ranges = []

    # At most two extra definitions.
    for symbol in referenced_symbols:
        definition_patterns = [
            re.compile(r'^\s*def\s+' + re.escape(symbol) + r'\s*\('),
            re.compile(r'^\s*class\s+' + re.escape(symbol) + r'\b'),
        ]

        for i, line in enumerate(lines):
            if any(pat.search(line) for pat in definition_patterns):
                # Don't expand a definition already directly visible.
                if i in selected:
                    break

                start_def = max(0, i - 3)
                stop_def = min(len(lines), i + 22)

                priority_definition_ranges.append(
                    (symbol, start_def, stop_def)
                )
                break

        if len(priority_definition_ranges) >= 2:
            break

    # Also honor explicit line numbers/ranges from the raw issue.
    explicit_ranges = []

    for m in re.finditer(r'\b(\d{1,6})\s*[-–]\s*(\d{1,6})\b', issue):
        a, b = int(m.group(1)), int(m.group(2))
        if a <= b and a >= 1:
            explicit_ranges.append((a, b))

    for m in re.finditer(r'\blines?\s+(\d{1,6})\b', issue, flags=re.I):
        n = int(m.group(1))
        if n >= 1:
            explicit_ranges.append((n, n))

    for a, b in explicit_ranges:
        start_idx = max(0, a - 1 - 25)
        stop_idx = min(len(lines), b + 25)
        for j in range(start_idx, stop_idx):
            selected.add(j)

    if not selected:
        return "".join(lines[:180])

    # Priority definitions are emitted first so referenced helpers
    # cannot be lost when the final context is truncated.
    priority_result = []

    seen_priority_lines = set()

    for symbol, start_def, stop_def in priority_definition_ranges:
        priority_result.append(
            f"\n[REFERENCED SYMBOL DEFINITION: {symbol}]\n"
        )

        for i in range(start_def, stop_def):
            priority_result.append(f"{i+1}: {lines[i]}")
            seen_priority_lines.add(i)

    result = []
    last = None

    for i in sorted(selected):
        if i in seen_priority_lines:
            continue

        if last is not None and i > last + 1:
            result.append("\n...[SNIP]...\n")

        result.append(f"{i+1}: {lines[i]}")
        last = i

    priority_text = "".join(priority_result)
    normal_text = "".join(result)

    # Hard cap applies to the complete returned context, including
    # referenced definitions.
    combined = priority_text + normal_text
    return combined[:8000]



def executable_content(text):
    """
    Remove blank lines and full-line comments so comment-only edits
    can be detected generically.
    """
    result = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue
        result.append(line.rstrip())
    return "\n".join(result)


def is_comment_only_edit(old_text, new_text):
    return (
        old_text != new_text
        and executable_content(old_text) == executable_content(new_text)
    )



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

    # Preserve short/code-like identifiers from the raw issue that the
    # normal text retrieval stage may filter out (e.g. MOD, PK, SQL).
    for extra_term in extra_issue_code_terms(issue):
        if extra_term.lower() not in {t.lower() for t in terms}:
            terms.append(extra_term)

    print("[read] augmented terms:", terms, file=sys.stderr)

    if not files:
        print("No candidate files found.")
        sys.exit(2)

    print("\nCANDIDATE FILES:")
    for i, path in enumerate(files):
        print(f"[{i}] {path}")

    shown = {}
    feedback = ""
    info_steps = 0
    total_steps = MAX_READS + MAX_EDIT_TRIES

    for step in range(1, total_steps + 1):
        unread = [i for i in range(len(files)) if i not in shown]

        editable = [
            i for i in shown
            if not _TEST_RE.search(files[i])
        ]

        # First action is always a normal repository read.
        if not shown:
            phase = "read"
            allow_info = True
            schema = build_schema("read", unread, [])
        else:
            phase = "adaptive"
            allow_info = info_steps < MAX_READS

            schema = build_schema(
                "adaptive",
                unread,
                sorted(editable),
                all_indices=list(range(len(files))),
                allow_info=allow_info,
            )

        print(
            f"\n===== STEP {step}/{total_steps} "
            f"phase={phase} info_steps={info_steps}/{MAX_READS} ====="
        )

        prompt = build_prompt(
            issue,
            files,
            shown,
            phase,
            feedback,
            allow_info=allow_info,
        )

        print(f"[prompt chars: {len(prompt)}]", file=sys.stderr)

        obj = ask(prompt, schema)

        print("MODEL ACTION:")
        print(json.dumps(obj, indent=2)[:3000])

        action = obj["action"]
        idx = obj["index"]

        # -------------------------
        # READ
        # -------------------------
        if action == "read":
            if not allow_info and shown:
                feedback = (
                    "READ REJECTED: information-gathering limit reached. "
                    "Use the evidence already collected and make an edit."
                )
                print(feedback)
                continue

            if idx not in unread:
                feedback = (
                    "READ REJECTED: choose an unread candidate file."
                )
                print(feedback)
                continue

            shown[idx] = read_relevant(
                repo, files[idx], terms, issue
            )

            info_steps += 1

            print(
                f"\nREAD [{idx}] {files[idx]} "
                f"({len(shown[idx])} chars)"
            )
            feedback = ""
            continue

        # -------------------------
        # TARGETED INSPECT
        # -------------------------
        if action == "inspect":
            if not allow_info:
                feedback = (
                    "INSPECT REJECTED: information-gathering limit reached. "
                    "Make the best edit using the evidence already collected."
                )
                print(feedback)
                continue

            query = obj.get("query", "").strip()

            if not query:
                feedback = (
                    "INSPECT REJECTED: provide the exact symbol or identifier "
                    "you want to inspect in 'query'."
                )
                print(feedback)
                continue

            inspected = inspect_symbol(
                repo,
                files[idx],
                query,
            )

            # Put targeted inspection FIRST so it survives prompt truncation.
            previous = shown.get(idx, "")
            shown[idx] = (
                f"[TARGETED INSPECTION: {query}]\n"
                f"{inspected}\n\n"
                f"[PREVIOUS CONTEXT]\n"
                f"{previous}"
            )

            info_steps += 1

            print(
                f"\nINSPECT [{idx}] {files[idx]} "
                f"query={query!r} ({len(inspected)} chars)"
            )
            feedback = ""
            continue

        # -------------------------
        # EDIT
        # -------------------------
        if action != "edit":
            feedback = "ACTION REJECTED: choose read, inspect, or edit."
            print(feedback)
            continue

        if idx not in editable:
            feedback = (
                "EDIT REJECTED: edit only a SOURCE file that has already "
                "been read or inspected."
            )
            print(feedback)
            continue

        # The adaptive call only decides WHAT to do.
        # Generate the actual patch in a separate strict-schema call so
        # old_text/new_text cannot be omitted.
        strict_schema = build_schema(
            "edit_strict",
            [],
            sorted(editable),
        )

        strict_prompt = build_prompt(
            issue,
            files,
            shown,
            "adaptive",
            feedback,
            allow_info=False,
        )

        strict_prompt += (
            "\n\nYou have already decided to EDIT. "
            "Now return the complete edit object only. "
            "You MUST include analysis, action='edit', index, old_text, and new_text. "
            "old_text must be 1-6 exact consecutive lines from a SOURCE file. "
            "new_text must be the smallest functional replacement."
        )

        print(
            f"[strict edit prompt chars: {len(strict_prompt)}]",
            file=sys.stderr,
        )

        obj = ask(strict_prompt, strict_schema)

        print("STRICT EDIT ACTION:")
        print(json.dumps(obj, indent=2)[:3000])

        idx = obj["index"]

        if idx not in editable:
            feedback = (
                "EDIT REJECTED: edit only a SOURCE file that has already "
                "been read or inspected."
            )
            print(feedback)
            continue

        if not obj.get("old_text", "").strip():
            feedback = (
                "EDIT REJECTED: old_text is empty. "
                "Copy 1-6 exact lines from the source file."
            )
            print(feedback)
            continue

        if "new_text" not in obj:
            feedback = "EDIT REJECTED: new_text is missing."
            print(feedback)
            continue

        idx = obj["index"]

        idx = obj["index"]

        if is_comment_only_edit(obj["old_text"], obj["new_text"]):
            feedback = (
                "EDIT REJECTED: the proposed change only modifies comments or "
                "blank lines and does not change executable code. "
                "Produce a functional code change."
            )
            print(feedback)
            continue

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
