import argparse
import ast
import json
import os
import re
import subprocess
import sys
import textwrap
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



class _RemovePassOnlyIf(ast.NodeTransformer):
    """Remove if-blocks whose only executable effect is pass."""

    def visit_If(self, node):
        node = self.generic_visit(node)

        if (
            not node.orelse
            and node.body
            and all(isinstance(stmt, ast.Pass) for stmt in node.body)
        ):
            return None

        return node


def _normalized_executable_ast(text):
    """
    Return a normalized AST for a code fragment.

    Pass-only if blocks are removed so edits such as:

        if condition:
            pass

    are treated as having no functional effect.
    """
    try:
        tree = ast.parse(textwrap.dedent(text))
    except (SyntaxError, ValueError):
        return None

    tree = _RemovePassOnlyIf().visit(tree)
    ast.fix_missing_locations(tree)

    return ast.dump(tree, include_attributes=False)


def is_obvious_noop_edit(old_text, new_text):
    """
    Detect edits whose executable behaviour is unchanged after removing
    pass-only conditional blocks.

    This is generic and contains no repository-specific logic.
    """
    old_ast = _normalized_executable_ast(old_text)
    new_ast = _normalized_executable_ast(new_text)

    if old_ast is None or new_ast is None:
        return False

    return old_text != new_text and old_ast == new_ast


def build_schema(
    phase,
    unread,
    editable,
    all_indices=None,
    allow_info=True,
    allow_inspect=True,
    allowed_actions=None,
):
    # Stage 1: choose only WHAT action to perform.
    # No index/query/edit text is generated at this stage.
    if phase == "decision":
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": allowed_actions or ["edit"],
                },
            },
            "required": ["action"],
            "additionalProperties": False,
        }

    # Stage 2: strict targeted inspection request.
    if phase == "inspect_strict":
        return {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["inspect"]},
                "index": {
                    "type": "integer",
                    "enum": all_indices or [],
                },
                "query": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 100,
                },
            },
            "required": ["action", "index", "query"],
            "additionalProperties": False,
        }

    # Stage 2: strict repository-wide search request.
    if phase == "search_strict":
        return {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["search"]},
                "query": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 100,
                },
            },
            "required": ["action", "query"],
            "additionalProperties": False,
        }

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
        if allow_inspect:
            actions.append("inspect")
        actions.append("search")

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



def search_repository(repo, query, max_files=6):
    """
    Generic repository-wide search requested by the model.
    Returns matching source files with short contextual snippets.
    No task-specific paths or symbols are hard-coded.
    """
    query = query.strip()
    if not query:
        return [], "Empty search query."

    matches = []

    for root, dirs, filenames in os.walk(repo):
        dirs[:] = [
            d for d in dirs
            if d not in {".git", "__pycache__", ".tox", ".venv", "venv", "build", "dist"}
        ]

        for filename in filenames:
            if not filename.endswith(".py"):
                continue

            full = os.path.join(root, filename)
            rel = os.path.relpath(full, repo)

            try:
                with open(full, encoding="utf-8", errors="replace") as f:
                    lines = f.readlines()
            except OSError:
                continue

            scored_hits = []

            for i, line in enumerate(lines):
                score = 0
                if query in line:
                    score += 2

                if re.search(r'\b' + re.escape(query) + r'\b', line):
                    score += 3

                if re.search(
                    r'^\s*(def|class)\s+' + re.escape(query) + r'\b',
                    line,
                ):
                    score += 8

                if score:
                    scored_hits.append((score, i))

            if not scored_hits:
                continue

            scored_hits.sort(key=lambda x: (-x[0], x[1]))
            file_score = sum(score for score, _ in scored_hits[:5])

            snippets = []
            used = set()

            for _, i in scored_hits[:3]:
                if any(abs(i - prev) < 8 for prev in used):
                    continue
                used.add(i)

                start = max(0, i - 4)
                stop = min(len(lines), i + 5)

                snippets.append(
                    "".join(
                        f"{j+1}: {lines[j]}"
                        for j in range(start, stop)
                    )
                )

            matches.append(
                (file_score, rel, "\n...[MATCH]...\n".join(snippets))
            )

    matches.sort(key=lambda x: (-x[0], x[1]))
    matches = matches[:max_files]

    paths = [path for _, path, _ in matches]

    text = "\n\n".join(
        f"### SEARCH MATCH: {path}\n{snippet}"
        for _, path, snippet in matches
    )

    if not matches:
        text = f"No repository-wide matches found for {query!r}."

    return paths, text



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



def extract_traceback_frames(issue):
    """
    Extract Python traceback file/line/function references from the raw issue.

    Examples handled:
        File ".../pkg/module.py", line 123, in func
        pkg/module.py:123
        pkg/module.py, line 123

    No repository-specific paths or symbols are hard-coded.
    """
    frames = []
    seen = set()

    patterns = [
        re.compile(
            r'File\s+["\']([^"\']+\.py)["\']\s*,\s*line\s+(\d+)'
            r'(?:\s*,\s*in\s+([A-Za-z_][A-Za-z0-9_<>]*))?'
        ),
        re.compile(
            r'([A-Za-z0-9_./\\-]+\.py)\s*:\s*(\d+)'
        ),
        re.compile(
            r'([A-Za-z0-9_./\\-]+\.py)\s*,\s*line\s+(\d+)'
        ),
    ]

    for pattern in patterns:
        for match in pattern.finditer(issue):
            path = match.group(1).replace("\\", "/")
            line_no = int(match.group(2))

            func = ""
            if match.lastindex and match.lastindex >= 3:
                func = match.group(3) or ""

            key = (path, line_no, func)

            if key not in seen:
                seen.add(key)
                frames.append(key)

    return frames


def build_traceback_context(repo, issue, files, radius=25, max_chars=9000):
    """
    Build focused source context directly around traceback line numbers.

    Candidate matching is suffix-based because traceback paths may be absolute
    while repository candidates are relative.

    Returns an empty string when the issue contains no usable traceback.
    In that case the existing agent behaviour is unchanged.
    """
    frames = extract_traceback_frames(issue)

    if not frames:
        return ""

    pieces = []
    used = set()

    for trace_path, line_no, func in frames:
        normalized_trace = trace_path.replace("\\", "/")

        candidate_idx = None

        # Prefer the longest matching suffix to avoid ambiguous basenames.
        matches = []

        for i, candidate in enumerate(files):
            candidate_norm = candidate.replace("\\", "/")

            if (
                normalized_trace.endswith(candidate_norm)
                or candidate_norm.endswith(normalized_trace)
                or normalized_trace.endswith("/" + candidate_norm)
            ):
                matches.append((len(candidate_norm), i))

        if not matches:
            # Conservative basename fallback.
            trace_name = os.path.basename(normalized_trace)

            basename_matches = [
                i
                for i, candidate in enumerate(files)
                if os.path.basename(candidate) == trace_name
            ]

            if len(basename_matches) == 1:
                candidate_idx = basename_matches[0]
        else:
            matches.sort(reverse=True)
            candidate_idx = matches[0][1]

        if candidate_idx is None:
            continue

        dedup_key = (candidate_idx, line_no)

        if dedup_key in used:
            continue

        used.add(dedup_key)

        full = os.path.join(repo, files[candidate_idx])

        try:
            with open(
                full,
                "r",
                encoding="utf-8",
                errors="replace",
            ) as f:
                lines = f.readlines()
        except OSError:
            continue

        if not lines:
            continue

        # Raw traceback line numbers are 1-based.
        center = max(0, min(len(lines) - 1, line_no - 1))

        start = max(0, center - radius)
        stop = min(len(lines), center + radius + 1)

        label = (
            f"\n### TRACEBACK FRAME [{candidate_idx}] "
            f"{files[candidate_idx]}:{line_no}"
        )

        if func:
            label += f" in {func}"

        label += "\n"

        snippet = "".join(
            f"{j + 1}: {lines[j]}"
            for j in range(start, stop)
        )

        pieces.append(label + snippet)

        if sum(len(x) for x in pieces) >= max_chars:
            break

    if not pieces:
        return ""

    result = "".join(pieces)

    return result[:max_chars]


def compact_shown_for_edit(files, shown, per_file_chars=2600, total_chars=8500):
    """
    Compact previously gathered evidence ONLY for the strict edit prompt.

    Broad reads remain available during normal read/inspect decisions.

    Targeted inspections are already prepended to shown[file], therefore the
    beginning of each entry is deliberately preserved.
    """
    parts = []
    used_chars = 0

    for i, txt in shown.items():
        if used_chars >= total_chars:
            break

        remaining = total_chars - used_chars
        limit = min(per_file_chars, remaining)

        if i >= 0:
            header = f"\n### FILE [{i}] {files[i]}\n"
        else:
            header = "\n### REPOSITORY SEARCH RESULTS\n"

        piece = header + txt[:limit] + "\n"

        parts.append(piece)
        used_chars += len(piece)

    return "".join(parts) or "\n(nothing read yet)\n"


def build_prompt(
    issue,
    files,
    shown,
    phase,
    feedback,
    allow_info=True,
    focused_context="",
    compact_for_edit=False,
):
    cand = "\n".join(
        f"[{i}] {p}" + ("   (already inspected/read)" if i in shown else "")
        for i, p in enumerate(files)
    )

    if compact_for_edit:
        contents = compact_shown_for_edit(
            files,
            shown,
        )
    else:
        contents = ""
        for i, txt in shown.items():
            if i >= 0:
                contents += (
                    f"\n### FILE [{i}] {files[i]}\n"
                    f"{txt[:FILE_CHARS_IN_PROMPT]}\n"
                )
            else:
                contents += (
                    f"\n### REPOSITORY SEARCH RESULTS\n"
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
                "3. action='search': search the entire repository for an exact "
                "symbol or short code term when the current candidate files appear "
                "insufficient. Set 'query' to the search term.\n"
                "4. action='edit': make the repair when you have enough evidence.\n\n"
                "Do not guess an edit if the current evidence is insufficient. "
                "Use inspect for a known file/symbol, or search when you need to "
                "discover relevant code elsewhere in the repository.\n\n"
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

TRACEBACK-FOCUSED EVIDENCE:
{focused_context if focused_context else "(no parseable traceback-focused source context)"}

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

    # Generic focused evidence derived only from traceback locations
    # already present in the raw SWE-bench problem statement.
    traceback_context = build_traceback_context(
        repo,
        issue,
        files,
    )

    if traceback_context:
        print(
            f"[traceback] focused context: "
            f"{len(traceback_context)} chars",
            file=sys.stderr,
        )
    else:
        print(
            "[traceback] no parseable focused context; "
            "using normal workflow",
            file=sys.stderr,
        )

    shown = {}
    feedback = ""
    info_steps = 0
    inspected_requests = set()
    force_diversify_next = False

    # Generic recovery after an edit targets text that does not belong
    # to the selected file. One recovery step temporarily re-enables
    # information gathering even when the normal information budget
    # has already been exhausted.
    force_recovery_info_next = False
    recovery_info_used = 0
    MAX_RECOVERY_INFO = 2

    total_steps = MAX_READS + MAX_EDIT_TRIES

    for step in range(1, total_steps + 1):
        unread = [i for i in range(len(files)) if i not in shown]

        editable = [
            i for i in shown
            if i >= 0 and not _TEST_RE.search(files[i])
        ]

        # Logging only: print every loop iteration.
        current_phase = "read" if not shown else "adaptive"
        print(
            f"\n===== STEP {step}/{total_steps} "
            f"phase={current_phase} info_steps={info_steps}/{MAX_READS} ====="
        )

        # First information action is always a normal repository read.
        if not shown:
            phase = "read"
            allow_info = True

            schema = build_schema(
                "read",
                unread,
                [],
            )

            prompt = build_prompt(
                issue,
                files,
                shown,
                phase,
                feedback,
                allow_info=True,
            )

            print(f"[prompt chars: {len(prompt)}]", file=sys.stderr)

            obj = ask(prompt, schema)

            print("MODEL ACTION:")
            print(json.dumps(obj, indent=2)[:3000])

            action = "read"
            idx = obj["index"]

        else:
            phase = "adaptive"

            # A rejected wrong-file edit gets one information-recovery
            # iteration. This prevents the agent from being trapped in:
            # bad edit -> reject -> bad edit -> reject.
            recovery_this_step = force_recovery_info_next

            if recovery_this_step:
                # One-shot flag. The current decision is forced to gather
                # information; the following iteration returns to normal.
                force_recovery_info_next = False

            allow_info = recovery_this_step or info_steps < MAX_READS

            # -------------------------------------------------
            # Stage 1: choose only the next action.
            # -------------------------------------------------
            if recovery_this_step:
                allowed_actions = []
            else:
                allowed_actions = ["edit"]

            if allow_info:
                if unread:
                    allowed_actions.append("read")

                if not force_diversify_next:
                    allowed_actions.append("inspect")

                allowed_actions.append("search")

            decision_schema = build_schema(
                "decision",
                unread,
                sorted(editable),
                allowed_actions=allowed_actions,
            )

            prompt_feedback = feedback

            if recovery_this_step:
                recovery_feedback = (
                    "The previous edit was rejected because its old_text did "
                    "not match the selected source file. For this iteration, "
                    "do not edit. Gather targeted information using read, "
                    "inspect, or repository search, then reconsider the fix."
                )

                prompt_feedback = (
                    f"{feedback}\n{recovery_feedback}".strip()
                    if feedback else recovery_feedback
                )

            if force_diversify_next:
                extra_feedback = (
                    "The previous targeted inspection duplicated an inspection "
                    "that was already completed. For this decision, choose a "
                    "different information path such as read or repository "
                    "search, or edit if the evidence is sufficient."
                )

                prompt_feedback = (
                    f"{feedback}\n{extra_feedback}".strip()
                    if feedback else extra_feedback
                )

            decision_prompt = build_prompt(
                issue,
                files,
                shown,
                phase,
                prompt_feedback,
                allow_info=allow_info,
            )

            decision_prompt += (
                "\n\nDECISION STAGE:\n"
                "Choose only the next action. Do not provide an index, query, "
                "old_text, or new_text at this stage."
            )

            print(
                f"[decision prompt chars: {len(decision_prompt)}]",
                file=sys.stderr,
            )

            decision = ask(decision_prompt, decision_schema)

            print("MODEL DECISION:")
            print(json.dumps(decision, indent=2)[:1000])

            action = decision["action"]

            # -------------------------------------------------
            # Stage 2A: READ
            # Only unread candidate indices are legal.
            # -------------------------------------------------
            if action == "read":
                read_schema = build_schema(
                    "read",
                    unread,
                    [],
                )

                read_prompt = build_prompt(
                    issue,
                    files,
                    shown,
                    "read",
                    feedback,
                    allow_info=True,
                )

                read_prompt += (
                    "\n\nYou already decided to READ another candidate. "
                    "Choose exactly one unread candidate index."
                )

                obj = ask(read_prompt, read_schema)

                print("STRICT READ ACTION:")
                print(json.dumps(obj, indent=2)[:1000])

                idx = obj["index"]

            # -------------------------------------------------
            # Stage 2B: INSPECT
            # Requires file index + exact query.
            # -------------------------------------------------
            elif action == "inspect":
                inspect_schema = build_schema(
                    "inspect_strict",
                    [],
                    [],
                    all_indices=list(range(len(files))),
                )

                inspect_prompt = build_prompt(
                    issue,
                    files,
                    shown,
                    "adaptive",
                    feedback,
                    allow_info=True,
                )

                inspect_prompt += (
                    "\n\nYou already decided to INSPECT. "
                    "Return action='inspect', the candidate-file index, and one "
                    "exact symbol/identifier in query. Do not edit code."
                )

                obj = ask(inspect_prompt, inspect_schema)

                print("STRICT INSPECT ACTION:")
                print(json.dumps(obj, indent=2)[:1500])

                idx = obj["index"]

            # -------------------------------------------------
            # Stage 2C: SEARCH
            # Requires only a repository-wide search query.
            # -------------------------------------------------
            elif action == "search":
                search_schema = build_schema(
                    "search_strict",
                    [],
                    [],
                )

                search_prompt = build_prompt(
                    issue,
                    files,
                    shown,
                    "adaptive",
                    feedback,
                    allow_info=True,
                )

                search_prompt += (
                    "\n\nYou already decided to SEARCH the repository. "
                    "Return action='search' and one short exact symbol or "
                    "code term in query. Do not provide an edit."
                )

                obj = ask(search_prompt, search_schema)

                print("STRICT SEARCH ACTION:")
                print(json.dumps(obj, indent=2)[:1500])

                # Existing search handler expects idx to exist even though
                # repository search itself doesn't need a file index.
                idx = 0

            # -------------------------------------------------
            # Stage 2D: EDIT
            # Requires a complete patch object.
            # -------------------------------------------------
            elif action == "edit":
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
                    focused_context=traceback_context,
                    compact_for_edit=bool(traceback_context),
                )

                if traceback_context:
                    strict_prompt += (
                        "\n\nTRACEBACK GUIDANCE:\n"
                        "Prioritize the traceback-focused evidence above. "
                        "Trace the reported execution path before considering "
                        "unrelated subsystems. Do not assume the most recently "
                        "inspected code is the bug location unless it explains "
                        "the original traceback.\n"
                    )

                strict_prompt += (
                    "\n\nYou already decided to EDIT. "
                    "Return the complete edit object only. "
                    "You MUST include analysis, action='edit', index, "
                    "old_text, and new_text. "
                    "old_text must be 1-6 exact consecutive lines from an "
                    "already read/inspected SOURCE file. "
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

            else:
                feedback = (
                    "ACTION REJECTED: invalid decision action."
                )
                print(feedback)
                continue

        # A non-inspect action satisfies the one-step diversification
        # requirement. Future inspect actions may be used again.
        if force_diversify_next and action != "inspect":
            force_diversify_next = False

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

            inspect_key = (idx, query)

            if inspect_key in inspected_requests:
                feedback = (
                    "INSPECT REJECTED: this exact file/query pair has already "
                    "been inspected. The next step must use a different "
                    "information path or make an edit."
                )
                force_diversify_next = True
                print(feedback)
                continue

            inspected_requests.add(inspect_key)

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
        # REPOSITORY-WIDE SEARCH
        # -------------------------
        if action == "search":
            if not allow_info:
                feedback = (
                    "SEARCH REJECTED: information-gathering limit reached. "
                    "Use the evidence already collected and make an edit."
                )
                print(feedback)
                continue

            query = obj.get("query", "").strip()

            if not query:
                feedback = (
                    "SEARCH REJECTED: provide a short exact symbol or code term "
                    "in 'query'."
                )
                print(feedback)
                continue

            new_paths, search_text = search_repository(repo, query)

            added = []

            for path in new_paths:
                if path not in files:
                    files.append(path)
                    added.append(path)

            # Keep search findings as context under a synthetic negative key.
            search_key = -(step + 1)
            shown[search_key] = search_text

            info_steps += 1

            print(
                f"\nSEARCH query={query!r} "
                f"matches={len(new_paths)} added_candidates={len(added)}"
            )

            for path in added:
                print(f"  + candidate [{files.index(path)}] {path}")

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

        # -------------------------------------------------
        # v7 generic edit-target validation
        # -------------------------------------------------
        obj["old_text"] = strip_line_numbers(
            obj.get("old_text", "")
        ).strip("\n")
        obj["new_text"] = strip_line_numbers(
            obj.get("new_text", "")
        ).strip("\n")

        candidate_old = obj["old_text"]
        candidate_new = obj["new_text"]

        if candidate_old.strip():
            target_path = os.path.join(repo, files[idx])

            try:
                with open(
                    target_path,
                    "r",
                    encoding="utf-8",
                    errors="replace",
                ) as f:
                    target_source = f.read()
            except OSError as exc:
                feedback = (
                    "EDIT REJECTED: unable to read the selected source file: "
                    f"{exc}"
                )
                print(feedback)
                continue

            # old_text must come from the SAME source file selected by index.
            # Use whitespace-tolerant matching consistent with apply_edit().
            _target_lines = target_source.split("\n")
            _old_lines = [line.rstrip() for line in candidate_old.split("\n")]

            _found_in_target = (
                candidate_old in target_source
                or any(
                    [
                        line.rstrip()
                        for line in _target_lines[
                            k:k + len(_old_lines)
                        ]
                    ] == _old_lines
                    for k in range(
                        len(_target_lines) - len(_old_lines) + 1
                    )
                )
            )

            if not _found_in_target:
                feedback = (
                    "EDIT REJECTED: old_text does not occur in the selected "
                    "source file. The proposed edit appears to target a "
                    "different code region or file."
                )

                if recovery_info_used < MAX_RECOVERY_INFO:
                    recovery_info_used += 1
                    force_recovery_info_next = True

                    feedback += (
                        " The next iteration will be an information-recovery "
                        "step; use read, inspect, or search before editing again."
                    )

                print(feedback)
                continue

            # Reject executable-looking edits that are still behaviourally empty.
            if (
                candidate_new.strip()
                and is_obvious_noop_edit(candidate_old, candidate_new)
            ):
                feedback = (
                    "EDIT REJECTED: the proposed executable change is an "
                    "obvious no-op after removing pass-only conditional blocks. "
                    "Produce a functional code change that affects behaviour."
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
    try:
        main()
    except Exception as exc:
        print(f"\nAGENT ERROR: {type(exc).__name__}: {exc}")
        sys.exit(6)
