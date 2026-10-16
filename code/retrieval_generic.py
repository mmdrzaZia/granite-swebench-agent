import collections, math, re, subprocess, sys

_STOP = set("""about above after again against also another because before being below between both
could does doing during each either every from further have having here into issue just like might
more most much must never only other over please same should since some such than that their them
then there these they this those through under until upon very want were what when where which while
will with without would your using used uses error errors problem description expected actual
instead behavior behaviour return returns returned raise raises raised call called calls works work
working none true false self args kwargs""".split())

_TEST_RE = re.compile(r"(^|/)(tests?|testing|testsuite|test_utils)(/|$)|(^|/)test_[^/]*\.py$|_tests?\.py$|(^|/)conftest\.py$")
_NOISE_RE = re.compile(r"(^|/)(docs?|examples?|benchmarks?|scripts?|asv_bench|tools|extras)(/|$)|(^|/)setup\.py$")
K_IMPL, K_TEST = 6, 3


def _git(repo, *args):
    p = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, errors="replace")
    return p.stdout


def _extract_terms(issue):
    """term -> (kind, weight). Uses ONLY the issue text."""
    terms = {}

    def add(t, kind, w):
        if len(t) < 3 or t.lower() in _STOP:
            return
        if t not in terms or w > terms[t][1]:
            terms[t] = (kind, w)

    for blk in re.findall(r"`([^`\n]{1,100})`", issue):
        for ident in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", blk):
            add(ident, "symbol", 3)
    for t in re.findall(r"\b[A-Z][a-z0-9]+(?:[A-Z][A-Za-z0-9]*)+\b", issue):
        add(t, "symbol", 3)
    for t in re.findall(r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b", issue):
        add(t, "symbol", 2.5)
    for t in re.findall(r"\b_{0,2}[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b", issue):
        add(t, "snake", 2.5)
    for t in re.findall(r"\b([A-Za-z_]\w*)\(", issue):
        add(t, "snake", 2.5)
    for t in re.findall(r", in (\w+)", issue):
        add(t, "symbol", 3)
    for m in re.finditer(r"\b[A-Z][a-z]{2,}\b", issue):          # mid-sentence Capitalized word
        before = issue[:m.start()].rstrip(" \t")
        if before and before[-1] not in ".!?:\n":
            add(m.group(), "cap", 2)
    for t in set(re.findall(r"\b[a-z]{5,}\b", issue)):
        add(t, "word", 1)
    ranked = sorted(terms.items(), key=lambda kv: (-kv[1][1], -len(kv[0])))
    return dict(ranked[:40])


def _resolve_paths(issue, tracked, tset):
    found = set()
    for p in re.findall(r"[\w./\\-]+\.py\b", issue):
        parts = p.replace("\\", "/").lstrip("./").split("/")
        for i in range(len(parts)):
            suf = "/".join(parts[i:])
            hits = [f for f in tracked if f == suf or f.endswith("/" + suf)]
            if hits:
                if len(hits) <= 3:
                    found.update(hits)
                break
    for d in re.findall(r"\b[a-z_]\w*(?:\.[a-z_]\w*){2,}\b", issue):
        base = d.replace(".", "/")
        for cand in (base + ".py", base + "/__init__.py", base.rsplit("/", 1)[0] + ".py"):
            if cand in tset:
                found.add(cand)
    return found


def retrieve_paths(repo, issue):
    tracked = [f for f in _git(repo, "ls-files", "*.py").splitlines()
               if f and not _NOISE_RE.search(f)]
    tset = set(tracked)
    N = max(len(tracked), 1)
    terms = _extract_terms(issue)
    score = collections.defaultdict(float)
    why = collections.defaultdict(list)

    # 1) occurrence evidence, IDF-weighted (rare terms count more; ubiquitous ones are dropped)
    live = {}
    for t, (kind, w) in terms.items():
        flags = ["-c", "-I", "-w", "-F"] + (["-i"] if kind == "word" else [])
        counts = {}
        for line in _git(repo, "grep", *flags, "-e", t, "--", "*.py").splitlines():
            path, _, c = line.rpartition(":")
            if path in tset:
                counts[path] = int(c)
        df = len(counts)
        if df == 0 or df > 0.3 * N:
            continue
        idf = math.log((N + 1) / (df + 1))
        if kind == "word" and idf < 2.5:
            continue
        live[t] = (kind, w, idf)
        for path, c in counts.items():
            score[path] += w * idf * min(1 + math.log2(c), 2.0)
            why[path].append(f"occ:{t}x{c}")

    # 2) exact symbol definitions
    sym = [t for t, (k, _, _) in live.items() if k in ("symbol", "cap", "snake")]
    if sym:
        alt = "|".join(sym)
        S = "[[:space:]]"
        pat_def = rf"^{S}*(async{S}+def|def|class){S}+({alt}){S}*[(:]"
        pat_asg = rf"^({alt}){S}*="
        def_files = collections.defaultdict(set)
        out = _git(repo, "grep", "-n", "-I", "-E", "-e", pat_def, "-e", pat_asg, "--", "*.py")
        for line in out.splitlines():
            path, _, rest = line.partition(":")
            m = re.search(rf"(?:class|def)\s+({alt})\b|^\d+:({alt})\s*=", rest)
            name = (m.group(1) or m.group(2)) if m else None
            if name and path in tset:
                def_files[name].add(path)
        for name, files in def_files.items():
            bonus = 20.0 / math.sqrt(len(files))
            for f in files:
                score[f] += bonus
                why[f].append(f"def:{name}")

        # 3) imports of those symbols
        pat_imp = rf"^{S}*(from{S}+[A-Za-z0-9_.]+{S}+)?import{S}.*({alt})"
        for line in _git(repo, "grep", "-n", "-I", "-E", "-e", pat_imp, "--", "*.py").splitlines():
            path, _, rest = line.partition(":")
            if path not in tset:
                continue
            for name in sym:
                if re.search(rf"\b{re.escape(name)}\b", rest):
                    score[path] += 3.0
                    why[path].append(f"imp:{name}")

    # 4) files / modules named explicitly in the issue
    for f in _resolve_paths(issue, tracked, tset):
        score[f] += 100.0
        why[f].append("named-in-issue")
    words = set(re.findall(r"[a-z_]{4,}", issue))
    for f in tracked:
        stem = f.rsplit("/", 1)[-1][:-3]
        if len(stem) >= 4 and stem in words and not _TEST_RE.search(f):
            score[f] += 4.0
            why[f].append(f"stem:{stem}")

    for f in list(score):
        if f.endswith("__init__.py"):
            score[f] *= 0.5

    # 5) two pools so tests never crowd out source files
    impl = sorted(((f, s) for f, s in score.items() if not _TEST_RE.search(f)), key=lambda x: -x[1])
    test = sorted(((f, s) for f, s in score.items() if _TEST_RE.search(f)), key=lambda x: -x[1])
    chosen = impl[:K_IMPL] + test[:K_TEST]

    print(f"[retrieve] terms kept: {sorted(live)}", file=sys.stderr)
    for f, s in chosen:
        print(f"[retrieve] {s:7.1f}  {f}  {' '.join(why[f][:6])}", file=sys.stderr)
    return [f for f, _ in chosen]


def build_matches(repo, paths, terms):
    """path -> list of 'path:line:text', same shape the old retrieve() produced."""
    matches, seen = {}, set()
    if not paths:
        return matches
    for term in terms:
        for line in _git(repo, "grep", "-n", "-I", "-i", "-F", "-e", term, "--", *paths).splitlines():
            parts = line.split(":", 2)
            if len(parts) < 3 or line in seen:
                continue
            seen.add(line)
            matches.setdefault(parts[0], []).append(line)
    return matches
