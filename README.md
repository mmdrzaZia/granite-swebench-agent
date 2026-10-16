# Local SWE-bench Agent Evaluation

This submission documents the iterative development and evaluation of a local coding agent on 10 SWE-bench Verified issues. The agent uses a local open-weight model through Ollama and operates by retrieving relevant repository files, reading/inspecting source code, optionally searching the repository, and proposing constrained `old_text` / `new_text` edits.

The submission separates two kinds of evidence:

1. **Frozen final-agent evaluation** — the same `granite_swe_agent_FINAL.py` was rerun on all 10 selected issues from their specified base commits.
2. **Development history** — selected earlier agent versions and runs are retained to show how the architecture evolved and where individual improvements helped or regressed behaviour.

The official SWE-bench evaluation harness was **not** run. Results are therefore reported as local repair outcomes based on source inspection, syntax checks, direct reproductions, and/or relevant repository tests where practical.

## Repository structure

```text
swebench-submission-clean/
├── README.md
├── report.pdf
├── code/
│   ├── granite_swe_agent_FINAL.py
│   └── retrieval_generic.py
├── results/
│   ├── 01_django-10880/
│   ├── 02_sklearn-14141/
│   ├── 03_django-11179/
│   ├── 04_django-16082/
│   ├── 05_sympy-16886/
│   ├── 06_sympy-15809/
│   ├── 07_sympy-13757/
│   ├── 08_sphinx-8269/
│   ├── 09_pytest-7982/
│   └── 10_xarray-3677/
└── development_history/
    ├── agent_versions/
    └── results/
```

## Final frozen-agent results

| # | SWE-bench issue | Final result | Main validation / failure evidence |
|---|---|---|---|
| 1 | `django__django-10880` | **FAIL** | Repository test exposed a regression: `COUNT(*)` became `COUNT( *)`. |
| 2 | `scikit-learn__scikit-learn-14141` | **PASS** | Direct source-level validation confirmed that `joblib` is included in `_get_deps_info()`. Repository pytest could not start because the historical checkout required compiled scikit-learn extensions. |
| 3 | `django__django-11179` | **FAIL** | Generated patch performed the deletion twice and included excessive reasoning comments. |
| 4 | `django__django-16082` | **FAIL** | `Combinable.MOD` was removed from an existing connector group without adding the required mixed-type replacement rules. |
| 5 | `sympy__sympy-16886` | **FAIL** | The bug was diagnosed correctly, but repeated malformed edits were rejected until the action limit was reached. No valid patch was applied. |
| 6 | `sympy__sympy-15809` | **FAIL** | Repeated inspect/edit recovery loop; no valid patch was applied before the step limit. |
| 7 | `sympy__sympy-13757` | **FAIL** | A valid patch was applied, but Python 3.8 functional validation showed that `x * Poly(x)` remained unevaluated. |
| 8 | `sphinx-doc__sphinx-8269` | **FAIL** | The run initially localized `linkcheck.py` correctly, then drifted into unrelated LaTeX/application context and reached the step limit without a valid edit. |
| 9 | `pytest-dev__pytest-7982` | **PASS** | Python 3.8.20 validation: `4 passed, 75 deselected`. |
| 10 | `pydata__xarray-3677` | **PASS** | Direct reproduction passed (`identical = True`) and relevant repository tests reported `24 passed, 870 deselected`. |

Each folder in `results/` contains the original issue text, base commit, final agent log, a concise machine-readable `result.txt`, and a detailed `validation_summary.txt`. When a valid patch was produced, `generated_patch.patch` is included. Relevant test or direct functional-validation output is also included where available.

## Final agent

The final agent implementation is:

```text
code/granite_swe_agent_FINAL.py
```

Repository retrieval support is implemented in:

```text
code/retrieval_generic.py
```

The final agent uses a generic workflow rather than task-specific hard-coding. Its main mechanisms include:

- issue-term-based repository retrieval;
- bounded source-file reading;
- targeted symbol inspection;
- generic repository-wide search;
- duplicate-inspection protection;
- action-specific schemas for read / inspect / search / edit;
- exact `old_text` / `new_text` edit application;
- syntax validation before accepting an edit;
- recovery steps after rejected edits;
- traceback-focused context when a parseable traceback is present.

## Development history

`development_history/` is supplementary evidence rather than the primary final benchmark result.

`development_history/agent_versions/` contains the major agent versions used during development. Selected development runs are retained under `development_history/results/` to show the progression of the system, including both failed and successful intermediate runs.

`granite_swe_agent_v7.py` is byte-identical to `granite_swe_agent_FINAL.py`; V7 was the version frozen for the final evaluation.

Examples include:

- `django__django-11179`: an earlier failure followed by a successful repair after generic line-aware context handling was added;
- `sphinx-doc__sphinx-8269`: V5 failed because of action/schema control loops, while V6 produced a successful repair candidate with `5 passed` in the relevant test file;
- `pydata__xarray-3677`: V6 suffered from context drift, while V7 added generic traceback-focused/compact edit context and produced a repair that passed the direct reproduction and 24 relevant tests.

These development results are intentionally kept separate from the frozen final-agent results. An earlier successful development run does not override a later failure in the final frozen evaluation.

## Validation conventions

The following conventions are used throughout the submission:

- **PASS** means the generated repair was supported by the strongest practical local validation available for that task.
- **FAIL** means the final agent either produced no valid patch, produced an obviously invalid/incomplete patch, or failed direct/repository-level functional validation.
- `deselected` tests are tests excluded by a pytest filter and are not failures.
- Environment-related import/build failures are distinguished from failures caused by an agent-generated patch.
- Historical projects that were incompatible with the general Python 3.10 environment were validated in isolated Python 3.8.20 environments where required.

## Important limitation

These results should **not** be described as official SWE-bench resolved scores because the official SWE-bench evaluation harness was not executed. The submission instead reports transparent local evidence for each repair attempt and keeps the complete agent logs needed to inspect the final behaviour.
