# Granite SWE-bench Agent

A local LLM-based software engineering agent for repository-level bug fixing on SWE-bench Verified tasks, built with Python and an open-weight Granite model served through Ollama.

The project focuses on understanding how a relatively small local model behaves in agentic software engineering workflows, especially around code retrieval, inspection, patch generation, validation, and recovery from failed actions.

## Motivation

Most agentic software engineering systems are designed around large hosted models. This project explores how far a smaller local model can go when the surrounding agent is carefully structured.

The main focus was not only whether the model could solve benchmark issues, but also understanding why it failed and iteratively improving the agent based on those failure modes.

## Agent Design

The agent evolved across multiple development versions and includes:

- Repository-level search and relevant-file retrieval
- Traceback-aware context selection
- Structured inspect/edit actions
- Duplicate-inspection protection
- Line-number normalization
- Patch and syntax validation
- No-op and malformed-edit detection
- Bounded recovery logic
- Step limits to reduce repetitive action loops

The final version uses a two-stage interaction process in which the model first inspects relevant code and then proposes structured edits.

## Local Model Setup

The agent was evaluated using:

- **Model:** Granite 4.2 8B
- **Serving:** Ollama
- **Temperature:** 0
- **Context size:** 8,192 tokens
- **Execution:** Fully local

The project was developed and tested under limited local hardware resources, which influenced model size and evaluation strategy.

## SWE-bench Verified Evaluation

The agent was evaluated on 10 SWE-bench Verified issues selected to be practical for local experimentation.

Each failed run was treated as debugging evidence and used to improve later versions of the agent. Improvements included changes to retrieval, action control, validation, context selection, and recovery behaviour.

The frozen final version produced locally validated repair candidates for 3 of the 10 tasks.

Some earlier development versions successfully repaired additional tasks that the final version did not, showing that added control and recovery logic did not uniformly improve performance across all issues.

These results are **local validation results only** and should not be interpreted as an official SWE-bench leaderboard score.

## Repository Structure

```text
granite-swebench-agent/
├── README.md
├── report.pdf
├── code/
│   ├── granite_swe_agent_FINAL.py
│   └── retrieval_generic.py
├── results/
│   ├── 01_django-10880/
│   ├── 02_sklearn-14141/
│   ├── ...
│   └── 10_xarray-3677/
└── development_history/
    ├── agent_versions/
    └── results/
