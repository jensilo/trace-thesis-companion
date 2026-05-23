# TRACE Thesis — Codebase Guide

## Tech Stack

| Concern | Library |
|---------|---------|
| Package manager | `uv` (`uv add`, `uv run`) |
| CLI | `typer` + `rich` |
| Agent orchestration | `langroid` (with `litellm` backend via `langroid[litellm]`) |
| Direct LLM calls | `litellm.completion()` — use for simple prompt/response loops |
| Data modelling | `pydantic` v2 |
| Semantic similarity | `bert-score` |

Use `langroid` agents only when multi-turn or tool-using agent behaviour is needed. For single-shot or repeated prompt/response patterns, use `litellm` directly.

## Packages

| Package | Entry point | Purpose |
|---------|-------------|---------|
| `src/squire/` | `uv run squire` | SQuIRE: generates synthetic interview transcripts from NICE requirements |
| `src/reqtrace/` | `uv run trace` | TRACE: extracts and classifies quality requirements from transcripts |
| `src/iso_labeller/` | `uv run iso-label` | Re-labels extracted requirements against ISO/IEC 25010 (analysis only) |

**No cross-package imports** between `squire` and `reqtrace`. They share `outputs/` on disk only. `reqtrace/utils.py` is an independent copy of `squire/utils.py`.

## Code Conventions

- **Python 3.11+** throughout. Use type hints, `pathlib`, and f-strings.
- **Model names** use the litellm format: `anthropic/claude-haiku-4-5`, `mistral/mistral-small-latest`, `openai/gpt-5-mini`.
- **Project IDs** are strings (`"1"` through `"15"`), matching the `ProjectID` column in NICE.csv.
- **Paths**: use `DATA_DIR` / `OUTPUT_DIR` constants from `config.py` — never hardcode paths.
- **Comments**: only write comments explaining *why* a non-obvious decision was made. Never comment what the code does — well-named identifiers do that.
- **Validation**: use `pydantic` at data boundaries. Do not add defensive error handling for scenarios that cannot occur.

## Architecture Overview

### SQuIRE (`squire` CLI)

`src/squire/main.py` is a thin router; all logic lives in dedicated modules.

- **`squire run`** — single transcript synthesis via `synthesis.py`. Uses `ScriptwriterAgent` (langroid). Default: three-step CoT (Selection → Outline → Generation). `--single-shot` skips the outline step.
- **`squire evaluate`** — G-Eval + BERTScore via `evaluation.py`. Two independent passes: quality (structuring, clarity, responsiveness, rigor) and meta (completeness, realism). Uses `litellm` directly.
- **`squire synthesize-corpus`** — 40 transcripts (5 projects × 2 interviewers × 4 stakeholders), async parallel via `corpus.py`.

### TRACE (`trace` CLI)

Package is named `reqtrace` — `trace` conflicts with the Python stdlib `trace` module.

- **`trace extract`** — two-step extraction pipeline via `extraction.py`: (1) draft plan, (2) structured JSON extraction with CoT + citation verification. Default model: `mistral/mistral-small-latest`.
- **`trace extract-corpus`** — parallel corpus extraction via `corpus.py`. Writes to `outputs/corpus_{corpus-ts}_{model}_trace_{trace-ts}/`.
- **`--evaluate` flag** — BERTScore + LLM judge scoring, 1:1 GT matching constraint, outputs precision/recall/F1.

## Key Output Fields

**TRACE extraction** (`src/reqtrace/models/requirement.py`):
`requirement`, `source_citation`, `citation_verified`, `citation_warning`, `classification`, `quality_attribute`, `confidence` (1--5), `rationale`, `follow_up_question`

**TRACE evaluation** (`src/reqtrace/models/evaluation.py`):
`overlap_score` (1--5), `quality_attribute_match`, `is_hit` (overlap ≥ 3), `is_duplicate` (1:1 constraint, conservative lower bound)

## Analysis

`notebooks/thesis_analysis.ipynb` — 65 cells. Read cell 04 first for path constants and the TRACE runs manifest (single source of truth for which runs are in scope).

```bash
cp notebooks/.env.dist notebooks/.env  # fill in HF_TOKEN
uv run jupyter notebook notebooks/thesis_analysis.ipynb
```
