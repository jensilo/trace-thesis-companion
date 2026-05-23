# TRACE — Trusted Requirements Automated Capture & Extraction

Companion repository for the master's thesis _Automated Extraction of Quality Requirements from Interview Transcripts: An Evaluation of LLM-driven Techniques_ (Karlstad University, 2026).

## Overview

Two artifacts are developed and evaluated in the thesis:

- **SQuIRE** (Synthetic Quality Interview Requirements Elicitation) — generates synthetic interview transcripts grounded in real requirements from the NICE dataset.
- **TRACE** (Trusted Requirements Automated Capture & Extraction) — extracts quality requirements from interview transcripts, classifies them by quality attribute, provides source citations, and assigns confidence scores.

The quantitative analysis comparing the two synthesis pipelines and multiple extraction configurations is in `notebooks/thesis_analysis.ipynb`.

## Repository Structure

```text
.
├── data/                           Input data
│   ├── NICE.csv                    NICE dataset (NFR corpus, multi-label)
│   ├── personas.json               Interviewer and stakeholder persona definitions
│   └── project_summaries.json      Project domain summaries used for synthesis
│
├── notebooks/
│   ├── thesis_analysis.ipynb       Full quantitative analysis (65 cells)
│   ├── generate_thesis_figures.py  Exports thesis-ready figures from analysis data
│   ├── tsne_coords.npz             Pre-computed embedding coordinates
│   ├── tsne_meta.json              Metadata for embedding visualisations
│   └── .env.dist                   HF token template (needed for embedding model)
│
├── outputs/
│   ├── corpus-20260227_105946/     Two-step SQuIRE corpus (40 transcripts)
│   ├── corpus-20260226_162106/     Three-step SQuIRE corpus (40 transcripts)
│   ├── corpus_..._trace_.../       TRACE extraction runs (10 in scope)
│   └── iso_label_.../              ISO/IEC 25010 re-labelling runs (5)
│
├── src/
│   ├── squire/                     SQuIRE synthesis tool
│   ├── reqtrace/                   TRACE extraction tool
│   └── iso_labeller/               ISO/IEC 25010 re-labelling tool (analysis only)
│
├── .env.example                    API key template (OpenAI, Anthropic, Mistral)
├── pyproject.toml
└── uv.lock
```

## Setup

Requirements: Python 3.11+, [uv](https://github.com/astral-sh/uv)

```bash
uv sync
cp .env.example .env
# Fill in OPENAI_API_KEY, ANTHROPIC_API_KEY, and MISTRAL_API_KEY
```

Three CLI entry points are installed:

```bash
uv run squire      # SQuIRE synthesis
uv run trace       # TRACE extraction
uv run iso-label   # ISO/IEC 25010 re-labelling
```

## Usage

### SQuIRE

```
Usage: squire [OPTIONS] COMMAND [ARGS]...

  SQuIRE — Synthetic Quality Interview Requirements Elicitation

Commands:
  run                 Synthesize a requirements elicitation interview transcript.
  evaluate            Evaluate an existing transcript with G-Eval (quality +
                      meta) and BERTScore.
  personas            List available personas.
  synthesize-corpus   Synthesize and evaluate a full corpus: 5 projects × 2
                      interviewers × 4 stakeholders = 40 transcripts.
  reevaluate-corpus   Re-evaluate all transcripts in an existing corpus.
  summarize-projects  Summarize NICE dataset projects from their requirements
                      and save to data/project_summaries.json.
```

Run `uv run squire COMMAND --help` for options on any subcommand.

### TRACE

```
Usage: trace [OPTIONS] COMMAND [ARGS]...

  TRACE — Trusted Requirements Automated Capture & Extraction

Commands:
  extract           Extract quality requirements from a transcript file.
  extract-corpus    Extract requirements from all transcripts in a corpus.
  evaluate-corpus   Evaluate all extractions in a trace corpus and rewrite
                    trace_corpus_meta.json.
```

Run `uv run trace COMMAND --help` for options on any subcommand.

### ISO/IEC 25010 re-labelling

This tool was used for a post-hoc construct validity analysis and is not needed to reproduce the main extraction results.

```
Usage: iso-label [OPTIONS] COMMAND [ARGS]...

  ISO/IEC 25010 re-labelling agent for NICE QRs.

Commands:
  run      Classify NICE QRs with ISO/IEC 25010:2023 sub-characteristics (n runs).
  analyse  Print per-NICE-label concordance summary across run directories.
```

## Reproducing the Analysis

The analysis notebook requires a Hugging Face token to load the Gemma embedding model used for transcript similarity analysis:

```bash
cp notebooks/.env.dist notebooks/.env
# Fill in HF_TOKEN with your Hugging Face API token
```

Then open the notebook:

```bash
uv run jupyter notebook notebooks/thesis_analysis.ipynb
```

All paths in the notebook are relative to `notebooks/` and resolve against `outputs/` and `data/` at the repo root. No additional path configuration is needed.

To regenerate thesis figures from pre-computed data:

```bash
uv run python notebooks/generate_thesis_figures.py
```

## Outputs

### SQuIRE Corpora

| Directory                | Pipeline                                      | Transcripts |
| ------------------------ | --------------------------------------------- | ----------- |
| `corpus-20260227_105946` | Two-step (Selection + Generation)             | 40          |
| `corpus-20260226_162106` | Three-step (Selection + Outline + Generation) | 40          |

Each corpus covers 5 projects × 2 interviewer personas × 4 stakeholder personas. Ground truth requirements are sourced from the NICE dataset.

### TRACE Extraction Runs

Ten runs in scope, covering a temperature sensitivity study (T = 0.0, 0.3, 0.5, 0.7), a cross-model comparison (Mistral Small, Mistral Medium, GPT-5-mini), and three primary stability replications at T = 0.5. Both SQuIRE corpora are covered. See notebook for the full manifest with metadata.

### ISO/IEC 25010 Re-labelling Runs

Five runs re-labelling extracted requirements against the ISO/IEC 25010 quality taxonomy, used to assess construct validity of the NICE label taxonomy.

## Data

The NICE dataset (`data/NICE.csv`) is sourced from:

> Rejithkumar, G., & Anish, P. R. (2025). _[Dataset] NICE: Non-Functional Requirements Identification, Classification, and Explanation Using Small Language Models_ [Data set]. Zenodo. <https://doi.org/10.5281/zenodo.14590935>

The dataset is redistributed here for reproducibility. Please cite the original work if you use it.

## How to Cite

If you use this code or data in your own work, please cite:

> Heise, J. (2026). _Automated Extraction of Quality Requirements from Interview Transcripts: An Evaluation of LLM-driven Techniques_. Master's thesis, Karlstad University.

```bibtex
@mastersthesis{heise2026trace,
  author  = {Heise, Jens},
  title   = {Automated Extraction of Quality Requirements from Interview
             Transcripts: An Evaluation of {LLM}-driven Techniques},
  school  = {Karlstad University},
  year    = {2026}
}
```

Author: [Jens Heise](https://github.com/jensilo)
