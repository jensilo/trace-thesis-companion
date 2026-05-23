from dotenv import load_dotenv

load_dotenv()

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from reqtrace.config import (
    DEFAULT_EVAL_JUDGE_MODEL,
    DEFAULT_EXTRACTION_MODEL,
    EXTRACTION_MODEL_ALIASES,
)

app = typer.Typer(
    name="trace",
    help="TRACE — Trusted Requirements Automated Capture & Extraction",
)
console = Console(force_terminal=True)


def _resolve_model(model: str, extractor: Optional[str]) -> str:
    if extractor:
        resolved = EXTRACTION_MODEL_ALIASES.get(extractor.lower())
        if not resolved:
            raise ValueError(
                f"Unknown --extractor '{extractor}'. "
                f"Valid aliases: {list(EXTRACTION_MODEL_ALIASES.keys())}"
            )
        return resolved
    return model


@app.command()
def extract(
    transcript: Path = typer.Argument(..., help="Path to transcript .txt file"),
    model: str = typer.Option(DEFAULT_EXTRACTION_MODEL, "--model", "-m", help="LiteLLM model string for extraction"),
    extractor: Optional[str] = typer.Option(None, "--extractor", help="Model alias: tiny|small|medium|large (overrides --model)"),
    evaluate: bool = typer.Option(False, "--evaluate", "-e", help="Run evaluation after extraction"),
    ground_truth: Optional[Path] = typer.Option(None, "--ground-truth", "-g", help="Ground truth JSON file (auto-discovers sidecar if omitted)"),
    eval_model: str = typer.Option(DEFAULT_EVAL_JUDGE_MODEL, "--eval-model", help="Judge model for evaluation"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Print draft plan to console"),
):
    """Extract quality requirements from a transcript file."""
    from reqtrace.extraction import run_extraction

    try:
        resolved_model = _resolve_model(model, extractor)
    except ValueError as e:
        console.print(f"[bold red]{e}[/bold red]")
        raise typer.Exit(code=1)

    if not transcript.exists():
        console.print(f"[bold red]Transcript not found: {transcript}[/bold red]")
        raise typer.Exit(code=1)

    try:
        result = run_extraction(transcript_file=transcript, model=resolved_model, debug=debug)
    except Exception as e:
        console.print(f"[bold red]Extraction failed: {e}[/bold red]")
        raise typer.Exit(code=1)

    if evaluate:
        from reqtrace.evaluation import run_evaluation

        try:
            run_evaluation(
                extraction_result=result,
                ground_truth_override=ground_truth,
                eval_model=eval_model,
            )
        except FileNotFoundError as e:
            console.print(f"[bold red]{e}[/bold red]")
            raise typer.Exit(code=1)
        except Exception as e:
            console.print(f"[bold red]Evaluation failed: {e}[/bold red]")
            raise typer.Exit(code=1)


@app.command("extract-corpus")
def extract_corpus(
    corpus: Optional[str] = typer.Option(None, "--corpus", "-c", help="Corpus name, timestamp, or path (defaults to latest squire corpus)"),
    model: str = typer.Option(DEFAULT_EXTRACTION_MODEL, "--model", "-m", help="LiteLLM model string for extraction"),
    extractor: Optional[str] = typer.Option(None, "--extractor", help="Model alias: tiny|small|medium|large (overrides --model)"),
    evaluate: bool = typer.Option(False, "--evaluate", "-e", help="Run evaluation after each extraction"),
    eval_model: str = typer.Option(DEFAULT_EVAL_JUDGE_MODEL, "--eval-model", help="Judge model for evaluation"),
    concurrency: int = typer.Option(5, "--concurrency", "-n", help="Max simultaneous extraction jobs"),
):
    """Extract requirements from all transcripts in a corpus."""
    from reqtrace.corpus import run_extract_corpus

    try:
        resolved_model = _resolve_model(model, extractor)
    except ValueError as e:
        console.print(f"[bold red]{e}[/bold red]")
        raise typer.Exit(code=1)

    try:
        run_extract_corpus(
            corpus=corpus,
            extraction_model=resolved_model,
            evaluate=evaluate,
            eval_model=eval_model,
            concurrency=concurrency,
        )
    except (ValueError, FileNotFoundError) as e:
        console.print(f"[bold red]{e}[/bold red]")
        raise typer.Exit(code=1)


@app.command("evaluate-corpus")
def evaluate_corpus(
    trace_corpus: Optional[str] = typer.Option(None, "--trace-corpus", "-t", help="Trace corpus name or path (defaults to latest)"),
    eval_model: str = typer.Option(DEFAULT_EVAL_JUDGE_MODEL, "--eval-model", help="Judge model for evaluation"),
    concurrency: int = typer.Option(5, "--concurrency", "-n", help="Max simultaneous evaluation jobs"),
):
    """Evaluate all extractions in a trace corpus and rewrite trace_corpus_meta.json."""
    from reqtrace.corpus import run_evaluate_corpus

    try:
        run_evaluate_corpus(
            trace_corpus=trace_corpus,
            eval_model=eval_model,
            concurrency=concurrency,
        )
    except (ValueError, FileNotFoundError) as e:
        console.print(f"[bold red]{e}[/bold red]")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
