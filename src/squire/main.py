from dotenv import load_dotenv

load_dotenv()

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from squire.config import (
    DEFAULT_QUALITY_JUDGE_MODEL,
    DEFAULT_META_JUDGE_MODEL,
    DEFAULT_QUALITY_SAMPLES,
    DEFAULT_META_SAMPLES,
    DEFAULT_SYNTHESIS_MODEL,
    DEFAULT_SUMMARIZE_DRAFT_MODEL,
    DEFAULT_SUMMARIZE_AGGREGATE_MODEL,
    DEFAULT_SUMMARIZE_SAMPLES,
    SYNTHESIS_MODEL_ALIASES,
    JUDGE_MODEL_ALIASES,
)

app = typer.Typer(
    name="squire",
    help="SQuIRE — Synthetic Quality Interview Requirements Elicitation",
)
console = Console(force_terminal=True)


def _resolve_synthesis_model(model: str, synther: Optional[str]) -> str:
    if synther:
        resolved = SYNTHESIS_MODEL_ALIASES.get(synther.lower())
        if not resolved:
            raise ValueError(f"Unknown --synther '{synther}'. Valid aliases: {list(SYNTHESIS_MODEL_ALIASES.keys())}")
        return resolved
    return model


def _resolve_judge_models(
    quality_judge: str,
    meta_judge: str,
    evaler: Optional[str],
) -> tuple[str, str]:
    if evaler:
        resolved = JUDGE_MODEL_ALIASES.get(evaler.lower())
        if not resolved:
            raise ValueError(f"Unknown --evaler '{evaler}'. Valid aliases: {list(JUDGE_MODEL_ALIASES.keys())}")
        return resolved, resolved
    return quality_judge, meta_judge


@app.command()
def run(
    project_id: str = typer.Option(..., "--project-id", "-p", help="Project ID to synthesize for"),
    model: str = typer.Option(DEFAULT_SYNTHESIS_MODEL, "--model", "-m", help="LLM for synthesis (full model name)"),
    synther: Optional[str] = typer.Option(None, "--synther", help="Synthesis model alias: 'gpt' or 'claude' (overrides --model)"),
    stakeholder: Optional[str] = typer.Option(None, "--stakeholder", "-s", help="Stakeholder persona ID"),
    interviewer: Optional[str] = typer.Option(None, "--interviewer", "-i", help="Interviewer persona ID"),
    single_shot: bool = typer.Option(False, "--single-shot", help="Use single-shot generation instead of multi-step"),
    refine: bool = typer.Option(False, "--refine", help="Enable refinement step (4-step pipeline; default: 3-step)"),
    evaluate_after: bool = typer.Option(False, "--evaluate", "-e", help="Run evaluation after synthesis"),
    quality_judge: str = typer.Option(DEFAULT_QUALITY_JUDGE_MODEL, "--quality-judge", help="Judge model for interview technique evaluation"),
    meta_judge: str = typer.Option(DEFAULT_META_JUDGE_MODEL, "--meta-judge", help="Judge model for meta evaluation (completeness + realism)"),
    evaler: Optional[str] = typer.Option(None, "--evaler", help="Evaluation model alias: 'gpt' or 'claude' (sets both quality and meta judges)"),
    quality_samples: int = typer.Option(DEFAULT_QUALITY_SAMPLES, "--quality-samples", help="Sample count for quality G-Eval"),
    meta_samples: int = typer.Option(DEFAULT_META_SAMPLES, "--meta-samples", help="Sample count for meta G-Eval"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug output"),
):
    """Synthesize a requirements elicitation interview transcript."""
    from squire.synthesis import run_synthesis

    try:
        resolved_model = _resolve_synthesis_model(model, synther)
    except ValueError as e:
        console.print(f"[bold red]{e}[/bold red]")
        raise typer.Exit(code=1)

    try:
        result = run_synthesis(
            project_id=project_id,
            model=resolved_model,
            stakeholder_id=stakeholder,
            interviewer_id=interviewer,
            single_shot=single_shot,
            no_refine=not refine,
            debug=debug,
        )
    except (ValueError, RuntimeError) as e:
        console.print(f"[bold red]{e}[/bold red]")
        raise typer.Exit(code=1)

    if evaluate_after:
        from squire.evaluation import run_evaluation

        try:
            resolved_quality, resolved_meta = _resolve_judge_models(quality_judge, meta_judge, evaler)
        except ValueError as e:
            console.print(f"[bold red]{e}[/bold red]")
            raise typer.Exit(code=1)

        console.print("\n[bold cyan]═══ Evaluation ═══[/bold cyan]")
        try:
            run_evaluation(
                transcript_file=result.transcript_path,
                quality_model=resolved_quality,
                meta_model=resolved_meta,
                quality_samples=quality_samples,
                meta_samples=meta_samples,
            )
        except (ValueError, FileNotFoundError) as e:
            console.print(f"[bold red]{e}[/bold red]")
            raise typer.Exit(code=1)


@app.command()
def evaluate(
    transcript_file: Optional[Path] = typer.Option(None, "--transcript", "-f", help="Transcript path (auto-discovers latest if omitted)"),
    quality_judge: str = typer.Option(DEFAULT_QUALITY_JUDGE_MODEL, "--quality-judge", help="Judge model for interview technique evaluation"),
    meta_judge: str = typer.Option(DEFAULT_META_JUDGE_MODEL, "--meta-judge", help="Judge model for meta evaluation (completeness + realism)"),
    evaler: Optional[str] = typer.Option(None, "--evaler", help="Evaluation model alias: 'gpt' or 'claude' (sets both quality and meta judges)"),
    quality_samples: int = typer.Option(DEFAULT_QUALITY_SAMPLES, "--quality-samples", help="Sample count for quality G-Eval"),
    meta_samples: int = typer.Option(DEFAULT_META_SAMPLES, "--meta-samples", help="Sample count for meta G-Eval"),
):
    """Evaluate an existing transcript with G-Eval (quality + meta) and BERTScore."""
    from squire.evaluation import run_evaluation

    try:
        resolved_quality, resolved_meta = _resolve_judge_models(quality_judge, meta_judge, evaler)
    except ValueError as e:
        console.print(f"[bold red]{e}[/bold red]")
        raise typer.Exit(code=1)

    try:
        run_evaluation(
            transcript_file=transcript_file,
            quality_model=resolved_quality,
            meta_model=resolved_meta,
            quality_samples=quality_samples,
            meta_samples=meta_samples,
        )
    except (ValueError, FileNotFoundError) as e:
        console.print(f"[bold red]{e}[/bold red]")
        raise typer.Exit(code=1)


@app.command()
def personas():
    """List available personas."""
    from squire.ingestion import load_personas

    all_personas = load_personas()
    interviewers = [p for p in all_personas if p.role == "Interviewer"]
    stakeholders = [p for p in all_personas if p.role == "Interviewee"]

    console.print("\n[bold]Interviewers:[/bold]")
    for p in interviewers:
        console.print(f"  [cyan]{p.id}[/cyan] — {p.name}")

    console.print("\n[bold]Stakeholders:[/bold]")
    for p in stakeholders:
        console.print(f"  [cyan]{p.id}[/cyan] — {p.name}")
    console.print()


@app.command("synthesize-corpus")
def synthesize_corpus(
    model: str = typer.Option(DEFAULT_SYNTHESIS_MODEL, "--model", "-m", help="LLM for synthesis (full model name)"),
    synther: Optional[str] = typer.Option(None, "--synther", help="Synthesis model alias: 'gpt' or 'claude' (overrides --model)"),
    single_shot: bool = typer.Option(False, "--single-shot", help="Use single-shot generation instead of multi-step"),
    refine: bool = typer.Option(False, "--refine", help="Enable refinement step (4-step pipeline; default: 3-step)"),
    quality_judge: str = typer.Option(DEFAULT_QUALITY_JUDGE_MODEL, "--quality-judge", help="Judge model for interview technique evaluation"),
    meta_judge: str = typer.Option(DEFAULT_META_JUDGE_MODEL, "--meta-judge", help="Judge model for meta evaluation (completeness + realism)"),
    evaler: Optional[str] = typer.Option(None, "--evaler", help="Evaluation model alias: 'gpt' or 'claude' (sets both quality and meta judges)"),
    quality_samples: int = typer.Option(5, "--quality-samples", help="Sample count for quality G-Eval"),
    meta_samples: int = typer.Option(5, "--meta-samples", help="Sample count for meta G-Eval"),
    concurrency: int = typer.Option(10, "--concurrency", "-c", help="Max simultaneous synthesis jobs"),
):
    """Synthesize and evaluate a full corpus: 5 projects × 2 interviewers × 4 stakeholders = 40 transcripts."""
    from squire.corpus import run_synthesize_corpus

    try:
        resolved_model = _resolve_synthesis_model(model, synther)
    except ValueError as e:
        console.print(f"[bold red]{e}[/bold red]")
        raise typer.Exit(code=1)

    try:
        resolved_quality, resolved_meta = _resolve_judge_models(quality_judge, meta_judge, evaler)
    except ValueError as e:
        console.print(f"[bold red]{e}[/bold red]")
        raise typer.Exit(code=1)

    try:
        run_synthesize_corpus(
            synthesis_model=resolved_model,
            quality_model=resolved_quality,
            meta_model=resolved_meta,
            single_shot=single_shot,
            no_refine=not refine,
            quality_samples=quality_samples,
            meta_samples=meta_samples,
            concurrency=concurrency,
        )
    except (ValueError, RuntimeError, FileNotFoundError) as e:
        console.print(f"[bold red]{e}[/bold red]")
        raise typer.Exit(code=1)


@app.command("reevaluate-corpus")
def reevaluate_corpus(
    corpus: Optional[str] = typer.Option(None, "--corpus", "-c", help="Corpus name, timestamp, or path (defaults to latest original corpus)"),
    quality_judge: str = typer.Option(DEFAULT_QUALITY_JUDGE_MODEL, "--quality-judge", help="Judge model for interview technique evaluation"),
    meta_judge: str = typer.Option(DEFAULT_META_JUDGE_MODEL, "--meta-judge", help="Judge model for meta evaluation (completeness + realism)"),
    evaler: Optional[str] = typer.Option(None, "--evaler", help="Evaluation model alias: 'gpt' or 'claude' (sets both quality and meta judges)"),
    quality_samples: int = typer.Option(5, "--quality-samples", help="Sample count for quality G-Eval"),
    meta_samples: int = typer.Option(5, "--meta-samples", help="Sample count for meta G-Eval"),
    concurrency: int = typer.Option(10, "--concurrency", "-n", help="Max simultaneous evaluation jobs"),
    no_bert: bool = typer.Option(False, "--no-bert", help="Skip BERTScore computation (faster; use when re-eval focus is G-Eval stability)"),
):
    """Re-evaluate all transcripts in an existing corpus. Writes logs + corpus_meta.json + corpus_meta_diff.json to a new corpus-{id}-reeval-{ts}/ directory."""
    from squire.corpus import run_reevaluate_corpus

    try:
        resolved_quality, resolved_meta = _resolve_judge_models(quality_judge, meta_judge, evaler)
    except ValueError as e:
        console.print(f"[bold red]{e}[/bold red]")
        raise typer.Exit(code=1)

    try:
        run_reevaluate_corpus(
            corpus=corpus,
            quality_model=resolved_quality,
            meta_model=resolved_meta,
            quality_samples=quality_samples,
            meta_samples=meta_samples,
            concurrency=concurrency,
            skip_bert=no_bert,
        )
    except (ValueError, FileNotFoundError) as e:
        console.print(f"[bold red]{e}[/bold red]")
        raise typer.Exit(code=1)


@app.command("summarize-projects")
def summarize_projects(
    project_id: Optional[str] = typer.Option(None, "--project-id", "-p", help="Single project ID to summarize (default: all 15)"),
    samples: int = typer.Option(DEFAULT_SUMMARIZE_SAMPLES, "--samples", "-n", help="Number of draft summaries per project"),
    draft_model: str = typer.Option(DEFAULT_SUMMARIZE_DRAFT_MODEL, "--draft-model", help="Model for per-project draft summaries"),
    aggregate_model: str = typer.Option(DEFAULT_SUMMARIZE_AGGREGATE_MODEL, "--aggregate-model", help="Model for aggregating drafts into final summary"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite existing entries in project_summaries.json"),
):
    """Summarize NICE dataset projects from their requirements and save to data/project_summaries.json."""
    from squire.summarize import run_summarize_projects

    try:
        run_summarize_projects(
            project_id=project_id,
            samples=samples,
            draft_model=draft_model,
            aggregate_model=aggregate_model,
            overwrite=overwrite,
        )
    except (ValueError, FileNotFoundError) as e:
        console.print(f"[bold red]{e}[/bold red]")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
