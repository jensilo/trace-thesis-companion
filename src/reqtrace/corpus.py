import asyncio
import datetime
import json
import statistics
import time
from pathlib import Path
from typing import Any

from rich.console import Console

from reqtrace.config import DEFAULT_EVAL_JUDGE_MODEL, DEFAULT_EXTRACTION_MODEL, OUTPUT_DIR
from reqtrace.extraction import model_short

console = Console(force_terminal=True)

CORPUS_CONCURRENCY = 10


# ── Corpus directory discovery ────────────────────────────────────────────────

def _find_squire_corpus_dir(corpus_arg: str | None) -> Path:
    """Resolve a squire corpus directory from a name, timestamp, path, or None (→ latest)."""
    if corpus_arg is None:
        dirs = [
            d for d in OUTPUT_DIR.iterdir()
            if d.is_dir() and d.name.startswith("corpus-") and "-reeval-" not in d.name
        ]
        if not dirs:
            raise FileNotFoundError("No squire corpus directories found in outputs/")
        return max(dirs, key=lambda d: d.stat().st_mtime)

    p = Path(corpus_arg)
    if p.exists() and p.is_dir():
        return p

    name = corpus_arg if corpus_arg.startswith("corpus-") else f"corpus-{corpus_arg}"
    p = OUTPUT_DIR / name
    if p.exists() and p.is_dir():
        return p

    raise FileNotFoundError(f"Corpus directory not found: {corpus_arg!r}. Looked in {OUTPUT_DIR}/")


def _trace_output_dir(corpus_dir: Path, extraction_model: str) -> Path:
    """Build the trace corpus output directory name."""
    org_ts = corpus_dir.name.removeprefix("corpus-")
    if "-reeval-" in org_ts:
        org_ts = org_ts.split("-reeval-")[0]
    trace_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    short = model_short(extraction_model)
    return OUTPUT_DIR / f"corpus_{org_ts}_{short}_trace_{trace_ts}"


# ── Per-transcript worker ─────────────────────────────────────────────────────

def _extract_one(
    transcript_path: Path,
    extraction_model: str,
    evaluate: bool,
    eval_model: str,
    trace_dir: Path,
    run_id: str,
) -> dict:
    """Blocking: extract (and optionally evaluate) one transcript."""
    from reqtrace.extraction import run_extraction
    from reqtrace.evaluation import run_evaluation

    start = time.time()
    try:
        result = run_extraction(
            transcript_file=transcript_path,
            model=extraction_model,
            output_dir=trace_dir,
            run_id=run_id,
        )

        eval_result = None
        if evaluate:
            eval_result = run_evaluation(
                extraction_result=result,
                eval_model=eval_model,
                output_dir=trace_dir,
            )

        elapsed = round(time.time() - start, 1)
        metrics = eval_result.metrics.model_dump() if eval_result else None

        return {
            "transcript": transcript_path.name,
            "transcript_path": str(transcript_path),
            "extraction_path": str(result.output_path),
            "model": extraction_model,
            "elapsed_seconds": elapsed,
            "extracted_count": len(result.extraction.requirements),
            "metrics": metrics,
        }
    except Exception as e:
        return {
            "transcript": transcript_path.name,
            "transcript_path": str(transcript_path),
            "model": extraction_model,
            "elapsed_seconds": round(time.time() - start, 1),
            "error": str(e),
        }


# ── Async orchestrator ────────────────────────────────────────────────────────

async def _async_extract_corpus(
    corpus_dir: Path,
    extraction_model: str,
    evaluate: bool,
    eval_model: str,
    concurrency: int,
    trace_dir: Path,
) -> list[dict]:
    transcripts = sorted(corpus_dir.glob("transcript_*.txt"))
    total = len(transcripts)

    semaphore = asyncio.Semaphore(concurrency)
    completed = 0
    lock = asyncio.Lock()
    wall_start = time.time()

    async def run_one(tx_path: Path) -> dict:
        nonlocal completed
        run_id = tx_path.stem  # use transcript stem as run_id for traceability
        async with semaphore:
            console.print(f"  [dim]→ {tx_path.name}[/dim]")
            result = await asyncio.to_thread(
                _extract_one,
                tx_path, extraction_model, evaluate, eval_model, trace_dir, run_id,
            )
        async with lock:
            completed += 1
            elapsed = result.get("elapsed_seconds", 0)
            if "error" in result:
                console.print(
                    f"  [red]✗[/red] [{completed}/{total}] {tx_path.stem}  "
                    f"({elapsed:.1f}s)  error: {result['error']}"
                )
            else:
                n = result.get("extracted_count", 0)
                m = result.get("metrics") or {}
                score_str = f"  extracted={n}"
                if m:
                    score_str += f"  P={m.get('precision', 0):.2f}  R={m.get('recall', 0):.2f}  F1={m.get('f1', 0):.2f}"
                console.print(
                    f"  [green]✓[/green] [{completed}/{total}] {tx_path.stem}  "
                    f"({elapsed:.1f}s){score_str}"
                )
        return result

    tasks = [run_one(tx) for tx in transcripts]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    total_elapsed = time.time() - wall_start

    result_dicts: list[dict] = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            result_dicts.append({
                "transcript": transcripts[i].name,
                "transcript_path": str(transcripts[i]),
                "model": extraction_model,
                "error": str(r),
                "elapsed_seconds": 0,
            })
        else:
            result_dicts.append(r)

    successful = [r for r in result_dicts if "error" not in r]
    failed = [r for r in result_dicts if "error" in r]

    console.print()
    console.print(f"[bold]Corpus extraction complete[/bold] in {total_elapsed:.1f}s")
    console.print(f"  Successful: [green]{len(successful)}[/green] / {total}")
    if failed:
        console.print(f"  Failed:     [red]{len(failed)}[/red]")
        for f in failed:
            console.print(f"    [red]✗[/red] {f['transcript']}: {f['error']}")

    return result_dicts


# ── Aggregation ───────────────────────────────────────────────────────────────

def _compute_corpus_meta(
    corpus_id: str,
    trace_run_id: str,
    results: list[dict],
    extraction_model: str,
    eval_model: str | None,
    total_elapsed: float,
) -> dict[str, Any]:
    successful = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]

    def _agg(values: list[float]) -> dict:
        if not values:
            return {"mean": None, "std_dev": None, "count": 0}
        return {
            "mean": round(statistics.mean(values), 3),
            "std_dev": round(statistics.stdev(values), 3) if len(values) > 1 else 0.0,
            "count": len(values),
        }

    def _metrics_stats(subset: list[dict]) -> dict:
        precision, recall, f1 = [], [], []
        confidence, overlap = [], []
        for r in subset:
            m = r.get("metrics") or {}
            if m.get("precision") is not None:
                precision.append(m["precision"])
            if m.get("recall") is not None:
                recall.append(m["recall"])
            if m.get("f1") is not None:
                f1.append(m["f1"])
            if m.get("avg_confidence") is not None:
                confidence.append(m["avg_confidence"])
            if m.get("avg_overlap_score") is not None:
                overlap.append(m["avg_overlap_score"])
        return {
            "count": len(subset),
            "precision": _agg(precision),
            "recall": _agg(recall),
            "f1": _agg(f1),
            "avg_confidence": _agg(confidence),
            "avg_overlap_score": _agg(overlap),
        }

    per_transcript = sorted(
        [
            {
                "transcript": r["transcript"],
                "extracted_count": r.get("extracted_count", 0),
                "elapsed_seconds": r.get("elapsed_seconds"),
                "metrics": r.get("metrics"),
                "error": r.get("error"),
            }
            for r in results
        ],
        key=lambda x: (x.get("metrics") or {}).get("recall", 0.0),
        reverse=True,
    )

    return {
        "corpus_id": corpus_id,
        "trace_run_id": trace_run_id,
        "model": extraction_model,
        "eval_model": eval_model,
        "generated_at": datetime.datetime.now().isoformat(),
        "total_elapsed_seconds": round(total_elapsed, 1),
        "summary": {
            "total": len(results),
            "successful": len(successful),
            "failed": len(failed),
            "failures": [{"transcript": r["transcript"], "error": r.get("error", "")} for r in failed],
        },
        "overall": _metrics_stats(successful),
        "by_model": {extraction_model: _metrics_stats(successful)},
        "per_transcript": per_transcript,
    }


# ── Trace corpus directory discovery ─────────────────────────────────────────

def _find_trace_corpus_dir(trace_corpus_arg: str | None) -> Path:
    """Resolve a trace corpus directory from a name, timestamp, path, or None (→ latest)."""
    if trace_corpus_arg is None:
        dirs = [
            d for d in OUTPUT_DIR.iterdir()
            if d.is_dir() and d.name.startswith("corpus_")
        ]
        if not dirs:
            raise FileNotFoundError("No trace corpus directories found in outputs/")
        return max(dirs, key=lambda d: d.stat().st_mtime)

    p = Path(trace_corpus_arg)
    if p.exists() and p.is_dir():
        return p

    p = OUTPUT_DIR / trace_corpus_arg
    if p.exists() and p.is_dir():
        return p

    raise FileNotFoundError(
        f"Trace corpus directory not found: {trace_corpus_arg!r}. Looked in {OUTPUT_DIR}/"
    )


# ── Per-transcript evaluation worker ─────────────────────────────────────────

def _evaluate_one(extraction_path: Path, eval_model: str, trace_dir: Path) -> dict:
    """Blocking: evaluate one extraction file against its ground truth."""
    from reqtrace.evaluation import run_evaluation
    from reqtrace.models.requirement import TraceExtraction

    start = time.time()
    try:
        raw = json.loads(extraction_path.read_text(encoding="utf-8"))
        extraction = TraceExtraction.model_validate(raw)
        transcript_path = Path(extraction.transcript_file)

        eval_result = run_evaluation(
            extraction_file=extraction_path,
            transcript_file=transcript_path,
            eval_model=eval_model,
            output_dir=trace_dir,
        )
        return {
            "transcript": transcript_path.name,
            "extracted_count": len(extraction.requirements),
            "elapsed_seconds": round(time.time() - start, 1),
            "metrics": eval_result.metrics.model_dump(),
            "error": None,
        }
    except Exception as e:
        return {
            "transcript": extraction_path.stem,
            "extracted_count": 0,
            "elapsed_seconds": round(time.time() - start, 1),
            "metrics": None,
            "error": str(e),
        }


async def _async_evaluate_corpus(
    trace_dir: Path,
    eval_model: str,
    concurrency: int,
) -> list[dict]:
    extraction_files = sorted(
        f for f in trace_dir.glob("trace_*.json")
        if f.name != "trace_corpus_meta.json"
    )
    total = len(extraction_files)

    semaphore = asyncio.Semaphore(concurrency)
    completed = 0
    lock = asyncio.Lock()
    wall_start = time.time()

    async def run_one(ex_path: Path) -> dict:
        nonlocal completed
        async with semaphore:
            console.print(f"  [dim]→ {ex_path.name}[/dim]")
            result = await asyncio.to_thread(_evaluate_one, ex_path, eval_model, trace_dir)
        async with lock:
            completed += 1
            elapsed = result.get("elapsed_seconds", 0)
            if result.get("error"):
                console.print(
                    f"  [red]✗[/red] [{completed}/{total}] {ex_path.stem}  "
                    f"({elapsed:.1f}s)  error: {result['error']}"
                )
            else:
                m = result.get("metrics") or {}
                console.print(
                    f"  [green]✓[/green] [{completed}/{total}] {ex_path.stem}  "
                    f"({elapsed:.1f}s)  "
                    f"P={m.get('precision', 0):.2f}  R={m.get('recall', 0):.2f}  F1={m.get('f1', 0):.2f}"
                )
        return result

    tasks = [run_one(f) for f in extraction_files]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    total_elapsed = time.time() - wall_start

    result_dicts: list[dict] = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            result_dicts.append({
                "transcript": extraction_files[i].stem,
                "extracted_count": 0,
                "elapsed_seconds": 0,
                "metrics": None,
                "error": str(r),
            })
        else:
            result_dicts.append(r)

    successful = [r for r in result_dicts if not r.get("error")]
    failed = [r for r in result_dicts if r.get("error")]

    console.print()
    console.print(f"[bold]Corpus evaluation complete[/bold] in {total_elapsed:.1f}s")
    console.print(f"  Successful: [green]{len(successful)}[/green] / {total}")
    if failed:
        console.print(f"  Failed:     [red]{len(failed)}[/red]")
        for f in failed:
            console.print(f"    [red]✗[/red] {f['transcript']}: {f['error']}")

    return result_dicts


def run_evaluate_corpus(
    trace_corpus: str | None = None,
    eval_model: str = DEFAULT_EVAL_JUDGE_MODEL,
    concurrency: int = CORPUS_CONCURRENCY,
) -> None:
    """Evaluate all extractions in a trace corpus directory and rewrite trace_corpus_meta.json."""
    trace_dir = _find_trace_corpus_dir(trace_corpus)

    extraction_files = [
        f for f in trace_dir.glob("trace_*.json")
        if f.name != "trace_corpus_meta.json"
    ]
    total = len(extraction_files)

    # Read existing meta to preserve corpus_id and model info
    meta_path = trace_dir / "trace_corpus_meta.json"
    existing_meta: dict = {}
    if meta_path.exists():
        existing_meta = json.loads(meta_path.read_text(encoding="utf-8"))

    corpus_id = existing_meta.get("corpus_id", trace_dir.name)
    trace_run_id = existing_meta.get("trace_run_id", trace_dir.name)
    extraction_model = existing_meta.get("model", "unknown")

    console.print(f"\n[bold cyan]══════════ TRACE Corpus Evaluation ══════════[/bold cyan]")
    console.print(f"  Trace corpus: {trace_dir.name}")
    console.print(f"  Extractions:  {total}")
    console.print(f"  Judge:        {eval_model}")
    console.print(f"  Concurrency:  {concurrency}")
    console.print()

    if total == 0:
        console.print("[red]No extraction files found in trace corpus directory.[/red]")
        return

    wall_start = time.time()
    results = asyncio.run(_async_evaluate_corpus(
        trace_dir=trace_dir,
        eval_model=eval_model,
        concurrency=concurrency,
    ))
    total_elapsed = time.time() - wall_start

    console.print("\n[bold cyan]Updating trace corpus meta...[/bold cyan]")
    meta = _compute_corpus_meta(
        corpus_id=corpus_id,
        trace_run_id=trace_run_id,
        results=results,
        extraction_model=extraction_model,
        eval_model=eval_model,
        total_elapsed=existing_meta.get("total_elapsed_seconds", 0) + round(total_elapsed, 1),
    )
    meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    console.print(f"  [bold green]Corpus meta updated:[/bold green] {meta_path}")


# ── Public entry point ────────────────────────────────────────────────────────

def run_extract_corpus(
    corpus: str | None = None,
    extraction_model: str = DEFAULT_EXTRACTION_MODEL,
    evaluate: bool = False,
    eval_model: str = DEFAULT_EVAL_JUDGE_MODEL,
    concurrency: int = CORPUS_CONCURRENCY,
) -> None:
    corpus_dir = _find_squire_corpus_dir(corpus)
    trace_dir = _trace_output_dir(corpus_dir, extraction_model)
    trace_dir.mkdir(parents=True)

    transcripts = list(corpus_dir.glob("transcript_*.txt"))
    total = len(transcripts)
    trace_run_id = trace_dir.name

    console.print(f"\n[bold cyan]══════════ TRACE Corpus Extraction ══════════[/bold cyan]")
    console.print(f"  Corpus:      {corpus_dir.name}")
    console.print(f"  Trace run:   {trace_run_id}")
    console.print(f"  Output:      {trace_dir}")
    console.print(f"  Transcripts: {total}")
    console.print(f"  Extractor:   {extraction_model}")
    if evaluate:
        console.print(f"  Evaluator:   {eval_model}")
    console.print(f"  Concurrency: {concurrency}")
    console.print()

    if total == 0:
        console.print("[red]No transcripts found in corpus directory.[/red]")
        return

    wall_start = time.time()
    results = asyncio.run(_async_extract_corpus(
        corpus_dir=corpus_dir,
        extraction_model=extraction_model,
        evaluate=evaluate,
        eval_model=eval_model,
        concurrency=concurrency,
        trace_dir=trace_dir,
    ))
    total_elapsed = time.time() - wall_start

    console.print("\n[bold cyan]Computing trace corpus meta-analysis...[/bold cyan]")
    meta = _compute_corpus_meta(
        corpus_id=corpus_dir.name,
        trace_run_id=trace_run_id,
        results=results,
        extraction_model=extraction_model,
        eval_model=eval_model if evaluate else None,
        total_elapsed=total_elapsed,
    )

    meta_path = trace_dir / "trace_corpus_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    console.print(f"  [bold green]Corpus meta:[/bold green] {meta_path}")
