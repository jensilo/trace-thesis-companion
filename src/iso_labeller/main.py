import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from iso_labeller.config import (
    DEFAULT_CONCURRENCY,
    DEFAULT_MODEL,
    DEFAULT_RUNS,
    DEFAULT_TEMPERATURE,
    OUTPUT_DIR,
)

app = typer.Typer(name="iso-label", help="ISO/IEC 25010 re-labelling agent for NICE QRs.")
console = Console(force_terminal=True)


@app.command()
def run(
    nice_csv: Annotated[Path, typer.Option("--nice-csv", help="Path to NICE.csv")] = Path(
        "data/NICE.csv"
    ),
    output_dir: Annotated[Path, typer.Option("--output-dir")] = OUTPUT_DIR,
    runs: Annotated[int, typer.Option("--runs", help="Number of independent runs")] = DEFAULT_RUNS,
    concurrency: Annotated[int, typer.Option("--concurrency")] = DEFAULT_CONCURRENCY,
    model: Annotated[str, typer.Option("--model")] = DEFAULT_MODEL,
    temperature: Annotated[float, typer.Option("--temperature")] = DEFAULT_TEMPERATURE,
) -> None:
    """Classify NICE QRs with ISO/IEC 25010:2023 sub-characteristics (n runs)."""
    import csv

    from iso_labeller.labelling import classify_all
    from iso_labeller.models.label import ISOLabelResult, RunOutput
    from iso_labeller.taxonomy import NICE_LABEL_COLUMNS

    if not nice_csv.exists():
        console.print(f"[red]NICE CSV not found: {nice_csv}[/red]")
        raise typer.Exit(code=1)

    with nice_csv.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        all_rows = [row for row in reader if row.get("IsQuality") == "1"]

    console.print(f"Loaded [bold]{len(all_rows)}[/bold] QR rows from {nice_csv}")

    for run_i in range(1, runs + 1):
        console.rule(f"Run {run_i}/{runs}")
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        raw_results = asyncio.run(
            classify_all(all_rows, model, temperature, concurrency)
        )

        results: list[ISOLabelResult] = []
        for row, (sub_chars, raw) in zip(all_rows, raw_results):
            nice_labels = [col for col in NICE_LABEL_COLUMNS if row.get(col) == "1"]
            results.append(
                ISOLabelResult(
                    project_id=int(row["ProjectID"]),
                    requirement_text=row["RequirementText"],
                    nice_labels=nice_labels,
                    iso_sub_chars=sub_chars,
                    raw_response=raw,
                )
            )

        run_output = RunOutput(
            model=model,
            timestamp=timestamp,
            temperature=temperature,
            n_requirements=len(results),
            results=results,
        )

        run_dir = output_dir / f"iso_label_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        out_file = run_dir / "results.json"
        out_file.write_text(run_output.model_dump_json(indent=2), encoding="utf-8")
        console.print(f"[green]✓[/green] Saved {len(results)} results → {out_file}")


@app.command()
def analyse(
    results_dir: Annotated[
        list[Path],
        typer.Option("--results-dir", help="One or more iso_label_* output directories"),
    ],
) -> None:
    """Print per-NICE-label concordance summary across run directories."""
    from collections import defaultdict

    from iso_labeller.taxonomy import NICE_LABEL_COLUMNS, NICE_TO_ISO_CHARS, score

    run_files: list[Path] = []
    for d in results_dir:
        for p in sorted(Path(".").glob(str(d))):
            candidate = p / "results.json" if p.is_dir() else p
            if candidate.exists():
                run_files.append(candidate)

    if not run_files:
        console.print("[red]No results.json files found.[/red]")
        raise typer.Exit(code=1)

    console.print(f"Loading {len(run_files)} run file(s)…")

    label_jaccards: dict[str, list[float]] = defaultdict(list)
    label_recalls: dict[str, list[float]] = defaultdict(list)
    label_misses: dict[str, list[bool]] = defaultdict(list)

    for run_file in run_files:
        data = json.loads(run_file.read_text(encoding="utf-8"))
        for item in data["results"]:
            s = score(item["iso_sub_chars"], item["nice_labels"])
            if not s["scoreable"]:
                continue
            for label in item["nice_labels"]:
                if label not in NICE_TO_ISO_CHARS:
                    continue
                label_jaccards[label].append(s["jaccard"])
                label_recalls[label].append(s["recall"])
                label_misses[label].append(s["complete_miss"])

    table = Table(title="NICE→ISO Concordance Summary", show_lines=False)
    table.add_column("NICE Label", style="bold")
    table.add_column("Mean J", justify="right")
    table.add_column("SD J", justify="right")
    table.add_column("Mean R", justify="right")
    table.add_column("Miss%", justify="right")
    table.add_column("n", justify="right")

    for label in NICE_LABEL_COLUMNS:
        jacs = label_jaccards.get(label)
        if not jacs:
            continue
        import statistics

        mean_j = statistics.mean(jacs)
        sd_j = statistics.stdev(jacs) if len(jacs) > 1 else 0.0
        mean_r = statistics.mean(label_recalls[label])
        miss_rate = sum(label_misses[label]) / len(label_misses[label]) * 100
        color = "green" if mean_j >= 0.7 else ("yellow" if mean_j >= 0.4 else "red")
        table.add_row(
            label,
            f"[{color}]{mean_j:.3f}[/{color}]",
            f"{sd_j:.3f}",
            f"{mean_r:.3f}",
            f"{miss_rate:.1f}%",
            str(len(jacs)),
        )

    console.print(table)
