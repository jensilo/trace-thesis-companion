import json
import re
from typing import Optional

import litellm
from rich.console import Console

from squire.config import (
    DATA_DIR,
    DEFAULT_SUMMARIZE_AGGREGATE_MODEL,
    DEFAULT_SUMMARIZE_DRAFT_MODEL,
    DEFAULT_SUMMARIZE_SAMPLES,
    SUMMARIZE_TEMPERATURE,
)
from squire.ingestion import load_requirements, get_requirements_by_project
from squire.utils import with_retry

console = Console(force_terminal=True)


def _get_all_project_ids() -> list[str]:
    """Return sorted unique project IDs from NICE.csv."""
    requirements = load_requirements()
    return sorted(set(r.project_id for r in requirements), key=lambda x: int(x))


def _draft_summary(project_id: str, requirements_text: str, model: str) -> str:
    """Call the draft model once to produce a ~100-word project summary."""
    prompt = (
        f"You are an experienced business analyst. Below are requirements for a software project.\n"
        "Recreate the initial brief summary for the inquiry to develop this project.\n"
        "Do not include specific requirements or quality metrics, instead just provide the abstract, "
        "high-level idea of the project. The system does not yet exist.\n\n"
        "Read the requirements carefully and summarize what this project is about in a factual, concise manner.\n\n"
        "Your summary must:\n"
        "- Be based SOLELY on the requirements listed below. Do NOT invent information.\n"
        "- Be truthful and factual — only state what is evident from the requirements.\n"
        "- Be no longer than 100 words.\n"
        "- Cut out implementation details, and technical details connected to specific requirements, keeping just broad strokes of the system.\n"
        "- Describe the system's goal, purpose and high-level functionality, without stating any explicit requirements or metrics.\n"
        "- Keep it deliberately abstract.\n\n"
        f"Requirements:\n{requirements_text}\n\n"
        "Write your summary now."
    )
    response = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=SUMMARIZE_TEMPERATURE,
    )
    return response.choices[0].message.content.strip()


def _aggregate_summaries(
    project_id: str, drafts: list[str], model: str
) -> dict[str, str]:
    """Combine n draft summaries into a single title (≤10 words) + description (≤100 words)."""
    joined = "\n\n---\n\n".join(f"Summary {i + 1}:\n{s}" for i, s in enumerate(drafts))
    prompt = (
        f"You are an experienced business analyst. You have {len(drafts)} draft summaries of the same "
        f"software project.\n"
        "Synthesize them into:\n"
        "1. A concise TITLE of at most 10 words capturing the project's core purpose.\n"
        "2. A unified DESCRIPTION of at most 100 words accurately representing the project.\n\n"
        "Rules:\n"
        "- Only include information present in the drafts. Do NOT invent new information.\n"
        "- Be factual and precise.\n"
        "- The description should cover the system's goal, purpose and high-level functionality.\n\n"
        f"Draft summaries:\n{joined}\n\n"
        'Respond in JSON exactly like this (no markdown fences):\n'
        '{"title": "...", "description": "..."}'
    )
    response = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=SUMMARIZE_TEMPERATURE,
    )
    content = response.choices[0].message.content.strip()

    # Strip optional markdown code fences
    content = re.sub(r"^```(?:json)?\s*", "", content)
    content = re.sub(r"\s*```$", "", content)

    data = json.loads(content)
    return {"title": data["title"].strip(), "description": data["description"].strip()}


def _save_summaries(path, existing: dict) -> None:
    sorted_summaries = [
        existing[pid] for pid in sorted(existing.keys(), key=lambda x: int(x))
    ]
    path.write_text(json.dumps(sorted_summaries, indent=4), encoding="utf-8")


def run_summarize_projects(
    project_id: Optional[str] = None,
    samples: int = DEFAULT_SUMMARIZE_SAMPLES,
    draft_model: str = DEFAULT_SUMMARIZE_DRAFT_MODEL,
    aggregate_model: str = DEFAULT_SUMMARIZE_AGGREGATE_MODEL,
    overwrite: bool = False,
) -> None:
    """Summarize one or all NICE projects and write results to data/project_summaries.json."""
    summaries_path = DATA_DIR / "project_summaries.json"

    # Load existing summaries keyed by project ID
    existing: dict[str, dict] = {}
    if summaries_path.exists():
        with open(summaries_path, "r", encoding="utf-8") as f:
            existing = {item["id"]: item for item in json.load(f)}

    all_ids = _get_all_project_ids()

    if project_id:
        if project_id not in all_ids:
            raise ValueError(
                f"Project ID '{project_id}' not found in NICE.csv. Available: {all_ids}"
            )
        to_process = [project_id]
    else:
        to_process = all_ids

    if not overwrite:
        already_done = [pid for pid in to_process if pid in existing]
        if already_done:
            console.print(
                f"[yellow]⚠ Already summarized (use --overwrite to replace): {already_done}[/yellow]"
            )
        to_process = [pid for pid in to_process if pid not in existing]
        if not to_process:
            console.print("[green]Nothing to do — all projects already summarized.[/green]")
            return

    console.print(
        f"\n[bold green]SQuIRE Summarize[/bold green] — {len(to_process)} project(s)"
    )
    console.print(f"  Draft model:     {draft_model}  (×{samples})")
    console.print(f"  Aggregate model: {aggregate_model}")
    console.print(f"  Temperature:     {SUMMARIZE_TEMPERATURE}\n")

    all_requirements = load_requirements()

    for pid in to_process:
        console.print(f"[bold cyan]Project {pid}[/bold cyan]")
        project_reqs = get_requirements_by_project(pid, all_requirements)
        if not project_reqs:
            console.print(f"  [red]No requirements found, skipping.[/red]")
            continue

        requirements_text = "\n".join(f"- {r.text}" for r in project_reqs)
        console.print(f"  Requirements: {len(project_reqs)}")

        # Step 1: Draft summaries
        drafts: list[str] = []
        for i in range(samples):
            console.print(f"  [dim]Draft {i + 1}/{samples}...[/dim]")
            try:
                drafts.append(
                    with_retry(_draft_summary, pid, requirements_text, draft_model)
                )
            except Exception as e:
                console.print(f"  [red]Draft {i + 1} failed: {e}[/red]")

        if not drafts:
            console.print(f"  [red]All drafts failed for project {pid}, skipping.[/red]")
            continue

        # Step 2: Aggregate
        console.print(f"  [dim]Aggregating {len(drafts)} draft(s)...[/dim]")
        try:
            result = with_retry(_aggregate_summaries, pid, drafts, aggregate_model)
        except Exception as e:
            console.print(f"  [red]Aggregation failed for project {pid}: {e}[/red]")
            continue

        existing[pid] = {
            "id": pid,
            "name": result["title"],
            "description": result["description"],
        }
        console.print(f"  [green]✓ {result['title']}[/green]")
        _save_summaries(summaries_path, existing)

    console.print(f"\n[bold green]Saved to:[/bold green] {summaries_path}")
