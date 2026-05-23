import json
import os
import datetime
from pathlib import Path
from typing import NamedTuple

from rich.console import Console
from langroid.language_models.openai_gpt import OpenAIGPTConfig
from langroid.utils.configuration import set_global, Settings

from squire.config import OUTPUT_DIR
from squire.ingestion import load_projects, load_requirements, load_personas, get_requirements_by_project
from squire.agents.scriptwriter import ScriptwriterAgent, ScriptwriterConfig
from squire.models.project import Project
from squire.models.persona import Persona

console = Console(force_terminal=True)


class SynthesisResult(NamedTuple):
    transcript_path: Path
    metadata_path: Path


def _resolve_project(project_id: str) -> Project:
    projects = load_projects()
    project = next((p for p in projects if p.id == project_id), None)
    if not project:
        raise ValueError(f"Project ID '{project_id}' not found.")
    return project


def _resolve_persona(persona_id: str | None, role: str) -> Persona:
    personas = load_personas()
    if persona_id:
        persona = next((p for p in personas if p.id == persona_id), None)
        if not persona:
            available = [p.id for p in personas if p.role == role]
            raise ValueError(f"Persona '{persona_id}' not found. Available {role}s: {available}")
        return persona
    defaults = {"Interviewer": "re_experienced", "Interviewee": "stakeholder_nontechnical_enduser"}
    return next(p for p in personas if p.id == defaults[role])


def _build_llm_config(model: str) -> OpenAIGPTConfig:
    # Langroid auto-loads OPENAI_API_KEY into api_key (env_prefix="OPENAI_") and
    # forwards it to litellm — which sends it to Anthropic, causing auth failure.
    # Explicitly passing the provider key overrides this before it reaches litellm.
    if model.startswith("litellm/anthropic/"):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set. Add it to your .env file.")
        return OpenAIGPTConfig(chat_model=model, api_key=api_key)
    return OpenAIGPTConfig(chat_model=model)


def run_synthesis(
    project_id: str,
    model: str,
    stakeholder_id: str | None = None,
    interviewer_id: str | None = None,
    single_shot: bool = False,
    no_refine: bool = False,
    debug: bool = False,
    output_dir: Path | None = None,
    run_id: str | None = None,
) -> SynthesisResult:
    set_global(Settings(debug=debug, cache=False))

    project = _resolve_project(project_id)
    stakeholder = _resolve_persona(stakeholder_id, "Interviewee")
    interviewer = _resolve_persona(interviewer_id, "Interviewer")

    all_requirements = load_requirements()
    project_reqs = get_requirements_by_project(project_id, all_requirements)
    if not project_reqs:
        raise ValueError(f"No requirements found for project '{project_id}'.")

    mode = "single-shot" if single_shot else ("3-step" if no_refine else "4-step")
    console.print(f"[bold green]SQuIRE Synthesis[/bold green] — {project.name}")
    console.print(f"  Model: {model} | Mode: {mode}")
    console.print(f"  Interviewer: {interviewer.name} ({interviewer.id})")
    console.print(f"  Stakeholder: {stakeholder.name} ({stakeholder.id})")
    console.print(f"  Available requirements: {len(project_reqs)}")

    llm_config = _build_llm_config(model)
    scriptwriter = ScriptwriterAgent(ScriptwriterConfig(llm=llm_config))

    result = scriptwriter.generate_transcript(
        project, project_reqs, stakeholder, interviewer,
        single_shot=single_shot, no_refine=no_refine,
    )

    if not result["transcript"]:
        raise RuntimeError("Scriptwriter produced an empty transcript.")

    transcript_path, metadata_path = _save_outputs(
        project=project,
        model=model,
        transcript=result["transcript"],
        selected_requirements=result["requirements"],
        stakeholder=stakeholder,
        interviewer=interviewer,
        no_refine=no_refine,
        output_dir=output_dir,
        run_id=run_id,
    )

    console.print(f"\n[bold green]Synthesis complete.[/bold green]")
    console.print(f"  Transcript: {transcript_path}")
    console.print(f"  Metadata:   {metadata_path}")

    return SynthesisResult(transcript_path=transcript_path, metadata_path=metadata_path)


def _save_outputs(
    project: Project,
    model: str,
    transcript: str,
    selected_requirements: list,
    stakeholder: Persona,
    interviewer: Persona,
    no_refine: bool = False,
    output_dir: Path | None = None,
    run_id: str | None = None,
) -> tuple[Path, Path]:
    out = output_dir or OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    req_section = _format_requirements_section(selected_requirements)
    full_transcript = transcript + "\n\n" + req_section

    stem = f"transcript_{project.id}_{run_id}_{timestamp}" if run_id else f"transcript_{project.id}_{timestamp}"
    transcript_path = out / f"{stem}.txt"
    transcript_path.write_text(full_transcript, encoding="utf-8")

    metadata = {
        "project_id": project.id,
        "project_name": project.name,
        "project_description": project.description,
        "model": model,
        "timestamp": timestamp,
        "interviewer_persona": interviewer.id,
        "stakeholder_persona": stakeholder.id,
        "no_refine": no_refine,
        "selected_requirements": [
            {
                "text": r.text,
                "line": r.line_number,
                "categories": r.categories,
            }
            for r in selected_requirements
        ],
    }
    metadata_path = transcript_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return transcript_path, metadata_path


def _format_requirements_section(requirements: list) -> str:
    lines = ["---", "Selected Requirements:"]
    for i, r in enumerate(requirements, 1):
        cats = ", ".join(r.categories) if r.categories else "uncategorised"
        line_ref = f"NICE.csv line {r.line_number}" if r.line_number is not None else ""
        parts = [f"{i}."]
        if line_ref:
            parts.append(f"[{line_ref}]")
        parts.append(f"[{cats}]")
        parts.append(r.text)
        lines.append(" ".join(parts))
    return "\n".join(lines)
