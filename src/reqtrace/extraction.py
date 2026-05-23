import datetime
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import litellm
from rich.console import Console

from reqtrace.config import (
    DEFAULT_EXTRACTION_MODEL,
    EVAL_REASONING_EFFORT,
    OUTPUT_DIR,
    REASONING_MODEL_PREFIXES,
)
from reqtrace.models.requirement import ExtractedRequirement, TraceExtraction
from reqtrace.utils import with_retry

console = Console(force_terminal=True)

# ── Prompts ──────────────────────────────────────────────────────────────────

_DRAFT_PLAN_PROMPT = """\
Read this interview transcript carefully.

Identify all passages where the interviewee discusses non-functional quality concerns: \
performance, security, usability, reliability, maintainability, scalability, and similar \
quality attributes. These may be explicit ("the system must respond within...") or \
implicit ("we always struggle with slow load times...").

List them briefly — for each passage, note the quality topic and approximately where it \
appears. This is a planning step; focus on coverage, not precision.

Transcript:
{transcript}"""

_EXTRACTION_PROMPT = """\
You are a Requirements Engineering expert. Extract all non-functional quality requirements \
from this interview transcript. Focus exclusively on quality concerns (performance, security, \
usability, availability, maintainability, etc.) — do not extract purely functional requirements.

Based on the following plan identifying candidate passages:
---
{draft_plan}
---

For each quality requirement you can identify, provide a structured analysis.

Output ONLY a JSON object — no additional text or markdown:
{{
  "requirements": [
    {{
      "requirement": "A clear, precise statement of the requirement in RE standard form",
      "source_citation": "An exact verbatim quote from the transcript that evidences this requirement (must appear word-for-word in the transcript)",
      "classification": "Functional or Non-Functional",
      "quality_attribute": "The quality attribute — use exactly one snake_case value from: availability, fault_tolerance, legal, look_and_feel, maintainability, operability, performance, portability, scalability, security, usability, other. Null only if Functional.",
      "confidence": "<integer 1-5: 1=very low evidence, 2=low, 3=medium, 4=high, 5=very high evidence>",
      "rationale": "Why this requirement was extracted, what the transcript evidence means, and what gaps remain (missing specifics, unclear scope, implicit assumptions)",
      "follow_up_question": "A concrete follow-up question that would increase confidence or add specificity"
    }}
  ]
}}

Transcript:
{transcript}"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def model_short(model: str) -> str:
    """Extract a short filename-safe identifier. E.g. mistral/mistral-small-latest → mistral-small."""
    return model.split("/")[-1].removesuffix("-latest")


def _completion_kwargs(model_name: str) -> dict[str, Any]:
    if any(model_name.lower().startswith(p) for p in REASONING_MODEL_PREFIXES):
        return {"reasoning_effort": EVAL_REASONING_EFFORT}
    return {"temperature": 0.5}


def _call_llm(prompt: str, model: str, use_json_mode: bool = False) -> str:
    kwargs: dict[str, Any] = {**_completion_kwargs(model)}
    if use_json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    response = with_retry(
        lambda: litellm.completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
    )
    if not response or not response.choices:
        raise RuntimeError("Empty response from LLM")
    return response.choices[0].message.content.strip()


def _parse_json(content: str) -> dict:
    """Parse JSON from LLM response, stripping markdown fences if present."""
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content.rstrip())
    return json.loads(content)


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class ExtractionResult:
    extraction: TraceExtraction
    output_path: Path


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_extraction(
    transcript_file: Path,
    model: str = DEFAULT_EXTRACTION_MODEL,
    output_dir: Path | None = None,
    run_id: str | None = None,
    debug: bool = False,
) -> ExtractionResult:
    """Run the 2-step extraction pipeline on a transcript .txt file.

    Steps:
    1. Draft plan — free-form passage identification (used as context, not saved)
    2. Precise extraction — structured JSON with full ExtractedRequirement schema
    3. Citation verification — post-processing substring check (no LLM call)
    4. Write JSON to outputs/
    """
    raw_text = transcript_file.read_text(encoding="utf-8")
    # Strip the appended ground truth section SQuIRE writes after "---\nSelected Requirements:"
    # to prevent data leakage into the extraction prompt.
    separator = "\n---\nSelected Requirements:"
    transcript_text = raw_text.split(separator)[0].strip()
    out_dir = output_dir or OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    short = model_short(model)

    console.print(f"\n[bold green]TRACE Extraction[/bold green] — {transcript_file.name}")
    console.print(f"  Model: {model}")

    # Step 1: Draft plan
    console.print("[blue]Step 1: Drafting extraction plan...[/blue]")
    draft_prompt = _DRAFT_PLAN_PROMPT.format(transcript=transcript_text)
    draft_plan = _call_llm(draft_prompt, model)
    if debug:
        console.print(f"[dim]{draft_plan}[/dim]")

    # Step 2: Precise extraction — try json_mode first, fall back if unsupported
    console.print("[blue]Step 2: Extracting requirements...[/blue]")
    extract_prompt = _EXTRACTION_PROMPT.format(draft_plan=draft_plan, transcript=transcript_text)
    try:
        content = _call_llm(extract_prompt, model, use_json_mode=True)
        data = _parse_json(content)
    except litellm.BadRequestError:
        content = _call_llm(extract_prompt, model, use_json_mode=False)
        data = _parse_json(content)
    except json.JSONDecodeError:
        # json_mode succeeded but returned garbage — retry without
        content = _call_llm(extract_prompt, model, use_json_mode=False)
        data = _parse_json(content)

    raw_requirements: list[dict] = data.get("requirements", [])

    # Step 3: Citation verification + construct validated models
    requirements = _build_requirements(raw_requirements, transcript_text)

    # Step 4: Serialize
    stem = run_id or f"{timestamp}_{short}"
    output_path = out_dir / f"trace_{stem}.json"
    extraction = TraceExtraction(
        transcript_file=str(transcript_file),
        model=model,
        timestamp=timestamp,
        requirements=requirements,
    )
    output_path.write_text(extraction.model_dump_json(indent=2), encoding="utf-8")

    _print_summary(requirements, output_path)
    return ExtractionResult(extraction=extraction, output_path=output_path)


# Maps every plausible LLM variant (lowercased) → canonical snake_case value
_QUALITY_ATTRIBUTE_CANONICAL: dict[str, str] = {
    # canonical forms
    "availability": "availability",
    "fault_tolerance": "fault_tolerance",
    "legal": "legal",
    "look_and_feel": "look_and_feel",
    "maintainability": "maintainability",
    "operability": "operability",
    "performance": "performance",
    "portability": "portability",
    "scalability": "scalability",
    "security": "security",
    "usability": "usability",
    "other": "other",
    # space/symbol variants
    "fault tolerance": "fault_tolerance",
    "look and feel": "look_and_feel",
    "look & feel": "look_and_feel",
}


def _normalise_quality_attribute(raw: str | None) -> str | None:
    if not raw:
        return None
    return _QUALITY_ATTRIBUTE_CANONICAL.get(raw.strip().lower(), raw.strip())


def _build_requirements(raw: list[dict], transcript_text: str) -> list[ExtractedRequirement]:
    transcript_lower = transcript_text.lower()
    requirements: list[ExtractedRequirement] = []

    for r in raw:
        citation = r.get("source_citation", "")
        verified = bool(citation) and citation.lower() in transcript_lower

        raw_conf = r.get("confidence", 3)
        try:
            conf = max(1, min(5, int(float(str(raw_conf)))))
        except (ValueError, TypeError):
            conf = 3

        raw_cls = r.get("classification", "Non-Functional")
        classification = "Functional" if str(raw_cls).lower().startswith("f") else "Non-Functional"

        requirements.append(ExtractedRequirement(
            requirement=r.get("requirement", ""),
            source_citation=citation,
            citation_verified=verified,
            citation_warning=None if verified else "Citation not found in transcript",
            classification=classification,
            quality_attribute=_normalise_quality_attribute(r.get("quality_attribute")),
            confidence=conf,
            rationale=r.get("rationale", ""),
            follow_up_question=r.get("follow_up_question", ""),
        ))

    return requirements


def _print_summary(requirements: list[ExtractedRequirement], output_path: Path) -> None:
    total = len(requirements)
    nf = sum(1 for r in requirements if r.classification == "Non-Functional")
    unverified = sum(1 for r in requirements if not r.citation_verified)
    conf_dist = {str(i): sum(1 for r in requirements if r.confidence == i) for i in range(1, 6)}

    console.print(f"\n[bold green]Extracted {total} requirements[/bold green]  "
                  f"(Functional: {total - nf}, Non-Functional: {nf})")
    console.print(f"  Confidence distribution: {conf_dist}")
    if unverified:
        console.print(f"  [yellow]⚠ {unverified} citation(s) not verified[/yellow]")
    console.print(f"  [blue]Saved:[/blue] {output_path.name}")
