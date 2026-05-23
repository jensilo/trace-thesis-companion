import json
import re
import statistics
import datetime
from pathlib import Path
from typing import Any

import litellm
from bert_score import score as bert_score_fn
from rich.console import Console

from squire.config import (
    DATA_DIR,
    DEFAULT_EVAL_REASONING_EFFORT,
    DEFAULT_QUALITY_JUDGE_MODEL,
    DEFAULT_META_JUDGE_MODEL,
    DEFAULT_QUALITY_SAMPLES,
    DEFAULT_META_SAMPLES,
    OUTPUT_DIR,
    REASONING_MODEL_PREFIXES,
)
from squire.ingestion import load_requirements, get_requirements_by_project, load_personas
from squire.utils import with_retry

console = Console(force_terminal=True)

QUALITY_DIMENSIONS = ["structuring", "clarity", "responsiveness", "rigor"]
META_DIMENSIONS = ["completeness", "realism"]

QUALITY_EVAL_PROMPT = """\
You are a rigorous assessor of requirements engineering interview technique.

You will be given a project summary and an interview transcript.
You do NOT know who conducted the interview or who the stakeholder is.
You do NOT know whether this was AI-generated or human-conducted.
Evaluate the interview purely on its technical craft — score what you actually observe.

Behavioral reference — use these anchors when scoring:
- A novice RE tends to ask generic or surface-level questions, miss follow-up opportunities, \
accept vague answers without probing, and fail to recall or connect earlier statements.
- An experienced RE probes ambiguous responses, recalls and builds on earlier statements, \
challenges inconsistencies, and uses domain knowledge to pursue precision.
This distinction is especially relevant to structuring and rigor, but may surface across all dimensions.

Score the following 4 dimensions, each on a scale from 1 (worst) to 10 (best):

structuring: Did the interviewer establish and communicate a clear structure? Did they \
steer the conversation, control digressions, and recall earlier statements to build on them?

clarity: Did the interviewer ask clear, simple, short questions? Did they interpret and \
extend the stakeholder's statements to confirm understanding and sharpen meaning?

responsiveness: Did the interviewer let the stakeholder finish, tolerate pauses, respond \
empathetically, and follow up on new aspects the stakeholder introduced unprompted?

rigor: Did the interviewer challenge inconsistencies and probe incomplete answers? Did they \
demonstrate domain knowledge and pursue important issues with adequate depth?

Scoring rules:
- Scores of 6 and above require you to quote specific supporting evidence from the transcript. \
State it explicitly in your reasoning.
- Recognise genuine strengths where they exist. An interview that demonstrates real skill in \
a dimension scores 7–9; exceptional craft scores 9–10.
- An adequate but unremarkable performance scores 5–6. Significant gaps score 3–4.
- Be precise: name what is present and what is absent. Vague generalisations are not acceptable.

Before assigning scores, reason through each dimension in turn: identify specific transcript \
moments, consider whether the behaviour is more consistent with a novice or an experienced RE, \
and weigh evidence on both sides. Encode your reasoning in the JSON fields below.

Output ONLY a JSON object with no additional text:
{
  "dimensions": {
    "structuring": {"reasoning": "...", "score": <integer 1-10>},
    "clarity": {"reasoning": "...", "score": <integer 1-10>},
    "responsiveness": {"reasoning": "...", "score": <integer 1-10>},
    "rigor": {"reasoning": "...", "score": <integer 1-10>}
  }
}"""

META_EVAL_PROMPT = """\
You are an objective evaluator of an AI-generated requirements elicitation transcript.
This transcript was produced by a language model simulating a requirements engineering \
interview between a Requirements Engineer and a Stakeholder.

You will be given:
- A project summary
- The intended interviewer persona
- The intended stakeholder persona
- The list of requirements that were supposed to be elicited through the interview
- The transcript itself

Score the following 2 dimensions, each on a scale from 1 (worst) to 10 (best):

completeness: Were all encoded requirements meaningfully elicited and discussed through \
genuine dialogue? A requirement that is merely mentioned in passing, or that appears as a \
verbatim specification statement rather than emerging organically through exchange, does \
not count as properly elicited. Strong coverage of most requirements with natural emergence \
scores 7–9; full, well-distributed coverage scores 9–10.

realism: Does this read like a plausible human interview? \
Assess whether the stakeholder's communication style, vocabulary, and level of precision \
match their described persona. Assess whether the interviewer's behaviour is consistent \
with their described background and experience level. Natural speech markers (hesitations, \
digressions, self-corrections), persona-consistent imprecision, and requirements that are \
discovered rather than stated all increase this score. A transcript that maintains \
consistent personas and avoids robotic turn-taking scores 7–9; one that also includes \
genuine imperfection and surprise scores 9–10.

Scoring rules:
- Scores of 6 and above require you to quote specific supporting evidence from the transcript. \
State it explicitly in your reasoning.
- Score what you observe. Where the transcript genuinely succeeds, say so; where it falls \
short, name the specific failure.
- Be precise. Vague criticism or vague praise is not acceptable.

Before assigning scores, work through the following steps:
1. Go through the list of encoded requirements one by one and note for each whether it was \
elicited organically through dialogue, mentioned only in passing, or not covered at all.
2. Identify 2–3 moments where the stakeholder's speech either matches or contradicts their \
described persona (vocabulary, precision, communication style).
3. Identify 2–3 moments where the interviewer's behaviour either matches or contradicts their \
described experience level — consider the novice/experienced distinction explicitly.
4. Use this evidence to assign and justify your scores.

Output ONLY a JSON object with no additional text:
{
  "dimensions": {
    "completeness": {"reasoning": "...", "score": <integer 1-10>},
    "realism": {"reasoning": "...", "score": <integer 1-10>}
  }
}"""


def _completion_kwargs(model_name: str) -> dict:
    """Return the appropriate generation-control kwargs for this model family.

    Reasoning models (GPT o-series, gpt-5) use reasoning_effort; everything
    else (Claude, older GPT) uses temperature=0 for deterministic output.
    """
    if any(model_name.lower().startswith(p) for p in REASONING_MODEL_PREFIXES):
        return {"reasoning_effort": DEFAULT_EVAL_REASONING_EFFORT}
    return {"temperature": 0.0}


def calculate_bert_score(candidates: list[str], references: list[str], job_label: str = "") -> dict[str, float] | None:
    pfx = f"[dim]{job_label}[/dim] " if job_label else ""
    console.print(f"{pfx}[blue]Calculating BERTScore...[/blue]")
    P, R, F1 = bert_score_fn(candidates, references, lang="en", verbose=False)
    return {
        "precision": P.mean().item(),
        "recall": R.mean().item(),
        "f1": F1.mean().item(),
    }


def evaluate_quality_g_eval(
    transcript: str,
    project_summary: str,
    model_name: str = DEFAULT_QUALITY_JUDGE_MODEL,
    n: int = DEFAULT_QUALITY_SAMPLES,
    job_label: str = "",
) -> dict[str, Any]:
    pfx = f"[dim]{job_label}[/dim] " if job_label else ""
    console.print(f"{pfx}[blue]Running Quality G-Eval with {model_name} (n={n})...[/blue]")

    prompt = f"{QUALITY_EVAL_PROMPT}\n\nProject Summary:\n{project_summary}\n\nInterview Transcript:\n{transcript}"

    if "/" not in model_name:
        model_name = f"anthropic/{model_name}"

    return _run_g_eval(prompt, QUALITY_DIMENSIONS, model_name, n, job_label=job_label)


def evaluate_meta_g_eval(
    transcript: str,
    project_summary: str,
    encoded_requirements: str,
    interviewer_description: str,
    stakeholder_description: str,
    model_name: str = DEFAULT_META_JUDGE_MODEL,
    n: int = DEFAULT_META_SAMPLES,
    job_label: str = "",
) -> dict[str, Any]:
    pfx = f"[dim]{job_label}[/dim] " if job_label else ""
    console.print(f"{pfx}[blue]Running Meta G-Eval with {model_name} (n={n})...[/blue]")

    prompt = f"{META_EVAL_PROMPT}\n\nProject Summary:\n{project_summary}\n\n"
    if interviewer_description:
        prompt += f"Interviewer Persona:\n{interviewer_description}\n\n"
    if stakeholder_description:
        prompt += f"Stakeholder Persona:\n{stakeholder_description}\n\n"
    if encoded_requirements:
        prompt += f"Requirements to be Elicited:\n{encoded_requirements}\n\n"
    prompt += f"Interview Transcript:\n{transcript}"

    if "/" not in model_name:
        model_name = f"anthropic/{model_name}"

    return _run_g_eval(prompt, META_DIMENSIONS, model_name, n, job_label=job_label)


def _run_g_eval(
    prompt: str,
    dimensions: list[str],
    model_name: str,
    n: int,
    job_label: str = "",
) -> dict[str, Any]:
    pfx = f"[dim]{job_label}[/dim] " if job_label else ""
    dimension_samples: dict[str, list[float]] = {d: [] for d in dimensions}
    raw_responses: list[dict[str, Any]] = []

    for i in range(n):
        console.print(f"{pfx}  Sample {i + 1}/{n}...")
        try:
            response = with_retry(
                lambda: litellm.completion(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    **_completion_kwargs(model_name),
                )
            )
            if not response or not response.choices:
                console.print(f"{pfx}  [red]Sample {i + 1}: empty response.[/red]")
                raw_responses.append({"sample": i + 1, "error": "empty response"})
                continue

            content = response.choices[0].message.content.strip()
            parsed = _parse_dimensions(content, dimensions)

            if parsed:
                for dim, score in parsed.items():
                    if dim in dimension_samples:
                        dimension_samples[dim].append(score)
                raw_responses.append({"sample": i + 1, "raw": content, "parsed": parsed})
            else:
                console.print(f"{pfx}  [red]Sample {i + 1}: could not parse dimensions.[/red]")
                raw_responses.append({"sample": i + 1, "raw": content, "error": "parse failure"})
        except Exception as e:
            console.print(f"{pfx}  [red]Sample {i + 1}: {e}[/red]")
            raw_responses.append({"sample": i + 1, "error": str(e)})

    dimensions_result: dict[str, dict[str, Any]] = {}
    dimension_means: list[float] = []
    for dim in dimensions:
        samples = dimension_samples[dim]
        if samples:
            mean = statistics.mean(samples)
            std_dev = statistics.stdev(samples) if len(samples) > 1 else 0.0
            dimensions_result[dim] = {"mean": round(mean, 2), "std_dev": round(std_dev, 2), "samples": samples}
            dimension_means.append(mean)
        else:
            dimensions_result[dim] = {"mean": 0.0, "std_dev": 0.0, "samples": []}

    overall_mean = round(statistics.mean(dimension_means), 2) if dimension_means else 0.0
    overall_std = round(statistics.stdev(dimension_means), 2) if len(dimension_means) > 1 else 0.0

    for dim in dimensions:
        r = dimensions_result[dim]
        console.print(f"{pfx}    {dim:<20s} {r['mean']:5.2f}  (σ={r['std_dev']:.2f})")
    console.print(f"{pfx}  [bold green]Overall:[/bold green] {overall_mean:.2f} (σ={overall_std:.2f})")

    return {
        "overall_mean": overall_mean,
        "overall_std_dev": overall_std,
        "dimensions": dimensions_result,
        "prompt": prompt,
        "raw_responses": raw_responses,
    }


def _parse_dimensions(content: str, dimensions: list[str]) -> dict[str, float] | None:
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)

    try:
        data = json.loads(content)
        dims = data.get("dimensions", {})
        if dims:
            return {k: float(v["score"]) for k, v in dims.items() if "score" in v}
    except (json.JSONDecodeError, KeyError, TypeError):
        pass

    result: dict[str, float] = {}
    for dim in dimensions:
        pattern = rf'"{dim}"\s*:\s*\{{[^}}]*"score"\s*:\s*(\d+(?:\.\d+)?)'
        match = re.search(pattern, content, re.DOTALL)
        if match:
            result[dim] = float(match.group(1))
    return result if result else None


def run_evaluation(
    transcript_file: Path | None = None,
    quality_model: str = DEFAULT_QUALITY_JUDGE_MODEL,
    meta_model: str = DEFAULT_META_JUDGE_MODEL,
    quality_samples: int = DEFAULT_QUALITY_SAMPLES,
    meta_samples: int = DEFAULT_META_SAMPLES,
    output_dir: Path | None = None,
    save_metadata: bool = True,
    skip_bert: bool = False,
    job_label: str = "",
) -> dict:
    """Run G-Eval (quality + meta) and BERTScore on a transcript.

    When save_metadata=False the original transcript's sidecar .json is not modified —
    useful for re-evaluation runs that write results to a separate corpus directory.
    Returns the evaluation result dict (g_eval_quality, g_eval_meta, bert_score).
    """
    pfx = f"[dim]{job_label}[/dim] " if job_label else ""
    transcript_file = _resolve_transcript(transcript_file)
    transcript_text = transcript_file.read_text(encoding="utf-8")

    project_id, project_name, project_description, stakeholder_desc, interviewer_desc, encoded_requirements = (
        _load_metadata(transcript_file)
    )

    all_requirements = load_requirements()
    reference_reqs = get_requirements_by_project(project_id, all_requirements)
    reference_text = "\n".join(r.text for r in reference_reqs)

    console.print(f"\n{pfx}[bold green]SQuIRE Evaluation[/bold green] — {project_name}")
    console.print(f"{pfx}  Quality Judge: {quality_model}  (n={quality_samples})")
    console.print(f"{pfx}  Meta Judge:    {meta_model}  (n={meta_samples})")
    console.print(f"{pfx}  Transcript: {transcript_file.name}")

    quality_results = evaluate_quality_g_eval(
        transcript_text,
        project_description,
        model_name=quality_model,
        n=quality_samples,
        job_label=job_label,
    )

    meta_results = evaluate_meta_g_eval(
        transcript_text,
        project_description,
        encoded_requirements=encoded_requirements,
        interviewer_description=interviewer_desc,
        stakeholder_description=stakeholder_desc,
        model_name=meta_model,
        n=meta_samples,
        job_label=job_label,
    )

    if skip_bert:
        bert_results = None
    else:
        console.print(f"{pfx}[bold]BERTScore...[/bold]")
        bert_results = calculate_bert_score([transcript_text], [reference_text], job_label=job_label)
        console.print(f"{pfx}[bold green]BERTScore F1:[/bold green] {bert_results['f1']:.4f}")

    if save_metadata:
        _save_evaluation_metadata(
            transcript_file, quality_model, meta_model, quality_results, meta_results, bert_results or {},
            job_label=job_label,
        )
    _write_evaluation_log(
        transcript_file, project_id, quality_model, meta_model, quality_results, meta_results, bert_results,
        output_dir=output_dir, job_label=job_label,
    )

    def _clean_dimensions(results: dict) -> dict:
        return {
            dim: {"mean": data["mean"], "std_dev": data["std_dev"], "samples": data["samples"]}
            for dim, data in results["dimensions"].items()
        }

    return {
        "g_eval_quality": {
            "judge_model": quality_model,
            "overall_mean": quality_results["overall_mean"],
            "overall_std_dev": quality_results["overall_std_dev"],
            "dimensions": _clean_dimensions(quality_results),
        },
        "g_eval_meta": {
            "judge_model": meta_model,
            "overall_mean": meta_results["overall_mean"],
            "overall_std_dev": meta_results["overall_std_dev"],
            "dimensions": _clean_dimensions(meta_results),
        },
        "bert_score": bert_results,
    }



def _resolve_transcript(transcript_file: Path | None) -> Path:
    if transcript_file:
        if transcript_file.suffix == ".json":
            transcript_file = transcript_file.with_suffix(".txt")
        if transcript_file.exists():
            return transcript_file
        raise FileNotFoundError(f"Transcript not found: {transcript_file}")

    if not OUTPUT_DIR.exists():
        raise FileNotFoundError("Outputs directory not found.")

    transcripts = list(OUTPUT_DIR.glob("transcript_*.txt"))
    if not transcripts:
        raise FileNotFoundError("No transcripts found in outputs directory.")

    latest = max(transcripts, key=lambda p: p.stat().st_mtime)
    console.print(f"[bold yellow]⚠ No transcript specified — using latest: {latest.name}[/bold yellow]")
    return latest


def _load_metadata(transcript_file: Path) -> tuple[str, str, str, str, str, str]:
    """Returns (project_id, project_name, project_description, stakeholder_desc, interviewer_desc, encoded_requirements)."""
    metadata_path = transcript_file.with_suffix(".json")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Transcript metadata not found: {metadata_path}. Cannot evaluate without it.")

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"Malformed transcript metadata: {metadata_path}: {e}") from e

    project_id = metadata.get("project_id", "")
    project_name = metadata.get("project_name", project_id)
    project_description = metadata.get("project_description", "")

    if not project_id:
        raise ValueError(f"Transcript metadata missing 'project_id': {metadata_path}")

    personas = load_personas()

    stakeholder_desc = ""
    stakeholder_id = metadata.get("stakeholder_persona", "")
    if stakeholder_id:
        persona = next((p for p in personas if p.id == stakeholder_id), None)
        if persona:
            stakeholder_desc = f"{persona.name} ({persona.id}): {persona.system_prompt_template.split('{')[0].strip()}"

    interviewer_desc = ""
    interviewer_id = metadata.get("interviewer_persona", "")
    if interviewer_id:
        persona = next((p for p in personas if p.id == interviewer_id), None)
        if persona:
            interviewer_desc = f"{persona.name} ({persona.id}): {persona.system_prompt_template.split('{')[0].strip()}"

    encoded_reqs = metadata.get("selected_requirements", [])
    req_texts = []
    for r in encoded_reqs:
        if isinstance(r, dict):
            req_texts.append(r.get("text", ""))
        else:
            req_texts.append(str(r))
    encoded_requirements = "\n".join(f"- {t}" for t in req_texts if t)

    return project_id, project_name, project_description, stakeholder_desc, interviewer_desc, encoded_requirements


def _save_evaluation_metadata(
    transcript_file: Path,
    quality_model: str,
    meta_model: str,
    quality_results: dict,
    meta_results: dict,
    bert_results: dict,
    job_label: str = "",
) -> None:
    metadata_path = transcript_file.with_suffix(".json")
    metadata: dict = {}
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass

    def _clean_dimensions(results: dict) -> dict:
        return {
            dim: {"mean": data["mean"], "std_dev": data["std_dev"], "samples": data["samples"]}
            for dim, data in results["dimensions"].items()
        }

    metadata["evaluation"] = {
        "g_eval_quality": {
            "judge_model": quality_model,
            "overall_mean": quality_results["overall_mean"],
            "overall_std_dev": quality_results["overall_std_dev"],
            "dimensions": _clean_dimensions(quality_results),
        },
        "g_eval_meta": {
            "judge_model": meta_model,
            "overall_mean": meta_results["overall_mean"],
            "overall_std_dev": meta_results["overall_std_dev"],
            "dimensions": _clean_dimensions(meta_results),
        },
        "bert_score": bert_results,
    }

    pfx = f"[dim]{job_label}[/dim] " if job_label else ""
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    console.print(f"{pfx}[blue]Evaluation saved to: {metadata_path.name}[/blue]")


def _write_evaluation_log(
    transcript_file: Path,
    project_id: str,
    quality_model: str,
    meta_model: str,
    quality_results: dict,
    meta_results: dict,
    bert_results: dict | None,
    output_dir: Path | None = None,
    job_label: str = "",
) -> None:
    log_dir = output_dir or OUTPUT_DIR
    # Name after the transcript stem so concurrent corpus runs never collide
    log_path = log_dir / f"{transcript_file.stem}_evaluation.log"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    lines: list[str] = []
    lines.append(f"SQuIRE Evaluation Log — {timestamp}")
    lines.append(f"Transcript: {transcript_file.name}")
    lines.append("=" * 80)

    lines.append(f"\n## QUALITY G-EVAL — Judge: {quality_model}\n")
    lines.append("### PROMPT\n")
    lines.append(quality_results.get("prompt", "(prompt not captured)"))
    lines.append("\n### RAW RESPONSES\n")
    for entry in quality_results.get("raw_responses", []):
        lines.append(f"--- Sample {entry.get('sample', '?')} ---")
        if "error" in entry:
            lines.append(f"ERROR: {entry['error']}")
        if "raw" in entry:
            lines.append(entry["raw"])
        if "parsed" in entry:
            lines.append(f"PARSED: {json.dumps(entry['parsed'])}")
        lines.append("")
    lines.append("### AGGREGATED RESULTS\n")
    lines.append(f"Overall Mean: {quality_results['overall_mean']:.2f}  (σ={quality_results['overall_std_dev']:.2f})")
    for dim in QUALITY_DIMENSIONS:
        d = quality_results["dimensions"].get(dim, {})
        lines.append(f"  {dim:<20s}  mean={d.get('mean', 0):.2f}  σ={d.get('std_dev', 0):.2f}  samples={d.get('samples', [])}")

    lines.append("\n" + "=" * 80)
    lines.append(f"\n## META G-EVAL — Judge: {meta_model}\n")
    lines.append("### PROMPT\n")
    lines.append(meta_results.get("prompt", "(prompt not captured)"))
    lines.append("\n### RAW RESPONSES\n")
    for entry in meta_results.get("raw_responses", []):
        lines.append(f"--- Sample {entry.get('sample', '?')} ---")
        if "error" in entry:
            lines.append(f"ERROR: {entry['error']}")
        if "raw" in entry:
            lines.append(entry["raw"])
        if "parsed" in entry:
            lines.append(f"PARSED: {json.dumps(entry['parsed'])}")
        lines.append("")
    lines.append("### AGGREGATED RESULTS\n")
    lines.append(f"Overall Mean: {meta_results['overall_mean']:.2f}  (σ={meta_results['overall_std_dev']:.2f})")
    for dim in META_DIMENSIONS:
        d = meta_results["dimensions"].get(dim, {})
        lines.append(f"  {dim:<20s}  mean={d.get('mean', 0):.2f}  σ={d.get('std_dev', 0):.2f}  samples={d.get('samples', [])}")

    lines.append("\n" + "=" * 80)
    lines.append("\n## BERT SCORE\n")
    if bert_results:
        lines.append(f"Precision: {bert_results['precision']:.4f}")
        lines.append(f"Recall:    {bert_results['recall']:.4f}")
        lines.append(f"F1:        {bert_results['f1']:.4f}")
    else:
        lines.append("(skipped)")

    pfx = f"[dim]{job_label}[/dim] " if job_label else ""
    log_path.write_text("\n".join(lines), encoding="utf-8")
    console.print(f"{pfx}[blue]Evaluation log saved to: {log_path.name}[/blue]")
