import json
import re
import statistics
import datetime
from pathlib import Path
from typing import Any

import litellm
from bert_score import score as bert_score_fn
from rich.console import Console

from reqtrace.config import (
    DEFAULT_EVAL_JUDGE_MODEL,
    EVAL_REASONING_EFFORT,
    OUTPUT_DIR,
    REASONING_MODEL_PREFIXES,
)
from reqtrace.extraction import ExtractionResult, model_short, run_extraction
from reqtrace.models.evaluation import RequirementEvaluation, TraceEvaluation, TraceMetrics
from reqtrace.models.requirement import ExtractedRequirement, TraceExtraction
from reqtrace.utils import with_retry

console = Console(force_terminal=True)

_OVERLAP_LABELS = {1: "Very Low", 2: "Low", 3: "Medium", 4: "High", 5: "Very High"}

_JUDGE_PROMPT = """\
You are evaluating an automatic requirements extraction from an interview transcript.

Extracted Requirement:
{requirement}

Confidence assigned by extractor: {confidence}/5
Rationale: {rationale}
Follow-up question: {follow_up_question}

Top matching ground truth requirements (by semantic similarity):
{gt_candidates}

Tasks — return scores for ALL five tasks:
1. Identify which candidate (1, 2, 3, or null) best matches the extracted requirement. \
A match means they describe the same underlying system property or constraint.
2. Score overlap (1–5):
   1 = Very Low — no meaningful overlap, almost nothing in common
   2 = Low — little overlap; some shared topic but different substance
   3 = Medium — moderate overlap; same area but not identical
   4 = High — significant overlap; same requirement, possibly different wording
   5 = Very High — nearly identical; only minor wording differences
3. Rate confidence_quality (1–5): Is the confidence level the extractor assigned \
appropriate given the transcript evidence?
4. Rate rationale_quality (1–5): Does the rationale clearly explain why this was extracted \
and identify meaningful gaps?
5. Rate gap_detection_quality (1–5): Does the gap analysis identify actionable missing \
information that a follow-up interview should address?

Output ONLY a JSON object — no additional text:
{{
  "matched_gt_index": <1, 2, 3, or null>,
  "matched_ground_truth": "<exact text of the matched GT requirement, or null>",
  "overlap_score": <integer 1-5>,
  "confidence_quality": <integer 1-5>,
  "rationale_quality": <integer 1-5>,
  "gap_detection_quality": <integer 1-5>
}}"""


def _completion_kwargs(model_name: str) -> dict[str, Any]:
    if any(model_name.lower().startswith(p) for p in REASONING_MODEL_PREFIXES):
        return {"reasoning_effort": EVAL_REASONING_EFFORT}
    return {"temperature": 0.0}


def _parse_json(content: str) -> dict:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content.rstrip())
    return json.loads(content)


class _GTRequirement:
    """Ground truth requirement with text and quality attribute categories."""
    __slots__ = ("text", "categories")

    def __init__(self, text: str, categories: list[str]) -> None:
        self.text = text
        self.categories = categories  # snake_case, e.g. ["usability", "performance"]


def load_ground_truth(
    transcript_file: Path,
    ground_truth_override: Path | None = None,
) -> tuple[list[_GTRequirement], str]:
    """Return (list of GT requirements with categories, source description string)."""
    if ground_truth_override:
        raw = json.loads(ground_truth_override.read_text(encoding="utf-8"))
        gt_list = raw if isinstance(raw, list) else raw.get("selected_requirements", [])
        source = str(ground_truth_override)
    else:
        sidecar = transcript_file.with_suffix(".json")
        if not sidecar.exists():
            raise FileNotFoundError(
                f"No ground truth sidecar found at {sidecar}. "
                "Use --ground-truth to specify a GT file explicitly."
            )
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        gt_list = metadata.get("selected_requirements", [])
        source = str(sidecar)

    items: list[_GTRequirement] = []
    for item in gt_list:
        if isinstance(item, dict):
            text = item.get("text", "")
            cats = [str(c).lower() for c in item.get("categories", [])]
        else:
            text = str(item)
            cats = []
        if text:
            items.append(_GTRequirement(text=text, categories=cats))
    return items, source


def _compute_bert_top3(
    extracted_texts: list[str],
    gt_texts: list[str],
) -> list[list[tuple[int, float]]]:
    """Return top-3 GT indices (with scores) for each extracted requirement.

    Uses one BERTScore call for the full cross-product matrix.
    Returns list[list[tuple(gt_idx, f1_score)]], outer indexed by extracted requirement.
    """
    M = len(extracted_texts)
    N = len(gt_texts)

    # Full cross-product: M * N pairs
    all_cands = [e for e in extracted_texts for _ in range(N)]
    all_refs = gt_texts * M
    _, _, F1 = bert_score_fn(all_cands, all_refs, lang="en", verbose=False)

    # Reshape to (M, N) — F1 is a flat tensor
    F1_matrix = F1.reshape(M, N)

    top3_per_req: list[list[tuple[int, float]]] = []
    for i in range(M):
        row = F1_matrix[i]
        top_k = min(3, N)
        top_indices = row.argsort(descending=True)[:top_k].tolist()
        top3_per_req.append([(j, row[j].item()) for j in top_indices])

    return top3_per_req


def _judge_requirement(
    req: ExtractedRequirement,
    gt_candidates: list[str],
    eval_model: str,
) -> dict:
    """Call the LLM judge for one extracted requirement. Returns raw parsed JSON dict."""
    if not gt_candidates:
        return {
            "matched_gt_index": None,
            "matched_ground_truth": None,
            "overlap_score": 1,
            "confidence_quality": 3,
            "rationale_quality": 3,
            "gap_detection_quality": 3,
        }

    candidates_block = "\n".join(
        f"{i + 1}. {text}" for i, text in enumerate(gt_candidates)
    )
    prompt = _JUDGE_PROMPT.format(
        requirement=req.requirement,
        confidence=req.confidence,
        rationale=req.rationale,
        follow_up_question=req.follow_up_question,
        gt_candidates=candidates_block,
    )

    kwargs: dict[str, Any] = {**_completion_kwargs(eval_model)}
    kwargs["response_format"] = {"type": "json_object"}

    try:
        response = with_retry(
            lambda: litellm.completion(
                model=eval_model,
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )
        )
    except litellm.BadRequestError:
        kwargs_no_json = {k: v for k, v in kwargs.items() if k != "response_format"}
        response = with_retry(
            lambda: litellm.completion(
                model=eval_model,
                messages=[{"role": "user", "content": prompt}],
                **kwargs_no_json,
            )
        )

    content = response.choices[0].message.content.strip()
    return _parse_json(content)


def _safe_int(value: Any, lo: int = 1, hi: int = 5, default: int = 3) -> int:
    try:
        return max(lo, min(hi, int(value)))
    except (TypeError, ValueError):
        return default


def run_evaluation(
    extraction_result: ExtractionResult | None = None,
    extraction_file: Path | None = None,
    transcript_file: Path | None = None,
    ground_truth_override: Path | None = None,
    eval_model: str = DEFAULT_EVAL_JUDGE_MODEL,
    output_dir: Path | None = None,
    job_label: str = "",
) -> TraceEvaluation:
    """Evaluate an extraction against ground truth requirements.

    Caller provides either extraction_result (from run_extraction) or extraction_file
    (path to a trace_*.json). transcript_file is needed for GT auto-discovery when
    extraction_file is passed.
    """
    pfx = f"[dim]{job_label}[/dim] " if job_label else ""
    out_dir = output_dir or OUTPUT_DIR

    if extraction_result is not None:
        extraction = extraction_result.extraction
        extraction_path = extraction_result.output_path
        tx_file = Path(extraction.transcript_file)
    elif extraction_file is not None:
        raw = json.loads(extraction_file.read_text(encoding="utf-8"))
        extraction = TraceExtraction.model_validate(raw)
        extraction_path = extraction_file
        tx_file = transcript_file or Path(extraction.transcript_file)
    else:
        raise ValueError("Provide either extraction_result or extraction_file.")

    gt_items, gt_source = load_ground_truth(tx_file, ground_truth_override)
    gt_texts = [g.text for g in gt_items]

    console.print(f"\n{pfx}[bold green]TRACE Evaluation[/bold green]")
    console.print(f"{pfx}  Extraction: {extraction_path.name}")
    console.print(f"{pfx}  GT source:  {Path(gt_source).name}")
    console.print(f"{pfx}  GT count:   {len(gt_items)}")
    console.print(f"{pfx}  Extracted:  {len(extraction.requirements)}")
    console.print(f"{pfx}  Judge:      {eval_model}")

    requirements = extraction.requirements
    per_requirement: list[RequirementEvaluation] = []

    if not requirements:
        metrics = TraceMetrics(
            hit_count=0, extracted_count=0, ground_truth_count=len(gt_items),
            precision=0.0, recall=0.0, f1=0.0,
            quality_attribute_accuracy=0.0,
            avg_confidence=0.0, avg_overlap_score=0.0,
            avg_confidence_quality=0.0, avg_rationale_quality=0.0,
            avg_gap_detection_quality=0.0,
        )
        return _write_evaluation(
            TraceEvaluation(
                extraction_file=str(extraction_path),
                ground_truth_source=gt_source,
                eval_model=eval_model,
                per_requirement=[],
                metrics=metrics,
            ),
            extraction_path, out_dir,
        )

    # BERTScore: compute full cross-product matrix once
    console.print(f"{pfx}[blue]Computing BERTScore similarity matrix...[/blue]")
    extracted_texts = [r.requirement for r in requirements]
    top3_per_req = _compute_bert_top3(extracted_texts, gt_texts)

    # LLM judge: one call per extracted requirement
    console.print(f"{pfx}[blue]Running LLM judge ({len(requirements)} requirements)...[/blue]")
    matched_gt_indices: set[int] = set()

    for i, (req, top3) in enumerate(zip(requirements, top3_per_req)):
        console.print(f"{pfx}  Requirement {i + 1}/{len(requirements)}: {req.requirement[:60]}...")
        candidates = [gt_texts[j] for j, _ in top3]
        raw_result = _judge_requirement(req, candidates, eval_model)

        matched_idx_raw = raw_result.get("matched_gt_index")
        matched_gt_text: str | None = None
        matched_gt_item: _GTRequirement | None = None
        is_duplicate = False

        if matched_idx_raw is not None:
            try:
                # 1-based index into the top3 candidates
                top3_pos = max(0, min(2, int(matched_idx_raw) - 1))
                global_gt_idx = top3[top3_pos][0]
                if global_gt_idx in matched_gt_indices:
                    is_duplicate = True
                else:
                    matched_gt_indices.add(global_gt_idx)
                matched_gt_item = gt_items[global_gt_idx]
                matched_gt_text = matched_gt_item.text
            except (TypeError, ValueError, IndexError):
                matched_gt_text = raw_result.get("matched_ground_truth")

        # Quality attribute match: extracted snake_case value must appear in GT categories
        qa = req.quality_attribute  # already normalised to snake_case or None
        if matched_gt_item is not None and qa is not None:
            qa_match = qa in matched_gt_item.categories
        else:
            qa_match = False

        overlap_score = _safe_int(raw_result.get("overlap_score"), 1, 5, 1)
        per_requirement.append(RequirementEvaluation(
            extracted_requirement=req.requirement,
            matched_ground_truth=matched_gt_text,
            overlap_score=overlap_score,
            overlap_label=_OVERLAP_LABELS.get(overlap_score, "Unknown"),
            quality_attribute_match=qa_match,
            confidence_quality=_safe_int(raw_result.get("confidence_quality")),
            rationale_quality=_safe_int(raw_result.get("rationale_quality")),
            gap_detection_quality=_safe_int(raw_result.get("gap_detection_quality")),
            is_hit=overlap_score >= 3,
            is_duplicate=is_duplicate,
        ))

    # Compute metrics
    non_duplicate_hits = [r for r in per_requirement if r.is_hit and not r.is_duplicate]
    hit_count = len(non_duplicate_hits)
    extracted_count = len(per_requirement)
    gt_count = len(gt_items)

    precision = hit_count / extracted_count if extracted_count > 0 else 0.0
    recall = hit_count / gt_count if gt_count > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # Quality attribute accuracy: among text-overlap hits, fraction with correct attribute
    text_hits = [r for r in per_requirement if r.overlap_score >= 3 and not r.is_duplicate]
    qa_accuracy = (
        sum(1 for r in text_hits if r.quality_attribute_match) / len(text_hits)
        if text_hits else 0.0
    )

    def _avg(field: str) -> float:
        vals = [getattr(r, field) for r in per_requirement]
        return round(statistics.mean(vals), 3) if vals else 0.0

    metrics = TraceMetrics(
        hit_count=hit_count,
        extracted_count=extracted_count,
        ground_truth_count=gt_count,
        precision=round(precision, 3),
        recall=round(recall, 3),
        f1=round(f1, 3),
        quality_attribute_accuracy=round(qa_accuracy, 3),
        avg_confidence=round(statistics.mean(r.confidence for r in requirements), 3) if requirements else 0.0,
        avg_overlap_score=_avg("overlap_score"),
        avg_confidence_quality=_avg("confidence_quality"),
        avg_rationale_quality=_avg("rationale_quality"),
        avg_gap_detection_quality=_avg("gap_detection_quality"),
    )

    console.print(f"\n{pfx}[bold green]Metrics[/bold green]")
    console.print(f"{pfx}  Precision: {precision:.3f}  Recall: {recall:.3f}  F1: {f1:.3f}")
    console.print(f"{pfx}  QA accuracy: {qa_accuracy:.3f}  (correct attribute among text-overlap hits)")
    console.print(f"{pfx}  Hits: {hit_count}/{extracted_count} extracted  ({hit_count}/{gt_count} GT covered)")

    result = TraceEvaluation(
        extraction_file=str(extraction_path),
        ground_truth_source=gt_source,
        eval_model=eval_model,
        per_requirement=per_requirement,
        metrics=metrics,
    )
    return _write_evaluation(result, extraction_path, out_dir)


def _write_evaluation(evaluation: TraceEvaluation, extraction_path: Path, out_dir: Path) -> TraceEvaluation:
    stem = extraction_path.stem  # e.g. trace_20260302_120000_mistral-small
    eval_path = out_dir / f"{stem}_eval.json"
    eval_path.write_text(evaluation.model_dump_json(indent=2), encoding="utf-8")
    console.print(f"  [blue]Evaluation saved:[/blue] {eval_path.name}")
    return evaluation
