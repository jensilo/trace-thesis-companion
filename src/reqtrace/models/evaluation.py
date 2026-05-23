from pydantic import BaseModel


class RequirementEvaluation(BaseModel):
    extracted_requirement: str
    matched_ground_truth: str | None
    overlap_score: int
    overlap_label: str
    quality_attribute_match: bool
    confidence_quality: int
    rationale_quality: int
    gap_detection_quality: int
    is_hit: bool        # overlap_score >= 3 (text retrieval only)
    is_duplicate: bool


class TraceMetrics(BaseModel):
    hit_count: int
    extracted_count: int
    ground_truth_count: int
    precision: float
    recall: float
    f1: float
    quality_attribute_accuracy: float   # fraction of text-retrieval hits where the quality attribute label is also correct
    avg_confidence: float
    avg_overlap_score: float
    avg_confidence_quality: float
    avg_rationale_quality: float
    avg_gap_detection_quality: float


class TraceEvaluation(BaseModel):
    extraction_file: str
    ground_truth_source: str
    eval_model: str
    per_requirement: list[RequirementEvaluation]
    metrics: TraceMetrics


class TraceCorpusMeta(BaseModel):
    corpus_id: str
    trace_run_id: str
    model: str
    eval_model: str | None
    generated_at: str
    total_elapsed_seconds: float
    overall: dict
    by_model: dict
    per_transcript: list[dict]
