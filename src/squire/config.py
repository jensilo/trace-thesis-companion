import os
from pathlib import Path

OUTPUT_DIR = Path("outputs")
DATA_DIR = Path(__file__).parent.parent.parent / "data"

# ── Synthesis model defaults ──────────────────────────────────────────────────
DEFAULT_SYNTHESIS_MODEL = "gpt-5-mini"

SYNTHESIS_MODEL_ALIASES: dict[str, str] = {
    "gpt": "gpt-5-mini",
    "gpt-pro": "gpt-5.2",
    "claude": "litellm/anthropic/claude-haiku-4-5"
}

# ── Evaluation model defaults ─────────────────────────────────────────────────
DEFAULT_QUALITY_JUDGE_MODEL = os.environ.get("SQUIRE_QUALITY_JUDGE_MODEL", "openai/gpt-5-mini")
DEFAULT_META_JUDGE_MODEL = os.environ.get("SQUIRE_META_JUDGE_MODEL", "openai/gpt-5-mini")
DEFAULT_QUALITY_SAMPLES = int(os.environ.get("SQUIRE_QUALITY_SAMPLES", "3"))
DEFAULT_META_SAMPLES = int(os.environ.get("SQUIRE_META_SAMPLES", "3"))

JUDGE_MODEL_ALIASES: dict[str, str] = {
    "gpt": "openai/gpt-5-mini",
    "claude": "anthropic/claude-haiku-4-5",
    "mistral": "mistral/mistral-small-latest",
    "mistral-pro": "mistral/mistral-large-latest"
}

# Reasoning models (GPT o-series / gpt-5) use reasoning_effort; others use temperature=0.
DEFAULT_EVAL_REASONING_EFFORT = "minimal"
REASONING_MODEL_PREFIXES = (
    "gpt-5", "o1", "o3", "o4",
    "openai/gpt-5", "openai/o1", "openai/o3", "openai/o4",
)

# ── Summarization defaults ────────────────────────────────────────────────────
DEFAULT_SUMMARIZE_DRAFT_MODEL = "anthropic/claude-haiku-4-5"
DEFAULT_SUMMARIZE_AGGREGATE_MODEL = "anthropic/claude-sonnet-4-6"
DEFAULT_SUMMARIZE_SAMPLES = 3
SUMMARIZE_TEMPERATURE = 0.7
