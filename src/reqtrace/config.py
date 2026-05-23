import os
from pathlib import Path

OUTPUT_DIR = Path("outputs")

DEFAULT_EXTRACTION_MODEL = "mistral/mistral-small-latest"

EXTRACTION_MODEL_ALIASES: dict[str, str] = {
    "tiny":   "mistral/ministral-8b-latest",
    "small":  "mistral/mistral-small-latest",
    "medium": "mistral/mistral-medium-latest",
    "large":  "mistral/mistral-large-latest",
    "gpt": "openai/gpt-5-mini",
}

DEFAULT_EVAL_JUDGE_MODEL = os.environ.get("TRACE_EVAL_MODEL", "openai/gpt-5-mini")

# Reasoning models use reasoning_effort; all others use temperature=0.
EVAL_REASONING_EFFORT = "minimal"
REASONING_MODEL_PREFIXES = (
    "gpt-5", "o1", "o3", "o4",
    "openai/gpt-5", "openai/o1", "openai/o3", "openai/o4",
)
