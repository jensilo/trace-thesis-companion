import os
from pathlib import Path

OUTPUT_DIR = Path("outputs")

DEFAULT_MODEL = os.environ.get("ISO_LABEL_MODEL", "mistral/mistral-small-latest")
DEFAULT_TEMPERATURE = 0.5
DEFAULT_CONCURRENCY = 10
DEFAULT_RUNS = 5
