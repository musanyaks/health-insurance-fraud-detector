"""Single source of truth for configuration. Paths resolve to repo root."""
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]   # was parents[2] — fixed


def _load() -> dict:
    with open(REPO_ROOT / "config" / "config.yaml") as f:
        return yaml.safe_load(f)


CONFIG = _load()


def path(key: str) -> Path:
    return (REPO_ROOT / CONFIG["paths"][key]).resolve()