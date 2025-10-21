"""Common filesystem paths for the pipeline."""

from __future__ import annotations

from pathlib import Path

# workspace/src/pipeline/paths.py -> parents[2] = workspace
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]

CONFIGS_DIR = WORKSPACE_ROOT / "configs"
DATA_DIR = WORKSPACE_ROOT / "data"
DOCS_DIR = WORKSPACE_ROOT / "docs"
LOGS_DIR = WORKSPACE_ROOT / "logs"
SCRIPTS_DIR = WORKSPACE_ROOT / "scripts"
STATE_DIR = WORKSPACE_ROOT / "state"

__all__ = [
    "WORKSPACE_ROOT",
    "CONFIGS_DIR",
    "DATA_DIR",
    "DOCS_DIR",
    "LOGS_DIR",
    "SCRIPTS_DIR",
    "STATE_DIR",
]
