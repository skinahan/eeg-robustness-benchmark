"""
Repository root and data-path helpers.

Used so scripts work on any machine (Windows, Linux, cluster) without hardcoded
absolute paths. ``get_project_root()`` is the directory containing this file.
"""
from __future__ import annotations

from pathlib import Path


def get_project_root() -> Path:
    """Absolute path to the repository root."""
    return Path(__file__).resolve().parent


def resolve_architecture_json_path(
    filename: str = "best_architecture_4_trial_178.json",
) -> Path:
    """
    Return path to a wiring JSON under ``outputs/architectures/`` or the
    alternate ``architecture_refinement/outputs/architectures/`` layout.
    """
    root = get_project_root()
    candidates = [
        root / "outputs" / "architectures" / filename,
        root / "architecture_refinement" / "outputs" / "architectures" / filename,
    ]
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError(
        "Architecture JSON not found. Tried:\n"
        + "\n".join(f"  - {p}" for p in candidates)
    )


def nas_pilot_dir(run_subdir: str) -> Path:
    """``architecture_refinement/outputs/nas_pilot/<run_subdir>`` under the repo root."""
    return (
        get_project_root()
        / "architecture_refinement"
        / "outputs"
        / "nas_pilot"
        / run_subdir
    )
