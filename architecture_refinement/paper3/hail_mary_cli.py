"""Shared argparse helpers for Hail Mary scripts (overwrite semantics)."""

from __future__ import annotations

import argparse
from pathlib import Path


def add_overwrite_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Default: overwrite output files. Use --no-overwrite to skip when output exists.
    """
    g = parser.add_mutually_exclusive_group()
    g.add_argument(
        "--overwrite",
        dest="overwrite",
        action="store_true",
        help="Overwrite output if it exists (default).",
    )
    g.add_argument(
        "--no-overwrite",
        dest="overwrite",
        action="store_false",
        help="Skip writing if the output file already exists.",
    )
    parser.set_defaults(overwrite=True)


def can_write_output(path: Path, *, overwrite: bool) -> bool:
    if overwrite:
        return True
    if path.exists():
        print(f"[skip] Output exists: {path} (use --overwrite to replace)")
        return False
    return True


def results_model_segment(model_name: str) -> str:
    """Matches utils.create_output_path(..., use_short_run_id=True) folder segment."""
    from utils import short_run_id

    return short_run_id(model_name, length=12)
