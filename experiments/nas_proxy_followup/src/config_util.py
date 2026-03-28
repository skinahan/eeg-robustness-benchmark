"""Load and merge YAML configs for nas_proxy_followup scripts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, MutableMapping

import yaml


def deep_merge(base: MutableMapping[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override into base (mutates base)."""
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            deep_merge(base[k], v)
        else:
            base[k] = v
    return dict(base)


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def load_merged_configs(config_paths: List[Path]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for p in config_paths:
        merged = deep_merge(merged, load_yaml(p))
    return merged


def experiment_root_from_config(cfg: Dict[str, Any]) -> Path:
    """Package root `experiments/nas_proxy_followup/` (parent of `src/`)."""
    root = cfg.get("nas_proxy_root")
    if root:
        return Path(root).resolve()
    return Path(__file__).resolve().parent.parent


def repo_root_from_config(cfg: Dict[str, Any]) -> Path:
    """Repository root (parent of `experiments/`)."""
    rr = cfg.get("repo_root")
    if rr:
        return Path(rr).resolve()
    # experiments/nas_proxy_followup/src/config_util.py -> parents: src, nas_proxy_followup, experiments, repo
    return Path(__file__).resolve().parent.parent.parent.parent


def resolve_path(cfg: Dict[str, Any], key: str, default_relative: str) -> Path:
    """Resolve paths in YAML (relative to repo root, as in `experiments/nas_proxy_followup/...`)."""
    raw = cfg.get(key, default_relative)
    p = Path(raw)
    if p.is_absolute():
        return p.resolve()
    return (repo_root_from_config(cfg) / p).resolve()


def add_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        action="append",
        dest="configs",
        default=[],
        metavar="PATH",
        help="YAML config file (repeat to merge, later overrides earlier)",
    )


def parse_config_paths(args: argparse.Namespace) -> List[Path]:
    if not args.configs:
        raise SystemExit("At least one --config is required")
    return [Path(p).resolve() for p in args.configs]
