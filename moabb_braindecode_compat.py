"""
braindecode 1.2.0 imports ``BNCI2014001`` from ``moabb.datasets``; MOABB >=1.4 only
registers ``BNCI2014_001`` (see MOABB docs). Import this module before any braindecode
import so the legacy name resolves to the official class.
"""
from pathlib import Path

import moabb.datasets as _moabb_datasets
from moabb.datasets import BNCI2014_001

_moabb_datasets.BNCI2014001 = BNCI2014_001


def _patch_moabb_download_path_sanitization() -> None:
    """Fix MOABB 1.5.x Windows bug: ``_sanitize_path`` replaces ``:`` in full paths.

    That turns ``C:\\`` into ``C-\\``, which is not an absolute path; downloads then
    land under the process cwd (e.g. ``.../moabb_experiments/C-/Users/...``). Upstream
    MOABB only sanitizes the relative segment; we drop ``:`` from the replacement map.
    """
    try:
        import moabb.datasets.download as _dl
    except ImportError:
        return

    def _sanitize_path(path: Path) -> Path:
        table = {ord(c): "-" for c in '*?"<>|'}
        return Path(str(path).translate(table))

    _dl._sanitize_path = _sanitize_path


_patch_moabb_download_path_sanitization()


def fix_moabb_lee2019_session_filter(dataset) -> None:
    """Work around MOABB Lee2019 + BaseDataset session filtering mismatch.

    Lee2019 uses dict keys ``str(session - 1)`` (``\"0\"``, ``\"1\"`` for the two days).
    ``selected_sessions=(1, 2)`` becomes ``{\"1\", \"2\"}`` in ``get_data``, which keeps
    only key ``\"1\"`` and drops session ``\"0\"``. That leaves a single session in the
    paradigm table so CrossSession / ``LeaveOneGroupOut`` fails. Clearing the filter
    restores both sessions while ``self.sessions`` still controls loading in
    ``_get_single_subject_data``.
    """
    code = getattr(dataset, "code", "") or ""
    if not str(code).startswith("Lee2019-"):
        return
    dataset._selected_sessions = None


__all__ = ["fix_moabb_lee2019_session_filter"]
