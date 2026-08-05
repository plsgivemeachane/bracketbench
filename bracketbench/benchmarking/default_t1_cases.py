"""Loader for the vendored T1 default case set (issue #12, spec #8).

The frozen fixture ``default_t1_cases.json`` holds 24 verified
ambiguous-yet-solvable cases (see :mod:`bracketbench.benchmarking.verifier`),
each ``{"valid_json", "bracket_index", "metadata"}``. This module exposes the
loader seam the post-ADR-0001 architecture consumes: callers pass the
``(valid_json, bracket_index)`` pairs explicitly to
:func:`bracketbench.repair.evaluate.evaluate` -- there is no module-level
default case list today.

``load_default_t1_cases`` returns the pair shape ``evaluate`` consumes;
``load_default_t1_records`` returns the full records (pairs plus metadata) for
per-cell failure attribution.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple

__all__ = ["load_default_t1_cases", "load_default_t1_records"]

# The fixture ships next to this module (committed, vendored -- not generated at
# import time; see scripts/generate_default_cases.py).
_FIXTURE_PATH = Path(__file__).resolve().with_name("default_t1_cases.json")


@lru_cache(maxsize=1)
def load_default_t1_records() -> List[Dict[str, Any]]:
    """Load the vendored fixture and return its 24 records.

    Returns:
        A list of ``{"valid_json": str, "bracket_index": int, "metadata": dict}``
        records, in fixture order.

    Raises:
        FileNotFoundError: If ``default_t1_cases.json`` is missing.
        ValueError: If the fixture is malformed (not a list of well-shaped
            records).
    """
    with open(_FIXTURE_PATH, "r", encoding="utf-8") as handle:
        records = json.load(handle)

    if not isinstance(records, list):
        raise ValueError(f"{_FIXTURE_PATH.name} must contain a JSON array")
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"fixture record {index} is not an object")
        if not isinstance(record.get("valid_json"), str):
            raise ValueError(f"fixture record {index} has no valid_json string")
        if not isinstance(record.get("bracket_index"), int):
            raise ValueError(f"fixture record {index} has no bracket_index int")
        if not isinstance(record.get("metadata"), dict):
            raise ValueError(f"fixture record {index} has no metadata object")
    return records


def load_default_t1_cases() -> List[Tuple[str, int]]:
    """Return the 24 default T1 cases as ``(valid_json, bracket_index)`` pairs.

    This is the pair shape :func:`bracketbench.repair.evaluate.evaluate`
    consumes, so the vendored set plugs into the existing seam with no
    interface change.
    """
    return [
        (record["valid_json"], record["bracket_index"])
        for record in load_default_t1_records()
    ]
