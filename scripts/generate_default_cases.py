"""One-time generator for the vendored T1 default case set (issue #12, spec #8).

Runs the 24-triple grid builder (``build_t1_grid``, issue #11) and writes the
frozen fixture ``bracketbench/benchmarking/default_t1_cases.json`` -- 24
records, each ``{"valid_json", "bracket_index", "metadata"}`` where metadata
carries the case's ``(tier, deletion_depth, seed)`` triple plus the tier's axis
levels.

The builder is deterministic (every case is reproducible from its
``(seed, tier, deletion_depth)`` triple), so re-running this script produces a
byte-identical fixture; parameter changes produce visible diffs between
benchmark versions (spec #8, user story 6).

Usage (from the repo root):

    py scripts/generate_default_cases.py
"""

from __future__ import annotations

import json
import pathlib
import sys

# Make the repo root importable when run as `py scripts/generate_default_cases.py`
# (script directories are otherwise the first entry on sys.path).
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from bracketbench.benchmarking.verifier import build_t1_grid  # noqa: E402


def main() -> None:
    """Build all 24 verified cases and write the frozen fixture file."""
    records = [
        {
            "valid_json": valid_json,
            "bracket_index": bracket_index,
            "metadata": metadata,
        }
        for valid_json, bracket_index, metadata in build_t1_grid()
    ]

    fixture_path = _REPO_ROOT / "bracketbench" / "benchmarking" / "default_t1_cases.json"
    payload = json.dumps(records, indent=2, ensure_ascii=True) + "\n"
    fixture_path.write_text(payload, encoding="utf-8")

    print(f"wrote {len(records)} verified cases to {fixture_path}")
    for tier, deletion_depth, seed in sorted(
        (r["metadata"]["tier"], r["metadata"]["deletion_depth"], r["metadata"]["seed"])
        for r in records
    ):
        print(f"  {tier:8s} {deletion_depth:10s} seed={seed}")


if __name__ == "__main__":
    main()
