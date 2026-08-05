"""Default case sets for all four tests (issue #7).

Centralizes the default case sets so ``evaluate(model)`` runs out-of-the-box with no
custom cases. T1 loads from the vendored fixture (issue #12); T2, T3, and T4 ship
curated constants. Each default set is also independently usable via its own loader.

The without-ipynb scoreboard weights are also defined here for convenience.
"""

from typing import Dict, List, Tuple

from bracketbench.benchmarking.default_t1_cases import load_default_t1_cases
from bracketbench.repair.t3_cases import DEFAULT_T3_CASES
from bracketbench.repair.t4 import DEFAULT_T4_CASES

__all__ = [
    "default_t1_cases",
    "DEFAULT_T2_CASES",
    "default_t3_cases",
    "default_t4_cases",
    "UNIFIED_SCOREBOARD_WEIGHTS",
    "WITHOUT_IPYNB_SCOREBOARD_WEIGHTS",
]


def default_t1_cases() -> List[Tuple[str, int]]:
    """Return the 24 vendored T1 default cases as ``(valid_json, bracket_index)`` pairs."""
    return load_default_t1_cases()


# T2 default cases: multi-breakage on simple valid JSON documents.
# Each tuple is (valid_json, bracket_indices) where bracket_indices is a non-empty
# list of per-step bracket indexes. These are small, deterministic, and cover 2-3
# breakages at different depths.
DEFAULT_T2_CASES: List[Tuple[str, List[int]]] = [
    # 2 breakages on a flat object: remove both the inner ] and the outer }
    ('{"a": [1, 2], "b": [3, 4]}', [0, 0]),
    # 2 breakages on a nested object: remove inner } and outer }
    ('{"outer": {"inner": {"x": 1}}}', [0, 0]),
    # 3 breakages: remove all three closing brackets in sequence
    ('{"a": {"b": [1]}, "c": [2]}', [0, 0, 0]),
]


def default_t3_cases() -> List[str]:
    """Return the curated T3 default cases (broken notebook JSON strings)."""
    return list(DEFAULT_T3_CASES)


def default_t4_cases() -> List[str]:
    """Return the curated T4 default cases (broken JSON strings from the wild)."""
    return list(DEFAULT_T4_CASES)


# Default unified-scoreboard weights (CONTEXT.md): 0.4*T1 + 0.3*T2 + 0.2*T3 + 0.1*T4.
UNIFIED_SCOREBOARD_WEIGHTS: Dict[str, float] = {
    "T1": 0.4,
    "T2": 0.3,
    "T3": 0.2,
    "T4": 0.1,
}

# Without-ipynb scoreboard: T3 weight 0, remaining weights renormalized over T1, T2, T4.
# Original: T1=0.4, T2=0.3, T4=0.1 -> sum=0.8 -> renormalized: T1=0.5, T2=0.375, T4=0.125.
WITHOUT_IPYNB_SCOREBOARD_WEIGHTS: Dict[str, float] = {
    "T1": 0.4 / 0.8,
    "T2": 0.3 / 0.8,
    "T3": 0.0,
    "T4": 0.1 / 0.8,
}
