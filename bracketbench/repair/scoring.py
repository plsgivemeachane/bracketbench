"""The T1/T2 4-tier scorer (ADR-0002 configurable tiers).

Each T1/T2 run yields one score in [0.0, 1.0], scaled to 0-100, on a four-tier ladder:

  - **0.0  unparseable**     — the repaired output does not parse via ``json.loads``.
  - **0.5  structural**      — parses, but is not value-equal to the Original object.
                              (default; configurable)
  - **0.9  value-fidelity**  — value-equal under lenient ``==`` (``1 == 1.0`` counts).
                              (default; configurable)
  - **1.0  exact-fidelity**  — type-aware deep equality (``1.0`` is NOT exact to ``1``).
                              (default; configurable)

Structural is a prerequisite for the fidelity tiers: an unparseable output scores 0.0
regardless of value. Tier scores come from a ``TierScoreConfig`` (ADR-0002), not module-level
constants.
"""

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TierScoreConfig:
    """The four tier scores (ADR-0002). Defaults match the MVP: 0.0 / 0.5 / 0.9 / 1.0."""

    unparseable: float = 0.0
    structural: float = 0.5
    value_fidelity: float = 0.9
    exact_fidelity: float = 1.0


@dataclass(frozen=True)
class ScoreResult:
    """The score for one T1/T2 run.

    ``tier`` is the ladder tier reached; ``score`` is the tier score scaled to 0-100.
    """

    tier: str
    score: int


class T1T2Scorer:
    """Scores repaired text against the Original object on the 4-tier ladder.

    The scorer takes a ``TierScoreConfig`` at construction (ADR-0002) so tier values can be
    tuned per-run without code changes.
    """

    def __init__(self, config: TierScoreConfig) -> None:
        self._config = config

    def score(self, repaired_text: str, original_object: Any) -> ScoreResult:
        """Score ``repaired_text`` against ``original_object`` on the 4-tier ladder.

        Args:
            repaired_text: The text the applier produced (the model's repair attempt).
            original_object: The Original object (``json.loads`` of the pre-breakage document).

        Returns:
            A ``ScoreResult`` with the tier reached and the score scaled to 0-100.
        """
        try:
            repaired_object = json.loads(repaired_text)
        except (json.JSONDecodeError, ValueError):
            return ScoreResult(tier="unparseable", score=self._scaled("unparseable"))

        # Structural tier: parses, but is it the right value?
        if not _lenient_equal(repaired_object, original_object):
            return ScoreResult(tier="structural", score=self._scaled("structural"))

        # Value-fidelity tier: value-equal under lenient ==.
        if not _type_aware_equal(repaired_object, original_object):
            return ScoreResult(tier="value_fidelity", score=self._scaled("value_fidelity"))

        # Exact-fidelity tier: type-aware deep equality.
        return ScoreResult(tier="exact_fidelity", score=self._scaled("exact_fidelity"))

    def _scaled(self, tier: str) -> int:
        """Scale a tier's [0.0, 1.0] score to an integer 0-100."""
        value = getattr(self._config, tier)
        return int(round(value * 100))


def _lenient_equal(repaired: Any, original: Any) -> bool:
    """Value equality under lenient ``==`` (CONTEXT.md).

    Uses Python ``==`` after ``json.loads``: key order, whitespace, and escapes are ignored
    by parsing, and ``1 == 1.0`` counts as equal. Recurses so that nested ``1``/``1.0``
    differences are also value-equal (they already are under ``==``).
    """
    return repaired == original


def _type_aware_equal(repaired: Any, original: Any) -> bool:
    """Type-aware deep equality (CONTEXT.md, ADR-0002 exact-fidelity).

    Value *and* JSON type must match. ``1.0`` is NOT exact-faithful to ``1`` because
    ``json.loads`` yields a ``float`` for one and an ``int`` for the other. Recurses through
    dicts and lists so a nested type mismatch is caught.
    """
    # bool is a subclass of int in Python, but JSON treats them as distinct types. Guard it
    # explicitly so True is not "type-equal" to 1.
    if isinstance(repaired, bool) or isinstance(original, bool):
        if isinstance(repaired, bool) != isinstance(original, bool):
            return False
        return repaired == original

    # Numbers: int vs float must differ even when value-equal (1 vs 1.0).
    if isinstance(repaired, (int, float)) and isinstance(original, (int, float)):
        if type(repaired) is not type(original):
            return False
        return repaired == original

    if type(repaired) is not type(original):
        return False

    if isinstance(repaired, dict):
        if repaired.keys() != original.keys():
            return False
        return all(
            _type_aware_equal(repaired[key], original[key]) for key in repaired
        )

    if isinstance(repaired, list):
        if len(repaired) != len(original):
            return False
        return all(
            _type_aware_equal(r, o) for r, o in zip(repaired, original)
        )

    return repaired == original
