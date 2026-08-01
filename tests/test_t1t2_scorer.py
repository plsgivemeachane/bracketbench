"""Tests for the T1/T2 4-tier scorer (ADR-0002 configurable tiers).

The scorer awards one score in [0.0, 1.0] per run, scaled to 0-100:
  - 0.0  unparseable       (repaired output does not parse)
  - 0.5  structural        (parses, but wrong value)          [default, configurable]
  - 0.9  value-fidelity    (lenient == : value-equal, types may differ)
  - 1.0  exact-fidelity    (type-aware deep equality)

Tier scores come from a config object (ADR-0002), not module-level constants. Tests target
the public ``T1T2Scorer.score`` seam only.
"""

import unittest

from bracketbench.repair.scoring import T1T2Scorer, TierScoreConfig, ScoreResult


class TestT1T2Scorer(unittest.TestCase):
    """Tests for the public ``T1T2Scorer.score(repaired_text, original_object)`` seam."""

    def _default_scorer(self) -> T1T2Scorer:
        return T1T2Scorer(TierScoreConfig())

    # --- Slice 1: unparseable output -> 0.0 tier -> 0 score ---

    def test_unparseable_output_scores_zero(self) -> None:
        """Repaired text that is not valid JSON gets the 0.0 tier (scaled to 0)."""
        scorer = self._default_scorer()
        result = scorer.score('{"a": 1', original_object={"a": 1})

        self.assertEqual(result.tier, "unparseable")
        self.assertEqual(result.score, 0)

    # --- Slice 2: parseable but wrong value -> structural tier (default 0.5 -> 50) ---

    def test_parseable_wrong_value_scores_structural_default(self) -> None:
        """Valid JSON that is not value-equal gets the structural tier (0.5 -> 50)."""
        scorer = self._default_scorer()
        # Parses fine, but to a different object.
        result = scorer.score('{"a": 2}', original_object={"a": 1})

        self.assertEqual(result.tier, "structural")
        self.assertEqual(result.score, 50)

    # --- Slice 3: value-fidelity tier (lenient ==, 1 == 1.0) ---

    def test_value_equal_under_lenient_eq_scores_value_fidelity(self) -> None:
        """1.0 is value-equal to 1 under lenient ==, so the value-fidelity tier (0.9 -> 90)."""
        scorer = self._default_scorer()
        # original has int 1; repaired parses to float 1.0 — value-equal, not type-equal.
        result = scorer.score('{"a": 1.0}', original_object={"a": 1})

        self.assertEqual(result.tier, "value_fidelity")
        self.assertEqual(result.score, 90)

    # --- Slice 4: exact-fidelity tier (type-aware deep equality) ---

    def test_exact_type_aware_equal_scores_exact_fidelity(self) -> None:
        """Type-and-value-equal output gets the exact-fidelity tier (1.0 -> 100)."""
        scorer = self._default_scorer()
        result = scorer.score('{"a": 1}', original_object={"a": 1})

        self.assertEqual(result.tier, "exact_fidelity")
        self.assertEqual(result.score, 100)

    def test_int_vs_float_blocks_exact_fidelity(self) -> None:
        """1.0 is NOT exact-faithful to 1 (type differs), so it stops at value-fidelity."""
        scorer = self._default_scorer()
        result = scorer.score('{"a": 1.0}', original_object={"a": 1})

        # value-equal (lenient ==) but not type-equal -> value_fidelity, not exact.
        self.assertEqual(result.tier, "value_fidelity")
        self.assertNotEqual(result.tier, "exact_fidelity")

    # --- Slice 5: tier scores are configurable, not hardcoded (ADR-0002) ---

    def test_structural_tier_score_is_configurable(self) -> None:
        """Tuning structural to 0.2 rescales the structural score without code changes."""
        config = TierScoreConfig(structural=0.2)
        scorer = T1T2Scorer(config)
        result = scorer.score('{"a": 2}', original_object={"a": 1})

        self.assertEqual(result.tier, "structural")
        self.assertEqual(result.score, 20)

    def test_value_fidelity_tier_score_is_configurable(self) -> None:
        """Tuning value-fidelity to 0.85 rescales that tier."""
        config = TierScoreConfig(value_fidelity=0.85)
        scorer = T1T2Scorer(config)
        result = scorer.score('{"a": 1.0}', original_object={"a": 1})

        self.assertEqual(result.tier, "value_fidelity")
        self.assertEqual(result.score, 85)

    # --- Slice 6: structural is a prerequisite for fidelity tiers ---

    def test_value_equal_but_unparseable_still_scores_zero(self) -> None:
        """If the text does not parse, fidelity is unreachable; the 0.0 tier stands."""
        scorer = self._default_scorer()
        result = scorer.score('{"a": 1', original_object={"a": 1})

        self.assertEqual(result.tier, "unparseable")
        self.assertEqual(result.score, 0)


if __name__ == "__main__":
    unittest.main()
