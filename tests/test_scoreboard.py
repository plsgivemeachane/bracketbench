"""Tests for the scoreboard calculator (ADR-0003, ADR-0002 weights).

A scoreboard is a weighted arithmetic mean over the four peer test scores, producing one
aggregate 0-100 number. A weight of 0 excludes a test from that scoreboard; the remaining
weights are renormalized over the nonzero-weighted tests. Tests target only the public
``Scoreboard.aggregate`` seam.
"""

import unittest

from bracketbench.repair.scoreboard import Scoreboard


class TestScoreboard(unittest.TestCase):
    """Tests for the public ``Scoreboard.aggregate(scores)`` seam."""

    # --- Slice 1: weighted mean over all four tests ---

    def test_weighted_mean_over_all_four_tests(self) -> None:
        """0.4*T1 + 0.3*T2 + 0.2*T3 + 0.1*T4 (unified default) over 0-100 scores."""
        board = Scoreboard(weights={"T1": 0.4, "T2": 0.3, "T3": 0.2, "T4": 0.1})
        # All 100 -> 100.
        self.assertEqual(
            board.aggregate({"T1": 100, "T2": 100, "T3": 100, "T4": 100}),
            100.0,
        )
        # 0.4*100 + 0.3*80 + 0.2*60 + 0.1*40 = 40 + 24 + 12 + 4 = 80
        self.assertEqual(
            board.aggregate({"T1": 100, "T2": 80, "T3": 60, "T4": 40}),
            80.0,
        )

    # --- Slice 2: weight 0 excludes a test (without-ipynb renormalization) ---

    def test_weight_zero_excludes_and_renormalizes(self) -> None:
        """T3 weight 0 + renormalize over T1/T2/T4 (the without-ipynb scoreboard)."""
        # T3 excluded; weights 0.4/0.3/0.1 sum to 0.8 -> renormalized to 0.5/0.375/0.125.
        board = Scoreboard(weights={"T1": 0.4, "T2": 0.3, "T3": 0.0, "T4": 0.1})
        # All nonzero tests = 100 -> 100.
        self.assertEqual(
            board.aggregate({"T1": 100, "T2": 100, "T4": 100}),
            100.0,
        )
        # T1=100, T2=0, T4=0 -> 100 * (0.4/0.8) = 50
        self.assertEqual(
            board.aggregate({"T1": 100, "T2": 0, "T4": 0}),
            50.0,
        )

    # --- Slice 3: a T1-only partial scoreboard is computable ---

    def test_t1_only_partial_scoreboard_is_computable(self) -> None:
        """Only T1 available; the aggregate is T1's score when the others are weight 0."""
        board = Scoreboard(weights={"T1": 1.0, "T2": 0.0, "T3": 0.0, "T4": 0.0})
        self.assertEqual(board.aggregate({"T1": 72}), 72.0)

    # --- Slice 4: missing tests are treated as unavailable (not as zero) ---

    def test_missing_tests_are_excluded_not_treated_as_zero(self) -> None:
        """A test absent from the scores dict is excluded, not scored 0 (no Original/no run)."""
        board = Scoreboard(weights={"T1": 0.4, "T2": 0.3, "T3": 0.2, "T4": 0.1})
        # Only T1 ran (score 60). Renormalize: 60 * (0.4/0.4) = 60.
        self.assertEqual(board.aggregate({"T1": 60}), 60.0)

    # --- Slice 5: all weights zero / no available tests is a defined no-op ---

    def test_no_available_tests_returns_zero(self) -> None:
        """If no test has both a score and a nonzero weight, the aggregate is 0."""
        board = Scoreboard(weights={"T1": 0.0, "T2": 0.0})
        self.assertEqual(board.aggregate({}), 0.0)


if __name__ == "__main__":
    unittest.main()
