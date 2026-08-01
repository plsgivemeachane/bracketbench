"""The scoreboard calculator (ADR-0003, ADR-0002 weights).

A scoreboard is a named weighting over the four peer test scores, producing one aggregate
0-100 number. The same four test scores feed every scoreboard; different scoreboards apply
different weights. A weight of 0 excludes a test from that scoreboard, and the remaining
weights are renormalized over the nonzero-weighted tests that actually ran.

A test that did not run (absent from the scores dict) is excluded, not scored 0 — this lets a
T1-only partial scoreboard be computable before T2/T3/T4 exist.
"""

from typing import Dict


class Scoreboard:
    """A weighted-mean aggregate over the four peer test scores.

    Args:
        weights: A map from test name (``"T1"``/``"T2"``/``"T3"``/``"T4"``) to its weight.
            A weight of 0 excludes a test. Weights are renormalized over the tests that
            actually have a score and a nonzero weight.
    """

    def __init__(self, weights: Dict[str, float]) -> None:
        self._weights = dict(weights)

    def aggregate(self, scores: Dict[str, float]) -> float:
        """Compute the weighted-mean aggregate over the available test scores.

        Args:
            scores: A map from test name to its 0-100 score. Only tests present here are
                considered "available"; absent tests are excluded (not scored 0).

        Returns:
            The aggregate score in 0-100. If no test has both a score and a nonzero weight,
            returns 0.0.
        """
        numerator = 0.0
        denominator = 0.0
        for test_name, weight in self._weights.items():
            if weight == 0:
                continue
            if test_name not in scores:
                continue
            numerator += weight * scores[test_name]
            denominator += weight

        if denominator == 0:
            return 0.0
        # Round to strip floating-point noise (e.g. 100.00000000000001); scores are 0-100
        # so 4 decimals is ample precision for a headline aggregate.
        return round(numerator / denominator, 4)
