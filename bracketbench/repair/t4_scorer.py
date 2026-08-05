"""The T4 structural-only scorer (CONTEXT.md).

T4 cases are genuinely broken JSON collected from the wild: there is no Original object, so
fidelity is unscorable. Only the Structural tier is reachable, and it is the ceiling — worth
100, not the T1/T2 ladder's 0.5. The scorer awards:

  - **0   unparseable**  — the repaired output does not parse via ``json.loads``.
  - **100 structural**   — the repaired output parses. Structural is the ceiling; there is
                          no higher tier to reach and no Original to compare against.

Structural is a binary pass/fail here: there is no value to compare, so "parses" is the only
graded property. Unlike the T1/T2 ladder (ADR-0002), tier scores are not configurable — the
ceiling is fixed at 100 per CONTEXT.md.
"""

import json

from bracketbench.repair.scoring import ScoreResult


class T4Scorer:
    """Scores repaired T4 text: 0 for unparseable, 100 for parseable (structural ceiling)."""

    def score(self, repaired_text: str) -> ScoreResult:
        """Score ``repaired_text`` on the T4 structural-only binary.

        Args:
            repaired_text: The text the applier produced (the model's repair attempt).

        Returns:
            A ``ScoreResult`` with tier ``"unparseable"`` (score 0) or ``"structural"``
            (score 100).
        """
        try:
            json.loads(repaired_text)
        except (json.JSONDecodeError, ValueError):
            return ScoreResult(tier="unparseable", score=0)
        return ScoreResult(tier="structural", score=100)
