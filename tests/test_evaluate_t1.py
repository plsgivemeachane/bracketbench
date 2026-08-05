"""End-to-end integration test for the T1 tracer bullet (issue #3).

The primary seam is the top-level ``evaluate`` entry point run against a stub model that
returns a canned edit script. This is the vertical slice: Breaker breaks a valid JSON
document, the T1 prompt is built, the stub model "emits" an edit script, the applier applies
it, the scorer scores the repaired text, and ``evaluate`` returns an Evaluation carrying the
T1 score (0-100) plus the computable partial scoreboard aggregate — with enough retained to
audit the run (raw edit script, repaired text, breakage record, tier reached).

This test never reaches inside internals; it drives the whole slice through ``evaluate``.
"""

import unittest
from typing import Dict, Any, Optional, List

from bracketbench.llms.base import LLMInterface
from bracketbench.repair.evaluate import evaluate, Evaluation


class StubModel(LLMInterface):
    """A stub LLM that returns a canned edit script for any prompt (no network)."""

    def __init__(self, canned_output: str, model_name: str = "stub") -> None:
        super().__init__(model_name)
        self._canned_output = canned_output
        self._is_initialized = True

    def initialize(self) -> None:
        self._is_initialized = True

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs: Any,
    ) -> str:
        return self._canned_output

    def generate_batch(
        self,
        prompts: List[str],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs: Any,
    ) -> List[str]:
        return [self.generate(p, max_tokens, temperature, **kwargs) for p in prompts]

    def get_model_info(self) -> Dict[str, Any]:
        return {"name": self.model_name, "provider": "stub"}


class TestEvaluateT1TracerBullet(unittest.TestCase):
    """The T1 vertical slice, driven through the public ``evaluate`` seam."""

    # --- The tracer bullet: a correct repair earns the exact-fidelity tier (100) ---

    def test_correct_repair_earns_exact_fidelity_tier(self) -> None:
        """Break '{"a": 1}', the stub emits the ADR-0006 repair script, evaluate scores 100."""
        valid_json = '{"a": 1}'
        bracket_index = 0
        # The worked-example repair script from ADR-0006.
        canned_script = '[{"old": ": 1", "new": ": 1}"}]'
        model = StubModel(canned_script)

        result = evaluate(model, t1_cases=[(valid_json, bracket_index)], t2_cases=[], t3_cases=[], t4_cases=[])

        self.assertIsInstance(result, Evaluation)
        # T1 score is 100 (exact-fidelity tier).
        self.assertEqual(result.t1_score, 100)
        self.assertEqual(result.tier, "exact_fidelity")
        # Audit data is retained.
        self.assertEqual(result.raw_edit_script, canned_script)
        self.assertEqual(result.repaired_text, '{"a": 1}')
        # The breakage record is retained (track, don't tell).
        self.assertEqual(result.breakage_record.break_type, "deleted_closing_bracket")
        self.assertEqual(result.breakage_record.deleted_char, "}")

    # --- A parseable-but-wrong repair earns the structural tier (50) ---

    def test_parseable_but_wrong_repair_earns_structural_tier(self) -> None:
        """A repair that yields valid but wrong JSON scores the structural tier (50)."""
        valid_json = '{"a": 1}'
        # Repair the brace AND change the value to 2 -> parses, but wrong value.
        canned_script = '[{"old": ": 1", "new": ": 2}"}]'
        model = StubModel(canned_script)

        result = evaluate(model, t1_cases=[(valid_json, 0)], t2_cases=[], t3_cases=[], t4_cases=[])

        self.assertEqual(result.t1_score, 50)
        self.assertEqual(result.tier, "structural")
        self.assertEqual(result.repaired_text, '{"a": 2}')

    # --- A malformed script earns the 0.0 tier (0) ---

    def test_malformed_script_earns_zero_tier(self) -> None:
        """A stub that emits non-JSON maps to the 0.0 tier; the broken text is scored as-is."""
        valid_json = '{"a": 1}'
        model = StubModel("not json at all")

        result = evaluate(model, t1_cases=[(valid_json, 0)], t2_cases=[], t3_cases=[], t4_cases=[])

        self.assertEqual(result.t1_score, 0)
        self.assertEqual(result.tier, "unparseable")
        # Repaired text is the broken text unchanged (applier discards the malformed script).
        self.assertEqual(result.repaired_text, '{"a": 1')

    # --- The partial scoreboard is computable from the T1 score alone ---

    def test_partial_scoreboard_aggregate_from_t1_only(self) -> None:
        """evaluate returns the computable partial scoreboard (T1-only) alongside T1."""
        valid_json = '{"a": 1}'
        model = StubModel('[{"old": ": 1", "new": ": 1}"}]')

        result = evaluate(model, t1_cases=[(valid_json, 0)], t2_cases=[], t3_cases=[], t4_cases=[])

        # T1-only partial scoreboard: T1=100, others unavailable -> aggregate 100.
        self.assertEqual(result.scoreboard, 100.0)


if __name__ == "__main__":
    unittest.main()

