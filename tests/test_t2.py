"""Tests for the T2 multi-breakage repair test (issue #4).

The primary seams are the Breaker's new multi-breakage composition (public
``Breaker.break_json_multi``) and the top-level ``evaluate`` entry point run against a stub
model (the same StubModel pattern as ``tests/test_evaluate_t1.py``). The T2 prompt must not
disclose how many breakages were applied ("track, don't tell"); T2 is scored on the same
4-tier ladder as T1 (reusing the shared scorer — no duplicate logic); the Evaluation carries
both 0-100 scores and the scoreboard aggregates include both.
"""

import unittest
from typing import Any, Dict, List, Optional

from bracketbench.breaker import BreakRecord, Breaker, BrokenJsonMulti
from bracketbench.llms.base import LLMInterface
from bracketbench.repair.evaluate import Evaluation, evaluate
from bracketbench.repair.t2 import build_t2_prompt


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


class TestT2MultiBreakage(unittest.TestCase):
    """T2 covers the Breaker's multi-breakage composition and the evaluate vertical slice."""

    # --- Slice 1: the Breaker composes N breakages, recording each, retaining Original ---

    def test_break_json_multi_composes_two_breakages_in_order(self) -> None:
        """Two sequential deletions, each from the current text, both recorded in order."""
        valid_json = '{"a": [1, 2], "b": {"c": 3}}'
        result = Breaker().break_json_multi(valid_json, [0, 1])

        self.assertIsInstance(result, BrokenJsonMulti)
        # The Original object is retained (pre-breakage parse).
        self.assertEqual(result.original_object, {"a": [1, 2], "b": {"c": 3}})
        # Both breakages recorded in application order.
        self.assertEqual(len(result.records), 2)
        self.assertIsInstance(result.records[0], BreakRecord)
        self.assertEqual(result.records[0].break_type, "deleted_closing_bracket")
        self.assertEqual(result.records[0].position, 11)
        self.assertEqual(result.records[0].deleted_char, "]")
        self.assertEqual(result.records[1].break_type, "deleted_closing_bracket")
        self.assertEqual(result.records[1].position, 26)
        self.assertEqual(result.records[1].deleted_char, "}")
        # The final broken text is the input minus both deleted brackets.
        self.assertEqual(result.broken_text, '{"a": [1, 2, "b": {"c": 3}')

    def test_break_json_multi_composes_three_breakages(self) -> None:
        """A third breakage is picked from the twice-broken text, not the original."""
        valid_json = '{"a": [1, [2, 3]], "b": {"c": [4, 5]}}'
        result = Breaker().break_json_multi(valid_json, [0, 1, 2])

        self.assertEqual(len(result.records), 3)
        self.assertEqual(
            [(r.position, r.deleted_char) for r in result.records],
            [(15, "]"), (34, "]"), (35, "}")],
        )
        self.assertEqual(result.broken_text, '{"a": [1, [2, 3], "b": {"c": [4, 5}')
        self.assertEqual(
            result.original_object, {"a": [1, [2, 3]], "b": {"c": [4, 5]}}
        )

    def test_break_json_multi_raises_on_out_of_range_index(self) -> None:
        """An index valid for the first step can go stale once brackets are deleted."""
        # After deleting the ']' of '{"a": [1], "b": 2}', only one closer remains, so a
        # second breakage at index 1 is out of range.
        with self.assertRaises(ValueError):
            Breaker().break_json_multi('{"a": [1], "b": 2}', [0, 1])

    def test_break_json_multi_raises_on_empty_indices(self) -> None:
        """T2 composes N > 1 breakages; an empty index list must raise ValueError."""
        with self.assertRaises(ValueError):
            Breaker().break_json_multi('{"a": 1}', [])

    def test_break_json_multi_raises_on_invalid_json(self) -> None:
        """Non-JSON input must raise ValueError, like break_json."""
        with self.assertRaises(ValueError):
            Breaker().break_json_multi("not json at all", [0, 1])

    # --- Slice 2: the T2 prompt does not disclose the breakage count ---

    def test_t2_prompt_instructs_edit_script_without_disclosing_count(self) -> None:
        """The prompt demands an edit script but never states how many breakages exist."""
        # Digit-free broken text, so any digit in the prompt would leak a count.
        prompt = build_t2_prompt('{"a": "x"')

        self.assertIn("EDIT SCRIPT", prompt)
        self.assertIn("one or more", prompt)
        # The T1 phrase "exactly one" must not appear, and no count may be embedded.
        self.assertNotIn("exactly one", prompt)
        self.assertFalse(any(ch.isdigit() for ch in prompt))

    # --- Slice 3: the evaluate vertical slice — a correct T2 repair earns 100 ---

    def test_correct_t2_repair_earns_exact_fidelity_tier(self) -> None:
        """A multi-element edit script that restores both brackets earns exact fidelity."""
        valid_json = '{"a": [1, 2], "b": {"c": 3}}'
        canned_script = (
            '[{"old": "2, \\"b\\"", "new": "2], \\"b\\""}, '
            '{"old": ": 3}", "new": ": 3}}"}]'
        )
        model = StubModel(canned_script)

        result = evaluate(model, t2_cases=[(valid_json, [0, 1])])

        self.assertIsInstance(result, Evaluation)
        self.assertEqual(result.t2_score, 100)
        self.assertEqual(result.t1_score, 0)
        self.assertEqual(len(result.t2_cases), 1)
        run = result.t2_cases[0]
        self.assertEqual(run.tier, "exact_fidelity")
        self.assertEqual(run.repaired_text, valid_json)
        self.assertEqual(run.raw_edit_script, canned_script)
        # The FULL breakage record (all N breakages) is retained for audit.
        self.assertEqual(len(run.breakage_records), 2)
        self.assertEqual(run.breakage_records[0].deleted_char, "]")
        self.assertEqual(run.breakage_records[1].deleted_char, "}")
        # T2-only partial scoreboard renormalizes over T2 -> 100.
        self.assertEqual(result.scoreboard, 100.0)

    # --- Slice 4: parseable-but-wrong earns the structural tier (same ladder as T1) ---

    def test_t2_repair_wrong_value_earns_structural_tier(self) -> None:
        """Repairing the brackets but changing a value scores structural (50), like T1."""
        valid_json = '{"a": [1, 2], "b": {"c": 3}}'
        canned_script = (
            '[{"old": "2, \\"b\\"", "new": "2], \\"b\\""}, '
            '{"old": ": 3}", "new": ": 4}}"}]'
        )
        model = StubModel(canned_script)

        result = evaluate(model, t2_cases=[(valid_json, [0, 1])])

        self.assertEqual(result.t2_score, 50)
        self.assertEqual(result.t2_cases[0].tier, "structural")
        self.assertEqual(result.t2_cases[0].repaired_text, '{"a": [1, 2], "b": {"c": 4}}')

    # --- Slice 5: a malformed script maps to the 0.0 tier (shared ladder) ---

    def test_malformed_t2_script_earns_zero_tier(self) -> None:
        """Non-JSON model output maps to 0; the broken text is scored as-is."""
        valid_json = '{"a": [1, 2], "b": {"c": 3}}'
        model = StubModel("not json at all")

        result = evaluate(model, t2_cases=[(valid_json, [0, 1])])

        self.assertEqual(result.t2_score, 0)
        self.assertEqual(result.t2_cases[0].tier, "unparseable")
        self.assertEqual(result.t2_cases[0].repaired_text, '{"a": [1, 2, "b": {"c": 3}')

    # --- Slice 6: T2 flows through the same 4-tier ladder as T1 (no duplicate scorer) ---

    def test_t2_uses_shared_ladder_reaching_value_fidelity_tier(self) -> None:
        """A type-mismatched (int vs float) but value-equal repair earns value-fidelity (90)."""
        valid_json = '{"a": [1, 2], "b": {"c": 3}}'
        canned_script = (
            '[{"old": "2, \\"b\\"", "new": "2], \\"b\\""}, '
            '{"old": ": 3}", "new": ": 3.0}}"}]'
        )
        model = StubModel(canned_script)

        result = evaluate(model, t2_cases=[(valid_json, [0, 1])])

        # The shared ladder's value-fidelity tier: default 0.9 scaled to 90.
        self.assertEqual(result.t2_score, 90)
        self.assertEqual(result.t2_cases[0].tier, "value_fidelity")

    # --- Slice 7: evaluate runs T1 and T2; both scores and the aggregate include both ---

    def test_evaluate_runs_t1_and_t2_with_both_scores(self) -> None:
        """One canned script repairs both cases; the Evaluation carries both 0-100 scores."""
        # Op 1 repairs the T1 break ('{"a": 1' -> '{"a": 1}') and is a zero-match no-op on
        # the T2 text; ops 2-3 repair the T2 breaks and are no-ops on the T1 text.
        canned_script = (
            '[{"old": "{\\"a\\": 1", "new": "{\\"a\\": 1}"}, '
            '{"old": "2, \\"b\\"", "new": "2], \\"b\\""}, '
            '{"old": ": 3}", "new": ": 3}}"}]'
        )
        model = StubModel(canned_script)

        result = evaluate(
            model,
            t1_cases=[('{"a": 1}', 0)],
            t2_cases=[('{"a": [1, 2], "b": {"c": 3}}', [0, 1])],
        )

        self.assertEqual(result.t1_score, 100)
        self.assertEqual(result.t2_score, 100)
        self.assertEqual(len(result.cases), 1)
        self.assertEqual(len(result.t2_cases), 1)
        # Unified scoreboard, renormalized over the two tests that ran: (0.4*100+0.3*100)/0.7.
        self.assertEqual(result.scoreboard, 100.0)

    def test_scoreboard_mixes_t1_and_t2_scores(self) -> None:
        """T1=50 and T2=100 aggregate to the weighted mean (0.4*50+0.3*100)/0.7 = 71.4286."""
        canned_script = (
            '[{"old": "{\\"a\\": 1", "new": "{\\"a\\": 2}"}, '
            '{"old": "2, \\"b\\"", "new": "2], \\"b\\""}, '
            '{"old": ": 3}", "new": ": 3}}"}]'
        )
        model = StubModel(canned_script)

        result = evaluate(
            model,
            t1_cases=[('{"a": 1}', 0)],
            t2_cases=[('{"a": [1, 2], "b": {"c": 3}}', [0, 1])],
        )

        self.assertEqual(result.t1_score, 50)
        self.assertEqual(result.t2_score, 100)
        self.assertAlmostEqual(result.scoreboard, 71.4286, places=4)


if __name__ == "__main__":
    unittest.main()
