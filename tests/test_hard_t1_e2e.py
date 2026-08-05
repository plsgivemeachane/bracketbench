"""End-to-end integration: a hard vendored case through ``evaluate`` (issue #13).

Proves that the hard default set exercises the ADR-0006 ambiguity contract the
way spec #8 promises, driven entirely through the existing public seam
(``evaluate`` with a ``StubModel``), mirroring ``tests/test_evaluate_t1.py``:

- A stub emitting the known-correct **``id``-anchored** repair script for one
  hard fixture case -> ``t1_score == 100``, ``tier == "exact_fidelity"``: the
  constructible path works end-to-end.
- A stub emitting the **naive short ``old``** (the repeating value just before
  the deletion) -> that edit is skipped as ambiguous (matches >= 2 times,
  ADR-0006) -> the broken text is scored as-is -> ``t1_score == 0``,
  ``tier == "unparseable"``: the ambiguity pressure bites.

The case is sourced from ``default_t1_cases.json`` (not hand-rolled), proving
the fixture is end-to-end consumable with no interface change to ``evaluate``,
``build_t1_case``, ``build_t1_prompt``, the applier, or the scorer.
"""

import json
import unittest
from typing import Any, Dict, List, Optional

from bracketbench.benchmarking.default_t1_cases import load_default_t1_records
from bracketbench.benchmarking.verifier import analyze_deletion
from bracketbench.llms.base import LLMInterface
from bracketbench.repair.applier import apply
from bracketbench.repair.evaluate import Evaluation, evaluate
from bracketbench.repair.t1 import build_t1_case


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


def _first_hard_record() -> dict:
    """The first ``hard``-tier record from the vendored fixture."""
    records = load_default_t1_records()
    for record in records:
        if record["metadata"]["tier"] == "hard":
            return record
    raise AssertionError("no hard-tier record in default_t1_cases.json")


class TestHardT1EndToEnd(unittest.TestCase):
    """A real hard case through ``evaluate``, both sides of the contract."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.record = _first_hard_record()
        cls.valid_json = cls.record["valid_json"]
        cls.bracket_index = cls.record["bracket_index"]
        cls.analysis = analyze_deletion(cls.valid_json, cls.bracket_index)

    def _correct_script(self) -> str:
        """The known-correct id-anchored repair: re-insert the deleted bracket."""
        old = self.analysis.id_anchored_old
        return json.dumps([{"old": old, "new": old + self.analysis.deleted_char}])

    def _naive_script(self) -> str:
        """The naive short old: the repeating value just before the deletion."""
        old = self.analysis.naive_old
        return json.dumps([{"old": old, "new": old + self.analysis.deleted_char}])

    # --- The constructible path: a surgical id-anchored script scores 100 ---

    def test_id_anchored_repair_scores_exact_fidelity(self) -> None:
        """The known-correct id-anchored script earns 100 / exact_fidelity."""
        model = StubModel(self._correct_script())

        result = evaluate(model, t1_cases=[(self.valid_json, self.bracket_index)], t2_cases=[], t3_cases=[], t4_cases=[])

        self.assertIsInstance(result, Evaluation)
        self.assertEqual(result.t1_score, 100)
        self.assertEqual(result.tier, "exact_fidelity")
        self.assertEqual(result.repaired_text, self.valid_json)
        self.assertEqual(result.raw_edit_script, self._correct_script())
        self.assertEqual(result.breakage_record.deleted_char,
                         self.analysis.deleted_char)

    def test_id_anchored_old_is_unique_in_the_broken_text(self) -> None:
        """Precondition: the constructible old matches exactly once."""
        self.assertEqual(self.analysis.id_anchored_occurrences, 1)
        self.assertEqual(self.analysis.broken_text.count(
            self.analysis.id_anchored_old), 1)

    # --- The ambiguity pressure: a vague short old is refused, not applied ---

    def test_naive_old_is_skipped_as_ambiguous_and_scores_zero(self) -> None:
        """The naive short old matches >= 2 times -> skipped -> 0 / unparseable."""
        self.assertGreaterEqual(self.analysis.naive_occurrences, 2)
        self.assertNotEqual(self.analysis.naive_old, self.analysis.id_anchored_old)

        model = StubModel(self._naive_script())
        result = evaluate(model, t1_cases=[(self.valid_json, self.bracket_index)], t2_cases=[], t3_cases=[], t4_cases=[])

        self.assertEqual(result.t1_score, 0)
        self.assertEqual(result.tier, "unparseable")
        # The broken text is scored as-is: nothing was applied.
        self.assertEqual(result.repaired_text, self.analysis.broken_text)

    def test_naive_script_is_skipped_for_multiple_matches(self) -> None:
        """Applier-level proof: the naive op is skipped with reason multiple_match."""
        case = build_t1_case(self.valid_json, self.bracket_index)
        apply_result = apply(case.broken_text, self._naive_script())
        self.assertEqual(len(apply_result.skipped), 1)
        self.assertEqual(apply_result.skipped[0].reason, "multiple_match")
        self.assertEqual(apply_result.skipped[0].old, self.analysis.naive_old)

    def test_case_is_sourced_from_the_vendored_fixture(self) -> None:
        """The case under test really comes from default_t1_cases.json."""
        self.assertEqual(self.record["valid_json"], self.valid_json)
        self.assertEqual(self.record["bracket_index"], self.bracket_index)
        self.assertEqual(self.record["metadata"]["tier"], "hard")


if __name__ == "__main__":
    unittest.main()
