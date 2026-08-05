"""Tests for the T4 real-world messy JSON test (issue #5).

T4 cases are genuinely broken JSON collected from the wild: there is no Original object, so
fidelity is unscorable and the Structural tier is the ceiling (worth 100, not the ladder's
0.5). The primary integration seam is the top-level ``evaluate`` entry point run against a
stub model; the T4-specific seams are ``T4Scorer.score`` and ``build_t4_prompt``. Tests
never reach inside internals — in particular, T4 scoring goes through the shared edit-script
applier exactly as T1 does.
"""

import json
import unittest
from typing import Any, Dict, List, Optional

from bracketbench.llms.base import LLMInterface
from bracketbench.repair.evaluate import evaluate, Evaluation
from bracketbench.repair.t4 import (
    DEFAULT_T4_CASES,
    T4Case,
    build_t4_case,
    build_t4_prompt,
)
from bracketbench.repair.t4_scorer import T4Scorer


class StubModel(LLMInterface):
    """A stub LLM that returns canned edit scripts for any prompt (no network).

    ``outputs`` is consumed front-to-back (one script per generated prompt); the last script
    repeats once exhausted, so a single-output stub is a constant responder.
    """

    def __init__(self, outputs: List[str], model_name: str = "stub") -> None:
        super().__init__(model_name)
        self._outputs = list(outputs)
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
        output = self._outputs[0]
        if len(self._outputs) > 1:
            self._outputs.pop(0)
        return output

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


class TestT4CaseAndPrompt(unittest.TestCase):
    """Tests for the public ``build_t4_case`` / ``build_t4_prompt`` seams."""

    def test_build_t4_case_carries_broken_text_only(self) -> None:
        """A T4 case wraps the broken text as-is; there is no Original object."""
        broken_text = '{"name": "test", "value": 42'

        case = build_t4_case(broken_text)

        self.assertIsInstance(case, T4Case)
        self.assertEqual(case.broken_text, broken_text)
        # No Original object, no breakage record (fidelity is unscorable for T4).
        self.assertFalse(hasattr(case, "original_object"))
        self.assertFalse(hasattr(case, "breakage_record"))

    def test_t4_prompt_instructs_edit_script_not_json(self) -> None:
        """The T4 prompt asks for an ADR-0006 edit script, never repaired JSON."""
        broken_text = '{"name": "test", "value": 42'

        prompt = build_t4_prompt(broken_text)

        self.assertIn("EDIT SCRIPT", prompt)
        self.assertIn('{"old": <string>, "new": <string>}', prompt)
        self.assertIn("find-and-replace", prompt)
        # The broken text is embedded so the model can localize edits against it.
        self.assertIn(broken_text, prompt)

    def test_default_t4_cases_are_genuinely_broken(self) -> None:
        """Every curated case fails ``json.loads`` as-is (real-world broken JSON)."""
        self.assertGreaterEqual(len(DEFAULT_T4_CASES), 3)
        for broken_text in DEFAULT_T4_CASES:
            with self.subTest(broken_text=broken_text):
                with self.assertRaises(json.JSONDecodeError):
                    json.loads(broken_text)


class TestT4Scorer(unittest.TestCase):
    """Tests for the public ``T4Scorer.score(repaired_text)`` seam."""

    def _scorer(self) -> T4Scorer:
        return T4Scorer()

    def test_unparseable_output_scores_zero(self) -> None:
        """Repaired text that is not valid JSON gets 0 (unparseable tier)."""
        result = self._scorer().score('{"a": 1')

        self.assertEqual(result.tier, "unparseable")
        self.assertEqual(result.score, 0)

    def test_parseable_output_scores_structural_ceiling(self) -> None:
        """Valid JSON gets the structural tier, which is the 100 ceiling for T4."""
        result = self._scorer().score('{"a": 1}')

        self.assertEqual(result.tier, "structural")
        self.assertEqual(result.score, 100)


class TestT4Evaluate(unittest.TestCase):
    """The T4 vertical slice, driven through the public ``evaluate`` seam."""

    def test_correct_repair_earns_structural_ceiling(self) -> None:
        """A stub emitting the right edit script yields t4_score 100 with audit data."""
        broken_text = '{"name": "test", "value": 42'
        canned_script = '[{"old": "42", "new": "42}"}]'
        model = StubModel([canned_script])

        result = evaluate(model, t1_cases=[], t2_cases=[], t3_cases=[], t4_cases=[broken_text])

        self.assertIsInstance(result, Evaluation)
        self.assertEqual(result.t4_score, 100)
        self.assertEqual(result.tier, "structural")
        # Audit data is retained.
        self.assertEqual(result.raw_edit_script, canned_script)
        self.assertEqual(result.repaired_text, '{"name": "test", "value": 42}')
        self.assertEqual(result.t4_runs[0].broken_text, broken_text)

    def test_malformed_script_scores_zero(self) -> None:
        """A stub that emits non-JSON maps to the unparseable tier; broken text scored as-is."""
        broken_text = '{"name": "test", "value": 42'
        model = StubModel(["not json at all"])

        result = evaluate(model, t1_cases=[], t2_cases=[], t3_cases=[], t4_cases=[broken_text])

        self.assertEqual(result.t4_score, 0)
        self.assertEqual(result.tier, "unparseable")
        self.assertEqual(result.repaired_text, broken_text)

    def test_t4_score_comes_from_applied_text_not_raw_output(self) -> None:
        """T4 reuses the edit-script applier: a no-op script leaves the text broken -> 0.

        The stub's raw output parses as JSON, but every operation is skipped by the applier
        (``old`` matches zero positions), so the repaired text stays unparseable.
        """
        broken_text = '{"name": "test", "value": 42'
        # "999" appears nowhere in the broken text -> zero_match -> skipped (ADR-0006).
        model = StubModel(['[{"old": "999", "new": "999}"}]'])

        result = evaluate(model, t1_cases=[], t2_cases=[], t3_cases=[], t4_cases=[broken_text])

        self.assertEqual(result.repaired_text, broken_text)
        self.assertEqual(result.t4_score, 0)

    def test_evaluate_runs_t1_and_t4_together(self) -> None:
        """Both tests run in one evaluate; the scoreboard aggregates T1 and T4."""
        model = StubModel(
            [
                '[{"old": ": 1", "new": ": 1}"}]',  # T1 repair: '{"a": 1' -> '{"a": 1}'
                '[{"old": "42", "new": "42}"}]',  # T4 repair
            ]
        )

        result = evaluate(
            model,
            t1_cases=[('{"a": 1}', 0)],
            t2_cases=[],
            t3_cases=[],
            t4_cases=['{"name": "test", "value": 42'],
        )

        self.assertEqual(result.t1_score, 100)
        self.assertEqual(result.t4_score, 100)
        # Both tests at 100 -> unified-weight aggregate 100.
        self.assertEqual(result.scoreboard, 100.0)

    def test_scoreboard_includes_t4_with_zero_score(self) -> None:
        """T4's score feeds the aggregate even when it is 0 (0.4*100 + 0.1*0)/0.5 = 80."""
        model = StubModel(
            [
                '[{"old": ": 1", "new": ": 1}"}]',  # T1 repair -> 100
                "garbage",  # T4 malformed script -> 0
            ]
        )

        result = evaluate(
            model,
            t1_cases=[('{"a": 1}', 0)],
            t2_cases=[],
            t3_cases=[],
            t4_cases=['{"name": "test", "value": 42'],
        )

        self.assertEqual(result.t1_score, 100)
        self.assertEqual(result.t4_score, 0)
        self.assertEqual(result.scoreboard, 80.0)

    def test_without_t4_cases_t4_is_unscored_and_scoreboard_is_t1_only(self) -> None:
        """No t4_cases -> t4_score 0 and the aggregate renormalizes over T1 alone."""
        model = StubModel(['[{"old": ": 1", "new": ": 1}"}]'])

        result = evaluate(model, t1_cases=[('{"a": 1}', 0)], t2_cases=[], t3_cases=[], t4_cases=[])

        self.assertEqual(result.t4_score, 0)
        self.assertEqual(result.scoreboard, 100.0)
        self.assertEqual(result.t4_runs, [])


if __name__ == "__main__":
    unittest.main()

