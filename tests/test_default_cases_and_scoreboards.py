"""Tests for default case sets, custom-case intake, and final scoreboards (issue #7)."""

import json
import unittest
from typing import Any, Dict, List, Optional

from bracketbench.default_cases import (
    DEFAULT_T2_CASES,
    UNIFIED_SCOREBOARD_WEIGHTS,
    WITHOUT_IPYNB_SCOREBOARD_WEIGHTS,
    default_t1_cases,
    default_t3_cases,
    default_t4_cases,
)
from bracketbench.repair.evaluate import Evaluation, evaluate
from bracketbench.llms.base import LLMInterface


class StubModel(LLMInterface):
    """A stub LLM that returns a canned edit script for any prompt (no network)."""

    def __init__(self, canned_output: str = "[]", model_name: str = "stub") -> None:
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


class TestDefaultCaseSets(unittest.TestCase):
    """Default case sets ship for all four tests (acceptance criteria 1-2)."""

    def test_default_t1_cases_load_24_cases(self) -> None:
        cases = default_t1_cases()
        self.assertEqual(len(cases), 24)
        for valid_json, bracket_index in cases:
            self.assertIsInstance(valid_json, str)
            self.assertIsInstance(bracket_index, int)
            json.loads(valid_json)

    def test_default_t2_cases_ship(self) -> None:
        self.assertGreaterEqual(len(DEFAULT_T2_CASES), 1)
        for valid_json, bracket_indices in DEFAULT_T2_CASES:
            self.assertIsInstance(valid_json, str)
            self.assertIsInstance(bracket_indices, list)
            self.assertGreater(len(bracket_indices), 0)
            json.loads(valid_json)

    def test_default_t3_cases_ship(self) -> None:
        cases = default_t3_cases()
        self.assertGreaterEqual(len(cases), 1)
        for case in cases:
            self.assertIsInstance(case, str)

    def test_default_t4_cases_ship(self) -> None:
        cases = default_t4_cases()
        self.assertGreaterEqual(len(cases), 1)
        for case in cases:
            self.assertIsInstance(case, str)


class TestCustomCaseIntake(unittest.TestCase):
    """Supplying custom cases per test type is supported (acceptance criterion 2)."""

    def test_custom_t1_cases_override_defaults(self) -> None:
        model = StubModel(canned_output="[]")
        result = evaluate(model, t1_cases=[('{"a": 1}', 0)], t2_cases=[], t3_cases=[], t4_cases=[])
        self.assertEqual(result.t1_score, 0)
        self.assertEqual(result.t2_score, 0)
        self.assertEqual(result.t3_score, 0)
        self.assertEqual(result.t4_score, 0)
        self.assertEqual(len(result.cases), 1)

    def test_custom_t4_cases_override_defaults(self) -> None:
        model = StubModel(canned_output="[]")
        result = evaluate(model, t1_cases=[], t2_cases=[], t3_cases=[], t4_cases=['{"a": 1'])
        self.assertEqual(result.t4_score, 0)
        self.assertEqual(len(result.t4_runs), 1)

    def test_empty_list_skips_test(self) -> None:
        model = StubModel(canned_output="[]")
        result = evaluate(model, t1_cases=[], t2_cases=[], t3_cases=[], t4_cases=[])
        self.assertEqual(result.t1_score, 0)
        self.assertEqual(result.t2_score, 0)
        self.assertEqual(result.t3_score, 0)
        self.assertEqual(result.t4_score, 0)
        self.assertEqual(result.scoreboard, 0.0)


class TestScoreboards(unittest.TestCase):
    """Unified and without-ipynb scoreboards (acceptance criteria 3-5)."""

    def test_unified_scoreboard_weights(self) -> None:
        self.assertAlmostEqual(UNIFIED_SCOREBOARD_WEIGHTS["T1"], 0.4)
        self.assertAlmostEqual(UNIFIED_SCOREBOARD_WEIGHTS["T2"], 0.3)
        self.assertAlmostEqual(UNIFIED_SCOREBOARD_WEIGHTS["T3"], 0.2)
        self.assertAlmostEqual(UNIFIED_SCOREBOARD_WEIGHTS["T4"], 0.1)

    def test_without_ipynb_excludes_t3_and_renormalizes(self) -> None:
        self.assertEqual(WITHOUT_IPYNB_SCOREBOARD_WEIGHTS["T3"], 0.0)
        total = sum(WITHOUT_IPYNB_SCOREBOARD_WEIGHTS.values())
        self.assertAlmostEqual(total, 1.0)

    def test_full_evaluate_run_with_defaults(self) -> None:
        model = StubModel(canned_output="[]")
        result = evaluate(model)
        self.assertIsInstance(result, Evaluation)
        self.assertIsInstance(result.t1_score, int)
        self.assertIsInstance(result.t2_score, int)
        self.assertIsInstance(result.t3_score, int)
        self.assertIsInstance(result.t4_score, int)
        self.assertIsInstance(result.scoreboard, float)
        self.assertGreater(len(result.cases), 0)
        self.assertGreater(len(result.t2_cases), 0)
        self.assertGreater(len(result.t3_cases), 0)
        self.assertGreater(len(result.t4_runs), 0)

    def test_scoreboard_weights_configurable(self) -> None:
        model = StubModel(canned_output="[]")
        custom_weights = {"T1": 1.0, "T2": 0.0, "T3": 0.0, "T4": 0.0}
        result = evaluate(
            model,
            t1_cases=[('{"a": 1}', 0)],
            t2_cases=[],
            t3_cases=[],
            t4_cases=[],
            scoreboard_weights=custom_weights,
        )
        self.assertEqual(result.scoreboard, float(result.t1_score))


if __name__ == "__main__":
    unittest.main()

