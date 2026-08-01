"""Regression test: the CLI runner must emit T1 4-tier scores, not difflib floats.

Locks down the fix for the "arbitrary results" bug. Before the fix, BenchmarkRunner ran
three placeholder tests scored by difflib string-similarity (e.g. 0.8235...), ignoring the
T1 pipeline in bracketbench/repair/evaluate.py. After the fix, the runner drives evaluate()
per model and yields only the 4-tier ladder scores {0, 50, 90, 100} with T1 audit metadata.

The seam is BenchmarkRunner.run driven by a stub LLM (no network).
"""

import unittest
from typing import Any, Dict, List, Optional

from bracketbench.llms.base import LLMInterface
from bracketbench.llms.manager import LLMManager
from bracketbench.benchmarking import BenchmarkRunner


# ADR-0006 worked-example repair script: for the T1 case built from {"a": 1} (bracket_index
# 0), the Breaker deletes the closing brace, and this script restores it -> exact-fidelity.
CANNED_EDIT_SCRIPT = '[{"old": ": 1", "new": ": 1}"}]'


class StubModel(LLMInterface):
    def __init__(self, canned: str = CANNED_EDIT_SCRIPT) -> None:
        super().__init__("stub")
        self._canned = canned
        self._is_initialized = True

    def initialize(self) -> None:
        self._is_initialized = True

    def generate(self, prompt, max_tokens=None, temperature=None, **kwargs):
        return self._canned

    def generate_batch(self, prompts, max_tokens=None, temperature=None, **kwargs):
        return [self.generate(p, max_tokens, temperature, **kwargs) for p in prompts]

    def get_model_info(self):
        return {"name": self.model_name, "provider": "stub"}


class TestRunnerEmitsT1TierScores(unittest.TestCase):
    """The CLI runner must score on the T1 4-tier ladder, not difflib similarity."""

    LEGIT_T1_SCORES = {0, 50, 90, 100}

    def test_runner_scores_are_t1_tier_values(self) -> None:
        manager = LLMManager()
        manager._llm_instances["stub"] = StubModel()

        runner = BenchmarkRunner(output_dir="results_test_runner", iterations=1)
        results = runner.run(llm_manager=manager, test_cases=["standard"], metrics=None)

        model_results = results["stub"]
        self.assertTrue(model_results, "runner produced no results")

        for result in model_results:
            self.assertIn(
                int(round(result.score)),
                self.LEGIT_T1_SCORES,
                msg="non-tier score " + str(result.score) + " -> difflib path still active",
            )
            self.assertEqual(result.metadata.get("test"), "T1")
            self.assertIn("tier", result.metadata, "T1 tier metadata missing")
            self.assertIn("repaired_text", result.metadata, "T1 audit data missing")

    def test_runner_does_not_emit_difflib_floats(self) -> None:
        """Guard against the exact symptom: arbitrary non-tier floats like 0.8235."""
        manager = LLMManager()
        manager._llm_instances["stub"] = StubModel()

        runner = BenchmarkRunner(output_dir="results_test_runner", iterations=1)
        results = runner.run(llm_manager=manager, test_cases=["standard"], metrics=None)

        for result in results["stub"]:
            # Every legitimate T1 score is one of {0, 50, 90, 100}; a float in (0, 1) or
            # any non-tier value means the legacy similarity scorer is back.
            self.assertNotAlmostEqual(
                result.score,
                0.8235294117647058,
                places=10,
                msg="difflib similarity score detected",
            )
            self.assertIn(int(round(result.score)), self.LEGIT_T1_SCORES)

    def test_runner_correct_repair_earns_hundred(self) -> None:
        """The stub knows the repair for the first default case -> exact-fidelity (100)."""
        manager = LLMManager()
        manager._llm_instances["stub"] = StubModel()

        runner = BenchmarkRunner(output_dir="results_test_runner", iterations=1)
        results = runner.run(llm_manager=manager, test_cases=["standard"], metrics=None)

        first = results["stub"][0]
        self.assertEqual(first.score, 100.0)
        self.assertEqual(first.metadata["tier"], "exact_fidelity")


if __name__ == "__main__":
    unittest.main()
