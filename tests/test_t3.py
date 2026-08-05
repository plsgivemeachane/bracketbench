"""Tests for the T3 complex-ipynb test (issue #6, ADR-0004).

T3 is scored by notebook-semantic static checking only — no Jupyter kernel is ever run. The
score is a weighted sum of four binary pass/fail checks (weights configurable per ADR-0002),
scaled to 0-100: (1) parses as JSON — prerequisite, failure scores 0; (2) passes the
canonical ``nbformat.validate`` validator; (3) cell-level integrity; (4) semantic
consistency. The canonical validator is used as a source of truth in these tests, never
mocked around.

Two seams are exercised: the ``T3Scorer`` static-check seam and the primary integration seam
``evaluate`` driven by a stub model (StubModel pattern from ``test_evaluate_t1``).
"""

import json
import unittest
from typing import Any, Dict, List, Optional

import nbformat

from bracketbench.llms.base import LLMInterface
from bracketbench.repair import t3_cases
from bracketbench.repair.evaluate import Evaluation, evaluate
from bracketbench.repair.scoreboard import Scoreboard
from bracketbench.repair.t3 import T3Case, build_t3_case, build_t3_prompt
from bracketbench.repair.t3_scorer import T3CheckConfig, T3Scorer


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


class PromptAwareStubModel(StubModel):
    """Returns a different canned script for T1 prompts than for T3 prompts."""

    def __init__(
        self, t1_output: str, t3_output: str, model_name: str = "stub"
    ) -> None:
        super().__init__(t1_output, model_name)
        self._t3_output = t3_output

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs: Any,
    ) -> str:
        if "Broken notebook:" in prompt:
            return self._t3_output
        return self._canned_output


class RepairAllT3StubModel(StubModel):
    """Emits the whole-text-replacement script repairing any T3 prompt's notebook to valid.

    Extracts the broken text from the prompt (the prompt embeds it verbatim) and returns the
    ADR-0006 auditable escape hatch: ``old`` = the entire broken text, ``new`` = the valid
    notebook. ``old`` matches exactly once, so the applier applies it.
    """

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs: Any,
    ) -> str:
        marker = "Broken notebook:\n"
        broken_text = prompt.split(marker, 1)[1]
        return json.dumps(
            [{"old": broken_text, "new": t3_cases.VALID_NOTEBOOK}]
        )


def _whole_text_script(broken_text: str, repaired_text: str) -> str:
    """The auditable whole-text-replacement edit script (ADR-0006 escape hatch)."""
    return json.dumps([{"old": broken_text, "new": repaired_text}])


class TestT3Scorer(unittest.TestCase):
    """The T3 static-check scorer seam (ADR-0004)."""

    def setUp(self) -> None:
        self.scorer = T3Scorer()

    # --- The valid notebook is genuinely canonical (source of truth, not mocked) ---

    def test_valid_fixture_passes_canonical_validator(self) -> None:
        """The reference notebook truly satisfies nbformat.validate."""
        nbformat.validate(json.loads(t3_cases.VALID_NOTEBOOK))

    # --- Check 1: the prerequisite gate ---

    def test_unparseable_text_scores_zero(self) -> None:
        """Text that does not parse as JSON is the prerequisite gate: T3 = 0."""
        result = self.scorer.score("this is not json")
        self.assertEqual(result.score, 0)
        self.assertEqual(result.tier, "unparseable")

    def test_json_that_is_not_a_notebook_scores_partial(self) -> None:
        """Valid JSON that is not notebook-shaped fails the notebook checks but not the gate."""
        result = self.scorer.score(json.dumps({"hello": "world"}))
        self.assertEqual(result.tier, "t3_partial")
        # Only the parses check passes: 1 of 4 equal weights.
        self.assertEqual(result.score, 25)

    # --- A fully valid notebook passes every check ---

    def test_valid_notebook_scores_100(self) -> None:
        """A notebook that passes nbformat.validate and the semantic checks scores 100."""
        result = self.scorer.score(t3_cases.VALID_NOTEBOOK)
        self.assertEqual(result.score, 100)
        self.assertEqual(result.tier, "t3_pass")

    # --- Check 3/4 breakages yield partial credit ---

    def test_corrupt_source_scores_partial(self) -> None:
        """A non-string `source` fails schema + cell integrity; parses + semantic pass."""
        result = self.scorer.score(t3_cases.CORRUPT_SOURCE_NOTEBOOK)
        self.assertEqual(result.tier, "t3_partial")
        self.assertEqual(result.score, 50)

    def test_outputs_on_markdown_scores_partial(self) -> None:
        """`outputs` on a non-code cell fails schema + cell integrity (2 of 4 checks pass)."""
        result = self.scorer.score(t3_cases.OUTPUTS_ON_MARKDOWN_NOTEBOOK)
        self.assertEqual(result.tier, "t3_partial")
        self.assertEqual(result.score, 50)

    def test_mangled_output_scores_partial(self) -> None:
        """An execute_result missing `data` fails all three post-gate checks."""
        result = self.scorer.score(t3_cases.MANGLED_OUTPUT_NOTEBOOK)
        self.assertEqual(result.tier, "t3_partial")
        self.assertEqual(result.score, 25)

    def test_missing_execution_count_scores_partial(self) -> None:
        """A code cell without `execution_count` fails schema + semantic consistency."""
        result = self.scorer.score(t3_cases.MISSING_EXECUTION_COUNT_NOTEBOOK)
        self.assertEqual(result.tier, "t3_partial")
        self.assertEqual(result.score, 50)

    def test_invalid_cell_type_scores_partial(self) -> None:
        """`cell_type` outside {code, markdown, raw} fails schema + cell integrity."""
        result = self.scorer.score(t3_cases.INVALID_CELL_TYPE_NOTEBOOK)
        self.assertEqual(result.tier, "t3_partial")
        self.assertEqual(result.score, 50)

    # --- Check 4: semantic consistency details ---

    def test_execution_count_as_string_fails_semantic_consistency(self) -> None:
        """`execution_count` must be an int or null; a string is a semantic violation."""
        notebook = json.loads(t3_cases.VALID_NOTEBOOK)
        notebook["cells"][0]["execution_count"] = "1"
        result = self.scorer.score(json.dumps(notebook))
        self.assertEqual(result.tier, "t3_partial")
        self.assertEqual(result.score, 50)

    def test_invalid_output_type_fails_semantic_consistency(self) -> None:
        """An output_type outside {stream, execute_result, error} is a violation."""
        notebook = json.loads(t3_cases.VALID_NOTEBOOK)
        notebook["cells"][0]["outputs"][0]["output_type"] = "execute_resultt"
        result = self.scorer.score(json.dumps(notebook))
        self.assertEqual(result.tier, "t3_partial")

    def test_mime_type_without_slash_fails_semantic_consistency(self) -> None:
        """A data key that is not a well-formed mime type ("/" missing) is a violation."""
        notebook = json.loads(t3_cases.VALID_NOTEBOOK)
        notebook["cells"][0]["outputs"] = [
            {
                "output_type": "execute_result",
                "data": {"textplain": "hello"},
                "metadata": {},
                "execution_count": 1,
            }
        ]
        result = self.scorer.score(json.dumps(notebook))
        self.assertEqual(result.tier, "t3_partial")

    # --- Configurable weights (ADR-0002) ---

    def test_check_weights_are_configurable(self) -> None:
        """A T3CheckConfig with only the schema check weighted scores 0/100 on it alone."""
        config = T3CheckConfig(
            parses=0.0, schema_valid=1.0, cell_integrity=0.0, semantic_consistency=0.0
        )
        scorer = T3Scorer(config)
        self.assertEqual(scorer.score(t3_cases.VALID_NOTEBOOK).score, 100)
        self.assertEqual(scorer.score(t3_cases.CORRUPT_SOURCE_NOTEBOOK).score, 0)


class TestT3CaseAndPrompt(unittest.TestCase):
    """The T3 case and prompt builders."""

    def test_build_t3_case_wraps_broken_text(self) -> None:
        """build_t3_case carries the broken notebook text as-is."""
        case = build_t3_case(t3_cases.CORRUPT_SOURCE_NOTEBOOK)
        self.assertIsInstance(case, T3Case)
        self.assertEqual(case.broken_text, t3_cases.CORRUPT_SOURCE_NOTEBOOK)

    def test_build_t3_prompt_asks_for_edit_script_not_json(self) -> None:
        """The prompt instructs an edit script (ADR-0006) and embeds the broken text."""
        prompt = build_t3_prompt(t3_cases.CORRUPT_SOURCE_NOTEBOOK)
        self.assertIn("EDIT SCRIPT", prompt)
        self.assertIn('"old"', prompt)
        self.assertIn('"new"', prompt)
        self.assertIn(t3_cases.CORRUPT_SOURCE_NOTEBOOK, prompt)


class TestT3EvaluateIntegration(unittest.TestCase):
    """The primary integration seam: evaluate runs T3 through the full pipeline."""

    def test_evaluate_scores_repaired_notebook_100(self) -> None:
        """A whole-text replacement that yields the valid notebook scores T3 = 100."""
        broken = t3_cases.CORRUPT_SOURCE_NOTEBOOK
        model = StubModel(_whole_text_script(broken, t3_cases.VALID_NOTEBOOK))

        result = evaluate(model, t1_cases=[], t2_cases=[], t3_cases=[broken], t4_cases=[])

        self.assertIsInstance(result, Evaluation)
        self.assertEqual(result.t3_score, 100)
        self.assertEqual(result.t3_cases[0].tier, "t3_pass")
        self.assertEqual(result.t3_cases[0].repaired_text, t3_cases.VALID_NOTEBOOK)

    def test_evaluate_scores_unrepaired_partial(self) -> None:
        """A no-op script leaves the breakage; the unchanged notebook earns the partial 50."""
        broken = t3_cases.MISSING_EXECUTION_COUNT_NOTEBOOK
        model = StubModel("[]")  # empty script: every `old` matches zero positions

        result = evaluate(model, t1_cases=[], t2_cases=[], t3_cases=[broken], t4_cases=[])

        # parses + cell integrity pass; schema + semantic consistency fail -> 2 of 4.
        self.assertEqual(result.t3_score, 50)
        self.assertEqual(result.t3_cases[0].tier, "t3_partial")

    def test_evaluate_scores_unparseable_zero(self) -> None:
        """A malformed script leaves the broken text unparseable -> the gate scores 0."""
        model = StubModel("not json at all")

        result = evaluate(model, t1_cases=[], t2_cases=[], t3_cases=[t3_cases.UNPARSEABLE_NOTEBOOK], t4_cases=[])

        self.assertEqual(result.t3_score, 0)
        self.assertEqual(result.t3_cases[0].tier, "unparseable")

    def test_evaluate_scores_default_case_set(self) -> None:
        """evaluate runs every curated T3 case and aggregates the mean."""
        model = RepairAllT3StubModel("")

        result = evaluate(model, t1_cases=[], t2_cases=[], t3_cases=list(t3_cases.DEFAULT_T3_CASES), t4_cases=[])

        self.assertEqual(result.t3_score, 100)
        self.assertEqual(len(result.t3_cases), len(t3_cases.DEFAULT_T3_CASES))
        self.assertTrue(all(r.tier == "t3_pass" for r in result.t3_cases))

    def test_evaluate_without_t3_cases_scores_zero(self) -> None:
        """No T3 cases -> t3_score defaults to 0 and the Evaluation still carries it."""
        model = StubModel("ignored")

        result = evaluate(model, t1_cases=[], t2_cases=[], t3_cases=[], t4_cases=[])

        self.assertEqual(result.t3_score, 0)
        self.assertEqual(result.t3_cases, [])

    def test_t1_and_t3_run_together_in_one_evaluation(self) -> None:
        """evaluate runs both tests; both scores land on the Evaluation."""
        t1_script = '[{"old": ": 1", "new": ": 1}"}]'
        broken = t3_cases.CORRUPT_SOURCE_NOTEBOOK
        model = PromptAwareStubModel(
            t1_script, _whole_text_script(broken, t3_cases.VALID_NOTEBOOK)
        )

        result = evaluate(
            model,
            t1_cases=[('{"a": 1}', 0)],
            t2_cases=[],
            t3_cases=[broken],
            t4_cases=[],
        )

        self.assertEqual(result.t1_score, 100)
        self.assertEqual(result.t3_score, 100)
        self.assertEqual(len(result.cases), 1)
        self.assertEqual(len(result.t3_cases), 1)

    def test_unified_scoreboard_includes_t3(self) -> None:
        """The default scoreboard aggregates T1 and T3 with the unified weights."""
        model = PromptAwareStubModel('[{"old": ": 1", "new": ": 1}"}]', "[]")

        result = evaluate(
            model,
            t1_cases=[('{"a": 1}', 0)],
            t2_cases=[],
            t3_cases=[t3_cases.CORRUPT_SOURCE_NOTEBOOK],
            t4_cases=[],
        )

        # T1=100; T3 = corrupted notebook unchanged: parses(0.25) + semantic(0.25) = 50.
        # Unified: (0.4*100 + 0.2*50) / 0.6 = 83.3333.
        self.assertEqual(result.t3_score, 50)
        self.assertAlmostEqual(result.scoreboard, 83.3333, places=4)

    def test_without_ipynb_scoreboard_renormalizes(self) -> None:
        """T3 weight 0 + renormalize over T1 (the without-ipynb scoreboard)."""
        model = PromptAwareStubModel('[{"old": ": 1", "new": ": 1}"}]', "[]")

        result = evaluate(
            model,
            t1_cases=[('{"a": 1}', 0)],
            t2_cases=[],
            t3_cases=[t3_cases.CORRUPT_SOURCE_NOTEBOOK],
            t4_cases=[],
            scoreboard_weights={"T1": 0.4, "T2": 0.3, "T3": 0.0, "T4": 0.1},
        )

        # Only T1 has a nonzero weight -> the aggregate is T1's score; T3 is still scored.
        self.assertEqual(result.t3_score, 50)
        self.assertEqual(result.scoreboard, 100.0)

        # Same scores, computed directly on the Scoreboard seam (T1=100, T3=50).
        board = Scoreboard({"T1": 0.4, "T2": 0.3, "T3": 0.0, "T4": 0.1})
        self.assertEqual(board.aggregate({"T1": 100, "T3": 50}), 100.0)


if __name__ == "__main__":
    unittest.main()

