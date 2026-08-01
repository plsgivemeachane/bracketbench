"""Feedback loop for the arbitrary results bug.

Drives the EXACT code path the CLI uses (main.py -> BenchmarkRunner.run) with a stub model
that emits a known-good ADR-0006 edit script for one T1 case, then asserts the result is a
real T1-tier score (one of 0/50/90/100) and that the T1 pipeline was exercised.

This is RED today: BenchmarkRunner runs three placeholder tests scored by difflib
string-similarity (e.g. 0.8235...), never touching repair/evaluate.py or the 4-tier ladder.
It goes GREEN once the runner is wired to the T1 pipeline (issue #3 -> CLI).

No network. Run:  py scripts\repro_arbitrary_results.py
Exit code 0 = green, 1 = red (bug present).
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional

from bracketbench.llms.base import LLMInterface
from bracketbench.llms.manager import LLMManager
from bracketbench.benchmarking import BenchmarkRunner


# A stub LLM that always returns the ADR-0006 worked-example repair script. For the T1 case
# built from valid_json {"a": 1} (bracket_index 0), the Breaker deletes the closing brace,
# so this edit script restores it exactly -> exact-fidelity tier (100).
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


# Only the four 4-tier scores are legitimate T1 outputs (CONTEXT.md / ADR-0002).
LEGIT_T1_SCORES = {0, 50, 90, 100}


def main() -> int:
    manager = LLMManager()
    manager._llm_instances["stub"] = StubModel()

    runner = BenchmarkRunner(output_dir="results_repro", iterations=1)
    results = runner.run(llm_manager=manager, test_cases=["standard"], metrics=None)

    model_results = results["stub"]
    if not model_results:
        print("RED: no results produced")
        return 1

    scores = [r.score for r in model_results]
    print("scores:", scores)

    bad_scores = [s for s in scores if s not in LEGIT_T1_SCORES]
    if bad_scores:
        print("RED: non-T1-tier scores present: " + str(bad_scores) + " (expected only " + str(sorted(LEGIT_T1_SCORES)) + ")")
        return 1

    has_tier = any("tier" in (r.metadata or {}) for r in model_results)
    if not has_tier:
        print("RED: results carry no T1 tier metadata -> repair/evaluate was not exercised")
        return 1

    print("GREEN: all scores are valid T1-tier scores and the T1 pipeline was exercised")
    return 0


if __name__ == "__main__":
    sys.exit(main())
