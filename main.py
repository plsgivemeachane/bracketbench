#!/usr/bin/env python3
"""Minimal CLI entry point for BracketBench.

The legacy CLI drove a generic multi-provider model registry and a ``BenchmarkRunner`` that
no longer exists (ADR-0001: "the CLI imported a non-existent ``BenchmarkRunner`` class").
That scaffolding has been stripped. This entry point wires the **focused** JSON-repair
product: it runs the T1 tracer bullet -- break a valid JSON document, have a model emit an
edit script (ADR-0005/0006), apply it, and score on the 4-tier ladder (ADR-0002) -- through
the public :func:`bracketbench.repair.evaluate.evaluate` seam.

Bring your own model by subclassing :class:`bracketbench.llms.base.LLMInterface`. Without a
real provider, ``--stub`` runs the worked example from ADR-0006 against a canned-repair
stub so the CLI is demonstrably alive.
"""

import argparse
import sys
from typing import List, Tuple

from bracketbench.llms.base import LLMInterface
from bracketbench.repair.evaluate import Evaluation, evaluate


class StubRepairModel(LLMInterface):
    """A stub model that emits the ADR-0006 worked-example repair script for any prompt.

    For ``{"a": 1}`` broken at ``bracket_index=0`` the Breaker deletes the closing ``}``,
    yielding ``{"a": 1``; the script ``[{"old": ": 1", "new": ": 1}"}]`` restores it ->
    exact-fidelity tier (100). Used only so the CLI is runnable without an API key.
    """

    def __init__(self) -> None:
        super().__init__("stub-repair-model", None)

    def initialize(self) -> None:
        self._is_initialized = True

    def generate(
        self,
        prompt: str,
        max_tokens: int = None,  # type: ignore[assignment]
        temperature: float = None,  # type: ignore[assignment]
        **kwargs,
    ) -> str:
        if not self.is_initialized():
            self.initialize()
        return '[{"old": ": 1", "new": ": 1}"}]'

    def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        return [self.generate(p, **kwargs) for p in prompts]

    def get_model_info(self) -> dict:
        return {"name": self.model_name, "provider": "stub"}


def _parse_cases(arg: str) -> List[Tuple[str, int]]:
    """Parse ``--cases`` as ``valid_json|bracket_index`` pairs, e.g. '{"a": 1}|0'."""
    cases: List[Tuple[str, int]] = []
    if not arg:
        return cases
    for chunk in arg.split(";;"):
        text, _, idx = chunk.rpartition("|")
        cases.append((text, int(idx)))
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(
        description="BracketBench: focused JSON-repair benchmark (T1 vertical slice).",
    )
    parser.add_argument(
        "--stub",
        action="store_true",
        help="Run the T1 worked example against the canned-repair stub model (no API key).",
    )
    parser.add_argument(
        "--cases",
        type=str,
        default='{"a": 1}|0',
        help="T1 cases as 'valid_json|bracket_index' pairs separated by ';;'.",
    )
    args = parser.parse_args()

    if not args.stub:
        parser.print_help()
        print(
            "\nNo model supplied. Re-run with --stub to run the T1 worked example, or "
            "subclass bracketbench.llms.base.LLMInterface and wire it into evaluate()."
        )
        return 0

    model = StubRepairModel()
    t1_cases = _parse_cases(args.cases)
    result: Evaluation = evaluate(model, t1_cases=t1_cases)

    print("BracketBench T1 run")
    print("-------------------")
    print(f"cases:      {len(result.cases)}")
    print(f"T1 score:   {result.t1_score}/100")
    print(f"tier:       {result.tier}")
    print(f"scoreboard: {result.scoreboard:.2f}")
    if result.cases:
        c = result.cases[0]
        print(f"\nfirst case:")
        print(f"  broken text:      {c.breakage_record!r}")
        print(f"  raw edit script:  {c.raw_edit_script}")
        print(f"  repaired text:    {c.repaired_text}")
        print(f"  tier:             {c.tier} ({c.score})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
