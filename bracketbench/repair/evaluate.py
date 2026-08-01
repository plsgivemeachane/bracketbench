"""The top-level ``evaluate`` entry point (issue #3 tracer bullet).

``evaluate`` runs the available tests against a model conforming to the existing
``LLMInterface`` and returns an ``Evaluation`` carrying per-test 0-100 scores plus the
computable scoreboard aggregate. For the T1 tracer bullet it runs T1 only: it builds each T1
case, constructs the T1 prompt, calls ``model.generate``, applies the edit script, scores the
repaired text, and aggregates into the partial scoreboard.

The ``Evaluation`` retains enough to audit a run: the model's raw edit-script output, the
applied repaired text, the breakage record, and the tier reached (per the issue's acceptance
criteria).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from bracketbench.breaker import BreakRecord
from bracketbench.llms.base import LLMInterface
from bracketbench.repair.applier import apply
from bracketbench.repair.scoring import ScoreResult, T1T2Scorer, TierScoreConfig
from bracketbench.repair.scoreboard import Scoreboard
from bracketbench.repair.t1 import build_t1_case, build_t1_prompt


@dataclass(frozen=True)
class T1RunResult:
    """The audit record for one T1 case run."""

    raw_edit_script: str
    repaired_text: str
    breakage_record: BreakRecord
    tier: str
    score: int


@dataclass(frozen=True)
class Evaluation:
    """The result of an ``evaluate`` run.

    Carries the T1 score (0-100), the computable partial scoreboard aggregate, and the audit
    data needed to inspect a run. For a single-case T1 run the headline audit fields
    (``raw_edit_script``, ``repaired_text``, ``breakage_record``, ``tier``) refer to that one
    case; ``cases`` holds the full per-case list for multi-case runs.
    """

    t1_score: int
    scoreboard: float
    tier: str
    raw_edit_script: str
    repaired_text: str
    breakage_record: BreakRecord
    cases: List[T1RunResult] = field(default_factory=list)


# Default unified-scoreboard weights (CONTEXT.md): 0.4*T1 + 0.3*T2 + 0.2*T3 + 0.1*T4.
_DEFAULT_SCOREBOARD_WEIGHTS: Dict[str, float] = {
    "T1": 0.4,
    "T2": 0.3,
    "T3": 0.2,
    "T4": 0.1,
}


def evaluate(
    model: LLMInterface,
    t1_cases: Optional[List[Tuple[str, int]]] = None,
    *,
    tier_config: Optional[TierScoreConfig] = None,
    scoreboard_weights: Optional[Dict[str, float]] = None,
) -> Evaluation:
    """Run T1 against ``model`` and return an ``Evaluation`` with the T1 score and aggregate.

    Args:
        model: An LLM conforming to ``LLMInterface``. Must be initialized or auto-initialize
            on first ``generate`` (the stub model does).
        t1_cases: A list of ``(valid_json, bracket_index)`` pairs defining the T1 cases to
            run. If empty or None, T1 scores nothing (t1_score defaults to 0).
        tier_config: The tier scores (ADR-0002). Defaults to ``TierScoreConfig()``.
        scoreboard_weights: The scoreboard weight map. Defaults to the unified scoreboard.

    Returns:
        An ``Evaluation`` carrying the T1 score (0-100), the partial scoreboard aggregate,
        and per-case audit data (raw edit script, repaired text, breakage record, tier).
    """
    if t1_cases is None:
        t1_cases = []
    config = tier_config if tier_config is not None else TierScoreConfig()
    scorer = T1T2Scorer(config)
    weights = (
        scoreboard_weights
        if scoreboard_weights is not None
        else dict(_DEFAULT_SCOREBOARD_WEIGHTS)
    )
    scoreboard = Scoreboard(weights)

    run_results: List[T1RunResult] = []
    for valid_json, bracket_index in t1_cases:
        case = build_t1_case(valid_json, bracket_index)
        prompt = build_t1_prompt(case.broken_text)
        raw_edit_script = model.generate(prompt)
        apply_result = apply(case.broken_text, raw_edit_script)
        score_result: ScoreResult = scorer.score(
            apply_result.repaired_text, case.original_object
        )
        run_results.append(
            T1RunResult(
                raw_edit_script=raw_edit_script,
                repaired_text=apply_result.repaired_text,
                breakage_record=case.breakage_record,
                tier=score_result.tier,
                score=score_result.score,
            )
        )

    t1_score = _mean([r.score for r in run_results]) if run_results else 0
    aggregate = scoreboard.aggregate({"T1": t1_score}) if run_results else 0.0

    # For the single-case tracer bullet, expose that case's audit fields on the Evaluation.
    if run_results:
        single = run_results[0]
        return Evaluation(
            t1_score=t1_score,
            scoreboard=aggregate,
            tier=single.tier,
            raw_edit_script=single.raw_edit_script,
            repaired_text=single.repaired_text,
            breakage_record=single.breakage_record,
            cases=run_results,
        )
    return Evaluation(
        t1_score=0,
        scoreboard=0.0,
        tier="unparseable",
        raw_edit_script="",
        repaired_text="",
        breakage_record=BreakRecord(break_type="none", position=-1, deleted_char=""),
        cases=[],
    )


def _mean(values: List[int]) -> int:
    """Integer mean of a list of 0-100 scores (rounds to nearest int)."""
    return int(round(sum(values) / len(values)))
