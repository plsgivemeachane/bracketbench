"""The top-level ``evaluate`` entry point (issues #3, #4, #5, #6).

``evaluate`` runs the available tests against a model conforming to the existing
``LLMInterface`` and returns an ``Evaluation`` carrying per-test 0-100 scores plus the
computable scoreboard aggregate. It runs T1 (single-breakage), T2 (multi-breakage), T3
(complex ipynb repair), and T4 (real-world messy JSON): for each case it builds the case,
constructs the prompt, calls ``model.generate``, applies the edit script, scores the
repaired text, and aggregates all into the scoreboard.

The ``Evaluation`` retains enough to audit a run: the model's raw edit-script output, the
applied repaired text, the breakage record(s), and the tier reached. T2 cases retain the
FULL breakage record (all N breakages).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from bracketbench.breaker import BreakRecord
from bracketbench.llms.base import LLMInterface
from bracketbench.repair.applier import apply
from bracketbench.repair.scoring import ScoreResult, T1T2Scorer, TierScoreConfig
from bracketbench.repair.scoreboard import Scoreboard
from bracketbench.repair.t1 import build_t1_case, build_t1_prompt
from bracketbench.repair.t2 import build_t2_case, build_t2_prompt
from bracketbench.repair.t3 import build_t3_case, build_t3_prompt
from bracketbench.repair.t3_scorer import T3CheckConfig, T3Scorer
from bracketbench.repair.t4 import build_t4_case, build_t4_prompt
from bracketbench.repair.t4_scorer import T4Scorer


@dataclass(frozen=True)
class T1RunResult:
    """The audit record for one T1 case run."""

    raw_edit_script: str
    repaired_text: str
    breakage_record: BreakRecord
    tier: str
    score: int


@dataclass(frozen=True)
class T2RunResult:
    """The audit record for one T2 case run.

    ``breakage_records`` carries ALL N breakages of the case, in application order.
    """

    raw_edit_script: str
    repaired_text: str
    breakage_records: List[BreakRecord]
    tier: str
    score: int


@dataclass(frozen=True)
class T3RunResult:
    """The audit record for one T3 case run."""

    raw_edit_script: str
    repaired_text: str
    tier: str
    score: int


@dataclass(frozen=True)
class T4RunResult:
    """The audit record for one T4 case run.

    T4 has no Original object and no breakage record, so the broken text itself identifies
    the case.
    """

    raw_edit_script: str
    repaired_text: str
    broken_text: str
    tier: str
    score: int


@dataclass(frozen=True)
class Evaluation:
    """The result of an ``evaluate`` run.

    Carries the T1, T2, T3, and T4 scores (0-100 each), the computable partial scoreboard
    aggregate, and the audit data needed to inspect a run. For a single-case T1 run the
    headline audit fields (``raw_edit_script``, ``repaired_text``, ``breakage_record``,
    ``tier``) refer to that one case; ``cases`` holds the full per-case T1 list,
    ``t2_cases`` the full per-case T2 list (each with its full breakage record),
    ``t3_cases`` the full per-case T3 list, and ``t4_runs`` the full per-case T4 list.
    """

    t1_score: int
    t2_score: int
    t3_score: int
    t4_score: int
    scoreboard: float
    tier: str
    raw_edit_script: str
    repaired_text: str
    breakage_record: BreakRecord
    cases: List[T1RunResult] = field(default_factory=list)
    t2_cases: List[T2RunResult] = field(default_factory=list)
    t3_cases: List[T3RunResult] = field(default_factory=list)
    t4_runs: List[T4RunResult] = field(default_factory=list)


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
    t2_cases: Optional[List[Tuple[str, List[int]]]] = None,
    t4_cases: Optional[List[str]] = None,
    *,
    t3_cases: Optional[List[str]] = None,
    tier_config: Optional[TierScoreConfig] = None,
    t3_check_config: Optional[T3CheckConfig] = None,
    scoreboard_weights: Optional[Dict[str, float]] = None,
) -> Evaluation:
    """Run T1, T2, T3, and T4 against ``model`` and return an ``Evaluation`` with all scores.

    Args:
        model: An LLM conforming to ``LLMInterface``. Must be initialized or auto-initialize
            on first ``generate`` (the stub model does).
        t1_cases: A list of ``(valid_json, bracket_index)`` pairs defining the T1 cases to
            run. If empty or None, T1 scores nothing (t1_score defaults to 0).
        t2_cases: A list of ``(valid_json, bracket_indices)`` pairs defining the T2 cases to
            run (each ``bracket_indices`` is a non-empty list of per-step bracket indexes).
            If empty or None, T2 scores nothing (t2_score defaults to 0).
        t4_cases: A list of broken-JSON text strings (no bracket_index) defining the T4
            cases to run. If empty or None, T4 scores nothing (t4_score defaults to 0).
        t3_cases: A list of broken notebook JSON strings defining the T3 cases to run. If
            empty or None, T3 scores nothing (t3_score defaults to 0).
        tier_config: The T1/T2 tier scores (ADR-0002). Defaults to ``TierScoreConfig()``.
        t3_check_config: The T3 check weights (ADR-0002). Defaults to ``T3CheckConfig()``.
        scoreboard_weights: The scoreboard weight map. Defaults to the unified scoreboard.

    Returns:
        An ``Evaluation`` carrying all test scores (0-100 each), the partial scoreboard
        aggregate over whichever tests ran, and per-case audit data.
    """
    if t1_cases is None:
        t1_cases = []
    if t2_cases is None:
        t2_cases = []
    if t3_cases is None:
        t3_cases = []
    if t4_cases is None:
        t4_cases = []
    config = tier_config if tier_config is not None else TierScoreConfig()
    scorer = T1T2Scorer(config)
    t3_scorer = T3Scorer(
        t3_check_config if t3_check_config is not None else T3CheckConfig()
    )
    t4_scorer = T4Scorer()
    weights = (
        scoreboard_weights
        if scoreboard_weights is not None
        else dict(_DEFAULT_SCOREBOARD_WEIGHTS)
    )
    scoreboard = Scoreboard(weights)

    # --- T1 ---
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

    # --- T2 ---
    t2_run_results: List[T2RunResult] = []
    for valid_json, bracket_indices in t2_cases:
        case = build_t2_case(valid_json, bracket_indices)
        prompt = build_t2_prompt(case.broken_text)
        raw_edit_script = model.generate(prompt)
        apply_result = apply(case.broken_text, raw_edit_script)
        score_result = scorer.score(
            apply_result.repaired_text, case.original_object
        )
        t2_run_results.append(
            T2RunResult(
                raw_edit_script=raw_edit_script,
                repaired_text=apply_result.repaired_text,
                breakage_records=case.breakage_records,
                tier=score_result.tier,
                score=score_result.score,
            )
        )

    # --- T3 ---
    t3_run_results: List[T3RunResult] = []
    for broken_text in t3_cases:
        case = build_t3_case(broken_text)
        prompt = build_t3_prompt(case.broken_text)
        raw_edit_script = model.generate(prompt)
        apply_result = apply(case.broken_text, raw_edit_script)
        score_result = t3_scorer.score(apply_result.repaired_text)
        t3_run_results.append(
            T3RunResult(
                raw_edit_script=raw_edit_script,
                repaired_text=apply_result.repaired_text,
                tier=score_result.tier,
                score=score_result.score,
            )
        )

    # --- T4 ---
    t4_run_results: List[T4RunResult] = []
    for broken_text in t4_cases:
        case = build_t4_case(broken_text)
        prompt = build_t4_prompt(case.broken_text)
        raw_edit_script = model.generate(prompt)
        apply_result = apply(case.broken_text, raw_edit_script)
        score_result = t4_scorer.score(apply_result.repaired_text)
        t4_run_results.append(
            T4RunResult(
                raw_edit_script=raw_edit_script,
                repaired_text=apply_result.repaired_text,
                broken_text=case.broken_text,
                tier=score_result.tier,
                score=score_result.score,
            )
        )

    t1_score = _mean([r.score for r in run_results]) if run_results else 0
    t2_score = _mean([r.score for r in t2_run_results]) if t2_run_results else 0
    t3_score = _mean([r.score for r in t3_run_results]) if t3_run_results else 0
    t4_score = _mean([r.score for r in t4_run_results]) if t4_run_results else 0

    # The scoreboard aggregates whichever tests ran; absent tests are excluded, not 0.
    scores: Dict[str, float] = {}
    if run_results:
        scores["T1"] = t1_score
    if t2_run_results:
        scores["T2"] = t2_score
    if t3_run_results:
        scores["T3"] = t3_score
    if t4_run_results:
        scores["T4"] = t4_score
    aggregate = scoreboard.aggregate(scores) if scores else 0.0

    # Headline audit fields come from the first run that happened (T1 > T2 > T3 > T4).
    if run_results:
        single = run_results[0]
        return Evaluation(
            t1_score=t1_score, t2_score=t2_score, t3_score=t3_score, t4_score=t4_score,
            scoreboard=aggregate, tier=single.tier,
            raw_edit_script=single.raw_edit_script, repaired_text=single.repaired_text,
            breakage_record=single.breakage_record,
            cases=run_results, t2_cases=t2_run_results, t3_cases=t3_run_results, t4_runs=t4_run_results,
        )
    if t2_run_results:
        single = t2_run_results[0]
        return Evaluation(
            t1_score=t1_score, t2_score=t2_score, t3_score=t3_score, t4_score=t4_score,
            scoreboard=aggregate, tier=single.tier,
            raw_edit_script=single.raw_edit_script, repaired_text=single.repaired_text,
            breakage_record=BreakRecord(break_type="none", position=-1, deleted_char=""),
            cases=[], t2_cases=t2_run_results, t3_cases=t3_run_results, t4_runs=t4_run_results,
        )
    if t3_run_results:
        single = t3_run_results[0]
        return Evaluation(
            t1_score=t1_score, t2_score=t2_score, t3_score=t3_score, t4_score=t4_score,
            scoreboard=aggregate, tier=single.tier,
            raw_edit_script=single.raw_edit_script, repaired_text=single.repaired_text,
            breakage_record=BreakRecord(break_type="none", position=-1, deleted_char=""),
            cases=[], t2_cases=t2_run_results, t3_cases=t3_run_results, t4_runs=t4_run_results,
        )
    if t4_run_results:
        single = t4_run_results[0]
        return Evaluation(
            t1_score=t1_score, t2_score=t2_score, t3_score=t3_score, t4_score=t4_score,
            scoreboard=aggregate, tier=single.tier,
            raw_edit_script=single.raw_edit_script, repaired_text=single.repaired_text,
            breakage_record=BreakRecord(break_type="none", position=-1, deleted_char=""),
            cases=[], t2_cases=t2_run_results, t3_cases=t3_run_results, t4_runs=t4_run_results,
        )
    return Evaluation(
        t1_score=0, t2_score=0, t3_score=0, t4_score=0,
        scoreboard=0.0, tier="unparseable",
        raw_edit_script="", repaired_text="",
        breakage_record=BreakRecord(break_type="none", position=-1, deleted_char=""),
        cases=[], t2_cases=[], t3_cases=[], t4_runs=[],
    )


def _mean(values: List[int]) -> int:
    """Integer mean of a list of 0-100 scores (rounds to nearest int)."""
    return int(round(sum(values) / len(values)))
