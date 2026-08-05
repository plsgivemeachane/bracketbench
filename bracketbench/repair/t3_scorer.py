"""The T3 notebook-semantic static scorer (ADR-0004, CONTEXT.md -> Scoring -> T3).

T3 does not use the T1/T2 4-tier JSON ladder; it has its own scoring path. A T3 score is a
weighted sum of **binary pass/fail checks**, scaled to 0-100, with weights supplied via a
``T3CheckConfig`` (configurable per ADR-0002):

  1. **Parses as JSON** — prerequisite gate; if it fails, T3 = 0 (tier ``unparseable``).
  2. **Passes ``nbformat.validate``** — the canonical notebook schema validator; notebook
     validity is not reinvented here.
  3. **Cell-level integrity** — every cell has the required keys for its ``cell_type``;
     ``source`` is a string or list of strings; ``cell_type`` is in {code, markdown, raw};
     ``outputs`` only present on code cells and each output well-formed.
  4. **Semantic consistency** — ``execution_count`` present on code cells (int or null);
     output types valid (stream / execute_result / error); mime types well-formed.

No Jupyter kernel is ever run (ADR-0004): every check is static and deterministic.
"""

import copy
import json
from dataclasses import dataclass
from typing import Any, Dict

import nbformat

from bracketbench.repair.scoring import ScoreResult

_VALID_CELL_TYPES = {"code", "markdown", "raw"}
_VALID_OUTPUT_TYPES = {"stream", "execute_result", "error"}


@dataclass(frozen=True)
class T3CheckConfig:
    """Configurable weights for the four T3 checks (ADR-0002). Defaults: equal weights.

    Each weight is the value of passing that binary check in the weighted-sum score.
    ``parses`` is a prerequisite gate as well as a check: an unparseable repaired text
    scores 0 regardless of the other weights.
    """

    parses: float = 0.25
    schema_valid: float = 0.25
    cell_integrity: float = 0.25
    semantic_consistency: float = 0.25


class T3Scorer:
    """Scores repaired notebook text on the four binary T3 checks (ADR-0004).

    The scorer takes a ``T3CheckConfig`` at construction (ADR-0002) so the check weights
    can be tuned per-run without code changes.
    """

    def __init__(self, config: T3CheckConfig = None) -> None:
        self._config = config if config is not None else T3CheckConfig()

    def score(self, repaired_text: str) -> ScoreResult:
        """Score ``repaired_text`` on the four binary T3 checks.

        Args:
            repaired_text: The text the applier produced (the model's repair attempt).

        Returns:
            A ``ScoreResult`` with tier ``t3_pass`` (all checks pass), ``t3_partial``
            (parses but at least one check fails), or ``unparseable`` (does not parse as
            JSON — score 0), and the weighted score scaled to 0-100.
        """
        try:
            notebook = json.loads(repaired_text)
        except (json.JSONDecodeError, ValueError):
            # Check 1 is the prerequisite gate: unparseable text scores 0 (CONTEXT.md).
            return ScoreResult(tier="unparseable", score=0)

        checks: Dict[str, bool] = {
            "parses": True,
            "schema_valid": _passes_canonical_validation(notebook),
            "cell_integrity": _cell_integrity_ok(notebook),
            "semantic_consistency": _semantic_consistency_ok(notebook),
        }

        weights = {
            "parses": self._config.parses,
            "schema_valid": self._config.schema_valid,
            "cell_integrity": self._config.cell_integrity,
            "semantic_consistency": self._config.semantic_consistency,
        }
        total_weight = sum(weights.values())
        if total_weight == 0:
            # All-zero weights are a defined no-op (mirrors the scoreboard).
            return ScoreResult(tier="t3_partial", score=0)

        numerator = sum(w * checks[name] for name, w in weights.items())
        score = int(round(numerator / total_weight * 100))
        tier = "t3_pass" if all(checks.values()) else "t3_partial"
        return ScoreResult(tier=tier, score=score)


def _passes_canonical_validation(notebook: Any) -> bool:
    """Check 2: the canonical notebook schema validator accepts the notebook.

    ``nbformat.validate`` raises several exception types depending on the failure mode
    (``NotebookValidationError`` for schema failures, ``AttributeError`` for non-dict
    input), so any exception counts as a failure. A deep copy protects the parsed object
    from validate's in-place normalization (it backfills missing cell ids).
    """
    try:
        nbformat.validate(copy.deepcopy(notebook))
    except Exception:
        return False
    return True


def _cell_integrity_ok(notebook: Any) -> bool:
    """Check 3: cell-level integrity (CONTEXT.md).

    Every cell must have a valid ``cell_type`` in {code, markdown, raw}; ``source`` must be
    a string or a list of strings; ``outputs`` may appear only on code cells; and each
    output must be well-formed (required keys for its ``output_type``).
    """
    if not isinstance(notebook, dict):
        return False
    cells = notebook.get("cells")
    if not isinstance(cells, list):
        return False
    for cell in cells:
        if not isinstance(cell, dict):
            return False
        if cell.get("cell_type") not in _VALID_CELL_TYPES:
            return False
        if not _text_ok(cell.get("source")):
            return False
        if cell.get("cell_type") != "code" and "outputs" in cell:
            return False
        outputs = cell.get("outputs")
        if outputs is not None:
            if not isinstance(outputs, list):
                return False
            for output in outputs:
                if not _output_well_formed(output):
                    return False
    return True


def _semantic_consistency_ok(notebook: Any) -> bool:
    """Check 4: semantic consistency (CONTEXT.md).

    Code cells must carry an ``execution_count`` (int or null); every output's
    ``output_type`` must be in {stream, execute_result, error}; and ``execute_result``
    outputs must carry ``data`` whose mime-type keys are well-formed (contain "/").
    """
    if not isinstance(notebook, dict):
        return False
    cells = notebook.get("cells")
    if not isinstance(cells, list):
        return False
    for cell in cells:
        if not isinstance(cell, dict):
            return False
        if cell.get("cell_type") == "code":
            if "execution_count" not in cell:
                return False
            execution_count = cell["execution_count"]
            # bool is a JSON type distinct from int; `true` is not a valid execution_count.
            if execution_count is not None and (
                not isinstance(execution_count, int) or isinstance(execution_count, bool)
            ):
                return False
        outputs = cell.get("outputs")
        if not isinstance(outputs, list):
            continue
        for output in outputs:
            if not isinstance(output, dict):
                return False
            if output.get("output_type") not in _VALID_OUTPUT_TYPES:
                return False
            if output.get("output_type") == "execute_result":
                data = output.get("data")
                if not isinstance(data, dict):
                    return False
                for mime_type in data:
                    if not isinstance(mime_type, str) or "/" not in mime_type:
                        return False
    return True


def _text_ok(value: Any) -> bool:
    """``source``/``text`` values are a string or a list of strings (nbformat shape)."""
    if isinstance(value, str):
        return True
    return isinstance(value, list) and all(isinstance(line, str) for line in value)


def _output_well_formed(output: Any) -> bool:
    """An output has the required keys for its ``output_type`` (nbformat spec)."""
    if not isinstance(output, dict):
        return False
    output_type = output.get("output_type")
    if output_type == "stream":
        return isinstance(output.get("name"), str) and _text_ok(output.get("text"))
    if output_type == "execute_result":
        return isinstance(output.get("data"), dict) and isinstance(
            output.get("metadata"), dict
        )
    if output_type == "error":
        return (
            isinstance(output.get("ename"), str)
            and isinstance(output.get("evalue"), str)
            and isinstance(output.get("traceback"), list)
            and all(isinstance(line, str) for line in output["traceback"])
        )
    return False
