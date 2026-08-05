"""The T2 test: multi-breakage repair (CONTEXT.md, ADR-0001).

A T2 case is built from a valid JSON document and a list of ``bracket_indices``: the Breaker
composes N (N > 1) breakages, each time removing a structural closing bracket from the
*current* text, while retaining the Original object and a record of every breakage
("track, don't tell"). The T2 prompt instructs the model to emit an edit script (ADR-0006)
but does NOT disclose how many breakages were applied.
"""

from dataclasses import dataclass, field
from typing import List

from bracketbench.breaker import BreakRecord, Breaker, BrokenJsonMulti


@dataclass(frozen=True)
class T2Case:
    """One T2 test case: a multi-broken JSON document plus its Original object and records.

    A case is fully determined by ``(valid_json, bracket_indices)`` and is therefore
    reproducible. The Original object is the ground truth a faithful repair must round-trip
    to; ``breakage_records`` carries every breakage in application order for audit.
    """

    valid_json: str
    bracket_indices: List[int]
    broken_text: str
    original_object: object
    breakage_records: List[BreakRecord] = field(default_factory=list)


def build_t2_case(valid_json: str, bracket_indices: List[int]) -> T2Case:
    """Build a T2 case by composing ``bracket_indices`` breakages onto ``valid_json``.

    Args:
        valid_json: A valid JSON document.
        bracket_indices: The per-step 0-based indexes of the structural closing bracket to
            remove, in application order (must be non-empty).

    Returns:
        A ``T2Case`` carrying the final broken text, the Original object, and the full
        per-breakage record.
    """
    broken: BrokenJsonMulti = Breaker().break_json_multi(
        valid_json, bracket_indices
    )
    return T2Case(
        valid_json=valid_json,
        bracket_indices=list(bracket_indices),
        broken_text=broken.broken_text,
        original_object=broken.original_object,
        breakage_records=list(broken.records),
    )


def build_t2_prompt(broken_text: str) -> str:
    """Build the T2 repair prompt that asks the model for an edit script (ADR-0006).

    The model is told to emit a JSON array of ``{"old", "new"}`` find-and-replace operations
    — NOT repaired JSON — so the benchmark tests *surgical repair* rather than regeneration
    (ADR-0005). Unlike T1, the prompt does NOT disclose how many breakages were applied
    ("track, don't tell"); the model must decide for itself how many edits are needed, and
    unmatched/ambiguous edits are simply skipped by the applier.
    """
    return (
        "You are repairing broken JSON. Below is a JSON document with one or more breakages: "
        "structural closing brackets (a '}' or ']') have been deleted. You are not told how "
        "many. Your job is to repair it.\n\n"
        "Output an EDIT SCRIPT, not repaired JSON. The edit script is a JSON array of "
        "find-and-replace operations, each an object {\"old\": <string>, \"new\": <string>}, "
        "applied in array order. Each operation replaces the substring `old` with `new`. "
        "If `old` matches zero positions the operation is skipped; if it matches more than "
        "one position it is skipped as ambiguous. Apply the minimal, unambiguous edits needed "
        "to repair the document.\n\n"
        "Return ONLY the JSON array of {\"old\", \"new\"} objects. No prose, no code fence.\n\n"
        f"Broken JSON:\n{broken_text}"
    )
