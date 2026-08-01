"""The T1 test: single-breakage repair (CONTEXT.md, ADR-0001).

A T1 case is built from a valid JSON document and a ``bracket_index``: the Breaker removes
one structural closing bracket, retaining the Original object and a record of what it broke
("track, don't tell"). The T1 prompt instructs the model to emit an edit script
(ADR-0006) — a JSON array of ``{"old", "new"}`` find-and-replace operations — NOT repaired
JSON.
"""

from dataclasses import dataclass

from bracketbench.breaker import Breaker, BreakRecord, BrokenJson


@dataclass(frozen=True)
class T1Case:
    """One T1 test case: a broken JSON document plus its Original object and breakage record.

    A case is fully determined by ``(valid_json, bracket_index)`` and is therefore
    reproducible. The Original object is the ground truth a faithful repair must round-trip to.
    """

    valid_json: str
    bracket_index: int
    broken_text: str
    original_object: object
    breakage_record: BreakRecord


def build_t1_case(valid_json: str, bracket_index: int = 0) -> T1Case:
    """Build a T1 case by breaking ``valid_json`` at its k-th structural closing bracket.

    Args:
        valid_json: A valid JSON document.
        bracket_index: 0-based index of the structural closing bracket to remove.

    Returns:
        A ``T1Case`` carrying the broken text, the Original object, and the breakage record.
    """
    broken: BrokenJson = Breaker().break_json(valid_json, bracket_index=bracket_index)
    return T1Case(
        valid_json=valid_json,
        bracket_index=bracket_index,
        broken_text=broken.broken_text,
        original_object=broken.original_object,
        breakage_record=broken.record,
    )


def build_t1_prompt(broken_text: str) -> str:
    """Build the T1 repair prompt that asks the model for an edit script (ADR-0006).

    The model is told to emit a JSON array of ``{"old", "new"}`` find-and-replace operations
    — NOT repaired JSON — so the benchmark tests *surgical repair* rather than regeneration
    (ADR-0005). The contract (skip-on-ambiguity, composition order) is the model's
    responsibility to satisfy; the prompt states the format and goal.

    Args:
        broken_text: The Broken JSON text the model must repair.

    Returns:
        The prompt string to pass to ``LLMInterface.generate``.
    """
    return (
        "You are repairing broken JSON. Below is a JSON document with exactly one breakage: "
        "one structural closing bracket (a '}' or ']') has been deleted. Your job is to "
        "repair it.\n\n"
        "Output an EDIT SCRIPT, not repaired JSON. The edit script is a JSON array of "
        "find-and-replace operations, each an object {\"old\": <string>, \"new\": <string>}, "
        "applied in array order. Each operation replaces the substring `old` with `new`. "
        "If `old` matches zero positions the operation is skipped; if it matches more than "
        "one position it is skipped as ambiguous. Apply the minimal, unambiguous edits needed "
        "to repair the document.\n\n"
        "Return ONLY the JSON array of {\"old\", \"new\"} objects. No prose, no code fence.\n\n"
        f"Broken JSON:\n{broken_text}"
    )
