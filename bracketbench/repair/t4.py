"""The T4 test: real-world messy JSON repair (CONTEXT.md).

A T4 case is a genuinely broken JSON document collected from the wild. There is no Original
object — the pre-breakage document was never known or valid — so fidelity is unscorable: T4
is scored structural-only, with Structural as the ceiling (worth 100, not the ladder's 0.5).
The T4 prompt instructs the model to emit an edit script (ADR-0006) — NOT repaired JSON
(ADR-0005) — and the script is applied with the shared edit-script applier.
"""

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class T4Case:
    """One T4 test case: a real-world broken JSON document.

    Unlike T1/T2 there is no Original object and no breakage record: the text is broken as
    collected, so the fidelity tiers do not apply. A case is just the broken text.
    """

    broken_text: str


def build_t4_case(broken_text: str) -> T4Case:
    """Build a T4 case from an already-broken JSON document.

    No breaking step is needed: the text is broken as collected from the wild.

    Args:
        broken_text: The genuinely broken JSON text.

    Returns:
        A ``T4Case`` carrying the broken text.
    """
    return T4Case(broken_text=broken_text)


def build_t4_prompt(broken_text: str) -> str:
    """Build the T4 repair prompt that asks the model for an edit script (ADR-0006).

    The model is told to emit a JSON array of ``{"old", "new"}`` find-and-replace operations
    — NOT repaired JSON (ADR-0005). Unlike T1, the breakage is not described: T4 text is
    broken in the wild with an unknown number of problems, so the prompt asks for the
    minimal, unambiguous edits that make the document parse.

    Args:
        broken_text: The Broken JSON text the model must repair.

    Returns:
        The prompt string to pass to ``LLMInterface.generate``.
    """
    return (
        "You are repairing broken JSON. Below is a JSON document collected from the real "
        "world: it is genuinely broken and may have any number of problems (missing "
        "brackets, missing commas, stray characters, truncation, etc.). Your job is to "
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


# The curated T4 case set: genuinely broken JSON, each with a plausible repair. These are
# representative shapes of real-world breakage — missing closing brace, trailing comma,
# missing comma, truncated array, stray token. None has an Original object; each fails
# ``json.loads`` as-is and is repairable (add the brace, drop the comma, etc.).
DEFAULT_T4_CASES: List[str] = [
    '{"name": "test", "value": 42',
    '{"items": [1, 2, 3,]',
    '{"key": "val" "key2": "val2"}',
    "[1, 2, 3",
    '{"a": "b"c}',
]
