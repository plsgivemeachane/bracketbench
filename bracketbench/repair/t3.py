"""The T3 test: complex ipynb repair (CONTEXT.md, ADR-0004).

A T3 case is a Jupyter Notebook (.ipynb) with real notebook-level problems: corrupt cell
``source``, mangled ``outputs``, cell-type inconsistency, etc. Unlike T1/T2 there is no
Breaker that records an Original object — the notebook text is already broken when the case
is built. The T3 prompt instructs the model to emit an edit script (ADR-0006) — a JSON array
of ``{"old", "new"}`` find-and-replace operations — NOT repaired JSON (ADR-0005). The
repaired notebook is scored by notebook-semantic static checking (ADR-0004): no code is ever
executed.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class T3Case:
    """One T3 test case: the broken notebook text (a JSON string of a .ipynb)."""

    broken_text: str


def build_t3_case(broken_notebook_text: str) -> T3Case:
    """Build a T3 case from a broken notebook's text.

    T3 cases are curated broken notebooks supplied as text; ``build_t3_case`` only wraps
    them so ``evaluate`` treats T1 and T3 cases uniformly (build case -> build prompt ->
    generate -> apply -> score).

    Args:
        broken_notebook_text: The text of a broken .ipynb (a JSON string, or text that
            should have been one).

    Returns:
        A ``T3Case`` carrying the broken notebook text.
    """
    return T3Case(broken_text=broken_notebook_text)


def build_t3_prompt(broken_text: str) -> str:
    """Build the T3 repair prompt that asks the model for an edit script (ADR-0006).

    The model repairs a broken Jupyter Notebook by emitting an edit script — a JSON array
    of ``{"old", "new"}`` find-and-replace operations — NOT repaired JSON, so the benchmark
    tests *surgical repair* rather than regeneration (ADR-0005). The repaired notebook is
    scored by static semantic checks only (ADR-0004): no code is executed.

    Args:
        broken_text: The broken notebook text (a JSON string of a .ipynb) to repair.

    Returns:
        The prompt string to pass to ``LLMInterface.generate``.
    """
    return (
        "You are repairing a broken Jupyter Notebook. Below is a .ipynb notebook (JSON) "
        "with notebook-level problems: corrupt cell source, mangled outputs, cell-type "
        "inconsistencies, and similar damage. Your job is to repair it into a structurally "
        "valid notebook.\n\n"
        "Output an EDIT SCRIPT, not repaired JSON. The edit script is a JSON array of "
        "find-and-replace operations, each an object {\"old\": <string>, \"new\": <string>}, "
        "applied in array order. Each operation replaces the substring `old` with `new`. "
        "If `old` matches zero positions the operation is skipped; if it matches more than "
        "one position it is skipped as ambiguous. Apply the minimal, unambiguous edits "
        "needed to repair the notebook.\n\n"
        "Return ONLY the JSON array of {\"old\", \"new\"} objects. No prose, no code fence.\n\n"
        f"Broken notebook:\n{broken_text}"
    )
