"""The edit-script applier (ADR-0006 contract).

The applier is the pure, deterministic function
``apply(broken_text: str, edit_script: str) -> ApplyResult``. The model emits a JSON array of
``{"old", "new"}`` find-and-replace operations; the applier applies them in array order, each
against the result of the previous one.

Ambiguity handling (ADR-0006):
  - **Zero matches** of ``old`` -> skip (no-op), continue.
  - **Multiple matches** of ``old`` -> skip as ambiguous, continue.
  - **Malformed script** (not a JSON array, or elements not ``{"old","new"}`` strings) ->
    discard the script and return the broken text unchanged (maps to the 0.0 tier downstream).

The applier records which operations were skipped and why for auditing, but this never affects
the returned text. Credit logic lives in the scorer, not here.
"""

import json
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class SkippedOp:
    """One operation that was skipped during application, with the reason.

    Reasons follow ADR-0006: ``zero_match`` (``old`` not found) or ``multiple_match``
    (``old`` found more than once — ambiguous).
    """

    index: int
    reason: str
    old: str


@dataclass(frozen=True)
class ApplyResult:
    """The result of applying an edit script to broken text.

    ``repaired_text`` is whatever the script leaves behind (or the broken text unchanged on a
    malformed script). ``skipped`` records no-op/ambiguous operations for auditing.
    ``malformed`` is True when the script could not be parsed into a valid operation list.
    """

    repaired_text: str
    malformed: bool = False
    skipped: List[SkippedOp] = field(default_factory=list)


def apply(broken_text: str, edit_script: str) -> ApplyResult:
    """Apply a model's edit-script string to ``broken_text``.

    Args:
        broken_text: The Broken JSON text produced by the Breaker.
        edit_script: The model's raw edit-script string output (a JSON array of
            ``{"old", "new"}`` objects per ADR-0006).

    Returns:
        An ``ApplyResult`` carrying the repaired text, whether the script was malformed,
        and the list of skipped operations with reasons (for auditing).
    """
    operations = _parse_script(edit_script)
    if operations is None:
        # Malformed script: discard it, return the broken text unchanged (0.0 tier downstream).
        return ApplyResult(repaired_text=broken_text, malformed=True)

    text = broken_text
    skipped: List[SkippedOp] = []
    for index, (old, new) in enumerate(operations):
        occurrences = text.count(old)
        if occurrences == 0:
            skipped.append(SkippedOp(index=index, reason="zero_match", old=old))
            continue
        if occurrences > 1:
            skipped.append(SkippedOp(index=index, reason="multiple_match", old=old))
            continue
        # Exactly one match: apply the find-and-replace.
        text = text.replace(old, new)

    return ApplyResult(repaired_text=text, malformed=False, skipped=skipped)


def _parse_script(edit_script: str) -> Optional[List[tuple]]:
    """Parse a model's edit-script string into a list of ``(old, new)`` string pairs.

    Returns ``None`` if the script is malformed (not a JSON array, or any element is not an
    object with string ``old`` and ``new`` fields). Per ADR-0006, a malformed script maps to
    the 0.0 tier; the caller returns the broken text unchanged.
    """
    try:
        parsed = json.loads(edit_script)
    except (json.JSONDecodeError, TypeError):
        return None

    if not isinstance(parsed, list):
        return None

    operations: List[tuple] = []
    for element in parsed:
        if not isinstance(element, dict):
            return None
        old = element.get("old")
        new = element.get("new")
        if not isinstance(old, str) or not isinstance(new, str):
            return None
        operations.append((old, new))

    return operations
