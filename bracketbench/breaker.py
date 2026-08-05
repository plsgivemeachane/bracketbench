"""Breaker: generates Broken JSON by removing structural closing brackets.

The Breaker transforms a valid JSON document into Broken JSON for T1/T2 test cases.
It removes structural closing brackets (a ``}`` or ``]`` that is NOT inside a string
literal), records what it broke, and retains the Original object obtained by parsing the
input *before* breaking. ``break_json`` applies ONE breakage (T1); ``break_json_multi``
composes N breakages sequentially, each time picking a structural closing bracket from the
*current* (already-broken) text (T2). See ``CONTEXT.md`` and ADR-0001.

This is the generator side of the benchmark; the model's edit-script output (ADR-0005) is
scored separately.
"""

import json
from dataclasses import dataclass
from typing import Any, List, Tuple


@dataclass
class BreakRecord:
    """What the Breaker did to the valid JSON document."""

    break_type: str
    position: int
    deleted_char: str


@dataclass
class BrokenJson:
    """The result of breaking a valid JSON document."""

    broken_text: str
    original_object: Any
    record: BreakRecord


@dataclass
class BrokenJsonMulti:
    """The result of composing N breakages onto a valid JSON document.

    ``records`` carries every breakage in application order. Each record's ``position`` is
    0-based in the text *as it stood before that deletion*, so positions of later records
    refer to the already-broken text, not the original document.
    """

    broken_text: str
    original_object: Any
    records: List[BreakRecord]


class Breaker:
    """Generates Broken JSON by removing structural closing brackets.

    ``bracket_index`` selects which structural closing bracket to remove (0-based, in
    source order). This is parameterized rather than random so test cases encode
    ``(valid_json, bracket_index)`` pairs and remain fully reproducible. For T2,
    ``break_json_multi`` takes a list of indexes, one per sequential breakage.
    """

    def break_json(self, valid_json: str, bracket_index: int = 0) -> BrokenJson:
        """Break ``valid_json`` by removing its k-th structural closing bracket.

        Args:
            valid_json: A valid JSON document.
            bracket_index: 0-based index of the structural closing bracket to remove,
                in source order.

        Returns:
            A BrokenJson carrying the broken text, the Original object, and a BreakRecord.

        Raises:
            ValueError: If ``valid_json`` is not valid JSON, or ``bracket_index`` is out
                of range.
        """
        # Capture the Original object BEFORE breaking.
        original_object = json.loads(valid_json)

        position, deleted_char, broken_text = self._delete_closing_bracket(
            valid_json, bracket_index
        )

        return BrokenJson(
            broken_text=broken_text,
            original_object=original_object,
            record=BreakRecord(
                break_type="deleted_closing_bracket",
                position=position,
                deleted_char=deleted_char,
            ),
        )

    def break_json_multi(
        self, valid_json: str, bracket_indices: List[int]
    ) -> BrokenJsonMulti:
        """Compose N breakages onto ``valid_json``, recording each.

        N sequential single-bracket deletions are applied: at each step the k-th structural
        closing bracket of the *current* text (already-broken after the previous step) is
        removed. The Original object is the parse of the pre-breakage document and is
        retained for Fidelity scoring ("track, don't tell").

        Args:
            valid_json: A valid JSON document.
            bracket_indices: The per-step 0-based indexes of the structural closing bracket
                to remove, in application order. Must be non-empty (T2 composes N > 1
                breakages).

        Returns:
            A BrokenJsonMulti carrying the final broken text, the Original object, and one
            BreakRecord per breakage in application order.

        Raises:
            ValueError: If ``valid_json`` is not valid JSON, ``bracket_indices`` is empty,
                or any index is out of range for the text at that step.
        """
        # Capture the Original object BEFORE breaking.
        original_object = json.loads(valid_json)
        if not bracket_indices:
            raise ValueError(
                "bracket_indices must contain at least one index "
                "(T2 composes N > 1 breakages)"
            )

        current_text = valid_json
        records: List[BreakRecord] = []
        for index in bracket_indices:
            position, deleted_char, current_text = self._delete_closing_bracket(
                current_text, index
            )
            records.append(
                BreakRecord(
                    break_type="deleted_closing_bracket",
                    position=position,
                    deleted_char=deleted_char,
                )
            )

        return BrokenJsonMulti(
            broken_text=current_text,
            original_object=original_object,
            records=records,
        )

    @staticmethod
    def _delete_closing_bracket(
        text: str, bracket_index: int
    ) -> Tuple[int, str, str]:
        """Delete the k-th structural closing bracket of ``text``.

        Returns the deleted character's position (in ``text``), the deleted character, and
        the resulting text. Raises ValueError if ``bracket_index`` is out of range.
        """
        positions = Breaker._find_structural_closing_brackets(text)
        if bracket_index < 0 or bracket_index >= len(positions):
            raise ValueError(
                f"bracket_index {bracket_index} out of range "
                f"(found {len(positions)} structural closing brackets)"
            )

        position, deleted_char = positions[bracket_index]
        return position, deleted_char, text[:position] + text[position + 1:]

    @staticmethod
    def _find_structural_closing_brackets(text: str) -> List[Tuple[int, str]]:
        """Return ``(position, char)`` for each ``}`` or ``]`` outside string literals.

        Tracks JSON string state: a ``"`` toggles in/out of a string, and a backslash
        escapes the next character (so an escaped quote does not end the string).
        Characters inside strings are ignored.
        """
        positions: List[Tuple[int, str]] = []
        in_string = False
        escaped = False

        for i, ch in enumerate(text):
            if escaped:
                escaped = False
                continue
            if in_string:
                if ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            # Outside a string.
            if ch == '"':
                in_string = True
            elif ch in ("}", "]"):
                positions.append((i, ch))

        return positions
