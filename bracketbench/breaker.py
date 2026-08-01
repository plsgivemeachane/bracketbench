"""Breaker: generates Broken JSON by removing a structural closing bracket.

The Breaker transforms a valid JSON document into Broken JSON for T1/T2 test cases.
It removes ONE structural closing bracket (a ``}`` or ``]`` that is NOT inside a string
literal), records what it broke, and retains the Original object obtained by parsing the
input *before* breaking. See ``CONTEXT.md`` and ADR-0001.

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


class Breaker:
    """Generates Broken JSON by removing a structural closing bracket.

    ``bracket_index`` selects which structural closing bracket to remove (0-based, in
    source order). This is parameterized rather than random so test cases encode
    ``(valid_json, bracket_index)`` pairs and remain fully reproducible.
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

        positions = self._find_structural_closing_brackets(valid_json)
        if bracket_index < 0 or bracket_index >= len(positions):
            raise ValueError(
                f"bracket_index {bracket_index} out of range "
                f"(found {len(positions)} structural closing brackets)"
            )

        position, deleted_char = positions[bracket_index]
        broken_text = valid_json[:position] + valid_json[position + 1:]

        return BrokenJson(
            broken_text=broken_text,
            original_object=original_object,
            record=BreakRecord(
                break_type="deleted_closing_bracket",
                position=position,
                deleted_char=deleted_char,
            ),
        )

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
