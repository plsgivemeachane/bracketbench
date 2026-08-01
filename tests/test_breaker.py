"""Tests for the Breaker module (T1/T2 broken-JSON generator).

The Breaker transforms a valid JSON document into Broken JSON by removing ONE structural
closing bracket, while retaining the Original object and a record of what was broken
("track, don't tell" - see CONTEXT.md).
"""

import unittest

from bracketbench.breaker import Breaker, BrokenJson, BreakRecord


class TestBreaker(unittest.TestCase):
    """Tests for the Breaker, tested only at the public Breaker.break_json seam."""

    # --- Slice 1: happy path - removes the k-th structural closing bracket ---

    def test_break_json_removes_first_closing_bracket(self) -> None:
        """Deleting bracket_index=0 of '{"a": 1}' removes the only '}'."""
        valid_json = '{"a": 1}'
        result = Breaker().break_json(valid_json, bracket_index=0)

        self.assertIsInstance(result, BrokenJson)
        # The closing } at index 7 is removed.
        self.assertEqual(result.broken_text, '{"a": 1')
        self.assertEqual(result.original_object, {"a": 1})
        self.assertIsInstance(result.record, BreakRecord)
        self.assertEqual(result.record.break_type, "deleted_closing_bracket")
        self.assertEqual(result.record.position, 7)
        self.assertEqual(result.record.deleted_char, "}")

    # --- Slice 4: brackets inside string literals are NOT structural ---

    def test_brackets_inside_string_literals_are_ignored(self) -> None:
        """A '}' or ']' inside a JSON string value must not be deleted."""
        # The string value "x}y]z" contains a } and a ]; only the final } is structural.
        valid_json = '{"a": "x}y]z"}'
        result = Breaker().break_json(valid_json, bracket_index=0)

        self.assertEqual(result.record.deleted_char, "}")
        # The deleted structural } is the last character of the document.
        self.assertEqual(result.record.position, len(valid_json) - 1)
        # The } and ] inside the string survive in broken_text.
        self.assertEqual(result.broken_text, '{"a": "x}y]z"')

    def test_escaped_quote_does_not_end_string(self) -> None:
        """An escaped \" inside a string must not close the string state."""
        # value = he said "}"  -- the } is inside the string, not structural.
        valid_json = '{"a": "he said \\"}\\""}'
        result = Breaker().break_json(valid_json, bracket_index=0)

        # Only the final } is structural.
        self.assertEqual(result.record.position, len(valid_json) - 1)
        self.assertEqual(result.record.deleted_char, "}")

    # --- Slice 5: invalid JSON input raises a clear error ---

    def test_invalid_json_input_raises_value_error(self) -> None:
        """Non-JSON input must raise ValueError, not silently misbehave."""
        with self.assertRaises(ValueError):
            Breaker().break_json("not json at all", bracket_index=0)

    def test_truncated_json_input_raises_value_error(self) -> None:
        """Truncated JSON (looks plausible but won't parse) must raise ValueError."""
        with self.assertRaises(ValueError):
            Breaker().break_json('{"a": 1', bracket_index=0)

    # --- Slice 6: bracket_index out of range raises a clear error ---

    def test_bracket_index_out_of_range_raises_value_error(self) -> None:
        """Asking for a bracket that doesn't exist must raise ValueError."""
        valid_json = '{"a": 1}'  # exactly one structural closing bracket
        with self.assertRaises(ValueError):
            Breaker().break_json(valid_json, bracket_index=1)

    def test_negative_bracket_index_raises_value_error(self) -> None:
        """A negative bracket_index must raise ValueError."""
        with self.assertRaises(ValueError):
            Breaker().break_json('{"a": 1}', bracket_index=-1)

    # --- Slice 7: selects the k-th structural bracket in nested JSON ---

    def test_selects_kth_bracket_in_nested_json(self) -> None:
        """bracket_index=0 picks the first structural closer, not the last."""
        valid_json = '{"a": [1, 2], "b": {"c": 3}}'
        # Structural closing brackets in source order: ] (after 2), } (inner), } (outer).
        result_inner = Breaker().break_json(valid_json, bracket_index=1)
        self.assertEqual(result_inner.record.deleted_char, "}")
        # The broken text is the input minus that one char.
        self.assertEqual(
            result_inner.broken_text,
            valid_json[:result_inner.record.position]
            + valid_json[result_inner.record.position + 1:],
        )
        # The Original object is the fully-parsed nested structure.
        self.assertEqual(result_inner.original_object, {"a": [1, 2], "b": {"c": 3}})


if __name__ == "__main__":
    unittest.main()
