"""Tests for the edit-script applier (ADR-0006 contract).

The applier is the pure, deterministic seam
``apply(broken_text: str, edit_script: str) -> str``. The model emits a JSON array of
``{"old", "new"}`` find-and-replace operations; the applier applies them in array order,
each against the result of the previous. Zero matches and multiple matches are both skipped
(no-op, continue). A malformed script maps to the 0.0 tier downstream, so the applier returns
the broken text unchanged.

These tests target only the public ``apply`` seam; they never reach inside the applier's
internals (the audit record is observable on the result, not asserted as a side effect).
"""

import unittest

from bracketbench.repair.applier import apply, ApplyResult


class TestEditScriptApplier(unittest.TestCase):
    """Tests for the public ``apply(broken_text, edit_script)`` seam."""

    # --- Slice 1: a single unambiguous edit repairs the breakage ---

    def test_single_unambiguous_edit_repairs_broken_text(self) -> None:
        """The worked example from ADR-0006: broken '{"a": 1' -> '{"a": 1}'."""
        broken_text = '{"a": 1'
        edit_script = '[{"old": ": 1", "new": ": 1}"}]'

        result = apply(broken_text, edit_script)

        self.assertEqual(result.repaired_text, '{"a": 1}')
        self.assertEqual(result.malformed, False)

    # --- Slice 2: multiple operations compose in array order ---

    def test_multiple_operations_compose_in_order(self) -> None:
        """Each edit applies against the result of the previous (sequential, not parallel)."""
        broken_text = '{"a": 1'  # missing the closing brace
        # First edit inserts the brace; second is a no-op against the now-repaired text.
        edit_script = '[{"old": ": 1", "new": ": 1}"}, {"old": "zzz", "new": "nope"}]'

        result = apply(broken_text, edit_script)

        self.assertEqual(result.repaired_text, '{"a": 1}')
        # The second edit matched zero positions and was skipped.
        self.assertEqual(len(result.skipped), 1)
        self.assertEqual(result.skipped[0].reason, "zero_match")

    # --- Slice 3: zero matches is a skip, not an error ---

    def test_zero_matches_is_skipped_not_an_error(self) -> None:
        """An edit whose `old` matches nowhere is skipped; application continues."""
        broken_text = '{"a": 1'
        edit_script = '[{"old": "does-not-appear", "new": "X"}, {"old": ": 1", "new": ": 1}"}]'

        result = apply(broken_text, edit_script)

        self.assertEqual(result.repaired_text, '{"a": 1}')
        self.assertEqual(len(result.skipped), 1)
        self.assertEqual(result.skipped[0].reason, "zero_match")

    # --- Slice 4: multiple matches is skipped as ambiguous ---

    def test_multiple_matches_is_skipped_as_ambiguous(self) -> None:
        """An `old` matching more than once is skipped as ambiguous, not first/all-matched."""
        broken_text = '{"a": 1, "b": 10}'  # "1" appears twice ("1" and "10")
        edit_script = '[{"old": "1", "new": "X"}]'

        result = apply(broken_text, edit_script)

        # Ambiguous edit skipped -> broken text unchanged.
        self.assertEqual(result.repaired_text, '{"a": 1, "b": 10}')
        self.assertEqual(len(result.skipped), 1)
        self.assertEqual(result.skipped[0].reason, "multiple_match")

    # --- Slice 5: a malformed script returns the broken text unchanged ---

    def test_malformed_script_returns_broken_text_unchanged(self) -> None:
        """A non-JSON-array script is discarded; the broken text is returned as-is (0.0 tier)."""
        broken_text = '{"a": 1'
        edit_script = "this is not json at all"

        result = apply(broken_text, edit_script)

        self.assertEqual(result.repaired_text, broken_text)
        self.assertTrue(result.malformed)

    def test_non_array_json_returns_broken_text_unchanged(self) -> None:
        """A JSON value that is not an array is also malformed."""
        broken_text = '{"a": 1'
        edit_script = '{"old": "x", "new": "y"}'  # object, not array

        result = apply(broken_text, edit_script)

        self.assertEqual(result.repaired_text, broken_text)
        self.assertTrue(result.malformed)

    def test_element_missing_keys_returns_broken_text_unchanged(self) -> None:
        """An array element without old/new is a malformed shape."""
        broken_text = '{"a": 1'
        edit_script = '[{"old": ": 1"}]'  # missing "new"

        result = apply(broken_text, edit_script)

        self.assertEqual(result.repaired_text, broken_text)
        self.assertTrue(result.malformed)

    # --- Slice 6: JSON string escaping handles quotes and newlines ---

    def test_edit_with_quoted_newline_applies_correctly(self) -> None:
        """JSON escaping in the script must round-trip newlines in `new`."""
        broken_text = '{"a": 1'
        # new = ": 1}\n" — a JSON-escaped newline.
        edit_script = '[{"old": ": 1", "new": ": 1}\\n"}]'

        result = apply(broken_text, edit_script)

        self.assertEqual(result.repaired_text, '{"a": 1}\n')

    # --- Slice 7: an edit that matches exactly once applies (whole-text replacement allowed) ---

    def test_whole_text_replacement_applies_when_old_matches_once(self) -> None:
        """The auditable escape hatch: old == entire broken text matches once, so it applies."""
        broken_text = '{"a": 1'
        edit_script = '[{"old": "{\\"a\\": 1", "new": "{\\"a\\": 1}"}]'

        result = apply(broken_text, edit_script)

        self.assertEqual(result.repaired_text, '{"a": 1}')
        self.assertEqual(len(result.skipped), 0)


if __name__ == "__main__":
    unittest.main()
