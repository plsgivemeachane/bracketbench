"""Tests for the ambiguous-yet-solvable verifier and the 24-triple grid builder
(issue #11, spec #8).

Tested only at the public seams: ``verify_ambiguous_yet_solvable``,
``analyze_deletion``, ``classify_deletion_depth``, ``build_grid_case`` and
``build_t1_grid``. Verification is exercised against the exact ``json.dumps``
text the Breaker sees (escapes/whitespace change which substrings match), not
against in-memory objects.
"""

import json
import unittest
from typing import Any, Dict, List, Tuple

from bracketbench.breaker import Breaker
from bracketbench.benchmarking.verifier import (
    DELETION_DEPTHS,
    TIER_AXES,
    analyze_deletion,
    build_grid_case,
    build_t1_grid,
    classify_deletion_depth,
    verify_ambiguous_yet_solvable,
)


class TestVerifyAmbiguousYetSolvable(unittest.TestCase):
    """Unit tests at the verification seam."""

    # --- Returns True for hand-crafted ambiguous-yet-solvable docs ---

    def test_hand_crafted_ambiguous_yet_solvable_returns_true(self) -> None:
        """The naive old 'active' repeats; an id-anchored old is unique -> True."""
        valid_json = (
            '{"id": 1, "status": "active", "x": 2, '
            '"y": {"id": 2, "status": "active"}}'
        )
        self.assertTrue(verify_ambiguous_yet_solvable(valid_json, bracket_index=1))

    def test_ambiguous_inner_object_returns_true(self) -> None:
        """Deleting a nested close: naive 'active' repeats, id-anchored unique."""
        valid_json = (
            '{"id": 1, "a": {"id": 2, "v": "active"}, '
            '"b": {"id": 3, "v": "active"}}'
        )
        self.assertTrue(verify_ambiguous_yet_solvable(valid_json, bracket_index=0))

    # --- Returns False for trivially solvable docs (naive old unique) ---

    def test_trivially_solvable_doc_returns_false(self) -> None:
        """Naive old '5' is unique -> the case is trivially solvable -> False."""
        valid_json = '{"id": 1, "x": 5}'
        self.assertFalse(verify_ambiguous_yet_solvable(valid_json, bracket_index=0))

    def test_flat_all_unique_doc_returns_false(self) -> None:
        """No repeating value before the close: naive old unique -> False."""
        valid_json = '{"id": 1, "a": "uniqueOne", "b": "uniqueTwo"}'
        self.assertFalse(verify_ambiguous_yet_solvable(valid_json, bracket_index=0))

    # --- Returns False for unsolvable docs (no unambiguous old) ---

    def test_unsolvable_doc_returns_false(self) -> None:
        """The id-anchored old 'id": 3' matches inside '"id": 30' -> unsolvable.

        The naive old (the id value '3') is ambiguous (it matches inside the
        later id 30), but the nearest-id-anchored old is also ambiguous: the
        span from '"id": 3' to the deletion point is a prefix of the text at
        '"id": 30'. No unambiguous old exists -> the case cannot be shipped.
        """
        valid_json = '{"a": {"id": 3}, "b": {"id": 30}}'
        self.assertFalse(verify_ambiguous_yet_solvable(valid_json, bracket_index=0))

    # --- Verification runs against the json.dumps text, not the object ---

    def test_verification_runs_against_dumps_text_not_object(self) -> None:
        """Escapes in the text matter: the in-memory value 'x\"y' has no
        substring match in the dumps output, but the escaped literal does."""
        # Three values that are the SAME in-memory string 'x"y' (1 backslash-escaped
        # quote each); in the text they render as "x\"y" with a backslash.
        valid_json = '{"id": 1, "a": "x\\"y", "b": "x\\"y", "c": "x\\"y"}'
        analysis = analyze_deletion(valid_json, bracket_index=0)
        # The naive old is the exact text literal, escapes included.
        self.assertEqual(analysis.naive_old, '"x\\"y"')
        self.assertEqual(analysis.naive_occurrences, 3)
        # The unescaped in-memory value never appears in the text.
        self.assertEqual(valid_json.count('x"y'), 0)
        self.assertTrue(verify_ambiguous_yet_solvable(valid_json, bracket_index=0))

    # --- Analysis exposes the pieces the verifier reasons about ---

    def test_id_anchored_old_spans_from_nearest_id_to_deletion(self) -> None:
        """The id-anchored old starts at the nearest id key and ends at the
        deletion point; the naive old is the exact preceding value literal."""
        valid_json = (
            '{"id": 1, "a": {"id": 2, "v": "active"}, '
            '"b": {"id": 3, "v": "active"}}'
        )
        analysis = analyze_deletion(valid_json, bracket_index=0)

        self.assertEqual(analysis.deleted_char, "}")
        self.assertEqual(analysis.naive_old, '"active"')
        self.assertEqual(analysis.naive_occurrences, 2)
        self.assertEqual(analysis.id_anchored_old, '"id": 2, "v": "active"')
        self.assertEqual(analysis.id_anchored_occurrences, 1)
        self.assertEqual(analysis.broken_text, valid_json.replace("}", "", 1))

    def test_analysis_aligns_with_breaker_positions(self) -> None:
        """The verifier's closer scan indexes brackets exactly like the Breaker."""
        valid_json = (
            '{"id": 1, "a": {"id": 2, "v": "active"}, '
            '"b": {"id": 3, "v": "active"}}'
        )
        for bracket_index in range(3):
            analysis = analyze_deletion(valid_json, bracket_index)
            broken = Breaker().break_json(valid_json, bracket_index)
            self.assertEqual(analysis.deletion_position, broken.record.position)
            self.assertEqual(analysis.deleted_char, broken.record.deleted_char)
            self.assertEqual(analysis.broken_text, broken.broken_text)

    def test_pathological_strings_do_not_count_as_brackets(self) -> None:
        """'}']' inside string literals are ignored by the closer scan."""
        # Each "value" contains a literal } inside the string (escaped quote).
        valid_json = '{"id": 1, "a": "x\\"}", "b": "x\\"}"}'
        closers_expected = 1  # only the root close is structural
        analysis = analyze_deletion(valid_json, bracket_index=0)
        self.assertEqual(analysis.nesting_depth, 1)
        # The naive old is the last value literal, which repeats.
        self.assertEqual(analysis.naive_old, '"x\\"}"')
        self.assertEqual(analysis.naive_occurrences, 2)
        self.assertEqual(analysis.id_anchored_occurrences, 1)
        self.assertEqual(len(Breaker()._find_structural_closing_brackets(valid_json)),
                         closers_expected)
        self.assertTrue(verify_ambiguous_yet_solvable(valid_json, bracket_index=0))

    # --- Input validation ---

    def test_out_of_range_bracket_index_raises(self) -> None:
        with self.assertRaises(ValueError):
            verify_ambiguous_yet_solvable('{"id": 1, "x": 5}', bracket_index=7)

    def test_invalid_json_raises(self) -> None:
        with self.assertRaises(ValueError):
            verify_ambiguous_yet_solvable('{"id": 1, "x": ', bracket_index=0)


class TestDeletionDepthClassification(unittest.TestCase):
    """Deletion-depth labels follow the per-doc convention (spec #8)."""

    # A depth-3 spine: b-close (innermost, depth 3), a-close (mid, depth 2),
    # root close (outermost, depth 1).
    NESTED = '{"id": 1, "a": {"id": 2, "b": {"id": 3, "x": 1}}}'

    def test_outermost_is_the_final_closing_bracket(self) -> None:
        self.assertEqual(classify_deletion_depth(self.NESTED, 2), "outermost")

    def test_innermost_is_maximum_nesting_depth(self) -> None:
        self.assertEqual(classify_deletion_depth(self.NESTED, 0), "innermost")

    def test_mid_depth_is_half_the_maximum_nesting_depth(self) -> None:
        # max depth 3 -> mid target round(3/2) == 2 -> the a-close.
        self.assertEqual(classify_deletion_depth(self.NESTED, 1), "mid_depth")

    def test_all_bracket_indices_classify(self) -> None:
        for bracket_index in range(3):
            label = classify_deletion_depth(self.NESTED, bracket_index)
            self.assertIn(label, DELETION_DEPTHS)


class TestGridBuilder(unittest.TestCase):
    """The 24-triple grid builder: count, verification, labels, reproducibility."""

    @classmethod
    def setUpClass(cls) -> None:
        # Build the full grid once; every test below is a pure assertion over it.
        cls.grid: List[Tuple[str, int, Dict[str, Any]]] = build_t1_grid()

    def test_grid_emits_exactly_24_cases(self) -> None:
        self.assertEqual(len(self.grid), 24)

    def test_every_case_is_valid_and_in_range(self) -> None:
        for valid_json, bracket_index, metadata in self.grid:
            with self.subTest(metadata=metadata):
                json.loads(valid_json)  # must not raise
                n_closers = len(Breaker()._find_structural_closing_brackets(valid_json))
                self.assertGreaterEqual(bracket_index, 0)
                self.assertLess(bracket_index, n_closers)

    def test_every_case_passes_verification(self) -> None:
        for valid_json, bracket_index, metadata in self.grid:
            with self.subTest(metadata=metadata):
                self.assertTrue(
                    verify_ambiguous_yet_solvable(valid_json, bracket_index),
                    f"case failed verification: {metadata}",
                )

    def test_ambiguity_and_constructibility_hold_per_case(self) -> None:
        for valid_json, bracket_index, metadata in self.grid:
            with self.subTest(metadata=metadata):
                analysis = analyze_deletion(valid_json, bracket_index)
                self.assertGreaterEqual(analysis.naive_occurrences, 2)
                self.assertEqual(analysis.id_anchored_occurrences, 1)

    def test_deletion_depth_labels_are_correctly_assigned(self) -> None:
        for valid_json, bracket_index, metadata in self.grid:
            with self.subTest(metadata=metadata):
                self.assertEqual(
                    classify_deletion_depth(valid_json, bracket_index),
                    metadata["deletion_depth"],
                )

    def test_metadata_matches_the_tier_table(self) -> None:
        for _, _, metadata in self.grid:
            with self.subTest(metadata=metadata):
                self.assertIn(metadata["tier"], TIER_AXES)
                self.assertIn(metadata["deletion_depth"], DELETION_DEPTHS)
                axes = TIER_AXES[metadata["tier"]]
                for axis in ("depth", "length_tokens", "ambiguity",
                             "string_pathology_rate"):
                    self.assertEqual(metadata[axis], axes[axis])

    def test_grid_covers_all_tier_depth_combinations_twice(self) -> None:
        combos: Dict[Tuple[str, str], List[int]] = {}
        for _, _, metadata in self.grid:
            combos.setdefault(
                (metadata["tier"], metadata["deletion_depth"]), []
            ).append(metadata["seed"])
        self.assertEqual(len(combos), 4 * 3)
        for key, seeds in combos.items():
            self.assertEqual(len(seeds), 2, f"cell {key} does not have 2 seeds")

    def test_seed_parity_is_disjoint_between_base_seeds(self) -> None:
        # Cells from base seed 1 only ever resolve to odd seeds, base seed 2 to
        # even ones, so no two cells can collide on the same resolved triple.
        seeds = [metadata["seed"] for _, _, metadata in self.grid]
        self.assertEqual(sum(1 for s in seeds if s % 2 == 1), 12)
        self.assertEqual(sum(1 for s in seeds if s % 2 == 0), 12)

    def test_every_case_is_reproducible_from_its_triple(self) -> None:
        for valid_json, bracket_index, metadata in self.grid:
            with self.subTest(metadata=metadata):
                rebuilt_valid_json, rebuilt_index, rebuilt_metadata = build_grid_case(
                    metadata["seed"], metadata["tier"], metadata["deletion_depth"]
                )
                self.assertEqual(rebuilt_valid_json, valid_json)
                self.assertEqual(rebuilt_index, bracket_index)
                self.assertEqual(rebuilt_metadata, metadata)


if __name__ == "__main__":
    unittest.main()
