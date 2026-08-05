"""Tests for the vendored T1 default case fixture and its loader (issue #12,
spec #8).

The fixture is frozen: these tests assert both exact-shape invariants (24
records, metadata matching the tier table) and behavioral invariants that must
hold for every shipped case (parses, bracket_index in range, passes
``verify_ambiguous_yet_solvable``). One test rebuilds the fixture from the
generator to prove it stays reproducible from source.
"""

import json
import unittest
from typing import Any, Dict, List, Tuple

from bracketbench.breaker import Breaker
from bracketbench.benchmarking.default_t1_cases import (
    load_default_t1_cases,
    load_default_t1_records,
)
from bracketbench.benchmarking.verifier import (
    TIER_AXES,
    analyze_deletion,
    build_t1_grid,
    verify_ambiguous_yet_solvable,
)


class TestDefaultT1Fixture(unittest.TestCase):
    """The committed fixture file: shape, validity, and per-case guarantees."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.records: List[Dict[str, Any]] = load_default_t1_records()

    def test_fixture_has_exactly_24_records(self) -> None:
        self.assertEqual(len(self.records), 24)

    def test_every_record_has_the_expected_shape(self) -> None:
        for record in self.records:
            with self.subTest(record=record):
                self.assertEqual(
                    set(record.keys()), {"valid_json", "bracket_index", "metadata"}
                )
                self.assertIsInstance(record["valid_json"], str)
                self.assertIsInstance(record["bracket_index"], int)
                self.assertEqual(
                    set(record["metadata"].keys()),
                    {
                        "tier",
                        "deletion_depth",
                        "seed",
                        "depth",
                        "length_tokens",
                        "ambiguity",
                        "string_pathology_rate",
                    },
                )

    def test_every_valid_json_parses(self) -> None:
        for record in self.records:
            with self.subTest(record=record):
                json.loads(record["valid_json"])  # must not raise

    def test_every_bracket_index_is_in_range(self) -> None:
        for record in self.records:
            with self.subTest(record=record):
                n_closers = len(
                    Breaker()._find_structural_closing_brackets(record["valid_json"])
                )
                self.assertGreaterEqual(record["bracket_index"], 0)
                self.assertLess(record["bracket_index"], n_closers)

    def test_every_record_passes_ambiguous_yet_solvable(self) -> None:
        for record in self.records:
            with self.subTest(record=record):
                self.assertTrue(
                    verify_ambiguous_yet_solvable(
                        record["valid_json"], record["bracket_index"]
                    ),
                    f"record failed verification: {record['metadata']}",
                )

    def test_metadata_matches_the_tier_table(self) -> None:
        for record in self.records:
            metadata = record["metadata"]
            with self.subTest(metadata=metadata):
                self.assertIn(metadata["tier"], TIER_AXES)
                axes = TIER_AXES[metadata["tier"]]
                for axis in ("depth", "length_tokens", "ambiguity",
                             "string_pathology_rate"):
                    self.assertEqual(metadata[axis], axes[axis])

    def test_deletion_depth_labels_are_consistent(self) -> None:
        # The fixture covers each (tier, deletion_depth) cell exactly twice.
        combos: Dict[Tuple[str, str], int] = {}
        for record in self.records:
            metadata = record["metadata"]
            key = (metadata["tier"], metadata["deletion_depth"])
            combos[key] = combos.get(key, 0) + 1
        self.assertEqual(len(combos), 12)
        for key, count in combos.items():
            self.assertEqual(count, 2, f"cell {key} should have 2 records")

    def test_breaker_and_verifier_agree_on_the_breakage(self) -> None:
        # Guards the verifier's closer scan against the Breaker's bracket index
        # on real (pathological) fixture data.
        for record in self.records:
            with self.subTest(record=record):
                analysis = analyze_deletion(
                    record["valid_json"], record["bracket_index"]
                )
                broken = Breaker().break_json(
                    record["valid_json"], record["bracket_index"]
                )
                self.assertEqual(analysis.deletion_position, broken.record.position)
                self.assertEqual(analysis.broken_text, broken.broken_text)

    def test_fixture_is_reproducible_from_the_generator(self) -> None:
        # Rebuilding the grid must reproduce the committed fixture byte-for-byte:
        # the vendored set stays in sync with the generator + verifier.
        rebuilt = build_t1_grid()
        self.assertEqual(len(rebuilt), 24)
        for (valid_json, bracket_index, metadata), record in zip(
            rebuilt, self.records
        ):
            self.assertEqual(valid_json, record["valid_json"])
            self.assertEqual(bracket_index, record["bracket_index"])
            self.assertEqual(metadata, record["metadata"])


class TestDefaultT1Loader(unittest.TestCase):
    """The loader seam: pairs for ``evaluate``, records with metadata."""

    def test_loader_returns_pairs_for_evaluate(self) -> None:
        pairs = load_default_t1_cases()
        records = load_default_t1_records()
        self.assertEqual(len(pairs), 24)
        for pair, record in zip(pairs, records):
            self.assertEqual(pair, (record["valid_json"], record["bracket_index"]))
        for valid_json, bracket_index in pairs:
            self.assertIsInstance(valid_json, str)
            self.assertIsInstance(bracket_index, int)

    def test_loader_is_stable_across_calls(self) -> None:
        self.assertEqual(load_default_t1_cases(), load_default_t1_cases())


if __name__ == "__main__":
    unittest.main()
