"""Tests for the synthetic JSON generator (Ticket B / issue #10).

A seeded, parameterized generator emitting valid JSON with independent control of four
axes: depth, length (token budget), ambiguity, and string-pathology rate. Pure function,
deterministic from seed, no I/O. See CONTEXT.md (Breaker / Original object) and ADR-0001.

Tested only at the public generate(...) seam, mirroring how test_breaker.py tests
Breaker().break_json. No internal helpers are exercised directly.
"""

import itertools
import json
import string
import unittest

from bracketbench.benchmarking.generator import GenerateError, generate


def _max_depth(obj: object) -> int:
    """Max nesting depth of a JSON-loaded Python object.

    A scalar (str/int/float/bool/None) has depth 0; a container's depth is 1 plus the
    max depth of its children (an empty container has depth 1).
    """
    if isinstance(obj, list):
        return 1 + max((_max_depth(x) for x in obj), default=0)
    if isinstance(obj, dict):
        return 1 + max((_max_depth(v) for v in obj.values()), default=0)
    return 0


def _leaf_values(obj: object):
    """Yield leaf values (scalars) of a JSON-loaded object, in document order."""
    if isinstance(obj, list):
        for x in obj:
            yield from _leaf_values(x)
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _leaf_values(v)
    else:
        yield obj


def _non_id_leaves(obj: object):
    """Yield leaf values, excluding the unique integer `id` fields (1..N).

    id fields are unique by construction and would skew ambiguity measurement.
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "id" and isinstance(v, int):
                continue
            yield from _non_id_leaves(v)
    elif isinstance(obj, list):
        for x in obj:
            yield from _non_id_leaves(x)
    else:
        yield obj


def _count_tokens(text: str) -> int:
    """Structural token count: split on JSON punctuation and whitespace.

    Mirrors the generator's internal token measure (see generator.py docstring).
    """
    cleaned = text
    for ch in "{}[]:,":
        cleaned = cleaned.replace(ch, " ")
    return len(cleaned.split())


def _is_pathological_string(py_value: object) -> bool:
    """True if the Python string carries a pathology the generator injects.

    Detected on the parsed value, where JSON escapes have resolved: an escaped quote
    becomes a literal ``"``, an escaped backslash becomes a single ``\\``, a unicode
    escape becomes its control character (ord < 32), and literal ``}``/``]`` survive.
    """
    if not isinstance(py_value, str):
        return False
    if any(c in py_value for c in ('"', "\\", "}", "]")):
        return True
    return any(ord(c) < 32 for c in py_value)


class TestGenerator(unittest.TestCase):
    """Tests for generate, only at the public seam."""

    # --- Slice 1: generate returns valid JSON ---

    def test_generate_returns_valid_json(self) -> None:
        """The emitted string must parse successfully via json.loads."""
        out = generate(depth=2, length_tokens=64, ambiguity=0.5,
                       string_pathology_rate=0.0, seed=1)
        self.assertIsInstance(out, str)
        json.loads(out)  # validity is free by construction

    def test_generate_returns_object_at_top_level(self) -> None:
        """The emitted document is a JSON object (a container the Breaker can target)."""
        out = generate(depth=1, length_tokens=32, ambiguity=0.0,
                       string_pathology_rate=0.0, seed=1)
        self.assertIsInstance(json.loads(out), dict)

    # --- Slice 2: determinism (same seed + params -> identical output) ---

    def test_same_seed_and_params_produce_identical_output(self) -> None:
        """generate is deterministic: same inputs must yield byte-identical output."""
        kwargs = dict(depth=3, length_tokens=128, ambiguity=0.5,
                      string_pathology_rate=0.3, seed=42)
        a = generate(**kwargs)
        b = generate(**kwargs)
        self.assertEqual(a, b)

    def test_different_seed_produces_different_output(self) -> None:
        """Different seeds should (almost surely) produce different documents."""
        kwargs = dict(depth=3, length_tokens=128, ambiguity=0.5,
                      string_pathology_rate=0.3)
        a = generate(seed=1, **kwargs)
        b = generate(seed=2, **kwargs)
        self.assertNotEqual(a, b)

    # --- Slice 2b: parameter validation ---

    def test_depth_below_one_raises(self) -> None:
        """depth must be >= 1."""
        with self.assertRaises(GenerateError):
            generate(depth=0, length_tokens=64, ambiguity=0.5,
                     string_pathology_rate=0.0, seed=1)

    def test_ambiguity_out_of_range_raises(self) -> None:
        """ambiguity must be in [0.0, 1.0]."""
        with self.assertRaises(GenerateError):
            generate(depth=2, length_tokens=64, ambiguity=1.5,
                     string_pathology_rate=0.0, seed=1)

    def test_string_pathology_rate_out_of_range_raises(self) -> None:
        """string_pathology_rate must be in [0.0, 1.0]."""
        with self.assertRaises(GenerateError):
            generate(depth=2, length_tokens=64, ambiguity=0.0,
                     string_pathology_rate=-0.1, seed=1)

    # --- Slice 3: depth matches the param exactly ---

    def test_depth_one_is_flat_object(self) -> None:
        """depth=1 -> a flat object whose children are all leaves."""
        out = generate(depth=1, length_tokens=64, ambiguity=0.3,
                       string_pathology_rate=0.0, seed=7)
        self.assertEqual(_max_depth(json.loads(out)), 1)

    def test_depth_three_reaches_three(self) -> None:
        """depth=3 -> the max nesting depth of the document is exactly 3."""
        out = generate(depth=3, length_tokens=256, ambiguity=0.3,
                       string_pathology_rate=0.0, seed=7)
        self.assertEqual(_max_depth(json.loads(out)), 3)

    def test_depth_five_reaches_five(self) -> None:
        """depth=5 -> the max nesting depth of the document is exactly 5."""
        out = generate(depth=5, length_tokens=512, ambiguity=0.3,
                       string_pathology_rate=0.0, seed=11)
        self.assertEqual(_max_depth(json.loads(out)), 5)

    # --- Slice 4: ambiguity fraction within tolerance ---

    def test_ambiguity_zero_no_repeating_leaves(self) -> None:
        """ambiguity=0.0 -> no leaf value appears more than once."""
        out = generate(depth=3, length_tokens=256, ambiguity=0.0,
                       string_pathology_rate=0.0, seed=3)
        leaves = list(_non_id_leaves(json.loads(out)))
        seen: dict = {}
        for v in leaves:
            seen[v] = seen.get(v, 0) + 1
        repeats = sum(1 for v, c in seen.items() if c >= 2)
        self.assertEqual(repeats, 0)

    def test_ambiguity_high_has_repeating_leaves(self) -> None:
        """ambiguity=1.0 -> a substantial fraction of distinct leaves repeat."""
        out = generate(depth=3, length_tokens=256, ambiguity=1.0,
                       string_pathology_rate=0.0, seed=3)
        leaves = list(_non_id_leaves(json.loads(out)))
        self.assertGreater(len(leaves), 0)
        seen: dict = {}
        for v in leaves:
            seen[v] = seen.get(v, 0) + 1
        repeating = sum(1 for v, c in seen.items() if c >= 2)
        rate = repeating / len(seen) if seen else 0.0
        self.assertGreater(rate, 0.3)

    # --- Slice 5: string_pathology_rate within tolerance ---

    def test_pathology_rate_mid_has_pathological_strings(self) -> None:
        """string_pathology_rate=0.5 -> ~half of string values are pathological."""
        out = generate(depth=3, length_tokens=512, ambiguity=0.3,
                       string_pathology_rate=0.5, seed=5)
        strings = [v for v in _non_id_leaves(json.loads(out)) if isinstance(v, str)]
        self.assertGreater(len(strings), 4, "need several strings to estimate a rate")
        patho = sum(1 for s in strings if _is_pathological_string(s))
        rate = patho / len(strings)
        self.assertGreater(rate, 0.2)
        self.assertLess(rate, 0.8)

    # --- Slice 6: string_pathology_rate=0 -> clean strings only ---

    def test_pathology_rate_zero_emits_clean_strings_only(self) -> None:
        """string_pathology_rate=0.0 -> no string contains escapes or ]/}."""
        out = generate(depth=3, length_tokens=512, ambiguity=0.3,
                       string_pathology_rate=0.0, seed=9)
        strings = [v for v in _non_id_leaves(json.loads(out)) if isinstance(v, str)]
        self.assertGreater(len(strings), 0)
        for s in strings:
            self.assertFalse(_is_pathological_string(s),
                             f"clean-mode string is pathological: {s!r}")
            self.assertTrue(all(c in (string.ascii_letters + string.digits) for c in s),
                            f"clean-mode string has non-alphanumeric chars: {s!r}")

    # --- Slice 7: length_tokens within tolerance ---

    def test_length_tokens_within_tolerance(self) -> None:
        """Emitted token count is within +/-15% of the target."""
        target = 512
        out = generate(depth=3, length_tokens=target, ambiguity=0.3,
                       string_pathology_rate=0.0, seed=2)
        actual = _count_tokens(out)
        self.assertGreaterEqual(actual, target * 0.85)
        self.assertLessEqual(actual, target * 1.15)

    def test_length_tokens_large_target(self) -> None:
        """A larger target still lands within tolerance."""
        target = 1024
        out = generate(depth=4, length_tokens=target, ambiguity=0.3,
                       string_pathology_rate=0.2, seed=13)
        actual = _count_tokens(out)
        self.assertGreaterEqual(actual, target * 0.85)
        self.assertLessEqual(actual, target * 1.15)


class TestGeneratorProperties(unittest.TestCase):
    """Property-style tests: run the core invariants across a param grid + seeds."""

    def test_validity_and_depth_hold_across_param_grid(self) -> None:
        """For every (depth, ambiguity, pathology, seed) in the grid: valid + depth match."""
        seeds = (1, 2, 3, 7, 42, 99)
        depths = (1, 2, 3, 4, 5)
        ambs = (0.0, 0.3, 0.7, 1.0)
        pathos = (0.0, 0.5, 1.0)
        for depth, amb, patho, seed in itertools.product(depths, ambs, pathos, seeds):
            with self.subTest(depth=depth, ambiguity=amb, pathology=patho, seed=seed):
                out = generate(depth=depth, length_tokens=128, ambiguity=amb,
                               string_pathology_rate=patho, seed=seed)
                obj = json.loads(out)  # must not raise
                self.assertEqual(_max_depth(obj), depth,
                                 f"depth mismatch at depth={depth} amb={amb} "
                                 f"patho={patho} seed={seed}")

    def test_determinism_holds_across_param_grid(self) -> None:
        """For every grid point, two calls with the same seed produce identical output."""
        for depth in (1, 3, 5):
            for amb in (0.0, 0.5, 1.0):
                for patho in (0.0, 0.5, 1.0):
                    for seed in (1, 7, 42):
                        with self.subTest(depth=depth, ambiguity=amb,
                                          pathology=patho, seed=seed):
                            a = generate(depth=depth, length_tokens=128, ambiguity=amb,
                                         string_pathology_rate=patho, seed=seed)
                            b = generate(depth=depth, length_tokens=128, ambiguity=amb,
                                         string_pathology_rate=patho, seed=seed)
                            self.assertEqual(a, b)

    def test_pathology_zero_is_clean_across_seeds(self) -> None:
        """string_pathology_rate=0 -> clean strings only, for every seed in a range."""
        for seed in range(1, 21):
            with self.subTest(seed=seed):
                out = generate(depth=3, length_tokens=256, ambiguity=0.3,
                               string_pathology_rate=0.0, seed=seed)
                strings = [v for v in _non_id_leaves(json.loads(out))
                           if isinstance(v, str)]
                for s in strings:
                    self.assertFalse(_is_pathological_string(s),
                                     f"seed={seed} clean-mode string pathological: {s!r}")

    def test_ambiguity_zero_is_unique_across_seeds(self) -> None:
        """ambiguity=0 -> no repeating non-id leaf, for every seed in a range."""
        for seed in range(1, 21):
            with self.subTest(seed=seed):
                out = generate(depth=3, length_tokens=256, ambiguity=0.0,
                               string_pathology_rate=0.0, seed=seed)
                leaves = list(_non_id_leaves(json.loads(out)))
                seen: dict = {}
                for v in leaves:
                    seen[v] = seen.get(v, 0) + 1
                repeats = sum(1 for c in seen.values() if c >= 2)
                self.assertEqual(repeats, 0, f"seed={seed} had repeats")


if __name__ == "__main__":
    unittest.main()
