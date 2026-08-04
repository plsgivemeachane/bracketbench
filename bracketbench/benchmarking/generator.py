"""Synthetic JSON generator (Ticket B / issue #10).

A seeded, parameterized generator that emits **valid JSON** with independent control of
four axes:

- **depth** -- target nesting depth of the emitted document. Guaranteed to *match* the
  param exactly: a depth-spine ensures the document reaches the requested depth, and every
  sibling value is capped so nothing exceeds it.
- **length_tokens** -- target token count, capped below the ~8k weak-model context floor.
- **ambiguity** -- fraction of leaf values that appear >=2 times. Each object keeps a
  unique integer ``id`` field (1..N); other leaf values are drawn from small repeating
  pools (with probability ``ambiguity``) or allocated fresh from a monotonic counter
  (guaranteed unique) with probability ``1 - ambiguity``.
- **string_pathology_rate** -- fraction of string values containing escaped quotes,
  backslashes, unicode escapes, and literal ``}``/``]`` inside the string. Built as Python
  strings with the pathology; ``json.dumps`` emits the valid JSON literal (validity free
  by construction).

``generate(...)`` is a pure function, deterministic from ``seed``, no I/O. The emitted
string is the kind of valid JSON the Breaker (see ``bracketbench.breaker``) consumes: a
top-level JSON object whose Original object is retained for Fidelity scoring.

Token length is measured by a structural token count -- split on JSON punctuation
(``{}[]:,``) and whitespace -- a pure-Python proxy with no dependency and no I/O. The
``length_tokens`` parameter is honoured within tolerance: the generator adds sibling
content until the structural token count reaches the lower tolerance bound, then stops as
soon as further content would overshoot the upper bound.

See ``CONTEXT.md`` and ADR-0001.
"""

from __future__ import annotations

import json
import random
import string
from typing import Any, List

__all__ = ["generate", "GenerateError"]


class GenerateError(ValueError):
    """Raised when generate() cannot satisfy the requested parameters."""


# Characters considered "clean" (no escaping, no structural look-alikes) for string values
# and keys when string_pathology_rate is 0.
_CLEAN_ALPHABET = string.ascii_letters + string.digits

# Pathology fragments, expressed as Python (pre-dump) characters that json.dumps renders
# as the intended awkward JSON escapes / literals:
#   "  -> JSON \"   (escaped quote)
#   \\ -> JSON \\   (escaped backslash)
#   \x01 -> JSON \u0001 (unicode escape; ensure_ascii=True default)
#   }  -> literal } inside the string (structural look-alike)
#   ]  -> literal ] inside the string (structural look-alike)
_PATHOLOGY_FRAGMENTS = ['"', '\\', '\x01', '}', ']']

# Tolerance for the length_tokens axis, as a fraction of the target. The generator aims to
# land within [target * (1 - TOL), target * (1 + TOL)] and stops as soon as further content
# would overshoot the upper bound.
_LENGTH_TOLERANCE = 0.15

# When the target token count is so small that the upper bound is below what one minimal
# object needs, the generator still emits the smallest valid object it can and does not
# raise -- "within tolerance" is best-effort for degenerate targets.
_MIN_TOKEN_BUDGET = 8


def _count_tokens(text: str) -> int:
    """Structural token count: split on JSON punctuation and whitespace."""
    cleaned = text
    for ch in "{}[]:,":
        cleaned = cleaned.replace(ch, " ")
    return len(cleaned.split())


def _make_clean_string(rng: random.Random, length: int = 4) -> str:
    """Build a clean alphanumeric string (no escapes, no structural look-alikes)."""
    return "".join(rng.choice(_CLEAN_ALPHABET) for _ in range(length))


def _encode_counter(n: int) -> str:
    """Encode a counter as a short, unique, clean-alphanumeric string.

    Used for fresh string values so that at ``ambiguity=0`` no two leaves collide.
    """
    alphabet = string.ascii_lowercase + string.digits  # 36 chars
    if n == 0:
        return "v0"
    chars: List[str] = []
    while n > 0:
        chars.append(alphabet[n % 36])
        n //= 36
    return "v" + "".join(reversed(chars))


class _AxisController:
    """Internal state for one generate() call.

    Holds the RNG and the per-axis bookkeeping (id counter, fresh-value counter,
    repeating value pools) so the recursive emitter can stay small. This is intentionally
    not part of the public interface -- ``generate`` is the seam.

    Ambiguity approximation: leaf values are drawn from a small repeating pool with
    probability ``ambiguity`` and allocated fresh (unique) otherwise. This makes the
    fraction of leaf *occurrences* whose value appears >=2 times approximate ``ambiguity``
    well at the extremes (0.0 -> no repeats, 1.0 -> all repeat) and reasonably in between.
    Pool size scales inversely with the rate so higher ambiguity yields more collision.
    """

    def __init__(self, params: dict) -> None:
        self.depth = params["depth"]
        self.length_tokens = params["length_tokens"]
        self.ambiguity = params["ambiguity"]
        self.string_pathology_rate = params["string_pathology_rate"]
        self.seed = params["seed"]
        self.rng = random.Random(self.seed)

        # Unique id per object, 1..N, as required by the ambiguity axis.
        self._next_id = 1
        # Monotonic counter for fresh (non-repeating) leaf values.
        self._fresh_counter = 0

        # Repeating value pools, sized to hit the target ambiguity rate. Empty when
        # ambiguity == 0 (every leaf is fresh -> unique).
        self._int_pool = self._build_int_pool(self.ambiguity)
        self._str_pool = self._build_str_pool(self.ambiguity)

    @staticmethod
    def _pool_size(rate: float) -> int:
        """Pool size scaling inversely with the rate; >=2 when rate > 0."""
        return max(2, int(round(2.0 / max(rate, 0.05))))

    @staticmethod
    def _build_int_pool(rate: float) -> List[int]:
        if rate <= 0.0:
            return []
        size = _AxisController._pool_size(rate)
        rng = random.Random(12345)  # deterministic pool contents, independent of seed
        return [rng.randint(1, 9) for _ in range(size)]

    @staticmethod
    def _build_str_pool(rate: float) -> List[str]:
        if rate <= 0.0:
            return []
        size = _AxisController._pool_size(rate)
        rng = random.Random(54321)  # deterministic, independent of seed
        return ["".join(rng.choice(_CLEAN_ALPHABET) for _ in range(4)) for _ in range(size)]

    def next_id(self) -> int:
        """Allocate the next unique object id (1..N)."""
        n = self._next_id
        self._next_id += 1
        return n

    def make_int_value(self) -> int:
        """An int leaf, drawn from the repeating pool with probability ``ambiguity``."""
        if self._int_pool and self.rng.random() < self.ambiguity:
            return self.rng.choice(self._int_pool)
        self._fresh_counter += 1
        # Fresh ints start above the pool range (1..9) so they never collide with it.
        return 1000 + self._fresh_counter

    def make_str_value(self) -> str:
        """A string leaf, with pathology governed by ``string_pathology_rate``."""
        # Base clean string: repeating pool (with prob ambiguity) or fresh unique.
        if self._str_pool and self.rng.random() < self.ambiguity:
            base = self.rng.choice(self._str_pool)
        else:
            self._fresh_counter += 1
            base = _encode_counter(self._fresh_counter)
        # Apply pathology independently of pool/fresh choice.
        if self.rng.random() < self.string_pathology_rate:
            frag = self.rng.choice(_PATHOLOGY_FRAGMENTS)
            return base[:1] + frag + base[1:]
        return base


def _emit_leaf(ctrl: _AxisController) -> Any:
    """Emit a leaf value (depth 0): an int or a string."""
    if ctrl.rng.random() < 0.5:
        return ctrl.make_int_value()
    return ctrl.make_str_value()


def _emit_value(ctrl: _AxisController, remaining: int, force_exact: bool) -> Any:
    """Emit a value whose depth is <= ``remaining`` (== remaining if force_exact).

    ``remaining`` is the max depth this value may have. A leaf has depth 0; a container
    has depth 1 + max(child depths). To force depth == remaining (>0), emit a container
    whose first child is itself forced to depth remaining-1.
    """
    if remaining <= 0:
        return _emit_leaf(ctrl)  # depth 0; force_exact satisfied (0 == 0)
    if force_exact or ctrl.rng.random() < 0.5:
        if ctrl.rng.random() < 0.4:
            return _emit_list(ctrl, remaining, force_exact)
        return _emit_dict(ctrl, remaining, force_exact)
    return _emit_leaf(ctrl)


def _emit_dict(ctrl: _AxisController, remaining: int, force_exact: bool) -> dict:
    """Emit a JSON object (with a unique ``id``) of depth ==/<= ``remaining``."""
    obj: dict[str, Any] = {"id": ctrl.next_id()}
    child_budget = remaining - 1
    # First key carries the forced-depth child (if any); extras are depth-capped.
    key = _make_clean_string(ctrl.rng, 3)
    obj[key] = _emit_value(ctrl, child_budget, force_exact=force_exact)
    n_extra = ctrl.rng.randint(0, 2)
    for _ in range(n_extra):
        key = _make_clean_string(ctrl.rng, 3)
        obj[key] = _emit_value(ctrl, child_budget, force_exact=False)
    return obj


def _emit_list(ctrl: _AxisController, remaining: int, force_exact: bool) -> list:
    """Emit a JSON array of depth ==/<= ``remaining`` (arrays carry no ``id``)."""
    child_budget = remaining - 1
    n = ctrl.rng.randint(1, 3)
    items: List[Any] = [_emit_value(ctrl, child_budget, force_exact=force_exact)]
    for _ in range(n - 1):
        items.append(_emit_value(ctrl, child_budget, force_exact=False))
    return items


def generate(
    depth: int,
    length_tokens: int,
    ambiguity: float,
    string_pathology_rate: float,
    seed: int,
) -> str:
    """Emit a valid JSON document with the requested axis values.

    Args:
        depth: Target nesting depth of the emitted document (>= 1). A document with
            ``depth=1`` is a flat object of leaves. The document's max depth matches
            ``depth`` exactly.
        length_tokens: Target token count (structural token count, split on JSON
            punctuation and whitespace), capped below the ~8k weak-model context floor.
            Honoured within tolerance (see ``_LENGTH_TOLERANCE``).
        ambiguity: Fraction of leaf values that should appear >=2 times, in [0.0, 1.0].
            ``0.0`` emits all-unique leaves; ``1.0`` emits all-repeating leaves.
        string_pathology_rate: Fraction of string values containing escaped quotes,
            backslashes, unicode escapes, and literal ``}``/``]`` inside the string, in
            [0.0, 1.0]. ``0.0`` emits clean alphanumeric strings only.
        seed: Seed for the deterministic RNG. Same seed + params always yield identical
            output.

    Returns:
        A valid JSON string (a top-level object).

    Raises:
        GenerateError: If any parameter is out of range.
    """
    if depth < 1:
        raise GenerateError(f"depth must be >= 1, got {depth}")
    if length_tokens < 1:
        raise GenerateError(f"length_tokens must be >= 1, got {length_tokens}")
    if not 0.0 <= ambiguity <= 1.0:
        raise GenerateError(f"ambiguity must be in [0.0, 1.0], got {ambiguity}")
    if not 0.0 <= string_pathology_rate <= 1.0:
        raise GenerateError(
            f"string_pathology_rate must be in [0.0, 1.0], got {string_pathology_rate}"
        )

    params = {
        "depth": depth,
        "length_tokens": length_tokens,
        "ambiguity": ambiguity,
        "string_pathology_rate": string_pathology_rate,
        "seed": seed,
    }
    ctrl = _AxisController(params)

    # Root object: always a dict, with a unique id, reaching exactly ``depth`` via a
    # forced depth-spine as its first value.
    root: dict[str, Any] = {"id": ctrl.next_id()}
    spine_key = _make_clean_string(ctrl.rng, 3)
    root[spine_key] = _emit_value(ctrl, depth - 1, force_exact=True)

    # Grow the document toward length_tokens by adding sibling keys whose values are
    # depth-capped at depth-1 (so the document's max depth stays exactly ``depth``).
    upper = max(length_tokens * (1.0 + _LENGTH_TOLERANCE), _MIN_TOKEN_BUDGET)
    target_lo = length_tokens * (1.0 - _LENGTH_TOLERANCE)

    while True:
        # Snapshot the token count before adding another sibling.
        if _count_tokens(json.dumps(root)) >= target_lo:
            break
        key = _make_clean_string(ctrl.rng, 3)
        if key in root:
            continue  # collision; try another key
        prev = dict(root)
        root[key] = _emit_value(ctrl, depth - 1, force_exact=False)
        if _count_tokens(json.dumps(root)) > upper:
            root = prev  # overshoot: discard the last sibling and stop.
            break

    return json.dumps(root)
