"""Ambiguous-yet-solvable verifier + 24-triple grid builder (issue #11, spec #8).

Spec #8's per-case guarantee for the shipped T1 hard set: each case must be
**ambiguous-yet-solvable** against the exact ``json.dumps`` output the Breaker sees:

- The naive short ``old`` -- the JSON literal of the leaf value immediately
  preceding the deleted closing bracket, i.e. the repeating value a weak model
  guesses as its find-and-replace anchor (ADR-0006's worked example calls this
  ``"1"`` in ``{"a": 1}``) -- must match **>= 2 times** in the broken text, so the
  edit is skipped as ambiguous under the ADR-0006 contract.
- An ``id``-anchored longer ``old`` -- the span from the nearest unique ``"id"``
  key to the deletion point -- must match **exactly once**, so a constructible
  edit exists that restores the document and yields exact-fidelity 100.

``verify_ambiguous_yet_solvable`` returns ``True`` only when both hold, and
``False`` otherwise: for docs where the naive ``old`` is unique (trivially
solvable -- no ambiguity pressure) and for docs where no unambiguous ``old``
exists (unsolvable -- the failure mode that would break the difficulty
gradient). Verification runs against the **text** the Breaker consumes
(escapes/whitespace change which substrings match), never against the in-memory
object.

``build_t1_grid`` enumerates the 24 ``(seed, tier, deletion_depth)`` triples
(4 tiers x 3 deletion-depths x 2 seeds) using ``generate`` from
:mod:`bracketbench.benchmarking.generator` (issue #10) with each tier's axis
levels. If a generated case fails verification, the builder reseeds or re-picks
the bracket at the same deletion-depth until the property holds.

Deletion-depth convention (per-doc, spec #8):

- **outermost** -- the document's outermost closing bracket (the final ``}``/``]``).
- **innermost** -- a closing bracket at maximum nesting depth.
- **mid_depth** -- a closing bracket at roughly half the maximum nesting depth.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from bracketbench.benchmarking.generator import generate

__all__ = [
    "TIER_AXES",
    "DELETION_DEPTHS",
    "DeletionAnalysis",
    "analyze_deletion",
    "verify_ambiguous_yet_solvable",
    "classify_deletion_depth",
    "build_grid_case",
    "build_t1_grid",
]

# The four difficulty tiers and their coupled axis levels (spec #8). Axes move
# together per tier; values are theoretical starting points (spec #8 Q10-B).
TIER_AXES: Dict[str, Dict[str, Any]] = {
    "easy": {
        "depth": 3,
        "length_tokens": 500,
        "ambiguity": 0.10,
        "string_pathology_rate": 0.00,
    },
    "medium": {
        "depth": 6,
        "length_tokens": 1500,
        "ambiguity": 0.40,
        "string_pathology_rate": 0.30,
    },
    "hard": {
        "depth": 9,
        "length_tokens": 3000,
        "ambiguity": 0.70,
        "string_pathology_rate": 0.60,
    },
    "extreme": {
        "depth": 12,
        "length_tokens": 5000,
        "ambiguity": 0.85,
        "string_pathology_rate": 0.80,
    },
}

# The three per-doc deletion-depth categories (spec #8). "mid-depth" is spelled
# ``mid_depth`` so the label is a clean metadata value.
DELETION_DEPTHS: Tuple[str, ...] = ("outermost", "mid_depth", "innermost")

# The two base seeds per (tier, deletion_depth) cell.
_GRID_SEEDS: Tuple[int, ...] = (1, 2)

# Reseeding stride. Cells with base seed 1 only ever try odd candidate seeds and
# cells with base seed 2 only even ones, so two cells can never collide on the
# same resolved (seed, tier, deletion_depth) triple.
_RESEED_STRIDE = 2

# Upper bound on reseeding attempts per cell before giving up. The naive-ambiguity
# property holds for a large fraction of random draws (int leaves repeat as digits
# almost always), so 60 attempts is a very wide net.
_MAX_ATTEMPTS = 60

# Every object in a generated document carries a unique integer id as its first
# field; ``json.dumps`` renders it as ``"id": <n>``. Keys are exactly 3 clean
# characters and string values are <= 5 characters, so this pattern can only
# match actual id anchors, never content inside strings.
_ID_ANCHOR_RE = re.compile(r'"id": (\d+)')


@dataclass(frozen=True)
class _Closer:
    """One structural closing bracket, with the context the verifier needs."""

    pos: int
    char: str
    depth: int
    leaf_start: int
    leaf_end: int


def _scan_closers(text: str) -> List[_Closer]:
    """Scan ``text`` for structural closing brackets with nesting depth and the
    extent of the leaf value immediately preceding each one.

    Mirrors the Breaker's string-state awareness: ``}``/``]`` inside string
    literals (pathological strings) are ignored, escaped quotes do not end a
    string, and the nesting depth of a closing bracket is the number of open
    containers at the moment it closes (1 for the root's close).

    ``leaf_start``/``leaf_end`` are the exact text extent of the most recently
    completed leaf value (string literal or number) in the document up to the
    bracket -- textually the value immediately before the deleted bracket, which
    is what a weak model would guess as its naive ``old``. A container that is
    empty at close time has no such leaf; the extent is ``(-1, -1)``.
    """
    closers: List[_Closer] = []
    stack: List[Tuple[str, int]] = []
    last_leaf: Tuple[int, int] = (-1, -1)
    num_start: int = -1
    in_string = False
    escaped = False
    string_start = 0

    for i, ch in enumerate(text):
        if escaped:
            escaped = False
            continue
        if in_string:
            if ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
                last_leaf = (string_start, i + 1)
            continue

        # A non-number character terminates an in-progress number literal.
        if num_start >= 0 and not ("0" <= ch <= "9"):
            last_leaf = (num_start, i)
            num_start = -1

        if ch == '"':
            in_string = True
            string_start = i
        elif ch in "{[":
            stack.append((ch, i))
        elif ch in "}]":
            closers.append(
                _Closer(
                    pos=i,
                    char=ch,
                    depth=len(stack),
                    leaf_start=last_leaf[0],
                    leaf_end=last_leaf[1],
                )
            )
            stack.pop()
        elif "0" <= ch <= "9":
            if num_start < 0:
                num_start = i
        # Whitespace, ':' and ',' need no handling: numbers were finalized above,
        # and a completed leaf stays the "last leaf" until the next one completes.

    return closers


def _find_id_anchors(text: str) -> List[Tuple[int, int]]:
    """Return ``(start, id_value)`` for every unique ``"id": <n>`` anchor."""
    return [(m.start(), int(m.group(1))) for m in _ID_ANCHOR_RE.finditer(text)]


@dataclass(frozen=True)
class DeletionAnalysis:
    """Everything the verifier needs to know about one T1 breakage, measured
    against the exact ``json.dumps`` text (not the in-memory object).

    ``naive_old`` is the JSON literal of the leaf value immediately before the
    deleted bracket (with quotes and escapes exactly as they appear in the
    text); ``id_anchored_old`` is the span from the nearest ``"id"`` key to the
    deletion point. Both are counted in the broken text, matching how the
    applier (ADR-0006) would see them.
    """

    valid_json: str
    bracket_index: int
    broken_text: str
    deleted_char: str
    deletion_position: int
    nesting_depth: int
    naive_old: str
    naive_occurrences: int
    id_anchored_old: str
    id_anchored_occurrences: int


def analyze_deletion(valid_json: str, bracket_index: int) -> DeletionAnalysis:
    """Analyze the breakage of ``valid_json`` at ``bracket_index``.

    Args:
        valid_json: A valid JSON document (the ``json.dumps`` output the Breaker
            sees).
        bracket_index: 0-based index (in source order) of the structural closing
            bracket to delete, as accepted by ``Breaker.break_json``.

    Returns:
        A ``DeletionAnalysis`` with the broken text, the naive short ``old`` and
        its occurrence count, and the ``id``-anchored ``old`` and its occurrence
        count -- all measured against the broken text.

    Raises:
        ValueError: If ``valid_json`` is not valid JSON, or ``bracket_index`` is
            out of range.
    """
    if not isinstance(valid_json, str) or not valid_json:
        raise ValueError("valid_json must be a non-empty JSON string")
    json.loads(valid_json)  # validity is a precondition of the seam

    closers = _scan_closers(valid_json)
    if bracket_index < 0 or bracket_index >= len(closers):
        raise ValueError(
            f"bracket_index {bracket_index} out of range "
            f"(found {len(closers)} structural closing brackets)"
        )
    closer = closers[bracket_index]

    deletion_position = closer.pos
    broken_text = (
        valid_json[:deletion_position] + valid_json[deletion_position + 1:]
    )

    if closer.leaf_start >= 0:
        naive_old = valid_json[closer.leaf_start:closer.leaf_end]
        naive_occurrences = broken_text.count(naive_old)
    else:
        # No leaf precedes the bracket (empty container): there is no naive old
        # to be ambiguous about, so the case cannot be ambiguous-yet-solvable.
        naive_old = ""
        naive_occurrences = 0

    anchors = _find_id_anchors(valid_json)
    before = [start for start, _ in anchors if start < deletion_position]
    if not before:
        # Cannot happen for generator output (every document opens with the root
        # object's id), but fail loudly rather than silently mis-verify.
        raise ValueError("no id anchor precedes the deletion point")
    anchor_start = max(before)
    id_anchored_old = valid_json[anchor_start:deletion_position]
    id_anchored_occurrences = broken_text.count(id_anchored_old)

    return DeletionAnalysis(
        valid_json=valid_json,
        bracket_index=bracket_index,
        broken_text=broken_text,
        deleted_char=closer.char,
        deletion_position=deletion_position,
        nesting_depth=closer.depth,
        naive_old=naive_old,
        naive_occurrences=naive_occurrences,
        id_anchored_old=id_anchored_old,
        id_anchored_occurrences=id_anchored_occurrences,
    )


def verify_ambiguous_yet_solvable(valid_json: str, bracket_index: int) -> bool:
    """Verify spec #8's per-case guarantee for one ``(valid_json, bracket_index)`` pair.

    Returns ``True`` only when **both** hold against the broken text:

    - the naive short ``old`` (the leaf value immediately before the deleted
      bracket) matches >= 2 times -- a naive edit would be skipped as ambiguous
      (ADR-0006), so the case is not trivially solvable;
    - the ``id``-anchored longer ``old`` (from the nearest unique ``"id"`` key to
      the deletion point) matches exactly once -- a constructible edit exists
      that yields exact-fidelity 100.

    Returns ``False`` for trivially solvable docs (naive ``old`` unique) and for
    unsolvable docs (no unambiguous ``old``). Raises ``ValueError`` for invalid
    inputs (see ``analyze_deletion``).
    """
    analysis = analyze_deletion(valid_json, bracket_index)
    return analysis.naive_occurrences >= 2 and analysis.id_anchored_occurrences == 1


def classify_deletion_depth(valid_json: str, bracket_index: int) -> str:
    """Return the deletion-depth label of ``bracket_index`` in ``valid_json``.

    Implements the per-doc convention (spec #8): the final closing bracket is
    ``outermost``; a bracket at maximum nesting depth is ``innermost``; a bracket
    at roughly half the maximum nesting depth is ``mid_depth``.
    """
    closers = _scan_closers(valid_json)
    if bracket_index < 0 or bracket_index >= len(closers):
        raise ValueError(
            f"bracket_index {bracket_index} out of range "
            f"(found {len(closers)} structural closing brackets)"
        )
    max_depth = max(closer.depth for closer in closers)
    closer = closers[bracket_index]

    if bracket_index == len(closers) - 1:
        return "outermost"
    if closer.depth == max_depth:
        return "innermost"
    target = max(1, round(max_depth / 2))
    if closer.depth == target:
        return "mid_depth"
    raise ValueError(
        f"bracket_index {bracket_index} is not at a deletion-depth position "
        f"(depth {closer.depth}, max {max_depth}, mid target {target})"
    )


def _candidate_bracket_indices(valid_json: str, deletion_depth: str) -> List[int]:
    """Bracket indices at the target deletion-depth, in source order."""
    closers = _scan_closers(valid_json)
    max_depth = max(closer.depth for closer in closers)

    if deletion_depth == "outermost":
        # The document's outermost closing bracket is the final one in source
        # order (the root's close).
        return [len(closers) - 1]
    if deletion_depth == "innermost":
        return [i for i, closer in enumerate(closers) if closer.depth == max_depth]
    if deletion_depth == "mid_depth":
        target = max(1, round(max_depth / 2))
        return [i for i, closer in enumerate(closers) if closer.depth == target]
    raise ValueError(f"unknown deletion_depth: {deletion_depth!r}")


def build_grid_case(
    seed: int, tier: str, deletion_depth: str
) -> Tuple[str, int, Dict[str, Any]]:
    """Build one verified ``(valid_json, bracket_index, metadata)`` case.

    Generates a document with ``tier``'s axis levels at ``seed`` and picks a
    bracket at the target ``deletion_depth``. If no candidate bracket of the
    generated document passes ``verify_ambiguous_yet_solvable``, the seed is
    advanced by ``_RESEED_STRIDE`` (keeping the base-seed parity so two cells
    never collide) and the document is regenerated, until the property holds.

    The returned ``metadata`` records the case's triple (``tier``,
    ``deletion_depth``, actual ``seed``) plus the tier's axis levels, so each
    case is reproducible from ``build_grid_case(metadata["seed"], metadata["tier"],
    metadata["deletion_depth"])`` and per-cell failure attribution is possible.
    The deletion position is a derived output: ``analyze_deletion(valid_json,
    bracket_index).deletion_position``.
    """
    if tier not in TIER_AXES:
        raise ValueError(f"unknown tier: {tier!r}")
    if deletion_depth not in DELETION_DEPTHS:
        raise ValueError(f"unknown deletion_depth: {deletion_depth!r}")
    axes = TIER_AXES[tier]

    for attempt in range(_MAX_ATTEMPTS):
        candidate_seed = seed + attempt * _RESEED_STRIDE
        valid_json = generate(
            depth=axes["depth"],
            length_tokens=axes["length_tokens"],
            ambiguity=axes["ambiguity"],
            string_pathology_rate=axes["string_pathology_rate"],
            seed=candidate_seed,
        )
        for bracket_index in _candidate_bracket_indices(valid_json, deletion_depth):
            if verify_ambiguous_yet_solvable(valid_json, bracket_index):
                metadata = {
                    "tier": tier,
                    "deletion_depth": deletion_depth,
                    "seed": candidate_seed,
                    "depth": axes["depth"],
                    "length_tokens": axes["length_tokens"],
                    "ambiguity": axes["ambiguity"],
                    "string_pathology_rate": axes["string_pathology_rate"],
                }
                return valid_json, bracket_index, metadata

    raise RuntimeError(
        f"could not build a verified case for seed={seed} tier={tier!r} "
        f"deletion_depth={deletion_depth!r} after {_MAX_ATTEMPTS} attempts"
    )


def build_t1_grid() -> List[Tuple[str, int, Dict[str, Any]]]:
    """Enumerate the 24 ``(seed, tier, deletion_depth)`` triples.

    4 tiers x 3 deletion-depths x 2 seeds, tier-major. Every emitted case passes
    ``verify_ambiguous_yet_solvable`` and is reproducible from its
    ``(seed, tier, deletion_depth)`` triple (see ``build_grid_case``).
    """
    return [
        build_grid_case(seed, tier, deletion_depth)
        for tier in TIER_AXES
        for deletion_depth in DELETION_DEPTHS
        for seed in _GRID_SEEDS
    ]
