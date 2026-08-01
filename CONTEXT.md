# BracketBench

A focused benchmark that tests how well LLMs repair broken JSON. It exists to put numbers
on the thesis that "AI sucks at fixing messy JSON."

## Language

### Tests

BracketBench runs **four peer tests**. Each is scored 0-100 independently. There is no
"Standard" container and no subtask grouping - the four tests are peers.

**Test 1 (T1) - single-breakage repair**:
A valid JSON document has exactly one breakage applied at one known, recorded location. The
model must repair it. Scored on the full 4-tier ladder (see Scoring). The canonical "minimal"
case.
_Avoid_: simple test, basic test, subtask 1

**Test 2 (T2) - multi-breakage repair**:
A valid JSON document has N breakages applied. The model is **not told** how many, but the
benchmark **tracks** what it broke and retains the Original object, so Fidelity stays
scorable. Scored on the full 4-tier ladder.
_Avoid_: advanced test, subtask 2

**Test 3 (T3) - Complex ipynb repair**:
A Jupyter Notebook (.ipynb) with real notebook-level problems (corrupt cell `source`, mangled
`outputs`, cell-type inconsistency, etc.). Scored by **notebook-semantic static checking** -
no code is executed. This exposes "writing new stuff inside the field of JSON," which is
genuinely hard. Has its own scoring path, distinct from the JSON ladder.
_Avoid_: ipynb test, notebook test, subtask 3

**Test 4 (T4) - real-world messy JSON**:
Genuinely broken JSON collected from the wild - no Original object exists. Only the
Structural tier is reachable, so Structural is the ceiling (worth 100) here. Fidelity tiers
do not apply.
_Avoid_: wild json, subtask 3 (T4 is its own peer test, not part of any Standard grouping)

### Breakage and repair

**Broken JSON**:
A JSON string that is no longer parseable (or no longer parses to the intended object)
because a mutation was applied to a once-valid document.
_Avoid_: corrupted json, invalid json (use these only when distinguishing parse-failure from
content-mismatch is not needed)

**Breaker**:
The mechanism that transforms a valid JSON document into Broken JSON - e.g. deleting a
bracket, swapping a quote, truncating, swapping a comma for a brace. A T1/T2 case is
"break valid JSON + remember the pre-breakage object." The Breaker always records what it
broke (see T2: "track, don't tell").
_Avoid_: mutator, corruptor

**Original object**:
The Python object obtained by `json.loads(valid_json)` *before* the Breaker ran. This is the
ground truth that a repaired output must round-trip to for a Fidelity pass. Exists for T1
and T2; does not exist for T4.
_Avoid_: source, target, golden

**Edit script**:
The model's output format for a repair task. A sequence of find-and-replace operations
(each: `old` substring -> `new` substring) that, when applied to the Broken JSON, yield the
repaired JSON. The repaired JSON is then scored by the normal ladder. This is what makes
BracketBench test *surgical repair* rather than *regeneration* (see ADR-0005).
_Avoid_: diff, patch, edits (use "edit script" for the model's output; "patch" reserved for
the applied result)

### Scoring

Scoring has two layers: each test produces an independent 0-100 score, and scoreboards
aggregate the four via configurable weights.

#### Per-test scoring

**T1 and T2** use a single four-tier ladder. Each run yields one score in [0.0, 1.0],
scaled to 0-100. The tier scores are **configurable** (see ADR-0002) and **per-test**; the
MVP defaults are shown.

**Structural tier** (partial, default 0.5):
Awarded when the model's repaired output parses successfully via `json.loads`. The model got
*valid JSON*, even if it is not the right JSON.
_Avoid_: syntax score, parse score

**Fidelity tier - value match** (default 0.9):
Awarded when the model's repaired output parses to an object value-equal to the Original
object under lenient equality (Python `==` after `json.loads`): key order, whitespace, and
escapes ignored, and `1 == 1.0` counts as equal. The model got the right values but not
necessarily the right JSON types.
_Avoid_: lenient fidelity, loose match

**Fidelity tier - exact match** (default 1.0):
Awarded when the model's repaired output parses to an object equal to the Original object
under type-aware deep equality: value *and* JSON type must match (so `1.0` is NOT
exact-faithful to `1`). The model got *exactly* the right JSON.
_Avoid_: strict fidelity, type-strict match

**T3** uses notebook-semantic static checking (see ADR-0004), a distinct scoring path - not
the JSON ladder above. T3 is a weighted sum of **binary pass/fail checks** (weights
configurable, consistent with ADR-0002), scaled to 100:
1. **Parses as JSON** (prerequisite; if not, T3 = 0).
2. **Passes `nbformat.validate`** (canonical notebook schema validity).
3. **Cell-level integrity** - every cell has the required keys for its `cell_type`; `source`
   is a string or list of strings; `cell_type` is in {code, markdown, raw}; `outputs` only
   present on code cells and each output is well-formed.
4. **Semantic consistency** - `execution_count` present on code cells; output types valid
   (stream / execute_result / error); mime types well-formed.

**T4** uses only the Structural tier, with Structural as the ceiling (worth 100, not 0.5).

**Repair**:
A model output (an Edit script) that, once applied to the Broken JSON, produces output that
earns at least the Structural tier. A *faithful* repair earns either Fidelity tier.
_Avoid_: fix (too ambiguous; "fix" is the colloquial verb, "repair" is the graded outcome)

#### Scoreboards

**Scoreboard**:
A named weighting over the four test scores, producing one aggregate 0-100 number. The same
four test scores feed every scoreboard; different scoreboards apply different weights. A
weight of 0 excludes a test from that scoreboard.
_Avoid_: leaderboard (use "scoreboard" for the weighting; "leaderboard" for the ranked table
of models on a scoreboard)

**Unified scoreboard** (default):
Weighted arithmetic mean of all four tests, e.g. `0.4*T1 + 0.3*T2 + 0.2*T3 + 0.1*T4`. The
default headline number.

**Without-ipynb scoreboard**:
A scoreboard where T3's weight is 0 and the remaining weights are renormalized over T1, T2,
T4. Used to compare models on JSON repair alone, excluding notebook semantics.
