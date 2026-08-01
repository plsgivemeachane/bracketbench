# Model output is an edit script, not a repaired JSON string

The model's output for a repair task is **not** a JSON string. It is an **edit script**: a
sequence of find-and-replace operations (each: `old` substring -> `new` substring) that, when
applied to the Broken JSON, *yield* the repaired JSON. The repaired JSON is then scored by
the normal ladder (Structural / Fidelity).

This is BracketBench's core differentiator. "Just output good JSON" tests *regeneration* —
the model throws the broken input away and produces valid JSON from scratch. That is an
easier, already-studied task. BracketBench tests *surgical repair*: the model must locate
the breakage and produce the minimal edit that fixes it. An edit-script output format forces
this — a model cannot regenerate, it must produce diffs against the given broken text.

Consequence: the prompt instructs the model to emit edit-tool calls, not JSON. The scoring
engine applies the edit script to the broken JSON before parsing/comparing. The edit-tool
contract (find/replace semantics, how multiple edits compose, ambiguity handling) is defined
separately and is a deferred grilling item.

This is hard to reverse: once results exist, changing the output format invalidates them, and
the benchmark's claim to measure "surgical repair" rather than "regeneration" depends on it.
