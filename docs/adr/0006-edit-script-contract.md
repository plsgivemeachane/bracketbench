# Edit-script contract: wire format, composition, and match behaviour

Status: accepted

ADR-0005 established that the model's output for a repair task is an **edit script** — a
sequence of find-and-replace operations — and deferred the precise contract ("find/replace
semantics, how multiple edits compose, ambiguity handling") to be "defined separately." This
ADR resolves that deferred item so the edit-script applier and the scorers can be built
against a single, decision-complete contract.

## Decision

The edit script is a **JSON array** of objects, each `{"old": <string>, "new": <string>}`,
applied **in array order**, each operation against the result of the previous one. The two
ambiguity cases are resolved as follows:

- **Zero matches** of `old` in the current text: the operation is **skipped (a no-op)**, and
  application continues with the remaining operations. It is not an error.
- **Multiple matches** of `old` in the current text: the operation is **skipped as
  ambiguous**, and application continues with the remaining operations. It is not an error.

The repaired text is whatever the script leaves behind after the last operation; that text is
then scored by the normal tier ladder (Structural / Fidelity). A malformed script (not a JSON
array, or elements not of the required shape) maps to the **0.0 (unparseable) tier** — the
script is discarded and the broken text is scored as-is.

### Worked example (T1, deleted-closing-bracket)

Breaking `{"a": 1}` at `bracket_index=0` removes the final `}`, yielding broken text
`{"a": 1`. A correct, unambiguous repair script is:

```json
[{"old": ": 1", "new": ": 1}"}]
```

`": 1"` occurs exactly once in `{"a": 1`, so the operation applies and yields `{"a": 1}`.
By contrast `[{"old": "1", "new": "1}"}]` is **ambiguous**: `"1"` is the last character and
also has no other occurrences here, so it would apply — but in `{"a": 1, "b": 10}` `"1"`
occurs twice and the same script would be skipped. The contract's job is to make that
distinction deterministic, not to rescue imprecise scripts.

## Rationale

**Wire format — JSON array of `{old, new}` objects.** The model's only output channel is the
existing `generate(prompt) -> str` seam (ADR-0005, parent spec), so the script is a string.
JSON is the natural encoding: the benchmark is *about* JSON, standard string escaping already
covers arbitrary substrings (including newlines, quotes, and the broken text's own quirks),
and a malformed script falls straight into the 0.0 tier that the ladder already defines. The
`old`/`new` pair is the minimal, lossless statement of one find-and-replace operation.

**Composition — array order, sequential.** This is the only reading consistent with
ADR-0005's "a sequence of ... operations that, when applied ... yield the repaired JSON."
Sequential application is also what makes a script *reproducible and auditable*: replaying
the array against the broken text deterministically reproduces the repaired text.

**Zero matches — skip.** Each edit is an independent localized operation against the current
text. Skipping a non-matching edit lets the model emit a defensive script (e.g. a second
fallback edit that only fires for an alternative breakage shape) without the whole script
dying on one miss. The tier ladder already scores the final repaired text, so the applier
should not *also* punish a no-op edit — that would double-count. In T2, any breakage the
script fails to repair almost always leaves the text unparseable and scores 0.0 anyway, so
skip is not a loophole that inflates scores.

**Multiple matches — skip as ambiguous.** Surgical repair *requires* unambiguous
localization: the model must point at the breakage, not spray edits. Refusing to guess when
`old` is ambiguous measures precision directly and is fully auditable (the applier records
which operations were skipped and why). It also keeps the applier a pure, deterministic
function with no hidden "first match" heuristic that a model could game.

## Considered options

### Wire format

- **Custom delimiter (e.g. `old|||new` per line).** Rejected. There is no standard escaping,
  so a delimiter occurring inside the broken text or the edit is ambiguous or unsafe. JSON's
  escaping already solves this for free.
- **Provider tool-calling (native `find_replace` tool).** Rejected. The retained model seam
  is `generate(prompt) -> str` (ADR-0005, parent spec); supporting tool-calls would couple
  the benchmark to provider-specific calling conventions and break the single, provider-agnostic
  output channel. A tool-call is a *prompt-instruction* concern, not a contract concern — the
  wire format is still the JSON array the model is told to emit.
- **Regex `old`.** Rejected. The Breaker's damage is a deleted character, not a pattern;
  regex is error-prone for the model, harder to audit, and unnecessary. Plain substring match
  is sufficient and far less ambiguous to reason about.

### Composition

- **Parallel / order-independent application.** Rejected. Operations that are valid
  individually can conflict when composed (the `new` of one edit creates a substring another
  edit's `old` then matches). Array order is the only reproducible reading.

### Zero matches

- **Fail the whole script (treat as unparseable).** Rejected. An all-or-nothing cliff
  conflates "the model produced one imprecise edit" with "the model produced no useful edits,"
  collapsing the tier ladder's ability to distinguish partial from total failure.
- **Partial credit awarded by the applier.** Rejected. Awarding credit is the *scorer's* job
  (the tier ladder). The applier must be a pure `text + script -> text` transform; mixing
  credit logic into it breaks the seam separation the parent spec fixes.

### Multiple matches

- **First match.** Rejected. Silently picks one occurrence, hiding imprecision from both the
  score and the audit. A model could game it by emitting a short, common `old` that happens
  to hit the breakage on the first occurrence.
- **All matches.** Rejected. Applying the replacement everywhere amplifies imprecision (one
  vague edit rewrites the whole document) and is regeneration-adjacent — the opposite of
  surgical repair.

## Consequences

- **The applier is a pure, deterministic function:**
  `apply(broken_text: str, edit_script: str) -> str`. Parsing failures and non-array inputs
  yield the broken text unchanged (mapping to the 0.0 tier downstream); per-operation
  skips yield the text as it stood at that step. The applier records (for auditing) which
  operations were skipped and the reason (`zero_match` / `multiple_match`), but this does
  not affect the returned text.
- **Malformed-script handling is delegated to the scorer, not duplicated in the applier.**
  The applier returns the broken text unchanged on a parse failure; the scorer then observes
  unparseable output and awards 0.0. This keeps the applier single-responsibility.
- **Whole-text replacement is a deliberate, auditable escape hatch.** A script of the form
  `[{"old": "<entire broken text>", "new": "<entire repaired text>"}]` *is* permitted by the
  letter of the contract: `old` matches exactly once, so the operation applies and yields the
  `new` text wholesale. This is regeneration wearing an edit-script costume, and it is the one
  way the contract can be gamed back toward regeneration. It is left open **on purpose**:
  closing it with a heuristic (e.g. "reject edits where `new` is longer than `old`") would
  inject subjective thresholds that are poison for a deterministic benchmark, and any such
  rule has legitimate counter-examples (a repair that *adds* a missing bracket is longer than
  the broken text). Instead, the mitigation is **transparency**: the model's raw edit-script
  output is retained on the result (parent spec user stories 28–29), so whole-text-replacement
  scripts are visible in every audit. A future ticket may add a *reportable* "regeneration
  suspicion" flag (e.g. single edit whose `old` length is the entire broken text) without
  changing the score — but that is out of scope for this contract.
- **T2 multi-breakage works naturally.** A correct T2 repair is a multi-element array, one
  edit per breakage, ordered so each `old` is unambiguous at its step. The "track, don't
  tell" property is unaffected: the model still is not told the count; it emits as many
  edits as it believes are needed, and unmatched/ambiguous edits are simply skipped.
- **Prompt-level instruction is the contract's only delivery surface, but is a separate
  concern.** This ADR fixes the *contract* the model is asked to satisfy. The exact prompt
  wording that tells the model to emit `[{old,new}]`, skip-on-ambiguity, etc., is deferred to
  the per-test prompt tickets — they build *on* this contract, they do not re-decide it.
- **This respects ADR-0005 and does not regress "surgical repair, not regeneration."** The
  model's only output is an edit script; it cannot emit repaired JSON except by producing an
  `old` that is the entire broken text (auditable, above). Substring-replacement with
  skip-on-ambiguity forces localization: a vague edit that matches many places is refused,
  not silently applied.
