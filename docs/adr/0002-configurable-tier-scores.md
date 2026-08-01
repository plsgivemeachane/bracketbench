# Tier scores are configurable, not hardcoded

The four scoring tiers (0.0 unparseable, structural-partial, 0.9 value-faithful, 1.0
exact-faithful) are not baked into the scoring engine as constants. They are supplied via a
configuration object so they can be tuned per-run without code changes.

We chose configurability over simple constants because the tier values are exactly the kind
of parameter a benchmark operator will want to sweep (e.g. "is the thesis still true if
structural-only is worth 0.2 instead of 0.5?"). Hardcoding them would force a code edit for
every experiment. The MVP ships with defaults of 0.0 / 0.5 / 0.9 / 1.0.

Consequence: the scoring engine takes a config object at construction rather than reading
module-level constants. Test fixtures that assert specific scores must construct the engine
with explicit config (or accept the defaults) rather than relying on magic numbers.
