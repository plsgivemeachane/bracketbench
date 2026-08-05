"""The JSON-repair product surface (ADR-0001).

This package holds the focused JSON-repair benchmark components: the edit-script applier
(ADR-0006), the T1/T2 4-tier scorer (ADR-0002), the T3 notebook-semantic static
scorer (ADR-0004), the T4 structural-only scorer, the scoreboard calculator
(ADR-0003), the T1, T2, T3, and T4 prompt + case builders, the curated T3 case set,
and the top-level ``evaluate`` entry point. It is intentionally separate from the
legacy generic ``benchmarking`` scaffolding flagged for stripping.
"""
