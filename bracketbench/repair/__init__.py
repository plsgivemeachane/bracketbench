"""The JSON-repair product surface (ADR-0001).

This package holds the focused JSON-repair benchmark components: the edit-script applier
(ADR-0006), the T1/T2 4-tier scorer (ADR-0002), the scoreboard calculator (ADR-0003), the T1
and T2 prompt + case builders, and the top-level ``evaluate`` entry point. It is
intentionally separate from the legacy generic ``benchmarking`` scaffolding flagged for
stripping.
"""
