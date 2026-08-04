"""Benchmarking utilities for BracketBench.

Holds the synthetic JSON :mod:`~bracketbench.benchmarking.generator` (Ticket B / issue
#10) -- a seeded, parameterized emitter of valid JSON that feeds the Breaker. The legacy
generic ``Benchmarking`` / ``BenchmarkRunner`` / ``TestCase`` / ``ScoringEngine`` /
``ResultsHandler`` scaffolding has been stripped per ADR-0001; the focused product's runner
lives in :mod:`bracketbench.repair.evaluate`.
"""

from .generator import GenerateError, generate

__all__ = ["generate", "GenerateError"]
