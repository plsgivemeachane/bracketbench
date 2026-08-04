"""BracketBench - a focused JSON-repair benchmark.

BracketBench tests the thesis that "AI sucks at fixing messy JSON." The product surface is
the JSON-repair path in :mod:`bracketbench.repair` (the edit-script applier, the T1/T2
4-tier scorer, the scoreboard, and the ``evaluate`` entry point) plus the
:mod:`bracketbench.breaker` and the synthetic :mod:`bracketbench.benchmarking.generator`.

The legacy generic LLM-harness scaffolding (multi-provider model registry, generic
TestCase, difflib similarity scoring) has been stripped per ADR-0001.
"""

__version__ = "0.1.0"
__author__ = "BracketBench Team"
__email__ = "contact@bracketbench.org"
