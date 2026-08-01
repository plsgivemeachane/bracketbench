# BracketBench is a focused JSON-repair benchmark, not a generic LLM harness

The existing codebase was scaffolded as a generic LLM benchmarking framework (generic
`TestCase`, difflib text-similarity scoring, a multi-provider model registry). We are
explicitly narrowing the product: BracketBench exists to test the thesis that "AI sucks at
fixing messy JSON." The two test types — Standard (broken valid JSON) and Complex (ipynb
files) — are the entire product, not one pluggable suite among many.

We chose focus over flexibility. A generic harness is a dime-a-dozen and the generic
scaffolding here was already broken (CLI imported a non-existent `BenchmarkRunner` class).
The JSON-repair thesis is what differentiates BracketBench, so the architecture should be
JSON-repair-specific: JSON breakers, repair prompts, and JSON-aware scoring. The generic
`TestCase` / similarity-scoring / model-registry code is to be stripped or repurposed rather
than extended.
