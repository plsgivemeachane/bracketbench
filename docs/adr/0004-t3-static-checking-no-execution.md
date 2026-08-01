# T3 (Complex ipynb) is scored by static semantic checking, no code execution

The Complex ipynb test (T3) is scored by notebook-semantic static checks, not by executing
the notebook. Checks detect corrupt cell `source`, mangled `outputs`, cell-type
inconsistency, and similar notebook-level problems. No Jupyter kernel is ever run.

We chose static checking over execution for three reasons:
1. Execution is non-deterministic, slow, and environment-dependent - poison for a
   reproducible benchmark.
2. Execution turns the benchmark into "can AI fix notebooks" (a different product) rather
   than "can AI repair JSON-shaped documents" (the thesis).
3. Static checks are fast, deterministic, and directly probe the JSON-structure-vs-notebook-
   semantics tension that makes T3 the hard case.

Consequence: T3 has its own scoring path, distinct from the T1/T2 4-tier JSON ladder. The
specific checks and their scoring weights are defined separately (see CONTEXT.md -> Scoring
-> T3, to be detailed). This is a deliberate deviation from "one scoring engine for all
tests" - T3's checks are notebook-aware, not JSON-aware.
