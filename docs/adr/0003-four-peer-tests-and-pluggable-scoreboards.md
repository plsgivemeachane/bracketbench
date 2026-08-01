# Four peer tests, not a Standard/Complex grouping

BracketBench has four peer tests, each scored 0-100 independently:

- T1 single-breakage JSON repair
- T2 multi-breakage JSON repair (track, don't tell)
- T3 Complex ipynb repair (notebook-semantic static checking)
- T4 real-world messy JSON (Structural-only)

There is no "Standard" container and no subtask grouping. The original vision ("2 things:
Standard + Complex") and an interim design ("Standard = 3 subtasks at 50/40/10") were both
retired in favor of four flat peers. The reason: the four tests have genuinely different
scoring paths (T1/T2 share the 4-tier JSON ladder; T3 is notebook-semantic; T4 is
Structural-only), so grouping T1+T2+T4 under "Standard" would imply a shared scoring model
they don't have. Flat peers keep each test's scoring honest.

Real-world messy JSON (T4) was promoted out of "Standard" into its own peer test because it
uniquely lacks an Original object and therefore can only ever reach the Structural tier -
making it structurally distinct from T1/T2, not a subtask of them.
