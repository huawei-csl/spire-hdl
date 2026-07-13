# Differential conformance harness

Checks that spire's three backends agree on combinational values, per `docs/README_semantics.md` (§2 table, §5
testing rules): the **Python simulator** (the specification), **IEEE-1364 evaluation of the emitted Verilog text**
(via the built-in reference evaluator — no external tools needed), and the **AIGER export** re-imported and
re-simulated. All comparisons are on masked bit patterns at each output's declared width.

Run with: `PYTHONPATH=src python -m pytest testing/differential` (a few seconds, deterministic — fixed seeds).

| File | Contents |
|---|---|
| `ieee_eval.py` | Mini IEEE-1364 expression evaluator + `VlogModule` (parses spire's emitted combinational text). Validated per-vector against real yosys elaboration when written; re-pinned every run by the leaf battery. |
| `harness.py` | Comparators (`diff_sim_vs_verilog`, `diff_sim_vs_aiger`), netlist/vector helpers, seeded random circuit generator. |
| `test_leaf_conformance.py` | Exhaustive single-operator battery with leaf operands and result-typed outputs. Doubles as the evaluator's calibration gate: on conformant shapes the evaluator must match the simulator on every vector. |
| `test_nested_shapes.py` | Compound subexpressions in every context that IEEE sizing rules could re-size (operand of a wider op, compare island, ternary branch, extension concat, shifts, flat emission), plus matching control shapes. |
| `test_fuzz.py` | Seeded random circuits compared across all three backends, in unsigned-only and mixed-signedness profiles. |

**xfail discipline.** Known divergences are marked `xfail(strict=True)` with a reason naming the issue-tracker entry
that owns them. Strict means a premature pass fails the suite: whoever fixes the emitter/exporter must remove the
mark in the same change, and nothing can silently start or stop diverging. Unmarked tests are conformance guarantees
— including a few shapes that are conformant only by the accident of current emission (commented inline); keeping
them green is part of any fix's job.

**Writing new shapes — the sharing pitfall.** Spire wraps a subexpression into a named wire once it is referenced
more than once (a second operand use, or an output drive), and a named wire is a width boundary that hides exactly
the inline-nesting behavior this harness exists to test. So: build each compound in a single expression, drive
outputs with `output_like` (output typed exactly like the node, so the assignment itself adds no resizing), and give
interior chain nodes exactly one consumer — the fuzz generator's chain bias and sink-only outputs exist for this
reason.
