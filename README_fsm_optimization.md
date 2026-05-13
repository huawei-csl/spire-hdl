# FSM and State-Encoding Optimization

Two opt-in context managers in
[`spirehdl/fsm/`](src/spirehdl/fsm/) add automatic optimization passes on
top of the basic [`State`](README_state_machines.md) API:

| Wrapper | What it does | When to use it |
|---|---|---|
| `optimized_fsm(reg, module, ...)` | Hopcroft DFA state minimization. Merges behaviourally-equivalent states by mutating the State Consts in place, then runs the peephole simplifier. | When your FSM has redundant states (e.g. paper-style "easy to write" form). |
| `optimized_encoding(state_cls, module, ...)` | Searches bit-assignments for a `State` subclass under an in-process synthesis metric (cells / wires / transistors via pyosys, or aig_gates / aig_depth via aigverse). | Whenever a `State` subclass appears in the design — FSM next-state, ALU opcode dispatch, decoder lookup, register-file tag, etc. |

Either wrapper is usable in isolation; **nest them** when both passes are
wanted. Critically, the user's FSM body inside the wrapper is *byte-identical*
to the un-optimised version — same `switch_/case_/if_/else_` you already
write. The optimization happens on `__exit__`.

```python
from spirehdl.spirehdl_state import (
    State, Encoding, state,
    optimized_fsm, optimized_encoding,
)
```

## Quick reference

```python
# Hopcroft minimisation only.
with optimized_fsm(reg, module=m, minimize=True, outputs=[out]):
    with switch_(reg):
        ...

# Bit-assignment search only.
with optimized_encoding(MyStates, module=m, objective="cells"):
    ...

# Both, composed via `with` nesting (inner runs first).
with optimized_encoding(MyStates, module=m):
    with optimized_fsm(reg, module=m, minimize=True):
        with switch_(reg):
            ...
```

---

## `optimized_fsm` — Hopcroft state minimisation

`optimized_fsm` extracts the FSM transition table from `reg._driver`, runs
Hopcroft to find behavioural-equivalence classes, mutates the State Consts so
every state in a class shares its representative's value, and runs
`apply_simplify(module)` to collapse the now-redundant mux branches.

### Worked example — case10 (the canonical 7→4 case)

```python
from spirehdl.spirehdl_state import State, Encoding, state, optimized_fsm
from spirehdl.spirehdl import Bool, UInt
from spirehdl.spirehdl_module import Module
from spirehdl.spirehdl_control_structures import case_, default, if_, else_, switch_

class S(State, encoding=Encoding.BINARY):
    S0 = state(); S1 = state(); S2 = state()
    S3 = state(); S4 = state(); S5 = state(); S6 = state()

m = Module("example", with_clock=True, with_reset=False)
x   = m.input(Bool(), "x")
out = m.output(UInt(1), "out")
reg = m.reg(S.typ, "state_reg", init=S.S0)

with optimized_fsm(reg, module=m, minimize=True, outputs=[out]):
    out <<= 0
    with switch_(reg):
        with case_(S.S0):
            out <<= 1
            with if_(x): reg <<= S.S2
            with else_(): reg <<= S.S1
        # … six more cases (S1..S6)
        with default():
            reg <<= S.S0
```

After the `with` exits, the State Consts are merged:

```python
S._values  # → {'S0': 0, 'S1': 1, 'S2': 2, 'S3': 0, 'S4': 2, 'S5': 5, 'S6': 2}
# Equivalence classes: {S0, S3}, {S1}, {S2, S4, S6}, {S5}
```

The user's emitted Verilog is byte-equivalent in shape; only the Const literals
inside the case selectors and the next-state branches change. `apply_simplify`
collapses the now-redundant duplicates so synthesis sees a tight 4-state FSM.

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `reg` | — | The FSM state register (an `m.reg(...)` instance). |
| `module` | — | The `Module` containing `reg`. Required so `apply_simplify` can run after rewriting. |
| `minimize` | `True` | Master switch. When `False`, the wrapper is a no-op marker. |
| `outputs` | `()` | Moore outputs whose drivers participate in the equivalence-class signature (initial Hopcroft partition keys on outputs). |
| `state_cls` | auto-inferred | Override the State subclass driving `reg`. Inference walks `reg._driver` for the first tagged Const. |

### Safety properties

- **Idempotent on already-minimal FSMs.** If no states merge, the wrapper exits without touching anything.
- **Per-name preservation.** Merged states keep their names (e.g. `S.S3` still exists), they just share the canonical's value. References to `S.S3` elsewhere in the design keep working.
- **Failure-safe.** When the input domain exceeds 65 536 combinations (`MAX_INPUT_COMBINATIONS`), `extract_transition_table` raises and the wrapper silently skips minimisation rather than producing a wrong result.

---

## `optimized_encoding` — bit-assignment search

`optimized_encoding` searches across permutations of bit-codes for the
states of a `State` subclass and picks the assignment that minimises a
chosen synthesis metric. Works for **any** use of a `State` subclass —
FSM next-state, ALU opcode dispatch, decoder lookup, register-file tag —
not just FSMs.

### Worked example — ALU opcode dispatch (no FSM)

```python
from spirehdl.spirehdl_state import State, Encoding, state, optimized_encoding
from spirehdl.spirehdl import UInt, mux
from spirehdl.spirehdl_module import Module

class Op(State, encoding=Encoding.BINARY):
    ADD = state(); SUB = state(); AND = state(); OR = state(); XOR = state()

m = Module("alu", with_clock=False, with_reset=False)
op = m.input(Op.typ, "op")
a  = m.input(UInt(8), "a")
b  = m.input(UInt(8), "b")
y  = m.output(UInt(8), "y")

with optimized_encoding(Op, module=m, objective="cells", search="auto"):
    y <<= mux(op == Op.ADD, a + b,
          mux(op == Op.SUB, a - b,
          mux(op == Op.AND, a & b,
          mux(op == Op.OR,  a | b,
                            a ^ b))))
```

On exit the wrapper enumerates / heuristically searches bit-assignments
for the 5 opcodes, synthesises each via Yosys, and commits the lowest-cell
encoding. The `Op` class is left in the chosen encoding for the rest of
the design (and any subsequent uses).

### Strategy ladder

| Strategy | When `"auto"` picks it | Behaviour |
|----------|------------------------|-----------|
| `predefined` | `n ≤ 2` | Tries `BINARY` and `GRAY` codes (no widening). 2 cost-fn calls. |
| `exhaustive` | `n! ≤ 5040` (so `n ≤ 7`) | All permutations of `n` codes from the universe of `2^width`. Up to ~5 040 cost-fn calls. |
| `swap` | otherwise | Pair-swap accept-on-improvement, 4 random restarts × 200 iters. ~800 cost-fn calls. |
| `anneal` | never (future work) | Reserved name; currently raises `NotImplementedError`. |

Override the choice explicitly:

```python
with optimized_encoding(MyStates, module=m, search="exhaustive"):
    ...

with optimized_encoding(MyStates, module=m, search="swap"):
    ...
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `state_cls` | — | The State subclass to re-encode. |
| `module` | — | The `Module` whose Verilog gets synthesised for each candidate assignment. |
| `objective` | `"cells"` | Synthesis metric to minimise. One of `cells`, `wires`, `transistors` (via in-process pyosys), `aig_gates`, `aig_depth` (via aigverse). |
| `search` | `"auto"` | Strategy (see ladder above). |
| `width` | `state_cls._width` | Width of the encoding. Width-changing search (widen for `ONEHOT`, etc.) is **future work** — must equal the current width. |
| `cost_fn` | `None` | Custom callable `assignment → float`. When `None`, defaults to `make_yosys_cost_fn`. |

### Custom cost functions

If you want a non-Yosys metric (Sky130 ADP, ABC depth, transistor count
via a different tool, total power, …), pass a `cost_fn`:

```python
def adp_cost(assignment: dict[str, int]) -> float:
    # mutate the State Consts to `assignment`, synthesise via OpenROAD,
    # read back area × delay, restore — return inf on failure.
    ...

with optimized_encoding(Op, module=m, cost_fn=adp_cost, search="auto"):
    ...
```

Contract: `cost_fn(assignment)` receives a dict mapping every state name
in `state_cls` to an integer code, returns a float (lower = better),
and must restore the State class to its prior encoding before returning
(otherwise the search will read stale values on the next iteration).
`make_yosys_cost_fn` already does this restoration for you.

---

## Composing the two wrappers

Nest them when you want both passes. Inner `__exit__` runs first, so
Hopcroft minimises the state set before the encoding search runs over the
surviving equivalence classes:

```python
with optimized_encoding(S, module=m, objective="cells"):
    with optimized_fsm(reg, module=m, minimize=True, outputs=[out]):
        with switch_(reg):
            ...
```

Order of operations:

1. Inner `optimized_fsm.__exit__`:
   - Extracts the transition table from `reg._driver`.
   - Runs Hopcroft → finds equivalence classes.
   - Mutates State Consts (every state in a class shares its representative's value).
   - Runs `apply_simplify(module)`.
2. Outer `optimized_encoding.__exit__`:
   - Detects the equivalence groups in `state_cls._values` (states that now share a value are in the same group).
   - Runs a **group-aware** encoding search — assigns one code per group, all names in a group get that code.
   - Mutates State Consts to the winning encoding.

Without nesting (just `optimized_encoding` alone), the search treats every
state name independently, which is the right behaviour when no merging
has happened.

---

## Under the hood (in one screen)

**Sentinel-based detection.** When you declare a `State` subclass, every
`Const` produced by `state()` is tagged with `_state_class = cls` and
`_state_name = name` in `State.__init_subclass__`. This makes detection
unambiguous — `getattr(c, "_state_class", None) is StateCls` — and
distinct from `(value, width)` matching which would collide with mask
constants, register inits, etc.

**Capture via `_SharedCache`.** Both wrappers snapshot
`len(_SharedCache.wires)` on `__enter__`. Every non-trivial Expr you build
inside the `with` block goes through `_maybe_share` and lands in the
cache. On `__exit__`, the diff `_SharedCache.wires[start_idx:]` plus the
wrapper's explicit-arg roots (the register / the State class itself)
covers everything we need to walk for State Const occurrences. **Zero new
DSL hooks.**

**Symbolic evaluation for transition extraction.** `_evaluator.eval_with`
is an `ExprVisitor[int]` that concretely evaluates an Expr DAG under a
signal-binding environment. Used by `extract_transition_table` to
enumerate `(state_value × input_combination) → (next_state, output)`
tuples. The operator table mirrors `spirehdl_simplify._fold_op2` so
symbolic eval never diverges from the peephole simplifier.

**In-place State Const mutation.** `apply_encoding(state_cls, assignment)`
mutates each `state_cls.NAME.value` directly. Because every `cls.NAME` is
a single shared `Const` object referenced everywhere in the Expr DAG,
the change propagates instantly — no DAG rewrite needed.

**Group-aware composition.** When `optimized_encoding` runs after
`optimized_fsm` (nested-with), it detects equivalence groups from the
current `state_cls._values` (states sharing a value) and constrains the
search so all names in a group keep the same code. Avoids re-spreading
states that Hopcroft just merged.

---

## Limitations / scope

- **Same-width re-encoding only** for now. Switching a 2-bit BINARY class
  to ONEHOT (which needs more bits) raises `NotImplementedError`. Workaround:
  declare the class with `encoding=Encoding.ONEHOT` from the start.
- **`anneal` strategy is reserved but unimplemented**; use `swap` for
  larger search spaces.
- **Input domain cap** on FSM transition-table extraction:
  `MAX_INPUT_COMBINATIONS = 65 536`. When exceeded, minimization is
  silently skipped (encoding search still runs).
- **Single-module scope.** `optimized_encoding(state_cls, module=m)`
  optimises one Module at a time. Multi-Module designs that share a
  State class need one wrapper per Module.
- **Mealy guards over wide data inputs** are enumerated by exhaustive
  product. Narrow inputs are fine; for wide combined input domains,
  consider partitioning the FSM.

## Tests

- [`testing/fsm/`](testing/fsm/) — unit tests per module plus three e2e
  tests:
  - `test_optimized_fsm.py` — case10 (7→4 classes), already-minimal FSM,
    `minimize=False` no-op, post-wrapper simulation parity.
  - `test_optimized_encoding.py` — ALU-style dispatch under a synthetic
    cost; also verifies that an empty `with` block is a no-op.
  - `test_nested_wrappers.py` — case10 with both wrappers nested,
    synthetic-cost and real-synth variants (both run unconditionally —
    the cost oracle uses in-process pyosys + aigverse).

Run them (from the spire-hdl repo root) with:

```bash
PYTHONPATH=src python -m pytest testing/fsm/ -q
```

## See also

- [`README_state_machines.md`](README_state_machines.md) — basic `State` /
  `Encoding` / `state()` API (this doc builds on it).
- [`README_optimization_decorators.md`](README_optimization_decorators.md) —
  the sibling `@abc_optimized` / `@flowy_optimized` decorators (these
  operate at AIG level; the FSM wrappers operate at expression-DAG level,
  so they compose cleanly with each other).
