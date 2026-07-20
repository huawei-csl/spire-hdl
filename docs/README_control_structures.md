# Control Structures

Spire provides `if_`/`elif_`/`else_` and `switch_`/`case_`/`default` as Python
context managers, so conditional hardware reads like ordinary control flow. Any
signal assignment (`<<=`) inside one of these blocks is guarded by the active
condition and lowered to a mux. When no branch matches, a combinational signal
keeps its previous driver and a register holds its current value.

> **Note:** this is a convenience layer. Everything here can also be written
> directly with `mux` from Spire's core ([`spire/expr.py`](../src/spire/expr.py));
> the context managers lower to exactly those muxes.

The constructs live in [`spire/control_structures.py`](../src/spire/control_structures.py).

```python
from spire.control_structures import if_, elif_, else_, switch_, case_, default
```

## `if_` / `elif_` / `else_`

The branches form a priority chain — the first true condition wins. Give the
signal a default driver *before* the chain; a combinational signal that is only
assigned inside conditional blocks has no fallback and raises `RuntimeError`.

```python
from spire import Component, IORecord, Input, Output, Bool, UInt

class Priority(Component):
    def __init__(self):
        self.io = IORecord(sel_a=Input(Bool()), sel_b=Input(Bool()), out=Output(UInt(2)))
        self.elaborate()

    def elaborate(self):
        out = self.io.out
        out <<= 0                       # default driver (required)
        with if_(self.io.sel_a):
            out <<= 1
        with elif_(self.io.sel_b):
            out <<= 2
        with else_():
            out <<= 3

# (sel_a, sel_b) -> out :  (0,0)->3   (0,1)->2   (1,0)->1   (1,1)->1
```

## `switch_` / `case_` / `default`

`case_` accepts several values to share one body, and `default()` catches
everything else.

```python
from spire import Component, IORecord, Input, Output, UInt

class Decode(Component):
    def __init__(self):
        self.io = IORecord(op=Input(UInt(2)), y=Output(UInt(4)))
        self.elaborate()

    def elaborate(self):
        y = self.io.y
        y <<= 0xF
        with switch_(self.io.op):
            with case_(0):
                y <<= 1
            with case_(1, 2):           # one body for several values
                y <<= 2
            with default():
                y <<= 3

# op -> y :  0->1   1->2   2->2   3->3
```

Switches nest: put a `switch_` (or an `if_` chain) inside a `case_` to build
multi-level decode logic.

## Registers

The same blocks guard register next-state assignments. A register written only
under a condition holds its value when the condition is false — i.e. a clock
enable.

```python
from spire import Component, IORecord, Input, Output, Bool, UInt, Simulator
from spire.expr import Register

class EnReg(Component):
    def __init__(self):
        self.io = IORecord(en=Input(Bool()), d=Input(UInt(4)), q=Output(UInt(4)))
        self.elaborate()

    def elaborate(self):
        r = Register(UInt(4), init=0, name="r")
        with if_(self.io.en):
            r <<= self.io.d             # updates only when en == 1, holds otherwise
        self.io.q <<= r

sim = Simulator(EnReg())
sim.eval()
for enable, data in [(1, 5), (0, 9), (1, 7)]:
    sim.set("en", enable).set("d", data)
    sim.step()
    print(sim.get("q"))         # 5, then 5 (held), then 7
```

## Composing with state machines

These are the same context managers the [State Machines](README_state_machines.md)
API builds on, so they nest directly inside an FSM `switch_(state_reg)` body to
express per-state transition and output logic.

## Tests

See [`testing/basic/test_control_structures.py`](../testing/basic/test_control_structures.py)
for the full behavioural test suite (priority, grouped cases, nested switches,
register hold, and the missing-default-driver error).

## Emission modes (`selection_topology`)

`switch_`, `if_`/`elif_`, and hand-nested `mux()` calls all lower to linear mux
cascades, which synthesis cannot rebalance (O(N) logic depth). `selection_topology`
sets a log-depth emission style for a scope — one object, two roles:

```python
from spire import selection_topology

# Region: applies to every selection cascade finalized inside — switch_,
# if_/elif_ chains, and hand-built chains alike. Rewrites eagerly on exit.
with selection_topology("onehot"):
    with switch_(op):
        with case_(A, B): y <<= ...

with selection_topology("tournament"):
    with if_(c0):   y <<= 1
    with elif_(c1): y <<= 2

with selection_topology("tournament"):
    y <<= hand_built_mux_chain

# Function: the same object as a decorator — rewrites the returned cascade.
@selection_topology("tournament")
def ff1_index(bits):
    chain = Const(0, UInt(5))
    for k in reversed(range(32)):
        chain = mux(bits[k], Const(k, UInt(5)), chain)
    return chain
```

Both roles are **eager**: the rewrite is baked into the expression graph, so
every backend (Verilog, AIGER export, Simulator, analyze) sees the same
structure. Signals without a cascade are skipped; an if_/elif_ chain may not
straddle a region boundary (fails loudly). For whole-design automatic
treatment without annotations, `to_verilog_file(..., selection_emission=True)`
auto-detects cascades above size thresholds
(`spire.selection_emission.SelectionEmissionConfig`).

### Worked example: register-file / RAM read mux

A 32-entry read port written as the natural loop. The loop builds a 31-deep
priority chain — O(N) logic depth if emitted as-is; since the `rd_ptr == i`
selects are provably one-hot, the region rewrites it into the flat one-hot
AND-OR network (what a Verilog `case` lowers to via `$pmux`), O(log N) deep:

```python
# cells[rd_ptr]: priority chain like the RTL's RAM read; emitted as
# one-hot AND-OR (the $pmux form) via the emission region.
with selection_topology("onehot"):
    rd = cells[31]                       # chain tail = the last entry:
    for i in range(30, -1, -1):          # reached exactly when rd_ptr == 31,
        rd = mux(rd_ptr == Const(i, UInt(5)), cells[i], rd)
    rd_q <<= rd                          # captured by the region, rewritten on exit
```

The tail `cells[31]` becomes the network's fallback term
(`~any_other_match & cells[31]`), so coverage stays exact. The same loop under
`selection_topology("bittree")` would instead index a mux tree directly with
`rd_ptr`'s five bits (no comparators at all) — the classic RAM-mux structure;
with 32 dense labels either form is legal, and `"auto"` picks for you.

### All modes and when they apply

Validation is **shape-based**: the analyzer judges each cascade's select
expressions, never the construct it came from.

| mode | requires | what is built / depth |
|---|---|---|
| `"chain"` | nothing | the plain serial mux cascade (default lowering). Depth O(N). |
| `"tournament"` | nothing — priority (first match wins) preserved by construction | balanced first-match tree: node `(sl \| sr, mux(sl, vl, vr))`. Depth O(log N), area ≈ the chain's. |
| `"onehot"` | **provably disjoint** arm selects: `sel == const` terms (or ORs of them) on one selector, pairwise-distinct constants | one-hot network: values AND-masked by their selects, OR-reduced in a balanced tree (the parallel `$pmux` form), plus a fallback term for the unmatched space. Cheapest log-depth form when legal. |
| `"bittree"` | `"onehot"`'s requirements; selector width capped by `bittree_max_sel_bits`. Missing labels are legal — absent leaves fill from the fallback (the `default()` value or the signal's prior driver) | mux tree indexed by the **selector bits** — arm compares vanish (no decoders). Depth K muxes. Niche: dense selectors; prefer `"onehot"` for sparse labels. |
| `"auto"` | nothing (never raises) | best legal form per cascade, subject to config thresholds; small cascades stay chains. |

Notes on disjointness:

* Redundant first-match gating (`cond & ~covered`) is **seen through** when
  provably dead — so eq-const `if_`/`elif_` chains qualify for `"onehot"` just
  like switches: the classifier evaluates `&`/`|`/`~` set-theoretically over
  the labels of one selector.
* Conditions whose exclusivity is real but not structurally provable
  (`a < 10` / `a >= 10`, one-hot state flags) are rejected for the one-hot
  modes — trusting an assertion would recreate Verilog's `parallel_case` bug
  class (silent OR-garbage on overlap). Use `"tournament"`: same O(log N)
  depth, no proof obligations.
* Genuinely overlapping arms: priority is semantics — only `"chain"` and
  `"tournament"` apply; `"auto"` degrades to them.
* Small cascades are deliberately left alone by `"auto"` — below ~8–16 arms
  the plain chain synthesizes as well or better.

Errors are raised at region exit, when the finished cascade is judged —
the message names the offending signal. The constructs themselves know
nothing about emission modes: `switch_`/`if_` build priority-correct
selections; `selection_topology` chooses (and validates) their physical topology
afterwards.

Independent of `selection_topology`, arm conditions that are **provably disjoint**
no longer emit `& ~covered` priority gating (it is provably redundant). This
applies uniformly to `switch_` cases with distinct constant labels *and* to
`if_`/`elif_` chains whose conditions are `sel == const` compares on one
selector — both constructs share one incremental disjointness tracker.
Overlapping, colliding, or non-classifiable conditions keep exact
first-match-wins semantics as before (only the offending arm and, for
non-classifiable conditions, later arms are gated).
