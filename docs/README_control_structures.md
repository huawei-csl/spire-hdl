# Control Structures

SpireHDL provides `if_`/`elif_`/`else_` and `switch_`/`case_`/`default` as Python
context managers, so conditional hardware reads like ordinary control flow. Any
signal assignment (`<<=`) inside one of these blocks is guarded by the active
condition and lowered to a mux. When no branch matches, a combinational signal
keeps its previous driver and a register holds its current value.

> **Note:** this is a convenience layer. Everything here can also be written
> directly with `mux` from SpireHDL's core ([`spirehdl/spirehdl.py`](../src/spirehdl/spirehdl.py));
> the context managers lower to exactly those muxes.

The constructs live in [`spirehdl/spirehdl_control_structures.py`](../src/spirehdl/spirehdl_control_structures.py).

```python
from spirehdl.spirehdl_control_structures import if_, elif_, else_, switch_, case_, default
```

## `if_` / `elif_` / `else_`

The branches form a priority chain — the first true condition wins. Give the
signal a default driver *before* the chain; a combinational signal that is only
assigned inside conditional blocks has no fallback and raises `RuntimeError`.

```python
from spirehdl.spirehdl import Bool, UInt
from spirehdl.spirehdl_module import Module

m = Module("Priority", with_clock=False, with_reset=False)
sel_a = m.input(Bool(), "sel_a")
sel_b = m.input(Bool(), "sel_b")
out = m.output(UInt(2), "out")

out <<= 0                       # default driver (required)
with if_(sel_a):
    out <<= 1
with elif_(sel_b):
    out <<= 2
with else_():
    out <<= 3

# (sel_a, sel_b) -> out :  (0,0)->3   (0,1)->2   (1,0)->1   (1,1)->1
```

## `switch_` / `case_` / `default`

`case_` accepts several values to share one body, and `default()` catches
everything else.

```python
from spirehdl.spirehdl import UInt
from spirehdl.spirehdl_module import Module

m = Module("Decode", with_clock=False, with_reset=False)
op = m.input(UInt(2), "op")
y = m.output(UInt(4), "y")

y <<= 0xF
with switch_(op):
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
from spirehdl.spirehdl import Bool, UInt
from spirehdl.spirehdl_module import Module
from spirehdl.spirehdl_simulator import Simulator

m = Module("EnReg", with_clock=True, with_reset=False)
en = m.input(Bool(), "en")
d = m.input(UInt(4), "d")
r = m.reg(UInt(4), "r")
r.set_init(0)

with if_(en):
    r <<= d                     # updates only when en == 1, holds otherwise

sim = Simulator(m)
sim.eval()
for enable, data in [(1, 5), (0, 9), (1, 7)]:
    sim.set("en", enable).set("d", data)
    sim.step()
    print(sim.get("r"))         # 5, then 5 (held), then 7
```

## Composing with state machines

These are the same context managers the [State Machines](README_state_machines.md)
API builds on, so they nest directly inside an FSM `switch_(state_reg)` body to
express per-state transition and output logic.

## Tests

See [`testing/basic/test_control_structures.py`](../testing/basic/test_control_structures.py)
for the full behavioural test suite (priority, grouped cases, nested switches,
register hold, and the missing-default-driver error).
