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
