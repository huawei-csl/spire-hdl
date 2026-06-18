# State Machines

SpireHDL ships with a small `State` base class plus an `Encoding` enum
(`BINARY` / `ONEHOT` / `GRAY`) for declaring finite-state-machine state sets
in a way that's IDE-friendly, type-checked, and composes with the existing
`switch_`/`case_`/`if_` control-flow context managers.

The pieces live in [`spire/state.py`](../src/spire/state.py).

## Quick start

Declare a state set as a class:

```python
from spire.state import State, Encoding, state

class TrafficFSM(State, encoding=Encoding.BINARY):
    RED    = state()
    GREEN  = state()
    YELLOW = state()
```

The class attributes (`TrafficFSM.RED`, etc.) become `Const` values you can
use anywhere an expression is expected:

```python
TrafficFSM.RED        # → Const(0, UInt(2))
TrafficFSM.GREEN      # → Const(1, UInt(2))
TrafficFSM.YELLOW     # → Const(2, UInt(2))
TrafficFSM.typ        # → UInt(2)   (matches encoding width)
TrafficFSM.names      # → ['RED', 'GREEN', 'YELLOW']
```

Build the machine by allocating a register of `TrafficFSM.typ`, then write
transitions with the existing `switch_`/`case_`/`if_` blocks:

```python
from spire.expr import Bool, UInt
from spire.component import Module
from spire.control_structures import case_, default, if_, switch_

m = Module("traffic", with_clock=True, with_reset=False)
go    = m.input(Bool(), "go")
light = m.output(UInt(2), "light")
state_reg = m.reg(TrafficFSM.typ, "state", init=TrafficFSM.RED)

light <<= state_reg   # output = current state (Moore)

with switch_(state_reg):
    with case_(TrafficFSM.RED):
        with if_(go):
            state_reg <<= TrafficFSM.GREEN
    with case_(TrafficFSM.GREEN):
        state_reg <<= TrafficFSM.YELLOW
    with case_(TrafficFSM.YELLOW):
        state_reg <<= TrafficFSM.RED
    with default():
        state_reg <<= TrafficFSM.RED

print(m.to_verilog())
```

The emitted Verilog uses `case` over the 2-bit register and assigns the next
state on each branch. The `init=TrafficFSM.RED` argument to `m.reg(...)`
becomes the register's reset / initial value.

## Encoding choices

`encoding=Encoding.<X>` controls the bit-width and the integer value of each
state Const. The three built-in encodings:

| Encoding | Width for $n$ states | Example ($n = 4$)        | When to use |
|----------|----------------------|--------------------------|-------------|
| `BINARY` (default) | $\lceil \log_2 n \rceil$ | `0, 1, 2, 3`             | Default. Smallest register, but next-state logic depends heavily on the bit-assignment. |
| `ONEHOT` | $n$                  | `0001, 0010, 0100, 1000` | One bit per state. Next-state and output logic become single-bit selects, often cheaper for small $n$ on FPGAs / wide-bit-mux ASIC libraries. Trades register area for combinational simplicity. |
| `GRAY`   | $\lceil \log_2 n \rceil$ | `00, 01, 11, 10`         | Same width as binary, but consecutive states differ by one bit. Useful when crossing clock domains or for low-power switching. |

The encoding only changes the values that the `State.<NAME>` constants hold;
everything downstream (`switch_/case_`, the register, the emitted Verilog) is
unchanged.

```python
class A(State, encoding=Encoding.ONEHOT):
    X = state(); Y = state(); Z = state()

A.typ       # → UInt(3)
A.X.value   # → 0b001
A.Y.value   # → 0b010
A.Z.value   # → 0b100
```

## Mealy vs Moore

The pattern above is a **Moore** machine: outputs depend only on the current
state (assigned outside the `case_` blocks, or to a constant inside them).

For **Mealy** outputs — output depends on state *and* input — drive the
output from inside the case body, gated by the input:

```python
with switch_(state_reg):
    with case_(MyFSM.WAIT):
        with if_(go):
            out <<= 1               # input-dependent output
            state_reg <<= MyFSM.RUN
        with else_():
            out <<= 0
```

## Reset / initial value

`m.reg(typ, name, init=expr)` declares the register's reset value. With
`Module(with_reset=True)`, the implicit `rst` input drives an async reset;
with `with_reset=False`, the `init=` value is just the initial state at
$t=0$ in simulation (synth tools may treat it as a power-on default).

For an explicit synchronous reset, mux it into the next-state expression:

```python
state_reg <<= mux(reset_signal, MyFSM.IDLE, next_state_expression)
```

## What `State` does under the hood

`__init_subclass__` inspects the class body, finds every attribute set to a
`state()` placeholder, computes the right `Const` value per the chosen
encoding, and replaces the placeholders. After class definition the
attributes are real `Const` expressions, which is why `MyFSM.IDLE` can be
typed directly into `==`, `case_(...)`, and `<<=` without any wrapping.

```python
class MyFSM(State):                 # encoding=Encoding.BINARY (default)
    A = state()                     # placeholder pre-__init_subclass__
    B = state()

# after class definition:
MyFSM.A      # Const(0, UInt(1))
MyFSM.B      # Const(1, UInt(1))
MyFSM.typ    # UInt(1)
MyFSM._width # 1
```

## See also

- The full simulation-verified examples are in
  [`testing/basic/test_fsm_examples.py`](../testing/basic/test_fsm_examples.py)
  (traffic-light, sequence detector, Mealy edge-detector, encoding-equivalence
  parity).
- The original minimal test —
  [`testing/basic/test_state_machine.py`](../testing/basic/test_state_machine.py)
  — covers the encoding-width contract and a 3-state demo.
- For control-flow context managers, see
  [`control_structures.py`](../src/spire/control_structures.py).
