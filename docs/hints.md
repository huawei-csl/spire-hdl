# Spire hints — key API & common mistakes

Distilled essentials for writing a correct, low-cost Spire design fast. For depth, read the topic
READMEs alongside this file (`./README_*.md`) and the examples in `../testing/examples/`.

Spire is a Python embedded DSL that generates synthesizable Verilog: define a `Component`, declare
its IO with `IORecord`, put the logic in `elaborate()`, then emit Verilog with
`MyComponent().to_verilog_file("design.v", name="<module_name>")`. The generated module name and
ports must match the specification exactly.

## Canonical pattern

```python
from spire import Component, IORecord, Input, Output, UInt
from spire.expr import Const, mux, cat

class Mult8(Component):
    def __init__(self):
        # Field names become signal names; Input/Output set direction + bit-precise type.
        self.io = IORecord(
            a=Input(UInt(8)),
            b=Input(UInt(8)),
            p=Output(UInt(16)),
        )
        self.elaborate()

    def elaborate(self):
        # Drive outputs with <<=; intermediate expressions need no explicit wires.
        self.io.p <<= self.io.a * self.io.b

# `name` sets the emitted Verilog module name (must match the spec).
Mult8().to_verilog_file("design.v", name="mult8")
```

## Key API

- `class MyDesign(Component)` — subclass `Component`. In `__init__`, set `self.io = IORecord(...)`
  then call `self.elaborate()`; put the logic in `elaborate(self)`.
- `IORecord(a=Input(UInt(8)), s=Output(UInt(9)), ...)` — declare ports. The keyword name becomes
  the signal name; `Input`/`Output` set direction and type. Access as `self.io.a`, `self.io.s`.
- `signal <<= expr` — drive a signal (combinational assignment).
- Types: `UInt(N)`, `SInt(N)`, `Bool()` — `UInt(N)`/`SInt(N)` take **exactly one** argument (the width).
- `Const(value, type)` — constant literal, e.g. `Const(0, UInt(1))`, `Const(3, SInt(8))` (from `spire.expr`).
- `Wire(UInt(N))` / `Register(UInt(N), init=0)` — internal wire / clocked register (from `spire`).
  **No need to name them** — names are auto-inferred from the Python variable.
- Operators: `+ - * & | ^ ~ << >> == != < <= > >=`
- `mux(sel, a, b)` — ternary `sel ? a : b` (from `spire.expr`).
- `cat(a, b, ...)` — concatenation, **LSB first** (`cat(a, b)` puts `a` in the lower bits) (from `spire.expr`).
- `signal[i]` bit-select; `signal[lo:hi]` slice (Python-style, exclusive upper bound).
- `MyComponent().to_verilog_file("design.v", name="<module_name>")` — emit Verilog. Optional
  `simplify=True` runs a peephole pass (constant folding, boolean identities, mux collapse) — often
  smaller output, but slower to compile and may time out on large circuits. Default off.
- No need to declare wires for intermediate expressions — create them inline (`t = b + c`).

## Sequential logic (clock / reset)

Use `Register` for state and drive it with `<<=`. Pass `with_clock=True` (and `with_reset=True`
when a register has a reset value) to `to_verilog_file`:

```python
from spire import Component, IORecord, Input, Output, UInt
from spire.expr import Register

class Mac(Component):
    def __init__(self):
        self.io = IORecord(a=Input(UInt(16)), b=Input(UInt(16)), acc_out=Output(UInt(32)))
        self.elaborate()

    def elaborate(self):
        acc = Register(UInt(32), init=0)          # name auto-inferred as "acc"
        acc <<= acc + self.io.a * self.io.b
        self.io.acc_out <<= acc

Mac().to_verilog_file("design.v", name="mac", with_clock=True, with_reset=True)
```

## Hierarchical designs

Instantiate a sub-`Component`, wire its `.io`, and its logic is embedded automatically:

```python
adder = Adder(width=8)      # another Component subclass
adder.io.a <<= self.io.x
adder.io.b <<= self.io.y
self.io.sum <<= adder.io.sum
```

## Common mistakes — avoid these

- **Wrong slice syntax:** do NOT use Verilog's `+:` part-select. Write `signal[lo : lo+N]`, not
  `signal[lo +: N]`.
- **Do not bypass Spire.** Express the whole design through the API — never write Verilog directly
  from Python (no `open(...).write(...)` / string templates).
- **Width-packing bug (the big one):** see below.

## Signal width inference — critical for correct output packing

Spire infers widths from expressions, and results can be **wider than you expect** (summing four
8-bit products → 10 bits; adding a 10-bit operand → 11–12 bits). `cat()` uses each signal's
**inferred** width, not the output port's width — so packing wide intermediates into an output bus
puts every element at the wrong bit offset.

**Fix — truncate each element to the required width before `cat()`:**

```python
# pack 11-bit elements explicitly, regardless of inferred width
self.io.y <<= cat(*[result[0:11] for result in results])
```

Tip: if unsure, inspect the generated Verilog wire widths to confirm element sizes.

## Synthesis-quality notes

- When accumulating in a loop, start from the first element rather than a zero constant, to avoid
  an extra adder level:
  ```python
  sum_val = a[i][0] * b[0][j]
  for k in range(1, 4):
      sum_val = sum_val + a[i][k] * b[k][j]
  ```
- To pin an intermediate to an exact width, assign it to an explicit `Wire`
  (`w = Wire(UInt(N)); w <<= expr`). Explicit wires are cut-points that help Yosys optimize each
  stage; precomputing products as explicit wires before an adder tree often improves delay. Exact
  ordering can shift timing — worth trying variations.

## How evaluation works

1. You write a `.py` file (e.g. `design.py`) using the API; it emits `design.v` via
   `MyComponent().to_verilog_file("design.v", name="<module_name>")`.
2. The evaluator runs your script, then checks correctness (Verilator) and cost (Yosys).
3. You may split helpers across multiple `.py` files — your working directory is on the Python
   path, so plain `from helper import build_adder` works.
