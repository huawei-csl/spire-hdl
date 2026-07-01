# SpireHDL hints — key API & common mistakes

Distilled essentials for writing a correct, low-cost SpireHDL design fast. For depth, read the
topic READMEs alongside this file (`../README*.md`).

SpireHDL is a Python embedded DSL that generates synthesizable Verilog: construct a `Module`
with the API, then emit Verilog with `m.to_verilog_file("design.v")`. The generated module name
and ports must match the specification exactly.

## Canonical pattern

```python
from spirehdl.spirehdl_module import Module
from spirehdl.spirehdl import UInt, Bool, SInt, Const, mux, cat

m = Module("mult8", with_clock=False, with_reset=False)
a = m.input(UInt(8), "a")
b = m.input(UInt(8), "b")
p = m.output(UInt(16), "p")

p <<= a * b

m.to_verilog_file("design.v")
```

## Key API

- `Module(name, with_clock=False, with_reset=False)` — create a module
- `m.input(UInt(N), "name")` / `m.output(UInt(N), "name")` / `m.wire(UInt(N), "name")` — ports / wire
- `signal <<= expr` — drive a signal (combinational assignment)
- Types: `UInt(N)`, `SInt(N)`, `Bool()` — `UInt(N)` takes **exactly one** argument (the width)
- `Const(value, type)` — constant literal, e.g. `Const(0, UInt(1))`, `Const(3, SInt(8))`
- Operators: `+ - * & | ^ ~ << >> == != < <= > >=`
- `mux(sel, a, b)` — ternary `sel ? a : b`
- `cat(a, b, ...)` — concatenation, **LSB first** (`cat(a, b)` puts `a` in the lower bits)
- `signal[i]` bit-select; `signal[lo:hi]` slice (Python-style, exclusive upper bound)
- `m.to_verilog_file("design.v")` — emit Verilog. Optional `simplify=True` runs a peephole pass
  (constant folding, boolean identities, mux collapse) — often smaller output, but slower to
  compile and may time out on large circuits. Default off.
- No need to declare wires for intermediate expressions — create them inline (`a = b + c`).

## Common mistakes — avoid these

- **Wrong slice syntax:** do NOT use Verilog's `+:` part-select. Write `signal[lo : lo+N]`, not
  `signal[lo +: N]`.
- **Do not bypass SpireHDL.** Express the whole design through the API — never write Verilog
  directly from Python (no `open(...).write(...)` / string templates).
- **Width-packing bug (the big one):** see below.

## Signal width inference — critical for correct output packing

SpireHDL infers widths from expressions, and results can be **wider than you expect** (summing
four 8-bit products → 10 bits; adding a 10-bit operand → 11–12 bits). `cat()` uses each signal's
**inferred** width, not the output port's width — so packing wide intermediates into an output
bus puts every element at the wrong bit offset.

**Fix — truncate each element to the required width before `cat()`:**

```python
# pack 11-bit elements explicitly, regardless of inferred width
y <<= cat(*[result[0:11] for result in results])
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
- To pin an intermediate to an exact width, assign it to a named wire
  (`w = m.wire(UInt(N), "name"); w <<= expr`). Named wires are explicit cut-points that help
  Yosys optimize each stage; precomputing products as explicit wires before an adder tree often
  improves delay. Exact ordering/naming can shift timing — worth trying variations.

## How evaluation works

1. You write a `.py` file (e.g. `design.py`) using the API; it emits `design.v` via
   `m.to_verilog_file("design.v")`.
2. The evaluator runs your script, then checks correctness (Verilator) and cost (Yosys).
3. You may split helpers across multiple `.py` files — your working directory is on the Python
   path, so plain `from helper import build_adder` works.
