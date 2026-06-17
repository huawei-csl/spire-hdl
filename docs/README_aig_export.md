# AIG / AAG Export & Import

SpireHDL can lower a `Module` to an **And-Inverter Graph** in AIGER ASCII
(`.aag`) form, and read AIG/AAG netlists back in as a `Component`. That makes it
easy to round-trip a design through external AIG tooling (ABC, mockturtle,
Yosys) and continue composing or simulating the result in Python. This is the
same machinery the [`@abc_optimized` / `@flowy_optimized`](README_optimization_decorators.md)
decorators use under the hood.

The exporter/importer live in [`spirehdl/spirehdl_aiger.py`](../src/spirehdl/spirehdl_aiger.py);
the import entry points are `Component.from_aag_lines` / `Component.from_aig_file`
in [`spirehdl/spirehdl_module.py`](../src/spirehdl/spirehdl_module.py).

## Export

```python
from spirehdl.spirehdl_aiger import AigerExporter

aag_lines = AigerExporter(module).get_aag()   # list[str], e.g. starts with "aag 200 8 0 8 192"
AigerExporter(module).write_aag("design.aag") # or write straight to a file
```

The AIG is a bit-level, purely combinational representation, so each packed port
(`a[7:0]`) becomes one input/output per bit.

## Import

Declare a `Component` whose IO matches the netlist's ports, then read the
netlist into it and materialize a `Module`:

```python
from dataclasses import dataclass
from spirehdl.spirehdl import UInt, Signal
from spirehdl.spirehdl_module import Component

@dataclass
class IO:
    a: Signal
    b: Signal
    y: Signal

class Imported(Component):
    def __init__(self):
        self.io = IO(
            a=Signal(typ=UInt(4), kind="input"),
            b=Signal(typ=UInt(4), kind="input"),
            y=Signal(typ=UInt(8), kind="output"),
        )

comp = Imported()
comp.from_aag_lines(aag_lines)                 # or: comp.from_aig_file("design.aig", map_file=...)
module = comp.to_module("from_aig")
```

With `group=True` (the default) the importer runs `IOCollector` to rebuild the
packed buses (`a[0] … a[N-1]` → `a[N-1:0]`) so the imported module exposes the
same ports as the component declaration. `from_aig_file` accepts a binary `.aig`
and converts it via Yosys; pass `map_file=` when the tool emitted a name map.

## Round-trip

Export and re-import are equivalence-preserving — a quick check:

```python
from spirehdl.spirehdl_simulator import Simulator

for av, bv in [(3, 5), (15, 15), (7, 9)]:
    a_out = Simulator(module).set("a", av).set("b", bv); a_out.eval()
    # ... compare against the original design driven with the same inputs
```

For an optimize-through-AIG flow, `refactor_module_to_aig(module, optimize=True)`
and `get_aig_stats(module)` in [`spirehdl/helpers.py`](../src/spirehdl/helpers.py)
export, optimize with aigverse, and re-import (regrouping IO) in one call.

## Examples

- [`testing/basic_examples/load_aig.py`](../testing/basic_examples/load_aig.py) — import an external `.aig`/`.aag` into a `Component` and simulate it.
- [`testing/optimize_and_check.py`](../testing/optimize_and_check.py) — export, optimize, and equivalence-check.
