# AIG / AAG Export & Import

Spire can lower a `Component` to an **And-Inverter Graph** in AIGER ASCII
(`.aag`) form, and read AIG/AAG netlists back in as a `Component`. That makes it
easy to round-trip a design through external AIG tooling (ABC, mockturtle,
Yosys) and continue composing or simulating the result in Python. This is the
same machinery the [`@abc_optimized` / `@flowy_optimized`](README_optimization_decorators.md)
decorators use under the hood.

The exporter/importer live in [`spire/aiger.py`](../src/spire/aiger.py);
the import entry points are `Component.from_aag_lines` / `Component.from_aig_file`
in [`spire/component.py`](../src/spire/component.py).

## Export

```python
# `comp` is any Component:
aag_lines = comp.to_aag("design")             # list[str], e.g. starts with "aag 200 8 0 8 192"

# To write straight to a file, run the exporter on the lowered netlist:
from spire.aiger import AigerExporter
AigerExporter(comp.to_netlist("design")).write_aag("design.aag")
```

The AIG is a bit-level, purely combinational representation, so each packed port
(`a[7:0]`) becomes one input/output per bit.

## Import

Declare an `ImportedComponent` whose IO matches the netlist's ports, then read
the netlist into it and materialize a `Netlist`:

```python
from spire import ImportedComponent, IORecord, Input, Output, UInt

class Imported(ImportedComponent):
    def __init__(self):
        self.io = IORecord(a=Input(UInt(4)), b=Input(UInt(4)), y=Output(UInt(8)))

comp = Imported()
# make_internal=False keeps the IO as ports so the result is a standalone netlist
# (use make_internal=True instead to inline the imported logic into a surrounding Component).
comp.from_aag_lines(aag_lines, make_internal=False)   # or: comp.from_aig_file("design.aig", map_file=...)
net = comp.to_netlist("from_aig")
```

With `group=True` (the default) the importer runs `IOCollector` to rebuild the
packed buses (`a[0] … a[N-1]` → `a[N-1:0]`) so the imported netlist exposes the
same ports as the component declaration. `from_aig_file` accepts a binary `.aig`
and converts it via Yosys; pass `map_file=` when the tool emitted a name map.

## Round-trip

Export and re-import are equivalence-preserving — a quick check:

```python
from spire import Simulator

for av, bv in [(3, 5), (15, 15), (7, 9)]:
    a_out = Simulator(net).set("a", av).set("b", bv); a_out.eval()
    # ... compare against the original design driven with the same inputs
```

For an optimize-through-AIG flow, `refactor_module_to_aig(module, optimize=True)`
and `get_aig_stats(module)` in [`spire/helpers.py`](../src/spire/helpers.py)
export, optimize with aigverse, and re-import (regrouping IO) in one call.

## Examples

- [`testing/basic_examples/load_aig.py`](../testing/basic_examples/load_aig.py) — import an external `.aig`/`.aag` into a `Component` and simulate it.
- [`testing/optimize_and_check.py`](../testing/optimize_and_check.py) — export, optimize, and equivalence-check.
