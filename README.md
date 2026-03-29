# Sprout-HDL

Sprout-HDL is a Python embedded domain-specific language (DSL) for building digital hardware in a concise, composable way.  It lets you describe logic with Python expressions, compile the result to synthesizable Verilog or AIG/AAG netlists, and iterate quickly with a built-in cycle-accurate simulator.

For cores, generators (e.g. arithmetic units), evaluation scripts, and extra tooling notes, see [`README_cores_extras.md`](README_cores_extras.md).

![Sprout HDL](imgs/sprout_hdl.png)

## Project overview

Sprout-HDL revolves around a small set of core modules:

- **[`sprouthdl/sprouthdl.py`](src/sprouthdl/sprouthdl.py)** – the expression DSL.  It provides bit-precise types such as `Bool`, `UInt`, and `SInt`, shared-expression caching, and the overloaded arithmetic / bitwise operators that make the Python syntax feel like an HDL.
- **[`sprouthdl/sprouthdl_module.py`](src/sprouthdl/sprouthdl_module.py)** – structural modeling helpers.  The `Module` class constructs ports, wires, and registers, produces Verilog, and exposes analysis utilities.  The `Component` base class lets you package reusable sub-designs and convert them to or from Sprout modules.  `IOCollector` can rebuild packed ports from bit-level signals when importing external netlists.
- **[`sprouthdl/sprouthdl_simulator.py`](src/sprouthdl/sprouthdl_simulator.py)** – a lightweight simulator that can drive inputs, tick clocks, inspect outputs or internal expressions, and capture probes for debugging—all without leaving Python.

Supporting packages add reusable arithmetic building blocks and importer utilities for external netlists when you need to mix handwritten Sprout code with pre-existing IP; see [`mutipliers_ext.py`](src/sprouthdl/arithmetic/int_multipliers/multipliers/mutipliers_ext.py) and [`multipliers_ext_optimized.py`](src/sprouthdl/arithmetic/int_multipliers/multipliers/multipliers_ext_optimized.py) for examples.

## Installation

```bash
git clone https://github.com/huawei-csl/sprout-hdl.git
cd sprout-hdl
pip install -e .
```

The library relies on the packages listed in `requirements.txt`.  Optional regression tests require Yosys/Pyosys and aigverse if you plan to exercise the external tooling integration flows.

## Quick start

### 1. Describe a module

```python
from sprouthdl.sprouthdl_module import Module
from sprouthdl.sprouthdl import Bool, UInt, mux, cat

m = Module("LogicDemo", with_clock=False, with_reset=False)
a = m.input(UInt(8), "a")
b = m.input(UInt(8), "b")
sel = m.input(Bool(), "sel")
sum_ = m.output(UInt(9), "sum")
mask = m.output(UInt(4), "mask")
out = m.output(UInt(8), "out")

sum_ <<= a + b              # automatic width growth
top_bits = cat(a[7], b[7])
mask <<= top_bits           # concatenate slices
a_and_b = a & b
b_or_a = a | b
out <<= mux(sel, a_and_b, b_or_a)

print(m.to_verilog())
```

The `Module` API checks that every output has a driver and every register has a next-state assignment before emitting Verilog (see [`sprouthdl_module.py`](src/sprouthdl/sprouthdl_module.py)).

### 2. Simulate the design

```python
from sprouthdl.sprouthdl_simulator import Simulator

sim = Simulator(m)
sim.set("a", 0x55).set("b", 0x0F).set("sel", 1)
sim.eval()                 # recompute combinational logic
print(sim.peek_outputs())   # {'sum': 0x64, 'mask': 0x9, 'out': 0x05}
```

The simulator keeps track of inputs, wires, outputs, and registers, supports `eval()` for combinational updates, `step()` for clocked designs, and exposes helpers such as `peek`, `peek_next`, and signal watching for deeper inspection ([`sprouthdl_simulator.py`](src/sprouthdl/sprouthdl_simulator.py)).

### 3. Integrate with external tooling

Modules can be exported to Verilog, AIG, or AAG for downstream synthesis, equivalence checking, or integration into larger verification environments.  Import helpers then let you bring optimized or third-party netlists back into Sprout for continued composition and simulation (see [`sprouthdl_module.py`](src/sprouthdl/sprouthdl_module.py) and [`multipliers_ext_optimized.py`](src/sprouthdl/arithmetic/int_multipliers/multipliers/multipliers_ext_optimized.py)).

## Modules and components in detail

- `Component` subclasses package reusable structures.  They can materialize new modules (`to_module`), import designs from Verilog or AIG formats (`from_verilog`, `from_aag_lines`), and retag ports as internals (`make_internal`).  Components also expose `get_spec()` to drive `IOCollector` regrouping when you import flattened designs (see [`sprouthdl_module.py`](src/sprouthdl/sprouthdl_module.py)).
- `Module` is typically used at the top level or as an intermediate representation while you are still wiring a design.  It offers constructors for inputs, outputs, wires, and registers; utilities for enumerating signals; Verilog emission with automatic width fitting; and a `module_analyze()` routine that reports combinational depth and node counts for timing exploration ([`sprouthdl_module.py`](src/sprouthdl/sprouthdl_module.py)).
- `IOCollector` helps rebuild packed buses (e.g., `a[0] … a[N-1]` → `a[N-1:0]`) after reading back designs from AIG/AAG files or external synthesizers ([`sprouthdl_module.py`](src/sprouthdl/sprouthdl_module.py)).
- Minimal end-to-end component example: [`testing/examples/simple_component.py`](testing/examples/simple_component.py).

Short component + hierarchy usage example:

```python
from dataclasses import dataclass
from sprouthdl.sprouthdl import UInt, Signal
from sprouthdl.sprouthdl_module import Component

class SimpleAdder(Component):
    def __init__(self, width=8):
        self.width = width
        @dataclass
        class IO:
            a: Signal
            b: Signal
            sum: Signal
        self.io = IO(
            a=Signal(name="a", typ=UInt(width), kind="input"),
            b=Signal(name="b", typ=UInt(width), kind="input"),
            sum=Signal(name="sum", typ=UInt(width + 1), kind="output"),
        )
        self.elaborate()

    def elaborate(self):
        self.io.sum <<= self.io.a + self.io.b

class Sum3Hier(Component):
    def __init__(self):
        @dataclass
        class IO:
            a: Signal
            b: Signal
            c: Signal
            sum: Signal
        self.io = IO(
            a=Signal(name="a", typ=UInt(8), kind="input"),
            b=Signal(name="b", typ=UInt(8), kind="input"),
            c=Signal(name="c", typ=UInt(8), kind="input"),
            sum=Signal(name="sum", typ=UInt(10), kind="output"),
        )
        self.elaborate()

    def elaborate(self):
        add_ab = SimpleAdder(width=8).make_internal()     # first sub-component
        add_abc = SimpleAdder(width=9).make_internal()    # second sub-component
        add_ab.io.a <<= self.io.a
        add_ab.io.b <<= self.io.b
        add_abc.io.a <<= add_ab.io.sum
        add_abc.io.b <<= self.io.c
        self.io.sum <<= add_abc.io.sum

module = Sum3Hier().to_module(name="Sum3Hier")
print(module.to_verilog())  # one top module, built from internal components
```

### Hierarchical design with components

Components are ideal for assembling hierarchical designs: they let you instantiate another component, adapt its IO, and even swap in a pre-synthesized netlist without leaving Python.  One common pattern wraps a reusable building block with `make_internal()` so that auxiliary logic can surround the core implementation while exposing a compact public interface (see [`mutipliers_ext.py`](src/sprouthdl/arithmetic/int_multipliers/multipliers/mutipliers_ext.py)).  A related flow imports an external AIG module, converts it into a `Component`, and calls `from_module(..., make_internal=True)` so the imported logic behaves like a native Sprout block inside a larger generator ([`multipliers_ext_optimized.py`](src/sprouthdl/arithmetic/int_multipliers/multipliers/multipliers_ext_optimized.py)).  These techniques extend to Verilog importers and make it straightforward to mix Sprout-authored code with IP produced by external flows.

## Aggregate data types

Sprout includes structured, bit-packable aggregates for cleaner interfaces and bulk assignments ([`aggregate/`](src/sprouthdl/aggregate)):

- `HDLAggregate` defines the base “pack to bits” API that powers all aggregates ([`hdl_aggregate.py`](src/sprouthdl/aggregate/hdl_aggregate.py)).
- `Array` offers N-dimensional indexing, packed assignment (`<<=`), and element-wise assignment (`@=`) for nested vectors or aggregates ([`aggregate_array.py`](src/sprouthdl/aggregate/aggregate_array.py)).
- `AggregateRecord` lets you declare bundle-like classes with named fields that remain packable to a flat bitvector ([`aggregate_record.py`](src/sprouthdl/aggregate/aggregate_record.py)).
- `FixedPoint` wraps a `Wire` or view with explicit total/frac widths and quantization helpers, keeping arithmetic readable while staying hardware-friendly ([`aggregate_fixed_point.py`](src/sprouthdl/aggregate/aggregate_fixed_point.py)).
- `AggregateRegister` stores any aggregate in a single register while preserving a structured view via `.value`/`.Q` ([`aggregate_register.py`](src/sprouthdl/aggregate/aggregate_register.py)).

Example:

```python
from sprouthdl.aggregate.aggregate_array import Array
from sprouthdl.aggregate.aggregate_record import AggregateRecord
from sprouthdl.aggregate.aggregate_fixed_point import FixedPoint, FixedPointType
from sprouthdl.aggregate.aggregate_register import AggregateRegister
from sprouthdl.sprouthdl import UInt, Wire

class Bus(AggregateRecord):
    data = Wire(UInt(8))
    valid = Wire(UInt(1))

payload = Array([Bus(), Bus()])
acc = FixedPoint(FixedPointType(width_total=16, width_frac=8))
acc_reg = AggregateRegister(FixedPoint, acc.ftype, name="acc_reg")

acc_reg <<= acc            # packed register write
payload[1] @= payload[0]   # element-wise copy between bundles
```

## Simulation notes

The simulator supports both combinational and sequential designs:

- `eval()` recomputes combinational logic and captures registered probes.
- `set()` and `get()` let you drive or inspect signals by name.
- `step()` advances the clock, committing register next-state expressions while honoring asynchronous resets.
- `watch()` and `peek_next()` provide scope-style visibility for debugging complex pipelines.

These capabilities align with the standard Sprout development flow: express a design, validate it in Python, then export it to your synthesis or verification stack.

## Slices
We follow the indexing of python also in Sprout-HDL signals. For example `sig[4:7]` creates a new expression containing of bits 4 and 5 (counted from lsb) of the original expression `sig`.


## Main development flow

1. **Model logic in Python.** Use `Module` in the the top-level file and DSL expressions to capture datapaths, state machines, and control logic.
2. **Factor reusable pieces.** Wrap recurring structures in `Component` subclasses so they can be instantiated, parameterized, or replaced with imported implementations.
3. **Simulate early and often.** Drive stimuli with the simulator, observe register evolution, and iterate on the Python source before handing designs to downstream tools.
4. **Export netlists.** Emit Verilog or AIG/AAG when you are ready for synthesis, formal checking, or integration with external flows.

## Examples

Check out the `testing/examples/` directory for practical examples:

- **`simple_component.py`** – A minimal example showing how to define a Component with IO ports and generate Verilog
- **`component_example.py`** – Comprehensive examples including hierarchical design and simulation
- **`module_with_component.py`** – Shows how to integrate Components within Module-based designs
- **`direct_expression_basics.py`** – Minimal direct expression examples (`y = a + b`) plus `+`, `-`, unary `-`, `Const(..., Int(...))`, typed/plain `False`, and a recursive Horner polynomial builder
- **`testing/riscv/rv32i.py`** – Minimal RV32I core example; see `testing/riscv/test_rv32i.py` for simulation-based checks.

See the [examples README](testing/examples/README.md) for detailed documentation and key concepts.

## Automatic arithmetic optimization

Sprout-HDL ships a library of configurable arithmetic building blocks: prefix adders (Kogge-Stone, Brent-Kung, Sklansky, ...), multipliers (stage-based with PPG/PPA/FSA selection), and subtractors.  Rather than requiring the user to pick the right topology, `replace_arithmetic_ops` can **automatically select the best configuration** for every `+`, `-`, and `*` operator in a design, guided by a pre-computed evaluation database.

Three optimization objectives are available:

| Objective | Minimizes | Good for |
|-----------|-----------|----------|
| `"area"`  | Yosys transistor count | Silicon area, power |
| `"delay"` | AIG depth (AND-gate levels) | Clock frequency |
| `"adp"`   | Area-delay product | Balanced designs |

### Example: 16-bit unsigned adder

The auto-config sweep evaluates all prefix-adder topologies and picks the best one per objective:

| Objective | Topology | Transistors | AIG Depth |
|-----------|----------|------------:|----------:|
| plain (Yosys `+`) | *(default synthesis)* | 648 | 32 |
| `area`    | Sparse Kogge-Stone (2) | 626 | 32 |
| `delay`   | Kogge-Stone            | 952 | 22 |
| `adp`     | Brent-Kung             | 746 | 26 |

The `adp` objective automatically finds the Brent-Kung topology as a compromise between the area-optimized Sparse Kogge-Stone and the speed-optimized Kogge-Stone.

### Example: 8-bit ALU (add + sub + mul)

A complete 8-bit ALU with addition, subtraction, and multiplication, comparing Yosys default synthesis (`*` operator) against auto-optimized replacements:

| Configuration | Transistors | AIG Depth |
|---------------|------------:|----------:|
| plain (Yosys `*`)  | 6168 | 127 |
| `area`             | 3004 |  73 |
| `delay`            | 3830 |  30 |
| `adp`              | 3084 |  30 |

The area objective achieves a **51% transistor reduction** over default synthesis.  The delay objective cuts critical-path depth from 127 to 30 AND-gate levels (**4.2x**), and the balanced `adp` objective achieves both nearly minimal area *and* minimal depth.

### Usage

```python
from sprouthdl.arithmetic.int_arithmetic_config import ArithmeticAutoConfig, replace_arithmetic_ops

# Define your design using plain operators
class ALU(Component):
    def elaborate(self):
        self.io.y_add <<= self.io.a + self.io.b
        self.io.y_sub <<= self.io.a - self.io.b
        self.io.y_mul <<= self.io.a * self.io.b

alu = ALU(width=8)

# Replace operators with optimized hardware — one line
replace_arithmetic_ops(alu, ArithmeticAutoConfig(objective="adp"))

module = alu.to_module("OptimizedALU")
print(module.to_verilog())
```

Each `+`, `-`, and `*` in the expression graph is independently replaced with the empirically best prefix-adder or stage-based multiplier configuration for its specific bit-width and signedness.  For widths not in the evaluation database, the nearest data point is selected using logarithmic interpolation.

See [`testing/low_level_arithmetic/int_adders/test_arithmetic_auto_config.py`](testing/low_level_arithmetic/int_adders/test_arithmetic_auto_config.py) for the full test and benchmark code.

## Next steps

- Explore the `testing/examples/` directory to see working examples of components and modules
- Explore the `sprouthdl/arithmetic` and `sprouthdl/arithmetic/floating_point` packages for more generators.
- Use `<module>.module_analyze()` to gauge combinational depth before synthesis.
- Integrate the simulator into your verification harness to shorten debug cycles.
