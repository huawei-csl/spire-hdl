<div align="center">
  <img src="imgs/spire-hdl.png" alt="SpireHDL" width="250">
</div>

<br>

A modern Python HDL that compiles concise, composable hardware descriptions to synthesizable Verilog and AIG/AAG netlists — with synthesis optimization and a cycle-accurate simulator built in.

- **Designed for humans and agents to be effective** — a small, regular surface that reads well to engineers and LLMs alike
- **Reduces area and delay vs. a traditional Verilog flow** — optimization is part of the compile, not an afterthought
- **Integrated with ABC and mockturtle** — modern synthesis optimization wired directly into the compilation pipeline
- **Arithmetic library with automated replacement** — swap adders, multipliers, and FP cores against an area/delay objective
- **Cycle-accurate Python simulator** — drive inputs, tick clocks, inspect any expression, and capture probes without leaving Python
- **Content-addressed optimization cache** — instant re-runs via `@flowy_optimized` / `@abc_optimized` decorators

# Overview

SpireHDL revolves around a small set of core modules:

- **[`spirehdl/spirehdl.py`](src/spirehdl/spirehdl.py)** – the expression DSL.  It provides bit-precise types such as `Bool`, `UInt`, and `SInt`, shared-expression caching, and the overloaded arithmetic / bitwise operators that make the Python syntax feel like an HDL.
- **[`spirehdl/spirehdl_module.py`](src/spirehdl/spirehdl_module.py)** – structural modeling helpers.  The `Module` class constructs ports, wires, and registers, produces Verilog, and exposes analysis utilities.  The `Component` base class lets you package reusable sub-designs and convert them to or from SpireHDL modules.  `IOCollector` can rebuild packed ports from bit-level signals when importing external netlists.
- **[`spirehdl/spirehdl_simulator.py`](src/spirehdl/spirehdl_simulator.py)** – a lightweight simulator that can drive inputs, tick clocks, inspect outputs or internal expressions, and capture probes for debugging—all without leaving Python.

### Further reading

Other markdown documents in this repository:

- [`README_cores_extras.md`](README_cores_extras.md) — cores, generators, evaluation scripts, and extra tooling notes
- [`README_arithmetic_optimization.md`](README_arithmetic_optimization.md) — automatic arithmetic replacement with `replace_arithmetic_ops` (adders, multipliers, subtractors)
- [`README_optimization_decorators.md`](README_optimization_decorators.md) — the `@abc_optimized` / `@flowy_optimized` circuit optimization decorators
- [`README_state_machines.md`](README_state_machines.md) — finite-state-machine declaration with the `State` / `Encoding` API and `switch_`/`case_` bodies
- [`README_memories.md`](README_memories.md) — the `Memory` primitive (FIFOs, ROMs, RAMs), port wiring with `<<=`, simulation, and reading current memory state
- [`testing/examples/README.md`](testing/examples/README.md) — example designs exercising SpireHDL features

## Installation

```bash
git clone https://github.com/huawei-csl/spire-hdl.git
cd spire-hdl
pip install -e .
```

The library relies on the packages listed in `requirements.txt`.  Optional regression tests require Yosys/Pyosys and aigverse if you plan to exercise the external tooling integration flows.

## Quick start

### 1. Describe a module

```python
from spirehdl.spirehdl_module import Module
from spirehdl.spirehdl import Bool, UInt, mux, cat

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

The `Module` API checks that every output has a driver and every register has a next-state assignment before emitting Verilog (see [`spirehdl_module.py`](src/spirehdl/spirehdl_module.py)).

**Registers** are created either via the standalone `Register` class or `Module.reg(...)`. Both take a `typ` and an optional reset value via the `init=` keyword (note: the keyword is `init`, not `reset_value` / `reset`). Assign the next-state expression with `<<=`:

```python
from spirehdl.spirehdl import Register, UInt

m = Module("Counter", with_clock=True, with_reset=True)
cnt = Register(UInt(8), init=0, name="cnt")       # or: cnt = m.reg(UInt(8), "cnt", init=0)
cnt <<= cnt + 1                                   # next-state = cnt + 1
m.output(UInt(8), "q") <<= cnt
```

### 2. Simulate the design

```python
from spirehdl.spirehdl_simulator import Simulator

sim = Simulator(m)
sim.set("a", 0x55).set("b", 0x0F).set("sel", 1)
sim.eval()                 # recompute combinational logic
print(sim.peek_outputs())   # {'sum': 0x64, 'mask': 0x9, 'out': 0x05}
```

The simulator keeps track of inputs, wires, outputs, and registers, supports `eval()` for combinational updates, `step()` for clocked designs, and exposes helpers such as `peek`, `peek_next`, and signal watching for deeper inspection ([`spirehdl_simulator.py`](src/spirehdl/spirehdl_simulator.py)).

### 3. Integrate with external tooling

Modules can be exported to Verilog, AIG, or AAG for downstream synthesis, equivalence checking, or integration into larger verification environments.  Import helpers then let you bring optimized or third-party netlists back into SpireHDL for continued composition and simulation (see [`spirehdl_module.py`](src/spirehdl/spirehdl_module.py) and [`multipliers_ext_optimized.py`](src/spirehdl/arithmetic/int_multipliers/multipliers/multipliers_ext_optimized.py)).

## Modules and components in detail

- `Component` subclasses package reusable structures.  They can materialize new modules (`to_module`), import designs from Verilog or AIG formats (`from_verilog`, `from_aag_lines`), and retag ports as internals (`make_internal`).  Components also expose `get_spec()` to drive `IOCollector` regrouping when you import flattened designs (see [`spirehdl_module.py`](src/spirehdl/spirehdl_module.py)).
- `Module` is typically used at the top level or as an intermediate representation while you are still wiring a design.  It offers constructors for inputs, outputs, wires, and registers; utilities for enumerating signals; Verilog emission with automatic width fitting; and a `module_analyze()` routine that reports combinational depth and node counts for timing exploration ([`spirehdl_module.py`](src/spirehdl/spirehdl_module.py)).
- `IOCollector` helps rebuild packed buses (e.g., `a[0] … a[N-1]` → `a[N-1:0]`) after reading back designs from AIG/AAG files or external synthesizers ([`spirehdl_module.py`](src/spirehdl/spirehdl_module.py)).
- Minimal end-to-end component example: [`testing/examples/simple_component.py`](testing/examples/simple_component.py).

Short component + hierarchy usage example:

```python
from dataclasses import dataclass
from spirehdl.spirehdl import UInt, Signal
from spirehdl.spirehdl_module import Component

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

Components are ideal for assembling hierarchical designs: they let you instantiate another component, adapt its IO, and even swap in a pre-synthesized netlist without leaving Python.  One common pattern wraps a reusable building block with `make_internal()` so that auxiliary logic can surround the core implementation while exposing a compact public interface (see [`mutipliers_ext.py`](src/spirehdl/arithmetic/int_multipliers/multipliers/mutipliers_ext.py)).  A related flow imports an external AIG module, converts it into a `Component`, and calls `from_module(..., make_internal=True)` so the imported logic behaves like a native SpireHDL block inside a larger generator ([`multipliers_ext_optimized.py`](src/spirehdl/arithmetic/int_multipliers/multipliers/multipliers_ext_optimized.py)).  These techniques extend to Verilog importers and make it straightforward to mix SpireHDL-authored code with IP produced by external flows.

## Aggregate data types

SpireHDL includes structured, bit-packable aggregates for cleaner interfaces and bulk assignments ([`aggregate/`](src/spirehdl/aggregate)):

- `HDLAggregate` defines the base “pack to bits” API that powers all aggregates ([`hdl_aggregate.py`](src/spirehdl/aggregate/hdl_aggregate.py)).
- `Array` offers N-dimensional indexing, packed assignment (`<<=`), and element-wise assignment (`@=`) for nested vectors or aggregates ([`aggregate_array.py`](src/spirehdl/aggregate/aggregate_array.py)).
- `AggregateRecord` lets you declare bundle-like classes with named fields that remain packable to a flat bitvector ([`aggregate_record.py`](src/spirehdl/aggregate/aggregate_record.py)).
- `FixedPoint` wraps a `Wire` or view with explicit total/frac widths and quantization helpers, keeping arithmetic readable while staying hardware-friendly ([`aggregate_fixed_point.py`](src/spirehdl/aggregate/aggregate_fixed_point.py)).
- `AggregateRegister` stores any aggregate in a single register while preserving a structured view via `.value`/`.Q` ([`aggregate_register.py`](src/spirehdl/aggregate/aggregate_register.py)).

Example:

```python
from spirehdl.aggregate.aggregate_array import Array
from spirehdl.aggregate.aggregate_record import AggregateRecord
from spirehdl.aggregate.aggregate_fixed_point import FixedPoint, FixedPointType
from spirehdl.aggregate.aggregate_register import AggregateRegister
from spirehdl.spirehdl import UInt, Wire

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

These capabilities align with the standard SpireHDL development flow: express a design, validate it in Python, then export it to your synthesis or verification stack.

## Slices
We follow the indexing of python also in SpireHDL signals. For example `sig[4:7]` creates a new expression containing of bits 4 and 5 (counted from lsb) of the original expression `sig`.


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

`replace_arithmetic_ops` automatically selects the best hardware configuration for every `+`, `-`, `*`, `==`, `!=` operator and detects MAC / inner product patterns for fusion.  See [`README_arithmetic_optimization.md`](README_arithmetic_optimization.md) for details, benchmarks, and examples including a 4-tap FIR filter with **54% area reduction** and **68% depth reduction**.

## Circuit optimization decorators

`@flowy_optimized` and `@abc_optimized` let you optimize any combinational function at the AIG level with a single decorator.  See [`README_optimization_decorators.md`](README_optimization_decorators.md) for usage, ABC script examples, and benchmark results.

## Next steps

- Explore the `testing/examples/` directory to see working examples of components and modules
- Explore the `spirehdl/arithmetic` and `spirehdl/arithmetic/floating_point` packages for more generators.
- Use `<module>.module_analyze()` to gauge combinational depth before synthesis.
- Integrate the simulator into your verification harness to shorten debug cycles.
