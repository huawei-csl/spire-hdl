<div align="center">
  <img src="imgs/spire-hdl.png" alt="SpireHDL" width="250">
</div>

<br>

<p align="center">
  <a href="https://github.com/huawei-csl/spire-hdl/actions/workflows/ci.yml"><img src="https://github.com/huawei-csl/spire-hdl/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://pypi.org/project/spire-hdl/"><img src="https://img.shields.io/pypi/v/spire-hdl.svg" alt="PyPI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-BSD--3--Clause--Clear-blue.svg" alt="License: BSD-3-Clause-Clear"></a>
</p>

A modern Python HDL that compiles concise, composable hardware descriptions to synthesizable Verilog and AIG netlists — with synthesis optimization and a cycle-accurate simulator built in.

- **Built for humans and agents alike:** a small surface that stays readable as a design grows
- **Reduces area and delay vs. a traditional Verilog flow:** optimization is part of the compile
- **Integrated with ABC and mockturtle:** modern synthesis optimization wired directly into the compilation pipeline
- **Arithmetic library with automated replacement:** swap adders, multipliers, and FP cores driven by an objective
- **Cycle-accurate Python simulator"** drive inputs, tick clocks, inspect expressions/outputs without leaving Python
- **Content-addressed optimization cache:** instant re-runs via `@abc_optimized` /`@flowy_optimized` decorators

## Optimizations built in 💡

SpireHDL supports **source-level optimization intent**: the designer marks *what* to optimize (e.g. a module, FSM, or arithmetic block) directly in the HDL source, and the compiler realizes it through synthesis-aware passes.

Because these passes run as part of the compile, the emitted Verilog is already small and fast before external tools see it. The numbers below are measured against a plain Yosys flow on the same RTL.

### 🔢 Arithmetic auto-replacement: `@arithmetic_optimized`

Drops in topology-tuned adders, multipliers, and MAC fusions against an `area` / `delay` / `adp` objective. On an 8-bit ALU (add + sub + mul):

- **−51% transistors** with the `area` objective
- **4.2× shorter critical path** with the `delay` objective
- balanced `adp` gets near-minimal area *and* near-minimal depth at once

MAC patterns (`a*b + c`) are fused into single column-reduction units, eliminating a full adder stage. See [`README_arithmetic_optimization.md`](docs/README_arithmetic_optimization.md).

### 🧩 ABC + mockturtle decorators: `@abc_optimized` / `@flowy_optimized`

One decorator stacks modern AIG synthesis (`resyn2`, `&deepsyn`, mockturtle) onto any `Module` or `Component`, with a content-addressed cache for instant re-runs:

- **−69% AIG gates** on an 8-bit multiplier (`resyn2`)
- **−83%** on a 16-bit multiplier
- stack with `@arithmetic_optimized` for compounding wins — ABC cleans up after the arithmetic rewriter

See [`README_optimization_decorators.md`](docs/README_optimization_decorators.md).

### 🎯 FSM + encoding search: `optimized_fsm` / `optimized_encoding`

Hopcroft state minimisation and bit-assignment search as two composable context managers. On a 7-state `case10` Moore FSM:

- **−19% cells** with `optimized_encoding` alone
- **−44% cells** with `optimized_fsm` alone
- **−69.5% cells** when both are nested — a ~3× reduction with two `with` blocks, no hand-tuned encoding tables

An 8-opcode CPU decoder sees **−66.7% cells** from a single `optimized_encoding`, because the search discovers an opcode layout where each wide OR collapses to one bit-test. See [`README_fsm_optimization.md`](docs/README_fsm_optimization.md).

### 🛠️ Fine-grained architecture selection: `arithmetic_generator`

Beyond the automatic passes above, the unified arithmetic generator lets you hand-pick the exact micro-architecture of an adder, multiplier, MAC, or matmul (partial-product generation, compression-tree topology, and final-stage adder), then emit Verilog/AIG, simulate, and collect Yosys metrics for direct comparison. See [`README_arithmetic_generator.md`](docs/README_arithmetic_generator.md).

## Overview

### 🪶 Minimal core

In its simplest form, SpireHDL only needs these core files. This is intentional — the HDL is kept to a minimal, self-contained core, and higher-level features are layered on top:

- **[`spirehdl/spirehdl.py`](src/spirehdl/spirehdl.py)** – the expression DSL. It provides bit-precise types such as `Bool`, `UInt`, and `SInt`, shared-expression caching, and the overloaded arithmetic / bitwise operators that make the Python syntax feel like an HDL.
- **[`spirehdl/spirehdl_module.py`](src/spirehdl/spirehdl_module.py)** – structural modeling helpers. The `Module` class constructs ports, wires, and registers, produces Verilog, and exposes analysis utilities. The `Component` base class lets you package reusable sub-designs and convert them to or from SpireHDL modules.
- **[`spirehdl/spirehdl_simulator.py`](src/spirehdl/spirehdl_simulator.py)** – a lightweight simulator that can drive inputs, tick clocks, inspect outputs or internal expressions, and capture probes for debugging—all without leaving Python.

### 📚 Further reading

Deeper guides for specific features:

- **[State machines](docs/README_state_machines.md)** — declaration with the `State` / `Encoding` API and `switch_` / `case_` bodies
- **[Control structures](docs/README_control_structures.md)** — `if_` / `elif_` / `else_` and `switch_` / `case_` / `default` context managers
- **[Memories](docs/README_memories.md)** — RAM / ROM / FIFO primitives, port wiring with `<<=`, simulation, and reading state
- **[Arithmetic optimization](docs/README_arithmetic_optimization.md)** — automatic replacement with optimized versions (adders, multipliers, MAC, etc)
- **[Optimization decorators](docs/README_optimization_decorators.md)** — `@abc_optimized` / `@flowy_optimized` circuit optimization
- **[FSM optimization](docs/README_fsm_optimization.md)** — `optimized_fsm` and `optimized_encoding` (state minimisation + encoding search)
- **[Arithmetic generators](docs/README_arithmetic_generator.md)** — evaluation scripts and extra tooling notes
- **[Custom Verilog](docs/README_custom_verilog.md)** — emit a raw Verilog block from a `Component`, with or without a Python sim model (blackbox)
- **[AIG / AAG export & import](docs/README_aig_export.md)** — lower a `Module` to an AIGER netlist and read AIG/AAG back in as a `Component`
- **[Verilog testbench](docs/README_verilog_testbench.md)** — turn a `Simulator` run into a self-checking, synthesizable Verilog testbench
- **[Examples](testing/examples/README.md)** — example designs exercising SpireHDL features

## Installation

Install the latest release from PyPI:

```bash
pip install spire-hdl
```

For development, install from source in editable mode:

```bash
git clone https://github.com/huawei-csl/spire-hdl.git
cd spire-hdl
pip install -e .
```

The library relies on the packages listed in `requirements.txt`.

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
q = m.output(UInt(8), "q")                        # bind to a name first
q <<= cnt                                          # (q <<= ... ; not m.output(...) <<= ...)
```

### 2. Simulate the design

```python
from spirehdl.spirehdl_simulator import Simulator

sim = Simulator(m)
sim.set("a", 0xC3).set("b", 0x99).set("sel", 1)
sim.eval()                 # recompute combinational logic
print(sim.peek_outputs())   # {'sum': 0x15c, 'mask': 0x3, 'out': 0x81}
```

The simulator keeps track of inputs, wires, outputs, and registers, supports `eval()` for combinational updates, `step()` for clocked designs, and exposes helpers such as `peek`, `peek_next`, and signal watching for deeper inspection ([`spirehdl_simulator.py`](src/spirehdl/spirehdl_simulator.py)).

### 3. Integrate with external tooling

Modules can be exported to Verilog or AIG for downstream synthesis, equivalence checking, or integration into larger verification environments. Import helpers then let you bring optimized or third-party netlists back into SpireHDL for continued composition and simulation (see [`spirehdl_module.py`](src/spirehdl/spirehdl_module.py) and [`multipliers_ext_optimized.py`](src/spirehdl/arithmetic/int_multipliers/multipliers/multipliers_ext_optimized.py)).

## Modules and components in detail

- `Component` subclasses package reusable structures. They can materialize new modules (`to_module`), import designs from Verilog or AIG formats (`from_verilog`, `from_aag_lines`), and retag ports as internals (`make_internal`). Components also expose `get_spec()` to drive `IOCollector` regrouping when you import flattened designs (see [`spirehdl_module.py`](src/spirehdl/spirehdl_module.py)).
- `Module` is typically used at the top level or as an intermediate representation while you are still wiring a design. It offers constructors for inputs, outputs, wires, and registers; utilities for enumerating signals; Verilog emission with automatic width fitting; and a `module_analyze()` routine that reports combinational depth and node counts for timing exploration ([`spirehdl_module.py`](src/spirehdl/spirehdl_module.py)).
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

class Sum3Hierarchical(Component):
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

module = Sum3Hierarchical().to_module(name="Sum3Hier")
print(module.to_verilog())  # one top module, built from internal components
```

### Hierarchical design with components

Components are how to build hierarchy: instantiate one inside another, adapt its IO, or drop in a pre-synthesized netlist — all without leaving Python. A common pattern wraps a reusable SpireHDL block (see [`multipliers_ext.py`](src/spirehdl/arithmetic/int_multipliers/multipliers/multipliers_ext.py)). It is also possible to import an external AIG module, turn it into a `Component`, and call `from_module(..., make_internal=True)` so it behaves like a native SpireHDL block inside a larger generator (see [`multipliers_ext_optimized.py`](src/spirehdl/arithmetic/int_multipliers/multipliers/multipliers_ext_optimized.py)). The same approach covers Verilog imports, so SpireHDL code and external IP mix freely.

## Aggregate data types

SpireHDL includes structured, bit-packable aggregates for cleaner interfaces and bulk assignments ([`aggregate/`](src/spirehdl/aggregate)). See [`README_aggregate_types.md`](docs/README_aggregate_types.md) for the full reference with an example for every type:

- `HDLAggregate` defines the base "pack to bits" API that powers all aggregates ([`hdl_aggregate.py`](src/spirehdl/aggregate/hdl_aggregate.py)).
- `Array` offers N-dimensional indexing, packed assignment (`<<=`), and element-wise assignment (`@=`) for nested vectors or aggregates ([`aggregate_array.py`](src/spirehdl/aggregate/aggregate_array.py)).
- `AggregateRecord` lets you declare bundle-like classes with named fields that remain packable to a flat bitvector ([`aggregate_record.py`](src/spirehdl/aggregate/aggregate_record.py)).
- `AggregateRecordDynamic` is the dataclass-friendly variant whose fields are defined per-instance, ideal for parameterized IO records ([`aggregate_record_dynamic.py`](src/spirehdl/aggregate/aggregate_record_dynamic.py)).
- `FixedPoint` wraps a `Wire` or view with explicit total/frac widths and quantization helpers, keeping arithmetic readable while staying hardware-friendly ([`aggregate_fixed_point.py`](src/spirehdl/aggregate/aggregate_fixed_point.py)).
- `FloatingPoint` provides an IEEE-style view with `add`/`mul` helpers parameterized by exponent / fraction widths ([`aggregate_floating_point.py`](src/spirehdl/aggregate/aggregate_floating_point.py)).
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

For waveforms, set `sim.trace_enabled = True` before driving the design, then dump the captured trace to a VCD file:

```python
from spirehdl.various.vcd_writer import write_vcd

sim.trace_enabled = True
sim.eval()
for _ in range(5):
    sim.step()
write_vcd(trace_by_names=sim.get_trace_by_names(), filename="run.vcd", top_module=m.name, timescale="1ns")
```

## Slices
SpireHDL signals follow Python's indexing convention. For example, `sig[4:7]` creates a new expression made up of bits 4, 5, and 6 (counted from the LSB) of the original expression `sig`.

## Examples

Check out the `testing/examples/` directory for practical examples:

- **[`simple_component.py`](testing/examples/simple_component.py)** – A minimal example showing how to define a Component with IO ports and generate Verilog
- **[`component_example.py`](testing/examples/component_example.py)** – Comprehensive examples including hierarchical design and simulation
- **[`module_with_component.py`](testing/examples/module_with_component.py)** – Shows how to integrate Components within Module-based designs
- **[`direct_expression_basics.py`](testing/examples/direct_expression_basics.py)** – Minimal direct expression examples (`y = a + b`) plus `+`, `-`, unary `-`, `Const(..., Int(...))`, typed/plain `False`, and a recursive Horner polynomial builder
- **[`testing/riscv/rv32i.py`](testing/riscv/rv32i.py)** – Minimal RV32I core example; see [`testing/riscv/test_rv32i.py`](testing/riscv/test_rv32i.py) for simulation-based checks.

See the [examples README](testing/examples/README.md) for detailed documentation and key concepts.
