# Verilog Testbench

`TestbenchGenSimulator` is a drop-in replacement for the Python
[`Simulator`](../src/spirehdl/spirehdl_simulator.py): it has the same
`set` / `get` / `eval` / `step` / `reset` API, but while it runs it records the
stimuli (and the outputs the Python simulator observed) and can then emit a
**synthesizable Verilog testbench** that replays the exact same interactions
against the generated RTL — handy for cross-checking the emitted Verilog under
Verilator or Icarus.

## Quick start

`TestbenchGenSimulator` is driven exactly like the simulator: set inputs,
`eval()` (or `step()`), read outputs — and it records every interaction. When the
trace is complete, write out the DUT and a testbench that replays it:

```python
from spirehdl.spirehdl_module import Module
from spirehdl.spirehdl import UInt
from spirehdl.spirehdl_verilog_testbench import TestbenchGenSimulator

m = Module("Add8", with_clock=False, with_reset=False)
a = m.input(UInt(8), "a")
b = m.input(UInt(8), "b")
y = m.output(UInt(9), "y")
y <<= a + b

tb = TestbenchGenSimulator(m)
for av, bv in [(1, 2), (100, 50), (255, 1)]:
    tb.set("a", av).set("b", bv).eval()      # same calls as Simulator
    print(tb.get("y"))                        # 3, 150, 256 — live simulation output

m.to_verilog_file("add8.v")                  # the DUT
tb.to_testbench_file("add8_tb.v", tb_module_name="Add8_tb")
```

Evaluation is delegated to a real Python `Simulator`, so the `tb` object is also a
live simulator — `tb.get(...)` / `tb.peek_outputs()` return current values as you
drive it. One object both exercises the design and emits the testbench; you don't
need a separate `Simulator(m)`.

The emitted `add8_tb.v` instantiates the DUT, applies each recorded stimulus, and
checks the outputs the Python simulator produced, finishing with `$finish`. For
clocked designs use `step()` (and `reset()`) just like the simulator; each
`step()` is lowered to a clock cycle.

## Driving from a vector table

For a checked test suite, `run_vectors_on_simulator` applies a table of
`(label, inputs, expected)` vectors, verifies each output, and raises on a
mismatch — and those same expected-value checks are baked into the emitted
testbench:

```python
from spirehdl.helpers import run_vectors_on_simulator

vectors = [
    ("1+2",    {"a": 1,   "b": 2},  {"y": 3}),
    ("100+50", {"a": 100, "b": 50}, {"y": 150}),
    ("255+1",  {"a": 255, "b": 1},  {"y": 256}),
]

tb = TestbenchGenSimulator(m)
run_vectors_on_simulator(tb, vectors, test_name="Add8")   # drives + checks, and records
tb.to_testbench_file("add8_tb.v", tb_module_name="Add8_tb")
```

It works on any `SimulatorBase`: pass a plain `Simulator(m)` to just verify, or a
`TestbenchGenSimulator(m)` (as here) to verify *and* record. The arithmetic
generator relies on exactly this interchangeability — see `_apply_actions` in
[`arithmetic_generator.py`](../src/spirehdl/arithmetic/arithmetic_generator.py),
which picks `TestbenchGenSimulator(module)` when a testbench output is requested
and `Simulator(module)` otherwise, then runs the same `run_vectors_on_simulator`
loop through whichever it chose.

The emitted `add8_tb.v` instantiates the DUT, applies each stimulus, and checks
the expected outputs, finishing with `$finish`.
For clocked designs, use `step()` (and `reset()`) just like the simulator; each
`step()` is lowered to a clock cycle in the testbench.

`to_testbench_str(...)` returns the same text as a string instead of writing a
file. Both take an optional `timescale` and a `dump_vcd` / `dumpfile` pair
(on by default) so the simulator run also produces a waveform.

## Testbench styles: inline vs. data-driven

`to_testbench_file` (above) emits a **self-contained** testbench: every stimulus
and its expected response is unrolled directly into the Verilog. That's convenient
for a handful of vectors but bloats the file for large suites.

In **data-driven** mode the vectors live in an external, whitespace-separated
`.dat` file and the testbench streams them in at runtime (via `$fopen` / `$fgets`),
checking each row as it goes. The `.dat` has one row per vector — inputs followed
by expected outputs — under a header naming the columns:

```text
# a b y
1 2 3
100 50 150
255 1 256
```

Emit the data file and a testbench that reads it, either in two steps:

```python
from spirehdl.spirehdl_verilog_testbench import write_vector_data_file

write_vector_data_file(vectors, "add8_vectors.dat")
tb.to_data_driver_testbench_file("add8_tb.v", data_file="add8_vectors.dat", with_clk=False)
```

…or in a single call that writes both the testbench and its `.dat`:

```python
tb.to_data_driver_testbench_file_incl_dat("add8_tb.v", vectors, "add8_vectors.dat", with_clk=False)
```

See [`testing/low_level_arithmetic/int_adders/int_adder_tb_sim.py`](../testing/low_level_arithmetic/int_adders/int_adder_tb_sim.py)
for an end-to-end example (DUT + inline testbench + data-driven testbench + VCD).
