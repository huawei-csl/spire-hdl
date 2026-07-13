# Simulation and Debugging

Debugging and waveform viewing are available directly on the Spire/Python level, without
involving Verilog: the `Simulator` executes the elaborated expression graph, every
internal signal, register and ad-hoc expression can be probed on the live simulation, and
a Python run traces straight to a standard VCD file for any wave viewer.

The two levels in detail:

1. **Spire level**: [`Simulator`](../src/spire/simulator.py) executes the elaborated
   expression graph directly. No external tools, instant turnaround, and full visibility
   into every internal signal, register and expression.
2. **Verilog level**: emit the design as Verilog and replay the very same run as a generated,
   self-checking Verilog testbench under Verilator or Icarus, with VCD waveforms.

Per the [semantics reference](README_semantics.md), the Spire simulator is the value
specification of a design and the emitted Verilog is the temporal/reset specification; the
testbench flow below is how the two are cross-checked in practice.

## A design to poke at

A small multiply-accumulate block with one internal wire (`p`), one register (`r`) and
both visible as outputs:

```python
from spire import Component, IORecord, Input, Output, Register, Simulator, UInt, Wire

class MacDemo(Component):
    def __init__(self):
        self.io = IORecord(a=Input(UInt(4)), b=Input(UInt(4)),
                           prod=Output(UInt(8)), acc=Output(UInt(16)))
        self.elaborate()

    def elaborate(self):
        p = Wire(UInt(8), name="p")
        p <<= self.io.a * self.io.b
        r = Register(UInt(16), init=0, name="r")
        r <<= r + p
        self.io.prod <<= p
        self.io.acc <<= r
```

## Driving the Spire simulator

`Simulator` accepts a `Component` directly (combinational designs) or a netlist when you
want the framework clock and reset:

```python
net = MacDemo().to_netlist("mac", with_clock=True, with_reset=True)
sim = Simulator(net)

sim.reset()                    # assert rst: registers take their init values
sim.deassert_reset()

for av, bv in [(3, 5), (2, 2), (15, 15)]:
    sim.set("a", av).set("b", bv)
    sim.eval()                 # combinational settle: prod is already valid
    print("prod =", sim.get("prod"), " acc(pre-edge) =", sim.get("acc"))
    sim.step()                 # one clock edge: r <= r + p

assert sim.get("acc") == 3 * 5 + 2 * 2 + 15 * 15   # 244
```

`eval()` recomputes the combinational network for the current inputs; `step(n=1)` clocks
the registers (and memories) `n` times. `get(name, signed=True)` converts the raw bits of
a signed output for you.

## Inspecting a live simulation

Everything in the graph is observable, not just the ports:

```python
sim.list_signals()             # every input/output/wire/reg name, e.g. ['clk', 'rst', 'a', 'b', ..., 'p', 'r']

sim.peek("p")                  # internal wire, by name
sim.peek("r")                  # register state (pre-edge)
sim.peek_next("r")             # what r will hold AFTER the next step(): its next-state function
sim.peek_outputs()             # {'prod': ..., 'acc': ...} in one call
sim.peek_inputs()              # the currently applied inputs

# peek() also takes expression objects, so ad-hoc probes need no design change:
mac = net.component
sim.peek(mac.io.a + mac.io.b)  # evaluate an expression that is not part of the design
```

For repeated probes, register a **watch**: it is captured at every `eval()`/`step()` and
read back by name.

```python
sim.watch("p", alias="product")
sim.set("a", 7).set("b", 9).eval()
assert sim.get_watch("product") == 63
```

Designs containing memories expose their storage as well: `sim.get_mem("mem_name")`
returns the array contents (see [README_memories.md](README_memories.md)).

## Waveforms from a Spire run

The simulator can trace every expression while it runs; the trace converts into a
standard VCD file with [`write_vcd`](../src/spire/various/vcd_writer.py). No Verilog and no external
tools are involved:

```python
from spire.various.vcd_writer import write_vcd

net = MacDemo().to_netlist("mac", with_clock=True, with_reset=True)
sim = Simulator(net)
sim.trace_enabled = True       # snapshot all traced expressions at every eval()/step()

sim.reset()
sim.deassert_reset()
for av, bv in [(3, 5), (2, 2), (15, 15), (1, 1)]:
    sim.set("a", av).set("b", bv).eval()
    sim.step()

write_vcd(sim.get_trace_by_names(), "mac_python.vcd", top_module="mac")
```

By default every expression in the design is traced (`sim.traced_expressions`); narrow it
to specific signals to keep the file small, e.g.
`sim.traced_expressions = [mac.io.acc, mac.io.prod]` before the run.

View the file in any wave viewer, for example [surfer](https://surfer-project.org) (a
modern open-source viewer) or GTKWave:

```sh
surfer mac_python.vcd
gtkwave mac_python.vcd
```

## Replaying the run as a Verilog testbench

[`TestbenchGenSimulator`](../src/spire/verilog_testbench.py) is a drop-in `Simulator`
replacement: drive it with the same `set`/`eval`/`step`/`reset` calls (it simulates live,
so `get()` works as usual) and it records every interaction. Afterwards it emits a
self-checking Verilog testbench that replays the exact run against the emitted Verilog and
compares each output against what the Spire simulator produced:

```python
from spire.verilog_testbench import TestbenchGenSimulator

net = MacDemo().to_netlist("mac", with_clock=True, with_reset=True)
tb = TestbenchGenSimulator(net)

tb.reset()
tb.deassert_reset()
for av, bv in [(3, 5), (2, 2), (15, 15)]:
    tb.set("a", av).set("b", bv).eval()
    tb.step()

net.to_verilog_file("mac.v")                            # the DUT
tb.to_testbench_file("mac_tb.v", tb_module_name="mac_tb")   # the recorded run as a testbench
```

Both simulators implement the same shared API (the `SimulatorBase` base class), and
`TestbenchGenSimulator` delegates every call to an internal real `Simulator`, so the whole
toolbox from above works identically while the run is being recorded: the inspection calls
(`peek`, `peek_next`, `peek_inputs`/`peek_outputs`, watches, `list_signals`, `get_mem`)
and the trace/VCD capture too. Set `tb.trace_enabled = True` before driving and one
recorded run yields all three artifacts: the Spire-side VCD
(`write_vcd(tb.get_trace_by_names(), ...)`), the DUT, and the replaying testbench.

The testbench dumps waveforms by default (`$dumpfile("dump.vcd")`; configurable via the
`dump_vcd`/`dumpfile` arguments). For vector-table driving, checked suites and the
data-driven testbench style (external `.dat` stimulus files), see
[README_verilog_testbench.md](README_verilog_testbench.md).

## Running the testbench under a real Verilog simulator

To compile and run the emitted testbench (Icarus/Verilator commands, `dump.vcd`), see
[README_verilog_testbench.md](README_verilog_testbench.md). The payoff of the recorded
flow: with `tb.trace_enabled = True` the SAME run yields the Spire-model VCD (via
`write_vcd`) and the Verilog VCD (`dump.vcd`), replaying identical stimuli. Opened side
by side (e.g. two surfer windows), any divergence between the Spire model and the emitted
Verilog shows up directly in the waves.

## See also

- [README_verilog_testbench.md](README_verilog_testbench.md): testbench styles in depth
  (inline vs data-driven), vector tables, `run_vectors_on_simulator`.
- [README_semantics.md](README_semantics.md): what the Spire simulator specifies vs what
  the emitted Verilog specifies (values vs temporal/reset behavior, power-on conventions).
- [README_memories.md](README_memories.md): simulating RAM/ROM/FIFO primitives and
  reading their storage.
- [`testing/low_level_arithmetic/int_adders/int_adder_tb_sim.py`](../testing/low_level_arithmetic/int_adders/int_adder_tb_sim.py):
  an end-to-end script (DUT, inline and data-driven testbenches, VCD).
