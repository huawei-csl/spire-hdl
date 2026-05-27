# Custom Verilog Components

SpireHDL's `Component` can opt to emit a hand-written Verilog block instead of
the auto-generated logic derived from its `elaborate()`. Two flavours are
supported:

- **Custom-with-sim**: the Component has both an `elaborate()` (Python sim
  model) and a `custom_verilog()` (hand-tuned Verilog). Simulation runs the
  Python; synthesis sees the user-supplied Verilog. Useful for hand-tuned
  RTL, external tool output, or anything where you want a different
  Verilog shape than SpireHDL would generate.
- **Blackbox**: the Component has only `custom_verilog()` — `elaborate()`
  is empty. There's no Python sim model, so the simulator stubs the
  outputs to 0. Useful for vendor IP, opaque macros (PLLs, SerDes, RAM
  macros), or any block where a Python model isn't practical.

The mechanism lives in [`spirehdl/spirehdl_module.py`](src/spirehdl/spirehdl_module.py)
(Component-side tagging) and [`spirehdl/spirehdl_simulator.py`](src/spirehdl/spirehdl_simulator.py)
(sim stub for blackboxes).

## Quick start — Component with both sim model and custom Verilog

Define a Component that adds two 8-bit numbers. The `elaborate()` provides
a Python reference model (used by `Simulator`); `custom_verilog()` returns
a hand-written Verilog block (used by `to_verilog`).

```python
from dataclasses import dataclass
from spirehdl.spirehdl import Bool, Signal, UInt, Wire
from spirehdl.spirehdl_module import Component


class CustomAdder(Component):
    def __init__(self):
        @dataclass
        class IO:
            a: Signal
            b: Signal
            sum: Signal
        self.io = IO(
            a   = Signal("a",   UInt(8), "input"),
            b   = Signal("b",   UInt(8), "input"),
            sum = Signal("sum", UInt(9), "output"),
        )
        self.elaborate()

    def elaborate(self):
        # Python sim model — used by Simulator(m). Free to use any spire-hdl
        # primitives (Registers, Wires, Memory, etc.).
        tmp = Wire(UInt(9))
        tmp <<= self.io.a + self.io.b
        self.io.sum <<= tmp

    def custom_verilog(self) -> str:
        # Hand-written Verilog — used by m.to_verilog(). The Signal names are
        # resolved at emit time, so uniquification (e.g. `a_1` if `a` collides
        # with a parent port) is already applied.
        return (
            f"  // hand-tuned adder\n"
            f"  assign {self.io.sum.name} = {self.io.a.name} + {self.io.b.name};"
        )
```

Compile and simulate:

```python
from spirehdl.spirehdl_simulator import Simulator

comp = CustomAdder()
m    = comp.to_module(name="CustomAdder", with_clock=False, with_reset=False)

# Verilog: the elaborate's `tmp` wire is suppressed; the hand-tuned block emits.
print(m.to_verilog())

# Simulation: uses elaborate's Python model.
sim = Simulator(m)
sim.set("a", 5).set("b", 7).eval()
assert sim.get("sum") == 12
```

Emitted Verilog (abbreviated):

```verilog
module CustomAdder (a, b, sum);
  input  [7:0] a, b;
  output [8:0] sum;
  // hand-tuned adder
  assign sum = a + b;
endmodule
```

No `tmp` wire — the framework tagged it `_no_emit_decl + _no_emit_drive` at
`to_module` time so the emitter skips it.

## Quick start — blackbox (only Verilog, no Python model)

For vendor IP or opaque macros, leave `elaborate()` empty:

```python
class VendorRAM(Component):
    def __init__(self):
        @dataclass
        class IO:
            clk: Signal
            addr: Signal
            data_in: Signal
            we: Signal
            data_out: Signal
        self.io = IO(
            clk      = Signal("clk",      Bool(),     "input"),
            addr     = Signal("addr",     UInt(10),   "input"),
            data_in  = Signal("data_in",  UInt(32),   "input"),
            we       = Signal("we",       Bool(),     "input"),
            data_out = Signal("data_out", UInt(32),   "output"),
        )
        self.elaborate()

    def elaborate(self):
        # Deliberately empty — this is a blackbox.
        pass

    def custom_verilog(self) -> str:
        return """
  // -- vendor 1024×32 single-port RAM macro --
  reg [31:0] mem [0:1023];
  always @(posedge clk) if (we) mem[addr] <= data_in;
  assign data_out = mem[addr];
"""
```

Compile and use:

```python
ram = VendorRAM()
m   = ram.to_module(name="VendorRAM", with_clock=True)
print(m.to_verilog())   # contains the hand-written RAM macro

# Simulation: the blackbox has no Python model — outputs read as 0.
sim = Simulator(m)
sim.set("addr", 5).set("data_in", 0xDEADBEEF).set("we", 1).step()
assert sim.get("data_out") == 0   # stub
```

## Embedding inside a larger Component

A custom-Verilog or blackbox Component slots into a parent Component via
`make_internal()` — same pattern as any other sub-Component:

```python
class TopWithVendorRAM(Component):
    def __init__(self):
        @dataclass
        class IO:
            clk: Signal
            x: Signal
            y: Signal
            result: Signal
        self.io = IO(
            clk    = Signal("clk",    Bool(),    "input"),
            x      = Signal("x",      UInt(10),  "input"),
            y      = Signal("y",      UInt(32),  "input"),
            result = Signal("result", UInt(32),  "output"),
        )
        self.elaborate()

    def elaborate(self):
        ram = VendorRAM().make_internal()
        ram.io.clk      <<= self.io.clk
        ram.io.addr     <<= self.io.x
        ram.io.data_in  <<= self.io.y
        ram.io.we       <<= 1
        self.io.result  <<= ram.io.data_out
```

`m.to_verilog()` produces a single flat module containing both the parent's
auto-emitted glue and the vendor RAM's custom block. The parent's `assign`
lines connect the parent inputs to the RAM's port wires; the vendor block
provides the RAM's behaviour.

## How it works

Two flags on `Signal`:

- `_no_emit_decl`: skip the wire/reg/mem declaration.
- `_no_emit_drive`: skip the `assign` or always-block update.

When `Component.to_module` (top-level) or `Component.make_internal`
(embedded) detects a `custom_verilog` method, it runs
`_apply_custom_verilog_tags()`:

1. Walks from each IO output through `_driver` chains.
2. Tags every reachable internal Signal with both flags (those drop out of
   the emitted Verilog entirely).
3. Tags IO outputs with only `_no_emit_drive` — their declaration stays so
   parent code can reference them, but their `assign` is suppressed.
4. Sets `self._is_blackbox = True` iff *no* output had an elaborate-set
   driver — used by the collector to decide whether the parent's
   input-side wiring needs explicit peer-seeding.

At emit time, `Module.to_verilog_lines`:

- Skips signals tagged `_no_emit_decl` in the wire/reg/mem decl loops.
- Skips signals tagged `_no_emit_drive` in the combinational-assigns loop
  and the sequential always block.
- Collects custom blocks (top-level via `module.component`, embedded via
  `_owning_component` back-edges on IO wires), dedupes by Component id,
  and emits them after the auto-emitted body. The custom blocks see the
  finalized port-wire names because `custom_verilog()` is called at emit
  time, after `collect_signals` has uniquified everything.

For **blackboxes specifically**, one extra mechanism is needed: the
collector's `visit_signal` seeds from peer IO wires when visiting any IO
wire of a blackbox Component. Without this, the parent's logic feeding
the blackbox's inputs would be unreachable (the blackbox's outputs have
no driver chain back to its inputs). The check is one attribute read —
`getattr(owner, "_is_blackbox", False)` — set by the tag function at
construction time.

For **simulation of blackbox outputs**, `_eval_signal_bits` recognises the
signature `_no_emit_drive=True AND _driver=None` and returns 0 instead of
raising. The contract is documented as "blackbox outputs read as 0 in
simulation, regardless of inputs."

## Caveats and contracts

- **Port-name matching.** The custom Verilog references Signal names by
  attribute (`self.io.foo.name`). Names are resolved at emit time, so
  uniquification produces matching names on both sides. If you hardcode
  names like `"sum"` in the custom string instead of using
  `self.io.sum.name`, a naming collision in the parent will silently
  produce broken Verilog.
- **Sim ↔ synth equivalence.** The framework cannot check that the Python
  `elaborate()` and the custom Verilog describe the same hardware. If
  they diverge, sim and synth disagree. For non-blackbox use, treat the
  Python as the reference and the Verilog as the optimised version — and
  cover both via tests.
- **Blackbox sim returns 0.** Not a bug — by design. If your test
  depends on a meaningful value from a blackbox, write an `elaborate()`
  reference model alongside the `custom_verilog()` (becomes the
  non-blackbox flavour) or stub the blackbox's outputs explicitly via
  the simulator's `set()` on the relevant inputs.
- **Hierarchical emission is not supported.** Each custom block is
  inlined into the parent module's body — there is no `module … endmodule
  + instantiation` per sub-Component. If you need that pattern (e.g. for
  multi-instance vendor IP that synthesizers prefer to instantiate by
  name), wrap the parent Verilog manually after `to_verilog`.

## Real-world examples: the primitives library

The `src/spirehdl/primitives/` package contains production-style uses of
the custom-with-sim pattern — both with non-trivial `elaborate()` reference
models and Yosys-friendly `custom_verilog()` outputs:

- [`MemoryPrimitive`](src/spirehdl/primitives/primitive_memory.py) —
  array-of-registers RAM with optional registered read, reset arm, and
  init values. Supports aggregate element types via user-side
  `to_bits`/`from_bits` at the port boundary. Emits the standard Yosys-
  inferable `reg [W-1:0] mem[0:D-1];` idiom in `custom_verilog()`.
- [`FIFOPrimitive`](src/spirehdl/primitives/primitive_fifo.py) — standard
  synchronous FIFO (push/pop/full/empty/count, registered head). Inlines
  its storage and pointer/count logic in a single `custom_verilog()`
  block.

Both are good templates for new custom-Verilog Components: they exercise
sequential state in `elaborate()` (Registers + mux trees), aggregate-type
support, and per-instance internal-name uniquification so multiple
instances inside the same parent don't collide on inlined names.

A standalone analysis of when to use this pattern vs. baking storage into
the SpireHDL core lives in
[`docs/memory_primitive_vs_builtin.md`](docs/memory_primitive_vs_builtin.md)
— useful background if you're deciding between "build it as a primitive"
and "extend the core".

## See also

- Tests covering both flavours:
  [`testing/test_component_custom_verilog.py`](testing/test_component_custom_verilog.py)
  (non-blackbox cases including embedded use, multiple instances, and
  sequential logic) and
  [`testing/test_blackbox_component.py`](testing/test_blackbox_component.py)
  (top-level, embedded, multi-instance blackboxes; the walker-reachability
  fix is exercised in `test_embedded_blackbox_parent_helper_is_reachable`).
- Primitives tests:
  [`testing/test_primitive_memory.py`](testing/test_primitive_memory.py)
  (10 cases including an `AggregateRecord` element-type round-trip) and
  [`testing/test_primitive_fifo.py`](testing/test_primitive_fifo.py)
  (8 cases including underflow/overflow safety and aggregate element types).
