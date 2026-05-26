# Memories

SpireHDL's `Memory` is the primitive for arrays of registers — FIFOs, ROMs,
scratchpad RAMs, lookup tables. It emits as a Verilog `reg [W-1:0] name[0:D-1];`
declaration plus an always block, in the shape that yosys's `memory` pass
recognises (so `memory_dff`, `memory_share`, `memory_bmux2rom` fire automatically).

The module lives in [`spirehdl/spirehdl.py`](src/spirehdl/spirehdl.py).

## Quick intro

A `Memory` is a `Signal` whose attributes are themselves Signals — the read /
write / reset ports. You wire those ports with `<<=` just like any other signal.
There are no `write()` / `reset()` / `registered_read()` methods; everything is
plain port wiring.

```python
from spirehdl.spirehdl import Bool, Memory, UInt
from spirehdl.spirehdl_module import Module

m = Module("scratch", with_reset=True)
we     = m.input(Bool(), "we")
clr    = m.input(Bool(), "clr")
addr_w = m.input(UInt(4), "addr_w")
addr_r = m.input(UInt(4), "addr_r")
din    = m.input(UInt(9), "din")
dout   = m.output(UInt(9), "dout")

ram = Memory(UInt(9), depth=16)        # name "ram" inferred from variable
ram.write_addr   <<= addr_w
ram.write_data   <<= din
ram.write_enable <<= we
ram.reset_enable <<= clr     # broadcast-clear all entries when high
# ram.reset_value defaults to Const(0) — override only for a non-zero clear value.
ram.read_addr    <<= addr_r
dout             <<= ram.read_data
```

That's the whole API. The shape above — single write port + optional
broadcast-clear + async read — covers most scratchpad-RAM use cases. (Building
a proper FIFO needs additional head/tail pointer logic on top; `Memory` is
just the storage element.)

## Ports

| Port | Direction | Type | Default | Required if… |
|---|---|---|---|---|
| `write_addr`   | in  | `UInt(clog2(depth))`  | —          | a write port is active |
| `write_data`   | in  | `elem_type`           | —          | a write port is active |
| `write_enable` | in  | `Bool()`              | `Const(1)` | optional gating |
| `reset_value`  | in  | `elem_type`           | `Const(0)` | optional |
| `reset_enable` | in  | `Bool()`              | —          | broadcast-clear arm |
| `read_addr`    | in  | `UInt(clog2(depth))`  | —          | `read_data` is consumed |
| `read_data`    | out | `elem_type`           | (derived)  | — |
| `read_enable`  | in  | `Bool()`              | `Const(1)` | only present with `registered_read=True` |

Defaults are applied at construction; override any of them with `<<=`. The
write port is "active" if `write_addr` is driven; both `write_addr` and
`write_data` must then be driven (validation raises otherwise). The reset arm
is active if `reset_enable` is driven.

## Examples

### Read-only ROM with initial values

```python
m = Module("rom8")
addr = m.input(UInt(3), "addr")
dout = m.output(UInt(8), "dout")

rom = Memory(UInt(8), depth=8,
             init=[0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80])
rom.read_addr <<= addr
dout          <<= rom.read_data
```

The `init=[…]` emits as a Verilog `initial begin name[i] = …; end` block.

### Synchronous read (one-cycle latency)

Pass `registered_read=True` to get a clocked output. Read latency = 1 cycle.

```python
re   = m.input(Bool(), "re")
rom  = Memory(UInt(8), depth=8, registered_read=True,
              init=[0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80])
rom.read_addr   <<= addr
rom.read_enable <<= re        # default is 1 (always-read)
dout            <<= rom.read_data
```

The `read_enable` port lets you hold the rdata register's previous value when
low — same idiom as a typical block-RAM `re` input.

### A real FIFO as a Component

`Memory` is just the storage element — a working FIFO also needs head/tail
pointers and a count register. Here it is as a SpireHDL
[`Component`](src/spirehdl/spirehdl_module.py), which makes the FIFO reusable
and parameterisable in depth and element type.

```python
from dataclasses import dataclass
from spirehdl.spirehdl import Bool, Memory, Register, Signal, UInt, mux
from spirehdl.spirehdl_module import Component


class Fifo(Component):
    """Synchronous FIFO. Storage is a Memory; control is head/tail/count regs.

    Depth must be a power of two so that pointer increments wrap naturally
    (the address signal is `clog2(depth)` bits wide, so `tail + 1` wraps at
    `depth` with no explicit modulo).
    """

    def __init__(self, elem_type, depth):
        if depth & (depth - 1):
            raise ValueError("FIFO depth must be a power of two")
        self.elem_type, self.depth = elem_type, depth
        self._addr_t  = UInt(max(1, (depth - 1).bit_length()))
        self._count_t = UInt(depth.bit_length())   # holds 0..depth inclusive

        @dataclass
        class IO:
            push:  Signal
            pop:   Signal
            din:   Signal
            dout:  Signal
            full:  Signal
            empty: Signal

        self.io = IO(
            push  = Signal("push",  Bool(),    "input"),
            pop   = Signal("pop",   Bool(),    "input"),
            din   = Signal("din",   elem_type, "input"),
            dout  = Signal("dout",  elem_type, "output"),
            full  = Signal("full",  Bool(),    "output"),
            empty = Signal("empty", Bool(),    "output"),
        )
        self.elaborate()

    def elaborate(self):
        head  = Register(self._addr_t,  init=0)
        tail  = Register(self._addr_t,  init=0)
        count = Register(self._count_t, init=0)

        # Flags from count.
        self.io.empty <<= count == 0
        self.io.full  <<= count == self.depth

        # Guarded request signals — ignore push when full, pop when empty.
        do_push = self.io.push & ~self.io.full
        do_pop  = self.io.pop  & ~self.io.empty

        # Storage: write at `tail`, async-read at `head`.
        store = Memory(self.elem_type, depth=self.depth)
        store.write_addr   <<= tail
        store.write_data   <<= self.io.din
        store.write_enable <<= do_push
        store.read_addr    <<= head
        self.io.dout       <<= store.read_data

        # Pointer + count next-state. The `+1` wraps naturally for power-of-two depth.
        head  <<= mux(do_pop,  head + 1, head)
        tail  <<= mux(do_push, tail + 1, tail)
        count <<= mux(do_push & ~do_pop, count + 1,
                  mux(~do_push & do_pop, count - 1, count))
```

Instantiate and use:

```python
from spirehdl.spirehdl_simulator import Simulator

fifo   = Fifo(UInt(8), depth=4)
module = fifo.to_module(name="Fifo8x4", with_clock=True, with_reset=True)

# Verilog
verilog = module.to_verilog()

# Simulation
sim = Simulator(module)
sim.deassert_reset()
for x in (0xA, 0xB, 0xC):
    sim.set("push", 1).set("din", x).step()      # push 3 values
sim.set("push", 0)
assert sim.get("empty") == 0 and sim.get("full") == 0

for expected in (0xA, 0xB, 0xC):
    assert sim.get("dout") == expected           # async read at `head`
    sim.set("pop", 1).step()
sim.set("pop", 0)
assert sim.get("empty") == 1
```

The Memory's reset arm isn't used here — FIFO reset is achieved by clearing
the pointers and count via their `init=0`, which is enough because `count`
tracks the number of valid entries. The Module's `with_reset=True` reset
auto-clears every internal Register to its `init`; the Memory storage itself
is not affected. To also clear the storage on reset, add an explicit
`rst: Signal` field to the Component's `IO` and wire
`store.reset_enable <<= self.io.rst`.

## Simulation

Memories work seamlessly with the built-in simulator. The Memory state lives
inside the simulator and is updated on each `step()`.

```python
from spirehdl.spirehdl_simulator import Simulator

sim = Simulator(m)
sim.deassert_reset()

# Drive writes
sim.set("we", 1).set("addr_w", 3).set("din", 0xAB).step()
sim.set("addr_w", 5).set("din", 0xCD).step()
sim.set("we", 0)

# Drive an async read
sim.set("addr_r", 3).eval()
assert sim.get("dout") == 0xAB
```

### Inspecting current memory state

Use `Simulator.get_mem(...)` to read the full array as a Python list. You can
pass either the Memory name (string) or the Memory object itself:

```python
sim.get_mem("ram")             # → [0, 0, 0, 0xAB, 0, 0xCD, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
sim.get_mem(ram)               # same, by reference
```

The returned list is a copy — mutating it does not affect simulation state.
Each entry is an unsigned bit-pattern of width `mem.typ.width`.

This is useful for assertions in tests, post-step debugging, or capturing
periodic memory snapshots in a longer simulation.

### Async vs. sync read in sim

- **Async** (`registered_read=False`, default): `mem.read_data` reflects
  `mem[read_addr]` combinationally. `sim.eval()` after setting `addr_r`
  is enough — no `step()` needed.
- **Sync** (`registered_read=True`): `mem.read_data` updates on the next
  clock edge; you must `step()` to advance, and the value lags `read_addr`
  by one cycle.

### Write-before-read on the same address

If a write and a registered read target the same address in the same cycle,
the rdata register captures the **old** value (pre-write). This matches the
Verilog non-blocking semantics emitted by the always block.

## Behaviour notes

- **Address width** is inferred as `max(1, ceil(log2(depth)))`. For
  `depth=1` you still get a 1-bit address.
- **Port wire names** in the emitted Verilog are `{mem.name}__waddr`,
  `{mem.name}__we`, etc. They appear as identity `assign` lines connecting
  your driver expressions to the always block (yosys folds them away
  during synthesis).
- **Memory directly in an expression raises.** Writing `out <<= mem` (without
  `.read_data`) is a user mistake; SpireHDL raises a clear `RuntimeError`
  instead of producing invalid Verilog like `assign out = mem;`.
- **Single port per kind (MVP).** One write port, one sync-read port, one
  reset arm. Async reads through `read_data` are also single-port. Multi-port
  support is a planned extension.
- **Reset wins over write.** If both `reset_enable` and `write_enable` are
  high in the same cycle, the broadcast clear takes priority — this matches
  the `if (reset) … else if (we) …` idiom yosys recognises.

## See also

- The FIFO example in
  [`benchmarks/dr_rtl_spirehdl/router/context/starting_point.py`](benchmarks/dr_rtl_spirehdl/router/context/starting_point.py)
  (16 × 9-bit FIFO).
- Tests in [`testing/test_memory.py`](testing/test_memory.py) cover Verilog
  emission, validation, and simulation patterns.
- The design rationale and the previous-vs-current API comparison live in
  [`docs/memory_refactor.md`](docs/memory_refactor.md) and
  [`docs/memory_design_vs_magma.md`](docs/memory_design_vs_magma.md).
