# Memories

Memory in Spire is provided by **Component primitives** — you instantiate one,
wire its `.io` ports, and embedding into the parent is handled automatically. Four primitives cover
the common cases:

| Primitive | Use for | Shape |
|---|---|---|
| [`MemoryPrimitive`](../src/spire/primitives/primitive_memory.py) | scratchpad RAM, single-port BRAM | 1 write + 1 read (async or registered), optional reset arm / write mask |
| [`RamPrimitive`](../src/spire/primitives/primitive_ram.py) | multi-port RAM, true dual-port, register files | N write + N read + N read/write (`rw`) ports over one array |
| [`RomPrimitive`](../src/spire/primitives/primitive_rom.py) | read-only memory, lookup tables | init-backed, 1 read (async or registered), no write port |
| [`FIFOPrimitive`](../src/spire/primitives/primitive_fifo.py) | ready-made synchronous FIFO | push / pop / full / empty / count |

```python
from spire.primitives import MemoryPrimitive, RamPrimitive, RomPrimitive, FIFOPrimitive
```

Each primitive emits its storage as a Verilog `reg [W-1:0] name[0:D-1];` array plus a
clock-only `always` block — the shape yosys's `memory` pass recognises (so `memory_dff`,
`memory_share`, `memory_bmux2rom` fire automatically). For simulation they are backed by
the core's O(1) `_MemoryArray` (an internal, sim-only storage object you never touch
directly). The two paths describe the same hardware; the tests pin the equivalence.

> **Why primitives, not a built-in `Memory` class?** The core was deliberately kept lean:
> all Verilog emission, port shapes, and flavors live in user-space primitives; the core
> only provides fast simulation storage. There is no user-facing `Memory` class — use the
> primitives below.

## MemoryPrimitive — single-port RAM / ROM

```python
from dataclasses import dataclass
from spire.expr import Bool, Signal, UInt
from spire.component import Component
from spire.primitives import MemoryPrimitive


class Scratch(Component):
    def __init__(self):
        @dataclass
        class IO:
            we: Signal; aw: Signal; ar: Signal; din: Signal; dout: Signal
        self.io = IO(
            we   = Signal(typ=Bool(), kind="input"),
            aw   = Signal(typ=UInt(4), kind="input"),
            ar   = Signal(typ=UInt(4), kind="input"),
            din  = Signal(typ=UInt(9), kind="input"),
            dout = Signal(typ=UInt(9), kind="output"),
        )
        self.elaborate()

    def elaborate(self):
        mem = MemoryPrimitive(UInt(9), depth=16, name="ram")
        mem.io.write_addr   <<= self.io.aw
        mem.io.write_data   <<= self.io.din
        mem.io.write_enable <<= self.io.we
        mem.io.read_addr    <<= self.io.ar
        self.io.dout        <<= mem.io.read_data

module = Scratch().to_netlist(name="scratch", with_clock=True, with_reset=True)
```

### Constructor

```python
MemoryPrimitive(elem_type, depth, *,
                init=None,                  # list[int] length==depth → ROM initial values
                registered_read=False,      # True → 1-cycle registered read + read_enable port
                with_reset_arm=False,       # True → reset_enable port broadcast-clears all cells
                reset_value=0,              # static clear value used by the reset arm
                mask_chunks=0,              # >1 → per-chunk write mask (byte/sub-word enables)
                read_under_write="readFirst",  # or "writeFirst" (async read) / "dontCare"
                name=None)
```

`elem_type` is an `HDLType` (`UInt`/`SInt`/`Bool`) **or** an `HDLComposite` — see
[Composite element types](#composite-element-types).

### Ports (`mem.io.*`)

| Port | Dir | Type | Present when |
|---|---|---|---|
| `write_addr`   | in  | `UInt(clog2(depth))` | always |
| `write_data`   | in  | `UInt(elem_w)`       | always |
| `write_enable` | in  | `Bool()`             | always |
| `read_addr`    | in  | `UInt(clog2(depth))` | always |
| `read_data`    | out | `UInt(elem_w)`       | always |
| `write_mask`   | in  | `UInt(mask_chunks)`  | `mask_chunks > 1` |
| `reset_enable` | in  | `Bool()`             | `with_reset_arm=True` |
| `read_enable`  | in  | `Bool()`             | `registered_read=True` |

### ROM with init + registered read

```python
rom = MemoryPrimitive(UInt(8), depth=8, registered_read=True,
                      init=[0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80],
                      name="rom")
rom.io.read_addr   <<= self.io.addr
rom.io.read_enable <<= self.io.re      # hold the registered output when low
self.io.dout       <<= rom.io.read_data
# write port is required at the boundary; tie it off for a pure ROM:
rom.io.write_addr <<= Const(0, UInt(3))
rom.io.write_data <<= Const(0, UInt(8))
rom.io.write_enable <<= Const(0, Bool())
```

`init=[…]` emits a Verilog `initial begin name[i] = …; end` block. `registered_read=True`
gives one cycle of read latency; `read_enable` low holds the previous output.

### Write mask (byte / sub-word enables)

`mask_chunks=N` splits each word into `N` equal chunks; `write_mask[i]` enables chunk `i`
(read-modify-write — unmasked chunks keep their old value).

```python
mem = MemoryPrimitive(UInt(16), depth=4, mask_chunks=2, name="mr")
mem.io.write_mask <<= self.io.byte_en   # 2-bit: bit0=low byte, bit1=high byte
```

### Read-under-write

`read_under_write="writeFirst"` forwards the just-written data onto a same-address async
read (the BRAM WRITE_FIRST idiom). `readFirst` (default) returns the old value; `dontCare`
emits readFirst and lets synthesis pick. *writeFirst is async-read only* — combine with the
default `registered_read=False`.

## RamPrimitive — multi-port / true dual-port

Multiple read/write ports and read/write (`rw`) ports over **one shared array** — the
shapes `MemoryPrimitive` can't express.

```python
RamPrimitive(elem_type, depth, *,
             num_read_ports=1, num_write_ports=1, rw_ports=0,
             mask_chunks=0, read_under_write="readFirst",
             registered_read=False, init=None, name=None)
```

Ports are **indexed** (`k` from 0):

| Port group | Names |
|---|---|
| write `k` | `w{k}_addr` `w{k}_data` `w{k}_en` (+ `w{k}_mask`) |
| read `k`  | `r{k}_addr` `r{k}_data` (+ `r{k}_en` if `registered_read`) |
| rw `k`    | `rw{k}_addr` `rw{k}_din` `rw{k}_write` `rw{k}_en` `rw{k}_dout` (+ `rw{k}_mask`) |

An `rw` port writes `din` when `en & write`, and `dout` reads `mem[addr]` (readFirst, or
writeFirst-forwarded from this port's own write). Multiple writers to the same address
resolve last-write-wins (registration order), matching Verilog NBA.

```python
# Simple dual-port: 1 write + 2 independent reads
ram = RamPrimitive(UInt(8), depth=4, num_write_ports=1, num_read_ports=2)
ram.io.w0_addr <<= wa; ram.io.w0_data <<= wd; ram.io.w0_en <<= we
ram.io.r0_addr <<= ra0; d0 <<= ram.io.r0_data
ram.io.r1_addr <<= ra1; d1 <<= ram.io.r1_data

# True dual-port (2RW): two ports that each read or write per cycle
dp = RamPrimitive(UInt(8), depth=4, rw_ports=2,
                  num_read_ports=0, num_write_ports=0)
for p, (addr, din, wr, en, dout) in (("rw0", a_sigs), ("rw1", b_sigs)):
    addr_p = getattr(dp.io, f"{p}_addr");  addr_p <<= addr   # bind first; `getattr(...) <<= x` is a syntax error
    din_p  = getattr(dp.io, f"{p}_din");   din_p  <<= din
    wr_p   = getattr(dp.io, f"{p}_write"); wr_p   <<= wr
    en_p   = getattr(dp.io, f"{p}_en");    en_p   <<= en
    dout <<= getattr(dp.io, f"{p}_dout")
```

## RomPrimitive — read-only memory / lookup table

Init-backed, single read port, **no write port** — the contents are fixed. Async read needs
no clock at all (combinational `assign read_data = rom[read_addr];`); `registered_read=True`
adds a 1-cycle registered output with a `read_enable` hold.

```python
RomPrimitive(elem_type, depth, init, *, registered_read=False, name=None)
# ports: read_addr, read_data (+ read_enable if registered_read)

rom = RomPrimitive(UInt(8), depth=8,
                   init=[0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80])
rom.io.read_addr <<= self.io.addr
self.io.dout     <<= rom.io.read_data        # async: combinational lookup, no step() needed
```

It's sugar over `MemoryPrimitive(..., init=…)` with the write port tied off — same emitted
storage idiom, but a clean read-only interface and no empty clocked block for the async case.

## FIFOPrimitive — ready-made synchronous FIFO

A complete sync FIFO (pointers + count + flags), one-cycle registered read latency. `depth`
must be a power of two ≥ 2.

```python
fifo = FIFOPrimitive(UInt(8), depth=4, name="fifo")
fifo.io.push <<= self.io.push
fifo.io.pop  <<= self.io.pop
fifo.io.din  <<= self.io.din
self.io.dout  <<= fifo.io.dout      # popped value appears one cycle after pop
self.io.full  <<= fifo.io.full
self.io.empty <<= fifo.io.empty
self.io.count <<= fifo.io.count
```

Ports: `push`, `pop`, `din` (in); `dout`, `full`, `empty`, `count` (out). Simultaneous
push+pop on a non-empty/non-full FIFO leaves `count` unchanged; underflow/overflow self-gate.

## Composite element types

Any primitive's `elem_type` may be an `HDLComposite` (record). Boundary ports are flat
`UInt(width)`; pack/unpack at the edge with `to_bits()` / `from_bits()`:

```python
class Bus(CompositeRecord):
    data  = Wire(UInt(8))
    valid = Wire(UInt(1))

mem = MemoryPrimitive(Bus, depth=16)               # 9-bit storage
bus_in = Bus(); bus_in.data <<= d; bus_in.valid <<= v
mem.io.write_data <<= bus_in.to_bits()
out = Bus(); out <<= mem.io.read_data                     # from_bits view
data_out <<= out.data; valid_out <<= out.valid
```

## Simulation

Primitives simulate with the built-in `Simulator` — no setup beyond building the module.

```python
from spire.simulator import Simulator

sim = Simulator(module)
sim.deassert_reset()
sim.set("we", 1).set("aw", 3).set("din", 0xAB).step()
sim.set("we", 0).set("ar", 3).eval()       # async read needs only eval(), no step()
assert sim.get("dout") == 0xAB
```

- **Async read** (`registered_read=False`): `read_data` reflects `mem[read_addr]`
  combinationally — `eval()` after setting the address is enough.
- **Registered read** (`registered_read=True`): output updates on the clock edge; `step()`
  to advance, value lags `read_addr` by one cycle.
- **Write-before-read** at the same address in one cycle: a registered read captures the
  **old** value (readFirst / Verilog NBA semantics).

### Inspecting array contents

`Simulator.get_mem(name)` returns the underlying array as a Python list copy. Use the
primitive's `name` (the inlined array name):

```python
sim.get_mem("ram")     # → [0, 0, 0, 0xAB, …]   (length == depth, unsigned bit-patterns)
```

## Notes

- **Reset wins over write.** If a `MemoryPrimitive` reset arm and a write fire in the same
  cycle, the broadcast clear takes priority (`if (reset) … else if (we) …`).
- **Register-bank fallback.** `MemoryPrimitive_via_reg` / `FIFOPrimitive_via_reg` are
  drop-in variants whose sim model is an explicit register file (O(depth), no array
  inference) — kept for comparison/debug. The synthesised Verilog is the same array idiom.
- **Internal store is not user-facing.** `_MemoryArray` (in
  `src/spire/memory.py`) is the sim backend the primitives wire up via a port
  factory; designs always go through the primitives' `.io`.

## See also

- Tests (in [`testing/memory/`](../testing/memory/)):
  [`test_memory.py`](../testing/memory/test_memory.py) (behaviour via primitives),
  [`test_primitive_memory.py`](../testing/memory/test_primitive_memory.py),
  [`test_primitive_ram.py`](../testing/memory/test_primitive_ram.py),
  [`test_primitive_fifo.py`](../testing/memory/test_primitive_fifo.py).
