# Interfaces

Reusable IO bundles for Component ports — handshakes, dataflow, and addressed memory ports —
living in [`spire/interfaces`](../src/spire/interfaces). An interface is just an
[`IORecord`](README_composite_types.md) subclass, so it composes, flips, connects, and simulates
like any other composite.

```python
from spire.interfaces import Flow, Stream, MemPort
```

## Philosophy in one minute

Declare an interface **once**, in its natural *source* / *master* polarity (the producer drives
the data). From that single definition you get the rest:

- **`Flipped(Interface(...))`** — the sink/slave side, with every leaf direction mirrored. One word
  flips a whole bundle.
- **`connect(a, b)`** — peer wiring: the `output` side drives the `input` side per leaf, in either
  argument order (so a backward `ready` wires itself).
- **`iface.view_as_flipped()`** — feedthrough: re-export a port without changing its declared
  direction (for passthroughs where both sides are *your* IO).
- **behaviour as methods** — because an interface *is* a class, helpers like `Stream.fire()` live on
  it.

Nesting an interface inside a Component's `IORecord` gives **hierarchical port names**
(`out_valid`, `out_ready`, `out_data`), so multiple instances never collide.

## The built-in interfaces

| Interface | Canonical (source/master) leaves | Notes |
|---|---|---|
| `Flow(payload)` | `valid`↑ `data`↑ | valid-only, no backpressure; `fire()` = `valid` |
| `Stream(payload)` | `valid`↑ `ready`↓ `data`↑ | ready/valid handshake; `fire()` = `valid & ready` |
| `MemPort(addr_w, data_w)` | `addr`↑ `wdata`↑ `wen`↑ `rdata`↓ | addressed access; `is_write()` = `wen` |

(↑ = driven by the source/master, ↓ = driven by the sink/slave. `Flipped(...)` swaps them.)

## Stream: declare, drive, transform

```python
from spire import Component, IORecord, Input, Output, UInt, Bool, Flipped, connect
from spire.interfaces import Stream

# A source — this module drives the stream.
class Producer(Component):
    def __init__(self, w=8):
        self.io = IORecord(out=Stream(UInt(w)))
        self.elaborate()
    def elaborate(self):
        self.io.out.valid <<= 1
        self.io.out.data  <<= 0xA5

# A transform stage — SINK on `inp` (declared Flipped), SOURCE on `out`.
class AddOne(Component):
    def __init__(self, w=8):
        self.io = IORecord(inp=Flipped(Stream(UInt(w))),   # one word mirrors every direction
                           out=Stream(UInt(w)))
        self.elaborate()
    def elaborate(self):
        i, o = self.io.inp, self.io.out      # Flipped lined the directions up: read inp.*, drive out.*
        o.data  <<= i.data + 1
        o.valid <<= i.valid
        i.ready <<= o.ready                   # backpressure passes straight through
```

## Peer connect — wire a source to a sink in one line

```python
producer, stage = Producer(), AddOne()
connect(stage.io.inp, producer.io.out)        # valid/data forward + ready backward, automatically
```

## Feedthrough — passthrough with no manual field wiring

When both ports are this module's own IO, view both as flipped (the "inside" orientation):

```python
from spire.interfaces import Stream

class Passthrough(Component):
    def __init__(self, w=8):
        self.io = IORecord(inp=Flipped(Stream(UInt(w))), out=Stream(UInt(w)))
        self.elaborate()
    def elaborate(self):
        connect(self.io.inp.view_as_flipped(), self.io.out.view_as_flipped())
```

`view_as_flipped()` does **not** mutate the declared port directions — the emitted module keeps
`inp_*` as inputs and `out_*` as outputs; only the connect-time orientation is flipped.

## Compose & simulate

Sub-components embed automatically when their IO is wired — there is no inlining step. A top-level
module just instantiates, connects, and exposes:

```python
from spire import Simulator

class Top(Component):
    def __init__(self):
        self.io = IORecord(y=Output(UInt(8)), y_valid=Output(Bool()),
                           y_ready=Input(Bool()), fired=Output(Bool()))
        self.elaborate()
    def elaborate(self):
        producer, stage = Producer(), AddOne()
        connect(stage.io.inp, producer.io.out)        # connect the two components' interfaces
        self.io.y          <<= stage.io.out.data       # expose the pipeline output
        self.io.y_valid    <<= stage.io.out.valid
        stage.io.out.ready <<= self.io.y_ready
        self.io.fired      <<= stage.io.out.fire()     # behaviour on the interface: valid & ready

sim = Simulator(Top())
sim.set("y_ready", 1).eval()
out = sim.peek_outputs()
assert out["y"] == 0xA6 and out["y_valid"] == 1 and out["fired"] == 1   # 0xA5 + 1, transfer fires
```

## MemPort: an interface carrying an address

The master drives `addr`; the slave (declared `Flipped`) receives it and drives `rdata`:

```python
from spire.interfaces import MemPort

class MemSlave(Component):                       # a tiny combinational "memory": rdata = addr + 1
    def __init__(self, aw=4, dw=8):
        self.io = IORecord(port=Flipped(MemPort(aw, dw)))
        self.elaborate()
    def elaborate(self):
        self.io.port.rdata <<= self.io.port.addr + 1   # receives addr, drives rdata

class MemTop(Component):                          # the master
    def __init__(self, aw=4, dw=8):
        self.io = IORecord(addr=Input(UInt(aw)), rdata=Output(UInt(dw)))
        self.elaborate()
    def elaborate(self):
        slave = MemSlave()
        slave.io.port.addr  <<= self.io.addr
        slave.io.port.wdata <<= 0
        slave.io.port.wen   <<= 0
        self.io.rdata <<= slave.io.port.rdata

sim = Simulator(MemTop()); sim.set("addr", 5).eval()
assert sim.peek_outputs()["rdata"] == 6           # addr + 1
```

## Defining your own

Subclass `IORecord`, declare leaves in source polarity, and add behaviour as methods:

```python
class AddrStream(IORecord):
    """A Stream whose payload is an address + data word."""
    def __init__(self, addr_w, data_w):
        super().__init__(valid=Output(Bool()), ready=Input(Bool()),
                         addr=Output(UInt(addr_w)), data=Output(UInt(data_w)))
    def fire(self):
        return self.valid & self.ready
```

`Flipped`, `connect`, and `view_as_flipped` work on it for free.

## See also

- Runnable versions of every example above: [`testing/test_interfaces.py`](../testing/test_interfaces.py)
- The trait/type hierarchy these build on: [`README_type_system.md`](README_type_system.md)
- The composite primitives (records, arrays, fixed/float): [`README_composite_types.md`](README_composite_types.md)
