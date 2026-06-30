"""Reusable interfaces (spire.interfaces): Flow / Stream / MemPort.

Covers declaration, Flipped/connect wiring, on-interface behaviour (`fire`), and end-to-end
simulation of small designs built from them.
"""
from spire import Component, IORecord, Input, Output, UInt, Bool, Flipped, connect, Simulator
from spire.expr import reset_shared_cache
from spire.interfaces import Flow, Stream, MemPort


# ---------------------------------------------------------------------------
# Stream: a producer -> +1 stage -> top, wired with connect()
# ---------------------------------------------------------------------------

class Producer(Component):
    """A source: drives the stream unconditionally."""
    def __init__(self, w=8):
        self.io = IORecord(out=Stream(UInt(w)))
        self.elaborate()

    def elaborate(self):
        self.io.out.valid <<= 1
        self.io.out.data <<= 0xA5


class AddOne(Component):
    """A transform stage: SINK on `inp` (Flipped), SOURCE on `out`."""
    def __init__(self, w=8):
        self.io = IORecord(inp=Flipped(Stream(UInt(w))), out=Stream(UInt(w)))
        self.elaborate()

    def elaborate(self):
        i, o = self.io.inp, self.io.out      # Flipped lined up directions: read inp.*, drive out.*
        o.data <<= i.data + 1
        o.valid <<= i.valid
        i.ready <<= o.ready                   # backpressure passes straight through


class StreamTop(Component):
    def __init__(self):
        self.io = IORecord(y=Output(UInt(8)), y_valid=Output(Bool()),
                           y_ready=Input(Bool()), fired=Output(Bool()))
        self.elaborate()

    def elaborate(self):
        producer, stage = Producer(), AddOne()
        connect(stage.io.inp, producer.io.out)      # one line wires valid/data fwd + ready back
        self.io.y <<= stage.io.out.data
        self.io.y_valid <<= stage.io.out.valid
        stage.io.out.ready <<= self.io.y_ready
        self.io.fired <<= stage.io.out.fire()        # behaviour lives on the interface: valid & ready


def test_stream_pipeline_simulates():
    reset_shared_cache()
    sim = Simulator(StreamTop())
    sim.set("y_ready", 1)
    sim.eval()
    out = sim.peek_outputs()
    assert out["y"] == 0xA6          # 0xA5 + 1
    assert out["y_valid"] == 1
    assert out["fired"] == 1          # valid & ready


def test_stream_fire_tracks_ready():
    reset_shared_cache()
    sim = Simulator(StreamTop())
    sim.set("y_ready", 0)
    sim.eval()
    assert sim.peek_outputs()["fired"] == 0   # valid high but ready low -> no transfer
    assert sim.peek_outputs()["y_valid"] == 1


def test_stream_connect_wires_both_directions():
    # connect() drives forward (valid/data) and backward (ready) in a single call.
    producer = Producer()
    stage = AddOne()
    connect(stage.io.inp, producer.io.out)
    assert stage.io.inp.valid._driver is producer.io.out.valid   # source -> sink
    assert stage.io.inp.data._driver is producer.io.out.data
    assert producer.io.out.ready._driver is stage.io.inp.ready    # sink -> source (automatic)


def test_flipped_stream_is_sink_polarity():
    src = Stream(UInt(8))
    assert [l.kind for l in src.to_list()] == ["output", "input", "output"]
    snk = Flipped(Stream(UInt(8)))
    assert isinstance(snk, Stream)
    assert [l.kind for l in snk.to_list()] == ["input", "output", "input"]


# ---------------------------------------------------------------------------
# Flow: valid-only (no backpressure)
# ---------------------------------------------------------------------------

class FlowProducer(Component):
    def __init__(self, w=8):
        self.io = IORecord(out=Flow(UInt(w)))
        self.elaborate()

    def elaborate(self):
        self.io.out.valid <<= 1
        self.io.out.data <<= 0x42


class FlowTop(Component):
    def __init__(self):
        self.io = IORecord(y=Output(UInt(8)), y_valid=Output(Bool()))
        self.elaborate()

    def elaborate(self):
        p = FlowProducer()
        self.io.y <<= p.io.out.data
        self.io.y_valid <<= p.io.out.valid


def test_flow_has_no_ready():
    assert [l.name for l in Flow(UInt(8)).to_list()] == ["valid", "data"]   # no `ready` leaf


def test_flow_simulates():
    reset_shared_cache()
    sim = Simulator(FlowTop())
    sim.eval()
    out = sim.peek_outputs()
    assert out["y"] == 0x42
    assert out["y_valid"] == 1


# ---------------------------------------------------------------------------
# MemPort: an interface carrying an address. Master drives addr; slave returns rdata.
# ---------------------------------------------------------------------------

class MemSlave(Component):
    """A tiny combinational 'memory': rdata = addr + 1. SINK on the port (Flipped)."""
    def __init__(self, aw=4, dw=8):
        self.io = IORecord(port=Flipped(MemPort(aw, dw)))
        self.elaborate()

    def elaborate(self):
        self.io.port.rdata <<= self.io.port.addr + 1   # receives addr, drives rdata


class MemTop(Component):
    """The master: drives the slave's port from an external address, exposes rdata."""
    def __init__(self, aw=4, dw=8):
        self.io = IORecord(addr=Input(UInt(aw)), rdata=Output(UInt(dw)))
        self.elaborate()

    def elaborate(self):
        slave = MemSlave()
        slave.io.port.addr <<= self.io.addr
        slave.io.port.wdata <<= 0
        slave.io.port.wen <<= 0
        self.io.rdata <<= slave.io.port.rdata


def test_memport_simulates():
    reset_shared_cache()
    sim = Simulator(MemTop())
    sim.set("addr", 5)
    sim.eval()
    assert sim.peek_outputs()["rdata"] == 6      # addr + 1

    sim.set("addr", 10)
    sim.eval()
    assert sim.peek_outputs()["rdata"] == 11


def test_memport_connect_mixed_directions():
    master = MemPort(4, 8)
    slave = Flipped(MemPort(4, 8))
    connect(slave, master)
    for nm in ("addr", "wdata", "wen"):              # master -> slave
        assert getattr(slave, nm)._driver is getattr(master, nm)
    assert master.rdata._driver is slave.rdata        # slave -> master


if __name__ == "__main__":
    for _name, _fn in list(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
    print("ok")
