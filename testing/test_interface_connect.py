"""Interface composition: flip() / Flipped / connect()."""
import pytest

from spire import Component, IORecord, Input, Output, UInt, Flipped, connect
from spire.expr import Bool, Wire


class Stream(IORecord):
    """Canonical (source/producer) polarity: valid+data out, ready in."""
    def __init__(self, payload):
        super().__init__(valid=Output(Bool()), ready=Input(Bool()), data=Output(payload))


class MemBus(IORecord):
    """Canonical (master) polarity: addr/wdata/wen out, rdata/rvalid in."""
    def __init__(self, aw, dw):
        super().__init__(
            addr=Output(UInt(aw)), wdata=Output(UInt(dw)), wen=Output(Bool()),
            rdata=Input(UInt(dw)), rvalid=Input(Bool()),
        )


def _kinds(rec):
    return [leaf.kind for leaf in rec.to_list()]


# ---- flip ----

def test_flip_involution():
    s = Stream(UInt(8))
    base = _kinds(s)                                  # [output, input, output]
    s.flip()
    assert _kinds(s) == ["input", "output", "input"]  # sink polarity
    s.flip()
    assert _kinds(s) == base                           # involution


def test_flip_leaves_wire_and_reg_untouched():
    rec = IORecord(p=Output(UInt(4)), w=Wire(UInt(4)))
    rec.flip()
    kinds = _kinds(rec)
    assert "input" in kinds    # the Output port flipped to input
    assert "wire" in kinds      # the wire is direction-less, untouched


def test_flipped_preserves_type():
    f = Flipped(Stream(UInt(8)))
    assert isinstance(f, Stream)
    assert _kinds(f) == ["input", "output", "input"]


def test_flip_recurses_into_nested_bundle():
    outer = IORecord(req=Stream(UInt(4)), done=Output(Bool()))
    outer.flip()
    by_name = {leaf.name: leaf.kind for leaf in outer.to_list()}
    assert by_name["req_valid"] == "input"
    assert by_name["req_ready"] == "output"
    assert by_name["req_data"] == "input"
    assert by_name["done"] == "input"


# ---- connect (peer) ----

def test_connect_stream_wires_both_directions():
    producer = Stream(UInt(8))             # source
    consumer = Flipped(Stream(UInt(8)))    # sink
    connect(consumer, producer)
    assert consumer.valid._driver is producer.valid   # forward
    assert consumer.data._driver is producer.data      # forward
    assert producer.ready._driver is consumer.ready    # backward (automatic)


def test_connect_is_symmetric():
    producer = Stream(UInt(8))
    consumer = Flipped(Stream(UInt(8)))
    connect(producer, consumer)                        # reversed arg order
    assert consumer.valid._driver is producer.valid
    assert producer.ready._driver is consumer.ready


def test_connect_membus_mixed_directions():
    master = MemBus(8, 32)
    slave = Flipped(MemBus(8, 32))
    connect(slave, master)
    for nm in ("addr", "wdata", "wen"):                # master -> slave
        assert getattr(slave, nm)._driver is getattr(master, nm)
    for nm in ("rdata", "rvalid"):                      # slave -> master
        assert getattr(master, nm)._driver is getattr(slave, nm)


def test_connect_width_mismatch_raises():
    with pytest.raises(ValueError):
        connect(Stream(UInt(8)), Flipped(Stream(UInt(16))))


def test_connect_same_direction_feedthrough_raises():
    # two sources -> matching leaves share a direction -> feedthrough, not supported here
    with pytest.raises(TypeError):
        connect(Stream(UInt(8)), Stream(UInt(8)))


def test_connect_leaf_count_mismatch_raises():
    with pytest.raises(ValueError):
        connect(Stream(UInt(8)), Flipped(MemBus(8, 32)))


# ---- emit: flipped interfaces give unique hierarchical ports ----

def test_source_and_sink_modules_emit_unique_ports():
    class Producer(Component):
        def __init__(self):
            self.io = IORecord(out=Stream(UInt(8)))
            self.elaborate()
        def elaborate(self):
            self.io.out.valid <<= 1
            self.io.out.data <<= 0xAB

    class Consumer(Component):
        def __init__(self):
            self.io = IORecord(inp=Flipped(Stream(UInt(8))))
            self.elaborate()
        def elaborate(self):
            self.io.inp.ready <<= 1

    pv = Producer().to_verilog("producer")
    assert "out_valid" in pv and "out_ready" in pv and "out_data" in pv
    cv = Consumer().to_verilog("consumer")
    assert "inp_valid" in cv and "inp_ready" in cv and "inp_data" in cv


# ---- view_as_flipped: non-mutating orientation for feedthrough ----

def test_view_as_flipped_does_not_mutate():
    s = Stream(UInt(8))
    before = _kinds(s)
    v = s.view_as_flipped()
    assert _kinds(s) == before                                  # real bundle untouched
    assert [k for _, k in v._directed_leaves()] == ["input", "output", "input"]  # view reports flipped


def test_view_as_flipped_involution():
    s = Stream(UInt(8))
    vv = s.view_as_flipped().view_as_flipped()
    assert [k for _, k in vv._directed_leaves()] == _kinds(s)


def test_feedthrough_passthrough_via_views():
    class Passthrough(Component):
        def __init__(self):
            self.io = IORecord(up=Flipped(Stream(UInt(8))), down=Stream(UInt(8)))
            self.elaborate()
        def elaborate(self):
            # both sides are this module's own IO -> view both flipped (inside orientation)
            connect(self.io.up.view_as_flipped(), self.io.down.view_as_flipped())

    c = Passthrough()
    assert c.io.down.valid._driver is c.io.up.valid     # up -> down
    assert c.io.down.data._driver is c.io.up.data
    assert c.io.up.ready._driver is c.io.down.ready      # ready down -> up
    # the view-based connect must NOT have mutated boundary directions
    k = {l.name: l.kind for l in c.io.to_list()}
    assert k["up_valid"] == "input" and k["down_valid"] == "output" and k["up_ready"] == "output"
    v = c.to_verilog("passthrough")
    assert "assign down_valid = up_valid" in v and "assign up_ready = down_ready" in v


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("ok")
