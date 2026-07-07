"""Composite IO leaf naming: field-path port names at any nesting depth.

Every enclosing record rebuilds its subtree's names from scratch, so a leaf port is always its
full field path with each segment appearing exactly once — regardless of nesting depth or
construction order. Field keys win: a `name=` given at leaf construction applies to standalone
signals only and is overridden inside records.
"""
from spire import Bool, Component, Input, IORecord, Output, UInt, Wire
from spire.composite.array import Array
from spire.composite.record import CompositeRecord
from spire.expr import reset_shared_cache
from spire.interfaces import Stream


def _leaf_names(composite):
    return [leaf.name for leaf in composite.to_list()]


def test_depth1_field_names_win_over_construction_names():
    reset_shared_cache()
    io = IORecord(addr=Input(UInt(4)), q=Output(UInt(4)), named=Input(UInt(4), name="my_addr"))
    assert _leaf_names(io) == ["addr", "q", "named"]


def test_depth2_single_path_segments():
    reset_shared_cache()
    io = IORecord(bus=CompositeRecord(hdr=CompositeRecord(addr=Input(UInt(4)), my=Input(UInt(4), name="my_addr")),
                                      data=Input(UInt(8))))
    assert _leaf_names(io) == ["bus_hdr_addr", "bus_hdr_my", "bus_data"]


def test_depth3_single_path_segments():
    reset_shared_cache()
    io = IORecord(top=CompositeRecord(mid=CompositeRecord(low=CompositeRecord(x=Input(UInt(2))))))
    assert _leaf_names(io) == ["top_mid_low_x"]


def test_stream_in_record():
    reset_shared_cache()
    io = IORecord(link=CompositeRecord(tx=Stream(UInt(8))))
    assert _leaf_names(io) == ["link_tx_valid", "link_tx_ready", "link_tx_data"]


def test_record_in_array_in_record():
    reset_shared_cache()
    io = IORecord(lanes=Array([CompositeRecord(v=Input(Bool()), d=Input(UInt(4))) for _ in range(2)]))
    assert _leaf_names(io) == ["lanes_0_v", "lanes_0_d", "lanes_1_v", "lanes_1_d"]


def test_construction_name_overridden_at_depth_too():
    reset_shared_cache()
    io = IORecord(bus=CompositeRecord(addr=Input(UInt(4), name="my_addr")))
    assert _leaf_names(io) == ["bus_addr"]


def test_wire_fields_get_field_path_names():
    reset_shared_cache()
    rec = CompositeRecord(top=Wire(UInt(4)), sub=CompositeRecord(inner=Wire(UInt(4))))
    assert _leaf_names(rec) == ["top", "sub_inner"]


class _Depth2Comp(Component):
    def __init__(self):
        self.io = IORecord(bus=CompositeRecord(hdr=CompositeRecord(addr=Input(UInt(4))), data=Input(UInt(4))),
                           q=Output(UInt(4)))
        self.elaborate()

    def elaborate(self):
        self.io.q <<= self.io.bus.hdr.addr + self.io.bus.data


def test_depth2_component_emits_path_ports():
    reset_shared_cache()
    v = _Depth2Comp().to_verilog("depth2")
    assert "bus_hdr_addr" in v and "bus_data" in v
    assert "hdr_hdr" not in v and "bus_bus" not in v


def test_get_ios_passes_composite_io_through_so_reads_never_rename():
    reset_shared_cache()
    c = _Depth2Comp()
    io = c.get_ios()
    assert c.get_ios() is io and c.io is io  # IORecord io is returned as-is, never re-wrapped
    leaf = io.to_list()[0]
    leaf.name = "bus_hdr_addr_1"  # emitter-style uniquification
    assert c.get_ios().to_list()[0].name == "bus_hdr_addr_1"  # a later read must not clobber it
