"""Memory primitive validation and emission hygiene.

writeFirst + write mask is rejected (forwarding can't blend chunks); init/reset element values
are validated against the element type and stored as bit patterns (negative literals like
`8'd-1` are illegal Verilog); out-of-range reads return the agreed 0 in every sim variant;
duplicate instance names propagate the collector-uniquified array name into the custom Verilog.
"""
import pytest

from spire import Component, Input, IORecord, Output, UInt, SInt, Bool
from spire.expr import reset_shared_cache
from spire.primitives import MemoryPrimitive, MemoryPrimitive_via_reg, RamPrimitive, RomPrimitive
from spire.simulator import Simulator


def test_writefirst_with_mask_rejected_on_both_primitives():
    reset_shared_cache()
    with pytest.raises(ValueError, match="unmasked"):
        MemoryPrimitive(UInt(8), depth=4, mask_chunks=2, read_under_write="writeFirst")
    with pytest.raises(ValueError, match="unmasked"):
        RamPrimitive(UInt(8), depth=4, rw_ports=1, mask_chunks=2, read_under_write="writeFirst")


def test_out_of_range_init_rejected():
    reset_shared_cache()
    with pytest.raises(ValueError, match="not representable"):
        RomPrimitive(UInt(8), depth=2, init=[300, 0])
    with pytest.raises(ValueError, match="not representable"):
        MemoryPrimitive(SInt(8), depth=2, init=[-129, 0])
    with pytest.raises(ValueError, match="reset_value"):
        MemoryPrimitive_via_reg(UInt(8), depth=2, with_reset_arm=True, reset_value=-1)


class _RomTop(Component):
    def __init__(self):
        self.io = IORecord(addr=Input(UInt(1)), dout=Output(UInt(8)))
        self.elaborate()

    def elaborate(self):
        rom = RomPrimitive(SInt(8), depth=2, init=[-1, -128], name="srom")
        rom.io.read_addr <<= self.io.addr
        self.io.dout <<= rom.io.read_data


def test_negative_init_emits_masked_literals_and_sim_matches():
    reset_shared_cache()
    top = _RomTop()
    m = top.to_module("srom_top", with_clock=False, with_reset=False)
    v = m.to_verilog()
    assert "8'd255" in v and "8'd128" in v and "'d-" not in v
    sim = Simulator(m)
    sim.set("addr", 0)
    sim.eval()
    assert sim.get("dout") == 0xFF


def _oor_top(cls):
    class Top(Component):
        def __init__(self):
            self.io = IORecord(wa=Input(UInt(3)), wd=Input(UInt(8)), we=Input(Bool()),
                               ra=Input(UInt(3)), rd=Output(UInt(8)))
            self.elaborate()

        def elaborate(self):
            mem = cls(UInt(8), depth=5)  # non-pow2: addresses 5..7 are drivable but out-of-range
            mem.io.write_addr <<= self.io.wa
            mem.io.write_data <<= self.io.wd
            mem.io.write_enable <<= self.io.we
            mem.io.read_addr <<= self.io.ra
            self.io.rd <<= mem.io.read_data

    reset_shared_cache()
    return Top().to_module("oor_top", with_clock=True, with_reset=True)


def test_out_of_range_read_is_zero_in_both_variants():
    for cls in (MemoryPrimitive, MemoryPrimitive_via_reg):
        sim = Simulator(_oor_top(cls))
        sim.reset(True)
        sim.deassert_reset()
        sim.set("wa", 0).set("wd", 0xAB).set("we", 1).set("ra", 7)
        sim.step()
        sim.set("we", 0)
        sim.eval()
        assert sim.get("rd") == 0, f"{cls.__name__}: out-of-range read must be the agreed 0"


def test_via_reg_ctor_parity_kwargs():
    reset_shared_cache()
    MemoryPrimitive_via_reg(UInt(8), depth=2, mask_chunks=0, read_under_write="readFirst")  # defaults fine
    with pytest.raises(NotImplementedError):
        MemoryPrimitive_via_reg(UInt(8), depth=2, mask_chunks=2)
    with pytest.raises(NotImplementedError):
        MemoryPrimitive_via_reg(UInt(8), depth=2, read_under_write="writeFirst")


class _DupTop(Component):
    def __init__(self):
        self.io = IORecord(wa=Input(UInt(1)), wd=Input(UInt(8)), we=Input(Bool()),
                           ra=Input(UInt(1)), r0=Output(UInt(8)), r1=Output(UInt(8)))
        self.elaborate()

    def elaborate(self):
        for out in (self.io.r0, self.io.r1):
            mem = MemoryPrimitive(UInt(8), depth=2, name="dup")
            mem.io.write_addr <<= self.io.wa
            mem.io.write_data <<= self.io.wd
            mem.io.write_enable <<= self.io.we
            mem.io.read_addr <<= self.io.ra
            out <<= mem.io.read_data


def test_duplicate_instance_names_emit_uniquified_arrays():
    reset_shared_cache()
    v = _DupTop().to_verilog("dup_top", with_clock=True, with_reset=True)
    decls = [l for l in v.splitlines() if l.strip().startswith("reg [7:0] dup")]
    names = {l.split()[2].split("[0:")[0] for l in decls}
    assert len(decls) == 2 and len(names) == 2, f"array decls must be uniquified: {decls}"
