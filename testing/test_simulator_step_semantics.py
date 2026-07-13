"""Clock-edge and reset semantics: memories sample pre-edge state; caches and watches stay coherent."""
import pytest

from spire.expr import Bool, Register, Signal, UInt, mux, reset_shared_cache
from spire.ir import Netlist
from spire.primitives import MemoryPrimitive
from spire.simulator import Simulator


def _mem_through_reg():
    """Write data flows through a register: the memory must sample the register's PRE-edge value."""
    reset_shared_cache()
    m = Netlist("mem_pre_edge", with_clock=True, with_reset=True)
    d = m.input(UInt(8), "d")
    q = m.output(UInt(8), "q")
    r = m.reg(UInt(8), "r", init=0)
    r <<= d
    mem = MemoryPrimitive(UInt(8), depth=4, name="mem")
    mem.io.write_addr <<= 0
    mem.io.write_data <<= r
    mem.io.write_enable <<= 1
    mem.io.read_addr <<= 0
    q <<= mem.io.read_data
    return m


def test_memory_write_samples_pre_edge_register():
    sim = Simulator(_mem_through_reg())
    sim.reset(True)
    sim.deassert_reset()
    sim.set("d", 5)
    sim.step()  # r: 0 -> 5; the write this edge must use r's OLD value (0)
    assert sim.get_mem("mem")[0] == 0
    sim.step()  # now the 5 lands
    assert sim.get_mem("mem")[0] == 5


def test_reset_deassert_invalidates_combinational_caches():
    reset_shared_cache()
    m = Netlist("rst_comb", with_clock=True, with_reset=True)
    a = m.input(UInt(4), "a")
    b = m.input(UInt(4), "b")
    y = m.output(UInt(4), "y")
    y <<= mux(m.rst, a, b)

    sim = Simulator(m)
    sim.set("a", 3)
    sim.set("b", 9)
    sim.reset(True)
    sim.eval()
    assert sim.get("y") == 3
    sim.reset(False)  # no explicit eval: the cached cone must still be dropped
    assert sim.get("y") == 9
    sim.reset(True)
    sim.deassert_reset()
    assert sim.get("y") == 9


def test_watch_sees_post_edge_state():
    reset_shared_cache()
    m = Netlist("watch_epoch", with_clock=True, with_reset=True)
    d = m.input(UInt(4), "d")
    q = m.output(UInt(5), "q")
    r = m.reg(UInt(4), "r", init=0)
    r <<= d
    w = Signal(typ=UInt(5), kind="wire", name="w")
    w <<= r + 1
    q <<= w

    sim = Simulator(m)
    sim.reset(True)
    sim.deassert_reset()
    sim.watch(w, "w")
    sim.set("d", 7)
    sim.step()
    assert sim.get_watch("w") == 8  # post-edge r (7) + 1, never the stale pre-edge cache


def test_watch_wire_outside_evaluated_cone():
    reset_shared_cache()
    m = Netlist("watch_side", with_clock=True, with_reset=True)
    a = m.input(UInt(4), "a")
    y = m.output(UInt(4), "y")
    y <<= a
    side = Signal(typ=UInt(5), kind="wire", name="side")
    side <<= a + 1
    m._signals.append(side)  # reachable only via the watch

    sim = Simulator(m)
    sim.watch(side, "side")
    sim.set("a", 4)
    sim.step()  # must not KeyError on the un-evaluated wire
    assert sim.get_watch("side") == 5


def test_peek_next_respects_asserted_reset():
    reset_shared_cache()
    m = Netlist("peek_rst", with_clock=True, with_reset=True)
    d = m.input(UInt(4), "d")
    y = m.output(UInt(4), "y")
    r = m.reg(UInt(4), "r", init=5)
    r <<= d
    y <<= r

    sim = Simulator(m)
    sim.set("d", 9)
    sim.reset(True)
    assert sim.peek_next("r") == 5  # reset wins over the driver
    sim.deassert_reset()
    assert sim.peek_next("r") == 9


def test_memory_signal_in_get_points_to_get_mem():
    sim = Simulator(_mem_through_reg())
    with pytest.raises(TypeError, match="use get_mem"):
        sim.get("mem")


def test_huge_variable_shift_is_zero():
    reset_shared_cache()
    m = Netlist("shift_huge", with_clock=False, with_reset=False)
    a = m.input(UInt(8), "a")
    n = m.input(UInt(32), "n")
    y = m.output(UInt(8), "y")
    y <<= a << n

    sim = Simulator(m)
    sim.set("a", 3)
    sim.set("n", 0xFFFFFFFF)
    sim.eval()
    assert sim.get("y") == 0
