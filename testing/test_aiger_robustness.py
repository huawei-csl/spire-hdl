"""AIGER exporter/reader robustness: idempotent failure-atomic builds, loud errors, latch inits."""
import os

import pytest

from spire import Component, Input, IORecord, Output, UInt, Bool
from spire.aiger import AigerExporter
from spire.component import ImportedComponent
from spire.expr import Wire, reset_shared_cache
from spire.ir import Netlist
from spire.primitives import MemoryPrimitive
from spire.simulator import Simulator


def _adder():
    reset_shared_cache()
    m = Netlist("add2", with_clock=False, with_reset=False)
    a = m.input(UInt(2), "a")
    b = m.input(UInt(2), "b")
    y = m.output(UInt(3), "y")
    y <<= a + b
    return m


def test_get_aag_idempotent():
    m = _adder()
    exp = AigerExporter(m)
    first = exp.get_aag()
    second = exp.get_aag()
    assert first == second and first[0].startswith("aag ")


def test_failed_build_is_retryable():
    reset_shared_cache()
    m = Netlist("halfbuilt", with_clock=False, with_reset=False)
    a = m.input(UInt(2), "a")
    y = m.output(UInt(2), "y")
    exp = AigerExporter(m)
    for _ in range(2):  # two failed attempts, then fix — the §12.1 scenario
        with pytest.raises(ValueError, match="no driver"):
            exp.get_aag()  # undriven output: loud, like the simulator
    y <<= a  # fix the design and retry with the SAME exporter
    lines = exp.get_aag()
    assert lines[0] == "aag 2 2 0 2 0", f"retry must not re-append inputs: {lines[0]}"


def test_memory_in_cone_gets_actionable_error():
    reset_shared_cache()

    class Top(Component):
        def __init__(self):
            self.io = IORecord(ra=Input(UInt(1)), rd=Output(UInt(4)))
            self.elaborate()

        def elaborate(self):
            mem = MemoryPrimitive(UInt(4), depth=2)
            mem.io.write_addr <<= 0
            mem.io.write_data <<= 0
            mem.io.write_enable <<= 0
            mem.io.read_addr <<= self.io.ra
            self.io.rd <<= mem.io.read_data

    with pytest.raises(NotImplementedError, match="does not support memories"):
        Top().to_aag("memtop", with_clock=True, with_reset=True)


def test_loop_guard_survives_a_raising_evaluation():
    reset_shared_cache()
    m = Netlist("guard", with_clock=False, with_reset=False)
    a = m.input(UInt(2), "a")
    y = m.output(UInt(2), "y")
    inner = Wire(UInt(2), name="inner")
    outer = Wire(UInt(2), name="outer")
    outer <<= inner + 0
    y <<= outer
    sim = Simulator(m)
    with pytest.raises(ValueError, match="no driver"):
        sim.eval()  # inner is undriven; raises while `outer` is on the visiting stack
    inner <<= a
    sim._invalidate()
    sim.set("a", 2)
    sim.eval()  # must NOT report a phantom combinational loop
    assert sim.get("y") == 2


def test_aag_latch_init_tokens(tmp_path):
    from spire.aig.aig_to_aag import read_aiger, get_aag_lines
    # Init tokens meaning power-on-0 (the uninitialized marker here) are accepted and normalized away.
    src = ["aag 3 1 1 1 1", "2", "4 6 4", "4", "6 2 4", "i0 a", "l0 st", "o0 y"]
    p = tmp_path / "t.aag"
    p.write_text("\n".join(src) + "\n")
    out = get_aag_lines(read_aiger(str(p)))
    assert "4 6" in out and "4 6 4" not in out
    # An init-1 latch is rejected loudly instead of silently misread as init-0.
    p.write_text("\n".join(["aag 3 1 1 1 1", "2", "4 6 1", "4", "6 2 4"]) + "\n")
    with pytest.raises(ValueError, match="init 1 is not supported"):
        read_aiger(str(p))


def test_extended_header_rejected_and_symbols_stripped():
    from spire.aig.aig_aigerverse import AagParseError, _read_aag
    with pytest.raises(AagParseError, match="1.9"):
        _read_aag(["aag 1 1 0 1 0 1", "2", "2"])
    d = _read_aag(["aag 1 1 0 1 0", "2", "2", "i0 a\r", "o0 y\r"])
    names = list(d.get("sym_i", {}).values()) or [v for k, v in d.items() if k == "sym_i"]
    flat = str(d)
    assert "\\r" not in flat and "a" in flat

