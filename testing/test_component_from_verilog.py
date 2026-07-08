"""Component.from_verilog / from_verilog_file: import round-trips and the sequential gate.

`from_verilog(str)` mirrors `to_verilog() -> str` (temp file, delegates to the file variant);
`from_verilog_file(path)` mirrors `to_verilog_file`. Sequential designs are rejected with one
clear error (AIGER latch import is not implemented). Misuse guards were removed in review:
bad paths / non-Verilog text reach yosys, whose errors end the process.
"""
import pytest

from spire import Component, Input, IORecord, Output, UInt
from spire.component import ImportedComponent
from spire.expr import Register, reset_shared_cache
from spire.simulator import Simulator


class _Add(Component):
    def __init__(self):
        self.io = IORecord(a=Input(UInt(4)), b=Input(UInt(4)), y=Output(UInt(5)))
        self.elaborate()

    def elaborate(self):
        self.io.y <<= self.io.a + self.io.b


def _shell():
    return ImportedComponent(IORecord(a=Input(UInt(4)), b=Input(UInt(4)), y=Output(UInt(5))))


def _check_exhaustive(shell):
    sim = Simulator(shell.to_netlist("reimported", with_clock=False, with_reset=False))
    for a in range(16):
        for b in range(16):
            sim.set("a", a).set("b", b)
            sim.eval()
            assert sim.get("y") == a + b, f"{a}+{b} -> {sim.get('y')}"


def test_from_verilog_string_roundtrip_exhaustive():
    reset_shared_cache()
    src = _Add().to_verilog("adder")          # to_verilog -> str; from_verilog takes it back
    _check_exhaustive(_shell().from_verilog(src))


def test_from_verilog_file_roundtrip_exhaustive(tmp_path):
    reset_shared_cache()
    p = tmp_path / "adder.v"
    _Add().to_verilog_file(str(p), "adder")
    _check_exhaustive(_shell().from_verilog_file(str(p)))


@pytest.mark.parametrize("with_reset", [False, True])
def test_sequential_designs_rejected_loudly(with_reset):
    """Both register flavors funnel to ONE catchable error (async-reset FFs used to kill the
    interpreter in yosys's AIGER backend; clock-only ones died in the minimal AAG reader)."""

    class Acc(Component):
        def __init__(self):
            self.io = IORecord(d=Input(UInt(4)), q=Output(UInt(4)))
            self.elaborate()

        def elaborate(self):
            r = Register(UInt(4), init=0, name="r")
            r <<= self.io.d
            self.io.q <<= r

    reset_shared_cache()
    src = Acc().to_verilog("acc", with_clock=True, with_reset=with_reset)
    shell = ImportedComponent(IORecord(d=Input(UInt(4)), q=Output(UInt(4))))
    with pytest.raises(NotImplementedError, match="combinational designs only"):
        shell.from_verilog(src)
