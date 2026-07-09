"""Component.from_verilog / from_verilog_file: combinational and sequential import round-trips.

`from_verilog(str)` mirrors `to_verilog() -> str` (temp file, delegates to the file variant);
`from_verilog_file(path)` mirrors `to_verilog_file`. Registers arrive as 1-bit AIGER latches
and become registers on the surrounding design's global clock. Misuse guards were removed in
review: bad paths / non-Verilog text reach yosys, whose errors end the process.
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


class _Acc(Component):
    """Accumulator: register feedback plus input, so latch state feeds the AND network."""

    def __init__(self):
        self.io = IORecord(d=Input(UInt(4)), q=Output(UInt(4)))
        self.elaborate()

    def elaborate(self):
        r = Register(UInt(4), init=0, name="r")
        r <<= r + self.io.d
        self.io.q <<= r


def test_from_verilog_sequential_roundtrip_differential():
    """Reimported registers must clock identically to the original: 50 cycles, exact match.
    The absorbed clock stays out of the port list, and latch init-0 matches the sim start."""
    reset_shared_cache()
    src = _Acc().to_verilog("acc", with_clock=True, with_reset=False)
    shell = ImportedComponent(IORecord(d=Input(UInt(4)), q=Output(UInt(4)))).from_verilog(src)

    reset_shared_cache()
    ref = Simulator(_Acc().to_netlist("orig", with_clock=True, with_reset=False))
    dut = Simulator(shell.to_netlist("reimported", with_clock=True, with_reset=False))
    for t, d in enumerate((7 * i + 3) % 16 for i in range(50)):
        for sim in (ref, dut):
            sim.set("d", d)
            sim.eval()
        assert ref.get("q") == dut.get("q"), f"cycle {t}: ref={ref.get('q')} dut={dut.get('q')}"
        ref.step()
        dut.step()


def test_from_verilog_reset_port_rejected_with_guidance():
    """A with_reset export folds rst into a DATA input, whose name collides with the
    framework-injected reset; the import must say so, not fail later or silently rename."""
    reset_shared_cache()
    src = _Acc().to_verilog("acc", with_clock=True, with_reset=True)
    shell = ImportedComponent(IORecord(d=Input(UInt(4)), q=Output(UInt(4))))
    with pytest.raises(ValueError, match="port named 'rst'.*reserved"):
        shell.from_verilog(src)
