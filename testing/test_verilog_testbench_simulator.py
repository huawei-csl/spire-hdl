import os
from pathlib import Path
import pytest

from spirehdl.spirehdl import UInt
from spirehdl.spirehdl_module import Module
from spirehdl.spirehdl_verilog_testbench import TestbenchGenSimulator


def build_accumulator():
    m = Module("Mac32", with_clock=True, with_reset=True)
    a = m.input(UInt(8), "a")
    b = m.input(UInt(8), "b")
    acc = m.reg(UInt(8), "acc", init=0)
    out = m.output(UInt(8), "out")

    acc <<= acc + a + b
    out <<= acc
    return m, a, b, acc, out


def test_verilog_testbench_basic_sequence(tmp_path: Path):
    m, a, b, acc, _ = build_accumulator()

    tb = TestbenchGenSimulator(m, clock_period=10.0, eval_delay=1.0)
    tb.reset(True)
    tb.step()  # capture reset behaviour
    tb.deassert_reset()
    tb.set(a, 1)
    tb.set(b, 2)
    tb.eval()
    assert tb._sim.peek_outputs()['out'] == 0
    tb.step()
    assert tb._sim.peek_outputs()["out"] == 3
    tb.eval()
    tb.set(acc, 5)
    tb.eval()

    out_path = tmp_path / "mac_tb.v"
    tb.to_testbench_file(out_path)
    text = out_path.read_text()

    assert "module Mac32_tb" in text
    assert "Mac32 dut" in text
    assert "a = 8'h01" in text
    assert "b = 8'h02" in text
    assert "dut.acc = 8'h05" in text
    assert "$fatal" in text  # expectations are emitted
    assert "#5" in text  # half-period delay

    # module to file
    str_m = m.to_verilog()
    verilog_path = tmp_path / "mac.v"
    verilog_path.write_text(str_m)

def test_verilog_testbench_requires_events(tmp_path):
    m = Module("Comb", with_clock=False, with_reset=False)
    a = m.input(UInt(4), "a")
    y = m.output(UInt(4), "y")
    y <<= a

    tb = TestbenchGenSimulator(m)
    with pytest.raises(RuntimeError):
        tb.to_testbench_file(tmp_path / "comb_tb.v")

    with pytest.raises(RuntimeError):
        tb.step()


if __name__ == "__main__":
    
    # make local tempdir for testing with os tempdir
    tmpdir = os.path.join(os.getcwd(), "temp_test/")
    os.makedirs(tmpdir, exist_ok=True)
    
    #pytest.main([__file__])
    test_verilog_testbench_basic_sequence(Path(tmpdir))
    test_verilog_testbench_requires_events(Path(tmpdir))
