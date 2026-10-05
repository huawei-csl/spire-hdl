"""The data-driven testbench must run under both Icarus and Verilator, including a reset column."""
import shutil
import subprocess

import pytest

from spire import Component, IORecord, Input, Output, Simulator
from spire.expr import Register, UInt, reset_shared_cache
from spire.verilog_testbench import TestbenchGenSimulator, write_vector_data_file


class _Acc(Component):
    def __init__(self):
        self.io = IORecord(a=Input(UInt(4)), q=Output(UInt(8)))
        self.elaborate()

    def elaborate(self):
        r = Register(UInt(8), init=3)
        r <<= (r + self.io.a)[0:8]
        self.io.q <<= r


def _vectors(net):
    sim = Simulator(net)
    sim.reset(True); sim.eval()
    vec = [("reset", {"rst": 1, "a": 0}, sim.peek_outputs())]
    sim.deassert_reset()
    for n in range(20):
        sim.set("a", n % 16).step()
        vec.append((f"c{n}", {"rst": 0, "a": n % 16}, sim.peek_outputs()))
    return vec


@pytest.mark.parametrize("tool", ["iverilog", "verilator"])
def test_data_driven_tb_runs(tool, tmp_path, monkeypatch):
    if not shutil.which(tool):
        pytest.skip(f"{tool} not installed")
    reset_shared_cache()
    net = _Acc().to_netlist("acc", with_clock=True, with_reset=True)
    monkeypatch.chdir(tmp_path)
    write_vector_data_file(_vectors(net), "acc.dat")
    TestbenchGenSimulator(net).to_data_driver_testbench_file("acc_tb.v", data_file="acc.dat",
                                                             input_stimuli=["rst", "a"], with_clk=True)
    net.to_verilog_file("acc.v")
    if tool == "iverilog":
        subprocess.run(["iverilog", "-g2012", "-o", "sim", "acc_tb.v", "acc.v"], check=True, capture_output=True)
        out = subprocess.run(["vvp", "-n", "sim"], capture_output=True, text=True).stdout
    else:
        subprocess.run(["verilator", "--binary", "--timing", "-Wno-fatal", "--top-module", "acc_tb", "acc_tb.v", "acc.v",
                        "-o", "Vacc"], check=True, capture_output=True)
        out = subprocess.run(["./obj_dir/Vacc"], capture_output=True, text=True).stdout
    assert "Finished: 21 passed, 0 failed" in out, out
    assert "completed successfully" in out, out
