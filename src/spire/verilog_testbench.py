"""Verilog testbench generator mirroring the Simulator API.

This module provides a drop-in replacement for :class:`spire.simulator.Simulator`
that records stimuli and expected results while delegating semantic evaluation to the Python
simulator.  The recorded trace can then be emitted as a synthesizable Verilog testbench that
replays the same interactions against the generated RTL (e.g. when running under Verilator).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Union

from spire import VERILOG_BANNER
from spire.expr import Expr, Signal
from spire.component import Component, Netlist
from spire.simulator_base import SimulatorBase
from spire.simulator import Simulator as PythonSimulator

if TYPE_CHECKING:
    from spire.arithmetic.int_multipliers.eval.testvector_generation import TestVectors


@dataclass
class _Event:
    action: str
    delay: float
    payload: dict


def write_vector_data_file(
    vectors: "TestVectors", data_file: Union[str, Path]
) -> None:
    """Write TestVectors to a whitespace-separated data file for Verilog testbenches."""

    _, first_inputs, first_outputs = vectors[0]
    input_names = list(first_inputs.keys())
    output_names = list(first_outputs.keys())

    with open(data_file, "w") as f:
        f.write("# " + " ".join(input_names + output_names) + "\n")
        for _, inputs, outputs in vectors:
            f.write(" ".join([str(inputs[k]) for k, _ in inputs.items()]) + " " +
                    " ".join([str(outputs[k]) for k, _ in outputs.items()]) + "\n")


class TestbenchGenSimulator(SimulatorBase):
    """Record Simulator-like interactions and emit a Verilog testbench.

    The public API mirrors :class:`~spire.simulator.Simulator`.  The methods
    delegate to the in-Python simulator to keep behaviour identical while logging the sequence
    of IO operations.  Once the trace is complete, call :meth:`write_testbench` to materialise
    the Verilog testbench that reproduces the stimuli and checks the observed outputs.
    """

    __test__ = False  # not a pytest class despite the Test* name

    def __init__(
        self,
        module: "Netlist | Component",
        *,
        clock_period: float = 10.0,
        eval_delay: float = 1.0,
    ) -> None:
        # Accept a Component directly and lower it to its netlist IR (same as Simulator), so this
        # stays a true drop-in replacement.
        if isinstance(module, Component):
            module = module.to_netlist()
        self.m = module
        self._sim = PythonSimulator(module)
        self.clock_period = clock_period
        self.eval_delay = eval_delay
        self._events: List[_Event] = []
        self.inputs = [s for s in self.m._ports if s.kind == "input"]
        self.outputs = [s for s in self.m._ports if s.kind == "output"]
        self.regs = [s for s in self.m._signals if s.kind == "reg"]
        self._by_name = {s.name: s for s in self.m._signals}
        self._half_period = clock_period / 2.0 if module.with_clock else None

    # ------------------------------------------------------------------
    # Simulator compatible API
    # ------------------------------------------------------------------
    def set(self, ref: Union[str, Signal], value: int) -> "TestbenchGenSimulator":
        sig = self._resolve(ref)
        if sig.kind not in {"input", "reg"}:
            raise ValueError("Only inputs and regs can be set directly.")
        self._sim.set(sig, value)
        self._events.append(
            _Event(
                action="set",
                delay=0.0,
                payload={
                    "signal": sig,
                    "value": value,
                },
            )
        )
        return self

    def get(self, ref: Union[str, Signal], *, signed: Optional[bool] = None) -> int:
        return self._sim.get(ref, signed=signed)

    def eval(self) -> "TestbenchGenSimulator":
        self._sim.eval()
        self._events.append(
            _Event(
                action="eval",
                delay=self.eval_delay,
                payload={"snapshot": self._snapshot()},
            )
        )
        return self

    def step(self, n: int = 1) -> "TestbenchGenSimulator":
        if not self.m.with_clock:
            raise RuntimeError("step() requires a clocked module.")
        for _ in range(n):
            self._sim.step(1)
            # Rising edge then falling edge with optional sampling delay afterwards.
            self._events.append(
                _Event("clock_high", self._half_period, {"clock": self.m.clk})
            )
            self._events.append(
                _Event("clock_low", self._half_period, {"clock": self.m.clk})
            )
            self._events.append(
                _Event(
                    action="sample",
                    delay=self.eval_delay,
                    payload={"snapshot": self._snapshot()},
                )
            )
        return self

    def reset(self, asserted: bool = True) -> "TestbenchGenSimulator":
        if not self.m.with_reset:
            return self
        self._sim.reset(asserted)
        value = 1 if asserted else 0
        self._events.append(
            _Event(
                action="set",
                delay=0.0,
                payload={
                    "signal": self.m.rst,
                    "value": value,
                },
            )
        )
        if asserted:
            # allow combinational paths to settle with reset asserted
            self._events.append(
                _Event(
                    action="eval",
                    delay=self.eval_delay,
                    payload={"snapshot": self._snapshot()},
                )
            )
        return self

    def deassert_reset(self) -> "TestbenchGenSimulator":
        if self.m.with_reset:
            self.reset(False)
        return self

    def peek_outputs(self) -> Dict[str, int]:
        return self._sim.peek_outputs()

    def peek_inputs(self) -> Dict[str, int]:
        return self._sim.peek_inputs()

    def get_mem(self, ref) -> List[int]:
        return self._sim.get_mem(ref)

    # Watch support mirrors the Python simulator by direct delegation.
    def watch(self, what, alias: Optional[str] = None):
        self._sim.watch(what, alias)
        return self

    def get_watch(self, name: str) -> int:
        return self._sim.get_watch(name)

    def clear_watches(self) -> None:
        self._sim.clear_watches()

    def list_signals(self) -> List[str]:
        return self._sim.list_signals()

    def peek(self, what):
        return self._sim.peek(what)

    def peek_next(self, reg_name):
        return self._sim.peek_next(reg_name)

    def log_expression_states(self, expr_list: Iterable[Expr]):
        return self._sim.log_expression_states(expr_list)

    # Trace/VCD capture: the inner simulator records the snapshots during the delegated
    # eval()/step() calls; these expose its switches and history, so one recorded run can
    # produce the Python-side VCD as well as the testbench.
    @property
    def trace_enabled(self) -> bool:
        return self._sim.trace_enabled

    @trace_enabled.setter
    def trace_enabled(self, value: bool) -> None:
        self._sim.trace_enabled = value

    @property
    def traced_expressions(self):
        return self._sim.traced_expressions

    @traced_expressions.setter
    def traced_expressions(self, exprs) -> None:
        self._sim.traced_expressions = exprs

    @property
    def trace_history(self):
        return self._sim.trace_history

    def record_expr_snapshot(self) -> None:
        self._sim.record_expr_snapshot()

    def get_traced_expr_names(self) -> Dict[int, str]:
        return self._sim.get_traced_expr_names()

    def get_trace_by_names(self) -> Dict[str, List[int]]:
        return self._sim.get_trace_by_names()

    # ------------------------------------------------------------------
    # Testbench emission
    # ------------------------------------------------------------------
    def to_testbench_lines(
        self,
        tb_module_name: Optional[str] = None,
        timescale: Optional[str] = "1ns/1ps",
        dump_vcd: bool = True,
        dumpfile: str = "dump.vcd",
    ) -> List[str]:
        """Emit a Verilog testbench that replays the recorded events."""

        if not self._events:
            raise RuntimeError("No events recorded – nothing to write.")

        tb_name = tb_module_name or f"{self.m.name}_tb"

        lines: List[str] = [VERILOG_BANNER, ""]

        if timescale:
            lines.append(f"`timescale {timescale}")
            lines.append("")

        lines.append(f"module {tb_name};")

        for s in self.inputs:
            decl = self._decl_line("reg", s)
            lines.append(f"  {decl}")
        for s in self.outputs:
            decl = self._decl_line("wire", s)
            lines.append(f"  {decl}")
        lines.append("")

        # DUT instantiation
        ports = []
        for p in self.m._ports:
            ports.append(f"    .{p.name}({p.name})")
        ports_csv = ",\n".join(ports)
        lines.append(f"  {self.m.name} dut (\n{ports_csv}\n  );")
        lines.append("")

        lines.append("  initial begin")

        if dump_vcd:
            self._append_dumpvars(lines, dumpfile)

        lines.append("")
        for s in self.inputs:
            init_literal = self._literal(0, s.typ.width)
            lines.append(f"    {s.name} = {init_literal};")
        lines.append("")

        for event in self._events:
            if event.delay > 0:
                lines.append(f"    #{self._format_delay(event.delay)};")
            if event.action == "set":
                sig: Signal = event.payload["signal"]
                literal = self._literal(event.payload["value"], sig.typ.width)
                if self.m._is_port(sig):
                    lines.append(f"    {sig.name} = {literal};")
                else:
                    lines.append(f"    dut.{sig.name} = {literal};")
            elif event.action == "eval":
                self._emit_snapshot(lines, event.payload["snapshot"], indent="    ")
            elif event.action in {"clock_high", "clock_low"}:
                clk = event.payload["clock"]
                val = "1'b1" if event.action == "clock_high" else "1'b0"
                lines.append(f"    {clk.name} = {val};")
            elif event.action == "sample":
                self._emit_snapshot(lines, event.payload["snapshot"], indent="    ")
            else:
                raise ValueError(f"Unknown event action: {event.action}")

        lines.append("    $display(\"Testbench completed successfully.\");")
        lines.append("    $finish;")
        lines.append("  end")
        lines.append("endmodule")

        return lines

    def to_testbench_str(
        self,
        tb_module_name: Optional[str] = None,
        timescale: Optional[str] = "1ns/1ps",
        dump_vcd: bool = True,
        dumpfile: str = "dump.vcd",
    ) -> str:
        lines = self.to_testbench_lines(
            tb_module_name=tb_module_name,
            timescale=timescale,
            dump_vcd=dump_vcd,
            dumpfile=dumpfile,
        )
        return "\n".join(lines) + "\n"

    def to_testbench_file(
        self,
        filepath: Union[str, Path],
        tb_module_name: Optional[str] = None,
        timescale: Optional[str] = "1ns/1ps",
        dump_vcd: bool = True,
        dumpfile: str = "dump.vcd",
    ) -> None:
        verilog_str = self.to_testbench_str(
            tb_module_name=tb_module_name,
            timescale=timescale,
            dump_vcd=dump_vcd,
            dumpfile=dumpfile,
        )
        with open(filepath, "w") as f:
            f.write(verilog_str)

    # ------------------------------------------------------------------
    # Data-driven Verilog testbench (reads vectors from file)
    # ------------------------------------------------------------------
    def to_data_driver_testbench_file(
        self,
        filepath: Union[str, Path],
        data_file: Union[str, Path],
        input_stimuli: Optional[List[str]] = None,
        outputs_expected: Optional[List[str]] = None,
        timescale: Optional[str] = "1ns/1ps",
        with_clk: bool = False,
        dump_vcd: bool = True,
        dumpfile: str = "dump.vcd",
    ) -> None:
        """
        Emit a Verilog testbench that reads test vectors from a data file and
        checks the DUT. Data lines should contain whitespace-separated values
        for the inputs (in the given order) followed by expected output(s).
        Optional header/comment lines that begin with '#' are ignored.

        input_stimuli: optional list of input signal names matching the column order
                       in the data file (do NOT include clk/rst here). If None,
                       defaults to all data inputs except clk/rst. Some inputs may
                       intentionally remain unstimulated.
        outputs_expected: optional list of output signal names corresponding to
                          expected value columns. If None, all outputs are checked.
        """
        if input_stimuli is None:
            data_inputs = [s for s in self.inputs if s.name not in ("clk", "rst")]
        else:
            data_inputs = [self._by_name[nm] for nm in input_stimuli]
        if not data_inputs:
            raise ValueError("No input stimuli specified or inferred.")

        # Include control signals for declaration/connection even if not in the data file
        all_inputs: List[Signal] = self.inputs

        if outputs_expected is None:
            expected_outputs = list(self.outputs)
        else:
            expected_outputs = [self._by_name[nm] for nm in outputs_expected]
        if not expected_outputs:
            raise ValueError("No outputs specified for checking.")

        tb_name = f"{self.m.name}_tb"

        lines: List[str] = [VERILOG_BANNER, ""]
        if timescale:
            lines.append(f"`timescale {timescale}")
            lines.append("")
        lines.append(f"module {tb_name};")
        for s in all_inputs:
            lines.append(f"  reg  {s.typ.range_str()} {s.name};")
        for s in expected_outputs:
            lines.append(f"  reg  {s.typ.range_str()} expected_{s.name};")
        for s in self.outputs:
            lines.append(f"  wire {s.typ.range_str()} {s.name};")
        lines.append("")
        # DUT instantiation
        ports = []
        for p in [*self.inputs, *self.outputs]:
            ports.append(f"    .{p.name}({p.name})")
        ports_csv = ",\n".join(ports)
        lines.append(f"  {self.m.name} dut (\n{ports_csv}\n  );")
        lines.append("")

        lines.append("  integer fd;")
        lines.append("  integer rc;")
        lines.append("  integer line_read;")
        lines.append("  integer line;")
        lines.append("  integer pass_cnt;")
        lines.append("  integer fail_cnt;")
        lines.append("  reg mismatch;")
        lines.append("  string line_buf;")
        lines.append("")
        lines.append("  initial begin")

        if dump_vcd:
            self._append_dumpvars(lines, dumpfile, trailing_blank=True)

        clk_sig = next((s for s in all_inputs if s.name == "clk"), None)
        if clk_sig is not None:
            lines.append(f"    {clk_sig.name} = 1'b0;")
        rst_sig = next((s for s in all_inputs if s.name == "rst"), None)
        if rst_sig is not None:
            lines.append(f"    {rst_sig.name} = 1'b0;")
        lines.append(f"    fd = $fopen(\"{data_file}\", \"r\");")
        lines.append("    if (fd == 0) begin")
        lines.append(f"      $display(\"Failed to open {data_file}\");")
        lines.append("      $finish;")
        lines.append("    end")
        lines.append("    line = 0; pass_cnt = 0; fail_cnt = 0;")
        lines.append("    while (!$feof(fd)) begin")
        fmt_parts = ["%d"] * (len(data_inputs) + len(expected_outputs))
        fmt = " ".join(fmt_parts)
        expected_names = [f"expected_{s.name}" for s in expected_outputs]
        lvalues = ", ".join([s.name for s in data_inputs] + expected_names)
        lines.append("      line_read = $fgets(line_buf, fd);")
        lines.append("      if (line_read == 0) begin")
        lines.append("        continue;")
        lines.append("      end")
        lines.append("      line = line + 1;")
        lines.append("      if (line_buf.len() == 0) begin")
        lines.append("        continue;")
        lines.append("      end")
        lines.append("      if (line_buf.substr(0, 0) == \"#\") begin")
        lines.append("        continue;")
        lines.append("      end")
        lines.append(f"      rc = $sscanf(line_buf, \"{fmt}\", {lvalues});")
        lines.append("      if (rc != {n}) begin".format(n=len(data_inputs) + len(expected_outputs)))
        lines.append("        $display(\"Skipping line %0d (rc=%0d)\", line, rc);")
        lines.append("        continue;")
        lines.append("      end")
        if with_clk and clk_sig is not None:
            lines.append("      #1;")
            lines.append(f"      {clk_sig.name} = 1'b1;")
            lines.append("      #1;")
            lines.append(f"      {clk_sig.name} = 1'b0;")
        else:
            lines.append("      #1; // allow combinational settle")
        lines.append("      mismatch = 0;")
        for s in expected_outputs:
            exp_name = f"expected_{s.name}"
            fmt_parts = [f" {inp.name}=%0d" for inp in data_inputs]
            fmt_parts.append(f" expected_{s.name}=%0d")
            fmt_parts.append(f" got={s.name}=%0d")
            fmt_str = "Mismatch line %0d:" + "".join(fmt_parts)
            fmt_args = ", ".join(["line"] + [inp.name for inp in data_inputs] + [exp_name, s.name])
            lines.append(f"      if ({s.name} !== {exp_name}) begin")
            lines.append("        mismatch = 1;")
            lines.append(f"        $display(\"{fmt_str}\", {fmt_args});")
            lines.append("      end")
        lines.append("      if (mismatch) begin")
        lines.append("        fail_cnt = fail_cnt + 1;")
        lines.append("      end else begin")
        lines.append("        pass_cnt = pass_cnt + 1;")
        lines.append("      end")
        lines.append("    end")
        lines.append("    $display(\"Finished: %0d passed, %0d failed\", pass_cnt, fail_cnt);")
        lines.append("    if (fail_cnt == 0 && pass_cnt > 0) begin")
        lines.append("      $display(\"Testbench completed successfully.\");")
        lines.append("    end")
        lines.append("    $finish;")
        lines.append("  end")
        lines.append("endmodule")

        with open(filepath, "w") as f:
            f.write("\n".join(lines) + "\n")

    def to_data_driver_testbench_file_incl_dat(
        self,
        filepath: Union[str, Path],
        vectors: "TestVectors",
        data_file: Union[str, Path],
        timescale: Optional[str] = "1ns/1ps",
        with_clk: bool = False,
        dump_vcd: bool = True,
        dumpfile: str = "dump.vcd",
    ) -> None:
        """Write vector data file and emit data-driven testbench from TestVectors."""
        if not vectors:
            raise ValueError("No test vectors provided.")

        _, first_inputs, first_outputs = vectors[0]
        input_names = list(first_inputs.keys())
        output_names = list(first_outputs.keys())

        write_vector_data_file(vectors, data_file)
        self.to_data_driver_testbench_file(
            filepath=filepath,
            data_file=data_file,
            input_stimuli=input_names,
            outputs_expected=output_names,
            timescale=timescale,
            with_clk=with_clk,
            dump_vcd=dump_vcd,
            dumpfile=dumpfile,
        )


    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _resolve(self, ref: Union[str, Signal]) -> Signal:
        if isinstance(ref, Signal):
            return ref
        if isinstance(ref, str):
            try:
                return self._by_name[ref]
            except KeyError as exc:
                raise KeyError(f"No signal named '{ref}' in module {self.m.name}.") from exc
        raise TypeError(f"Expected Signal or str, got {type(ref)}")

    def _snapshot(self) -> dict:
        outputs = self._sim.peek_outputs()
        watches = {}
        if hasattr(self._sim, "_watch_values"):
            watches = dict(self._sim._watch_values)
        return {"outputs": outputs, "watches": watches}

    def _literal(self, value: int, width: int) -> str:
        mask = (1 << width) - 1
        value &= mask
        if width == 1:
            return f"1'b{value & 1}"
        digits = max(1, (width + 3) // 4)
        return f"{width}'h{value:0{digits}x}"

    def _decl_line(self, kind: str, sig: Signal) -> str:
        sign = "signed " if sig.typ.signed else ""
        rng = sig.typ.range_str()
        if rng:
            return f"{kind} {sign}{rng} {sig.name};"
        return f"{kind} {sign}{sig.name};".replace("  ", " ")

    def _emit_snapshot(self, lines: List[str], snapshot: dict, *, indent: str) -> None:
        outputs = snapshot.get("outputs", {})
        if outputs:
            fmt_parts = []
            arg_names = ["$time"]
            for name, value in outputs.items():
                literal = self._literal(value, self._by_name[name].typ.width)
                lines.append(f"{indent}if ({name} !== {literal}) begin")
                lines.append(f"{indent}  $fatal(1, \"Expected {name}=%0h, got %0h\", {literal}, {name});")
                lines.append(f"{indent}end")
                fmt_parts.append(f" {name}=%0h")
                arg_names.append(name)
            fmt = "[%0t]" + "".join(fmt_parts)
            arg_list = ", ".join(arg_names)
            lines.append(f"{indent}$display(\"{fmt}\", {arg_list});")
        if snapshot.get("watches"):
            watch_names = ", ".join(sorted(snapshot["watches"].keys()))
            lines.append(f"{indent}// Watches captured in Python only: {watch_names}")

    def _format_delay(self, value: float) -> str:
        if abs(value - int(value)) < 1e-9:
            return str(int(value))
        return f"{value:g}"

    def _append_dumpvars(
        self, lines: List[str], dumpfile: str, *, trailing_blank: bool = False
    ) -> None:
        lines.append("")
        lines.append("    `ifndef STRINGIFY")
        lines.append("      `define STRINGIFY(x) `\"x`\"")
        lines.append("    `endif")
        lines.append("    string __vcd_file;")
        lines.append("    __vcd_file = \"\";")
        lines.append("    `ifdef VCD_FILE_PATH")
        lines.append("      __vcd_file = `STRINGIFY(`VCD_FILE_PATH);")
        lines.append("    `endif")
        lines.append("    if (__vcd_file != \"\") begin")
        lines.append("      $dumpfile(__vcd_file);")
        lines.append(f"    end else begin")
        lines.append(f'      $dumpfile("{dumpfile}");')
        lines.append("    end")
        lines.append("    $dumpvars();")
        if trailing_blank:
            lines.append("")


__all__ = ["TestbenchGenSimulator", "write_vector_data_file"]
