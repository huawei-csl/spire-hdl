from pyosys import libyosys as ys


import os
import tempfile
from typing import List, Tuple

from sprouthdl.aig.aig_aigerverse import file_to_lines


def verilog_to_aag_via_yosys(
    verilog_path: str,
    *,
    top: str | None = None,
    aag_out_path: str | None = None,
    tie_undriven: str | None = None,  # {"zero","one","random"} or None
    embed_symbols: bool = True,
    no_startoffset: bool = True,
    map_out_path: str | None = None,  # write_aiger -map
) -> Tuple[str, str | None]:
    """Run yosys: read_verilog → synth -flatten → aigmap → write_aiger -ascii […]. Returns (aag_path, map_path_or_None)."""
    # reset yosys design to avoid crosstalk between runs
    ys.run_pass("design -reset")
    if aag_out_path is None:
        fd, aag_out_path = tempfile.mkstemp(suffix=".aag")
        os.close(fd)
    if map_out_path is None:
        fd, map_out_path = tempfile.mkstemp(suffix=".map")
        os.close(fd)

    ys.run_pass("design -reset")
    ys.run_pass(f"read_verilog -sv {verilog_path}")
    ys.run_pass("hierarchy -check " + ("-auto-top" if top is None else f"-top {top}"))
    ys.run_pass("synth -flatten")
    if tie_undriven in {"zero", "one", "random"}:
        flag = {"zero": "-zero", "one": "-one", "random": "-random"}[tie_undriven]
        ys.run_pass(f"setundef -undriven {flag}")
    ys.run_pass("aigmap")

    opts = ["-ascii"]
    if embed_symbols:
        opts.append("-symbols")
    if no_startoffset:
        opts.append("-no-startoffset")
    if map_out_path:
        opts += ["-map", map_out_path]
    ys.run_pass(f"write_aiger {' '.join(opts)} {aag_out_path}")
    return aag_out_path, (map_out_path if os.path.getsize(map_out_path) > 0 else None)


def verilog_to_aag_lines_via_yosys(
    verilog_path: str,
    *,
    top: str | None = None,
    tie_undriven: str | None = None,  # {"zero","one","random"} or None
    embed_symbols: bool = True,
    no_startoffset: bool = True,
) -> List[str]:
    """Run yosys: read_verilog → synth -flatten → aigmap → write_aiger -ascii […]. Returns AAG lines."""
    aag_path, _ = verilog_to_aag_via_yosys(
        verilog_path,
        top=top,
        tie_undriven=tie_undriven,
        embed_symbols=embed_symbols,
        no_startoffset=no_startoffset,
    )
    return file_to_lines(aag_path)


def aig_file_to_aag_lines_via_yosys(aag_path: str, map_file: str|None = None) -> List[str]:
    # also works for binary files
    ys.run_pass("design -reset")
    aag_out_path = None
    map_out_path = None
    if aag_out_path is None:
        fd, aag_out_path = tempfile.mkstemp(suffix=".aag")
        os.close(fd)
    if map_out_path is None:
        fd, map_out_path = tempfile.mkstemp(suffix=".map")
        os.close(fd)
    ys.run_pass("design -reset")
    if map_file:
        opt_map = f"-map {map_file} "
    else:
        opt_map = ""
    ys.run_pass(f"read_aiger {opt_map}{aag_path}")
    ys.run_pass(f"write_aiger -symbols -ascii {aag_out_path}")
    aag_lines = file_to_lines(aag_out_path)

    # clean up, in case of [<index>] appearing twice in a line
    # e.g. "o1 y[10][10]" -> "o1 y[10]"
    aag_lines_clean: List[str] = []
    for line in aag_lines:
        if line.count('[') == 2:
            second_bracket: int = line.index('[', line.index('[') + 1)
            line = line[:second_bracket]
        aag_lines_clean.append(line)

    return aag_lines_clean