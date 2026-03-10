import contextlib
import ctypes
import hashlib
import os
import random
import sys
import tempfile
import time
from typing import Callable, Dict, List, Optional, Tuple

from aigverse import DepthAig, aig_cut_rewriting, aig_resubstitution, balancing, sop_refactoring
from pyosys import libyosys as ys

from sprouthdl.aig.aig_aigerverse import _get_aag_sym, conv_aag_into_aig, conv_aig_into_aag
from sprouthdl.arithmetic.int_multipliers.eval.testvector_generation import TestVectors
from sprouthdl.sprouthdl import Expr, Op2
from sprouthdl.sprouthdl_aiger import AigerExporter, AigerImporter
from sprouthdl.sprouthdl_module import IOCollector
from sprouthdl.sprouthdl_module import Module
from sprouthdl.sprouthdl_simulator import Simulator


DEFAULT_N_ITER_OPTIMIZATIONS = 3


@contextlib.contextmanager
def _suppress_output(stderr: bool = True):
    """Suppress all writes to stdout (and optionally stderr) at both Python and C/extension level."""
    sys.stdout.flush()
    if stderr:
        sys.stderr.flush()
    try:
        ctypes.CDLL(None).fflush(None)  # flush all C stdio buffers
    except Exception:
        pass
    saved_out = os.dup(1)
    saved_err = os.dup(2) if stderr else None
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    if stderr:
        os.dup2(devnull, 2)
    os.close(devnull)
    old_pyout = sys.stdout
    old_pyerr = sys.stderr
    sys.stdout = open(os.devnull, 'w')
    if stderr:
        sys.stderr = sys.stdout
    try:
        yield
    finally:
        sys.stdout.flush()
        sys.stdout.close()
        sys.stdout = old_pyout
        sys.stderr = old_pyerr
        try:
            ctypes.CDLL(None).fflush(None)
        except Exception:
            pass
        os.dup2(saved_out, 1)
        os.close(saved_out)
        if saved_err is not None:
            os.dup2(saved_err, 2)
            os.close(saved_err)


def _resolve_n_iter_optimizations(n_iter_optimizations: Optional[int]) -> int:
    """Resolve None to the shared default iteration count."""
    return DEFAULT_N_ITER_OPTIMIZATIONS if n_iter_optimizations is None else n_iter_optimizations


def optimize_aig_elaborate(aig_in, n_iter_optimizations: Optional[int] = None) -> dict:

    n_iter_optimizations = _resolve_n_iter_optimizations(n_iter_optimizations)

    # Track the best gate and depth
    best_gate = None
    best_depth = None

    best_stats = None
    best_aig = None

    # Two sequences are tested
    op_seq_tuple = ([aig_resubstitution, sop_refactoring, aig_cut_rewriting], [aig_resubstitution, sop_refactoring, aig_cut_rewriting, balancing])

    # Loop through the two sequences
    for op_seq in op_seq_tuple:
        # Load the aig file
        aig = aig_in.clone()
        # Loop through the sequences 3 times
        for i in range(n_iter_optimizations):
            for op in op_seq:
                # Perform the operation
                op(aig)
                # Perform analysis
                depth_aig = DepthAig(aig)
                # Extract stats
                stats = {
                    "num_pis": len(aig.pis()),
                    "num_pos": len(aig.pos()),
                    "num_gates": len(aig.gates()),
                    "size": aig.size(),
                    "depth": depth_aig.num_levels(),
                }
                # Store best stats dict
                if best_gate is None or stats["num_gates"] < best_gate:
                    best_gate = stats["num_gates"]
                    best_depth = stats["depth"]
                    best_stats = stats
                    best_aig = aig
                elif stats["num_gates"] == best_gate and stats["depth"] < best_depth:
                    best_depth = stats["depth"]
                    best_stats = stats
                    best_aig = aig

    return best_aig, best_stats


def optimize_aig_simple(aig, n_iter_optimizations: Optional[int] = None) -> dict:
    n_iter_optimizations = _resolve_n_iter_optimizations(n_iter_optimizations)
    for i in range(n_iter_optimizations):
        for optimization in [aig_resubstitution, sop_refactoring, aig_cut_rewriting]: #, balancing]: balancing increases size
            optimization(aig)
    return aig

def optimize_aag(aag_lines: List[str], n_iter_optimizations: Optional[int] = None, simple=False) -> List[str]:

    # convert to aigverse object
    aig = conv_aag_into_aig(aag_lines)    

    if simple:
        aig = optimize_aig_simple(aig, n_iter_optimizations=n_iter_optimizations)
    else:
        aig, _ = optimize_aig_elaborate(aig, n_iter_optimizations=n_iter_optimizations)

    # aig back to aag
    aag_optimized = conv_aig_into_aag(aig, symbols=_get_aag_sym(aag_lines))

    return aag_optimized


def refactor_module_to_aig(module: Module, optimize=True, n_iter_optimizations: Optional[int] = None) -> Module:
    # get AIG
    aag = AigerExporter(module).get_aag()
    if optimize:
        aag = optimize_aag(aag, n_iter_optimizations=n_iter_optimizations)
    m_aig = AigerImporter(aag).get_sprout_module()
    try:
        spec = module.component.get_spec()
    except (AttributeError, TypeError):
        spec = module.get_spec()
    IOCollector().group(m_aig, spec) # regroup I/Os to match original port widths
    return m_aig


def get_aig_stats(m: Module, n_iter_optimizations: Optional[int] = None, simple=False) -> dict:
    aag_lines = AigerExporter(m).get_aag()
    aig = conv_aag_into_aig(aag_lines)

    if simple:
        aig = optimize_aig_simple(aig, n_iter_optimizations=n_iter_optimizations)
    else:
        aig, _ = optimize_aig_elaborate(aig, n_iter_optimizations=n_iter_optimizations)

    depth_aig = DepthAig(aig)

    stats = {
        'num_pis': len(aig.pis()),
        'num_pos': len(aig.pos()),
        'num_gates': len(aig.gates()),
        'size': aig.size(),
        'depth': depth_aig.num_levels(),
    }
    return stats


# -- sim

def run_vectors(
    m: Module, vectors: TestVectors, 
    decoder: Callable[[int], float] | None = None, exprs: Optional[List[Expr]] = None,
    use_signed: bool = False,
    raise_on_fail: bool = True,
    print_on_pass: bool = False,
) -> None:
    sim = Simulator(m)
    sim.trace_enabled = True if exprs is not None else False
    sim.traced_expressions = exprs if exprs is not None else []
    run_vectors_on_simulator(
        sim, vectors, decoder=decoder, use_signed=use_signed,
        raise_on_fail=raise_on_fail, print_on_pass=print_on_pass,
    )
    return sim.trace_history


def run_vectors_on_simulator(
    sim: Simulator, vectors: TestVectors,
    decoder: Callable[[int], float] | None = None,
    use_signed: bool = False,
    raise_on_fail: bool = True,
    print_on_pass: bool = False,
    with_clk: bool = False,
    test_name: Optional[str] = None,
) -> None:
    
    """
    Generic runner:
      vectors: list of (label, inputs{name->int}, expected{name->int})
    Prints mismatches; raises AssertionError at the end if any failed.
    """

    states_list = []

    fails = 0
    for name, ins, outs in vectors:
        for k, v in ins.items():
            if k[0] == "_":
                continue
            sim.set(k, v)
        if with_clk:
            sim.step()
        sim.eval()
        bad = []
        for oname, exp in outs.items():
            if oname[0] == "_":
                continue
            got_raw = sim.peek(oname) #sim.get(oname)
            got_signed = sim.get(oname)
            got = got_signed if use_signed else got_raw
            if got != exp:
                if decoder and oname == "y":
                    bad.append(f"{oname}: got=0x{got:0X} ({decoder(got):.8g})  exp=0x{exp:0X} ({decoder(exp):.8g})")
                else:
                    bad.append(f"{oname}: got=0x{got:0X}  exp=0x{exp:0X}")
            else:
                if print_on_pass:
                    if decoder and oname == "y":
                        print(f"PASS {name}: {oname}=0x{got:0X} ({decoder(got):.8g})")
                    else:
                        print(f"PASS {name}: {oname}=0x{got:0X} ({got})")
                    pass
        if bad:
            fails += 1
            print(f"FAIL  {name}:  " + " | ".join(bad))
    
    test_name = "" if test_name is None else test_name + " - "
    print(f"{test_name}Number of vectors: {len(vectors)}, {fails} failures")
    if fails and raise_on_fail:
        raise AssertionError(f"{fails}/{len(vectors)} vectors failed")

    return fails

def get_switch_count(states_list) -> float:
    # get all ids from step 0
    ids = set(states_list[0].keys())
    # count number of switches per id
    switches = {i: 0 for i in ids}
    for i in ids:
        last = states_list[0][i]
        for s in states_list[1:]:
            if s[i] != last:
                switches[i] += 1
                last = s[i]
    # sum up all switches
    total_switches = sum(switches.values())
    return total_switches / len(states_list)  # average per vector


def sim_and_switch_count(module: Module, vectors: TestVectors) -> Module:

    m_aig = refactor_module_to_aig(module)

    # AIG network test sim
    run_vectors(m_aig, vectors[0:min(16, len(vectors))])  # smoke test

    exprs = m_aig.all_exprs()
    all_ands = [e for e in exprs if isinstance(e, Op2) and e.op == "&"]

    def run_and_count(vecs_run) -> int:
        states_list = run_vectors(m_aig, vecs_run, exprs=all_ands)
        return get_switch_count(states_list)             

    switches = run_and_count(vectors)

    return switches


def _run_yosys_metric_flow(read_cmd: str, deepsyn: bool = False, auto_top: bool = False, suppress_stderr: bool = True) -> dict:
    fd, stat_tmp_file = tempfile.mkstemp(suffix=".json")
    os.close(fd)

    abc_tmp_file = None
    if deepsyn:
        # create abc script temp file
        fd_abc, abc_tmp_file = tempfile.mkstemp(suffix=".abc")
        os.close(fd_abc)
        with open(abc_tmp_file, "w") as f:
            # I: stop after I iterations without improvement, J: random initializations, T: timeout (seconds)
            f.write("strash; &get -n; &deepsyn J 5 -T 100; &put\n")

    try:
        with _suppress_output(stderr=suppress_stderr):
            ys.run_pass("design -reset")
            ys.run_pass(read_cmd)
            if auto_top:
                ys.run_pass("hierarchy -check -auto-top")
            ys.run_pass("rename -top top")
            ys.run_pass("hierarchy -check")
            ys.run_pass("proc; opt; fsm; memory; opt")
            if deepsyn:
                ys.run_pass(f"abc -script {abc_tmp_file}")
            ys.run_pass("techmap; opt; abc -fast; opt")
            ys.run_pass("rename -wire -suffix _reg t:*DFF*")
            ys.run_pass(f"tee -q -o {stat_tmp_file} stat -top top -tech cmos -json")

        # read stats from json file
        import json
        with open(stat_tmp_file, "r") as f:
            stats = json.load(f)
    finally:
        os.remove(stat_tmp_file)
        if abc_tmp_file and os.path.exists(abc_tmp_file):
            os.remove(abc_tmp_file)

    stats = stats["modules"]["\\top"]
    # plus in case of registers, because they are not counted
    stats["estimated_num_transistors"] = int(stats["estimated_num_transistors"].replace("+", ""))
    return stats


def extract_yosys_metrics(aag_lines: list[str], deepsyn=False) -> dict:
    fd, aag_tmp_file = tempfile.mkstemp(suffix=".aag")
    os.close(fd)
    with open(aag_tmp_file, "w") as f:
        f.write("\n".join(aag_lines) + "\n")

    try:
        return _run_yosys_metric_flow(f"read_aiger {aag_tmp_file}", deepsyn=deepsyn, auto_top=False)
    finally:
        os.remove(aag_tmp_file)


def extract_yosys_metrics_from_verilog(verilog_lines: list[str], deepsyn=False) -> dict:
    fd, verilog_tmp_file = tempfile.mkstemp(suffix=".v")
    os.close(fd)
    with open(verilog_tmp_file, "w") as f:
        f.write("\n".join(verilog_lines) + "\n")

    try:
        return _run_yosys_metric_flow(f"read_verilog -sv {verilog_tmp_file}", deepsyn=deepsyn, auto_top=True)
    finally:
        os.remove(verilog_tmp_file)


def extract_yosys_metrics_from_verilog_file(filename: str, deepsyn=False) -> dict:
    with open(filename, "r") as f:
        verilog_lines = f.read().splitlines()
    return extract_yosys_metrics_from_verilog(verilog_lines, deepsyn=deepsyn)


def get_yosys_metrics(m: Module, n_iter_optimizations: Optional[int] = None, deepsyn=False, suppress_stderr: bool = True) -> int:
    with _suppress_output(stderr=suppress_stderr):
        print("Exporting AAG...")
        aag_lines = AigerExporter(m).get_aag()
        print("Exporting AAG done")

        if n_iter_optimizations is None or n_iter_optimizations > 0:
            aag_lines = optimize_aag(aag_lines, n_iter_optimizations=n_iter_optimizations)

        print("Optimizing AAG done")
    stat = extract_yosys_metrics(aag_lines, deepsyn=deepsyn)
    return stat

def get_transistor_count_from_stats(stats: dict) -> int:
    return stats["estimated_num_transistors"]

def get_yosys_transistor_count(m: Module, n_iter_optimizations: Optional[int] = None, deepsyn=False) -> int:
    stats = get_yosys_metrics(m, n_iter_optimizations=n_iter_optimizations, deepsyn=deepsyn)
    return get_transistor_count_from_stats(stats)
