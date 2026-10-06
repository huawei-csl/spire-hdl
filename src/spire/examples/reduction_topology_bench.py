"""Measure spire.reduce topologies — the numbers in docs/README_reductions.md.

Four small but representative designs, each built with every applicable
topology, measured two ways:

  * raw AIG (aigverse)      — what the simulator / AIG export / aig_adp see
  * yosys `synth` + `ltp`   — post-synthesis cells and gate levels

Run:  python testing/examples/reduction_topology_bench.py
"""

import os
import re
import shutil
import subprocess
import tempfile

from spire.component import Component, Netlist
from spire.expr import UInt, mux
from spire.reduce import argmax_, max_, prefix_scan, reduce_tree

import aigverse


def _netlist(n, w, name="Bench"):
    m = Netlist(name, with_clock=False, with_reset=False)
    return m, [m.input(UInt(w), f"x{i}") for i in range(n)]


def aig_stats(m):
    aag = Component.from_netlist(m).to_aag(name="M")
    text = aag if isinstance(aag, str) else "\n".join(str(x) for x in aag)
    with tempfile.NamedTemporaryFile("w", suffix=".aag", delete=False) as f:
        f.write(text if text.endswith("\n") else text + "\n")
        path = f.name
    aig = aigverse.read_ascii_aiger_into_aig(path)
    os.unlink(path)
    return aig.num_gates(), aigverse.DepthAig(aig).num_levels()


def yosys_stats(m):
    if shutil.which("yosys") is None:
        return None, None
    with tempfile.NamedTemporaryFile("w", suffix=".v", delete=False) as f:
        f.write(m.to_verilog())
        path = f.name
    out = subprocess.run(
        ["yosys", "-p", f"read_verilog {path}; synth -auto-top; ltp -noff; stat"],
        capture_output=True, text=True, timeout=300).stdout
    os.unlink(path)
    depth = re.findall(r"length=(\d+)", out)
    cells = re.findall(r"Number of cells:\s+(\d+)", out)
    return (int(cells[-1]) if cells else None), (int(depth[-1]) if depth else None)


# -- the designs --------------------------------------------------------------

def peak_select(topology, n=20, w=8):
    """Max of 20 values (peak detector / winner-take-all select)."""
    m, xs = _netlist(n, w)
    if topology == "loop":  # the natural hand-written running max
        acc = xs[0]
        for i in range(1, n):
            acc = mux(xs[i] > acc, xs[i], acc)
    else:
        acc = max_(xs, topology)
    m.output(UInt(w), "y").assign(acc)
    return m


def pool_argmax(topology, n=16, w=8):
    """Argmax of 16 activations (pooling winner: value + index)."""
    m, xs = _netlist(n, w)
    v, i = argmax_(xs, topology)
    m.output(UInt(w), "v").assign(v)
    m.output(UInt(4), "i").assign(i)
    return m


def watermark(topology, n=16, w=8):
    """Running max with ALL prefixes tapped (per-lane watermark)."""
    m, xs = _netlist(n, w)
    op = lambda a, b: mux(a >= b, a, b)
    if topology == "loop":
        outs, acc = [xs[0]], xs[0]
        for i in range(1, n):
            acc = op(acc, xs[i])
            outs.append(acc)
    else:
        outs = prefix_scan(op, xs, topology)
    for i, o in enumerate(outs):
        m.output(UInt(w), f"y{i}").assign(o)
    return m


def parity64(topology):
    """Parity of a 64-bit word (XOR reduce)."""
    m, xs = _netlist(64, 1)
    m.output(UInt(1), "y").assign(reduce_tree(lambda a, b: a ^ b, xs, topology))
    return m


CASES = [
    ("peak_select (max of 20)", peak_select, ["loop", "tree", "huffman", "matrix"]),
    ("pool_argmax (16 values)", pool_argmax, ["chain", "tree", "matrix"]),
    ("watermark (16 prefixes)", watermark, ["loop", "sklansky", "brentkung", "koggestone"]),
    ("parity64 (xor reduce)", parity64, ["chain", "tree"]),
]


if __name__ == "__main__":
    for title, build, topologies in CASES:
        print(f"\n== {title}")
        print(f"{'topology':12} {'AIG gates':>9} {'AIG depth':>9} {'syn cells':>9} {'syn depth':>9}")
        for topo in topologies:
            m = build(topo)
            g, d = aig_stats(m)
            c, ld = yosys_stats(m)
            print(f"{topo:12} {g:9d} {d:9d} "
                  f"{c if c is not None else '-':>9} {ld if ld is not None else '-':>9}")
