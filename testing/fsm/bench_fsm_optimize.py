"""Benchmark: measure AIG gates / depth for the two optimization wrappers.

Two cases, mirroring ``docs/README_fsm_optimization.md``:

1.  **Sequence-detector FSM (7 states with redundancy).** The canonical
    ``case10`` body — four pairs of states are behaviourally equivalent,
    so Hopcroft minimises 7 → 4 classes; the encoding search picks the
    best bit-assignment over the surviving classes.

2.  **8-opcode ALU dispatch.** Pure combinational mux-tree over a
    ``State`` enum. No FSM, so only ``optimized_encoding`` applies.

Both wrappers use ``search="swap"`` so the search is bounded (4
restarts × 200 iters = 800 ``cost_fn`` calls per case) and deterministic
(internal ``random.Random(0)``). The benchmark is the same code path the
unit tests exercise, run on slightly larger inputs to make the
optimisation effect visible.

Run::

    PYTHONPATH=src python testing/fsm/bench_fsm_optimize.py

Expected runtime: ~2–3 minutes total (aigverse is the bottleneck).
"""
from __future__ import annotations

import time

from spire.helpers import get_yosys_metrics
from spire.optimize.fsm._emit import restore_encoding
from spire.expr import Bool, UInt, mux
from spire.control_structures import case_, default, else_, if_, switch_
from spire.component import Netlist
from spire.state import (
    Encoding, State, optimized_encoding, optimized_fsm, state,
)


# ============================================================================
# Case 1 — 7-state FSM with redundancy (case10 from the unit tests).
# Equivalence classes under Hopcroft: {S0, S3}, {S1}, {S2, S4, S6}, {S5}.
# ============================================================================

class Seq(State, encoding=Encoding.BINARY):
    S0 = state(); S1 = state(); S2 = state()
    S3 = state(); S4 = state(); S5 = state(); S6 = state()


_SEQ_BASELINE = {f"S{i}": i for i in range(7)}


def _reset_seq() -> None:
    restore_encoding(Seq, _SEQ_BASELINE)


def _build_seq(name: str):
    m = Netlist(name, with_clock=True, with_reset=False)
    x   = m.input(Bool(), "x")
    out = m.output(UInt(1), "out")
    reg = m.reg(Seq.typ, "state_reg", init=Seq.S0)
    return m, x, out, reg


def _seq_body(reg, out, x) -> None:
    out <<= 0
    with switch_(reg):
        with case_(Seq.S0):
            out <<= 1
            with if_(x): reg <<= Seq.S2
            with else_(): reg <<= Seq.S1
        with case_(Seq.S1):
            out <<= 1
            with if_(x): reg <<= Seq.S5
            with else_(): reg <<= Seq.S3
        with case_(Seq.S2):
            out <<= 0
            with if_(x): reg <<= Seq.S4
            with else_(): reg <<= Seq.S5
        with case_(Seq.S3):
            out <<= 1
            with if_(x): reg <<= Seq.S6
            with else_(): reg <<= Seq.S1
        with case_(Seq.S4):
            out <<= 0
            with if_(x): reg <<= Seq.S2
            with else_(): reg <<= Seq.S5
        with case_(Seq.S5):
            out <<= 0
            with if_(x): reg <<= Seq.S3
            with else_(): reg <<= Seq.S4
        with case_(Seq.S6):
            out <<= 0
            with if_(x): reg <<= Seq.S6
            with else_(): reg <<= Seq.S5
        with default():
            reg <<= Seq.S0


def _seq_stats(m) -> dict:
    # Sequential — aigverse doesn't take latches, so use yosys cell/transistor
    # counts after the standard ``proc; opt; fsm; opt; techmap; abc -fast; opt``
    # flow (``via_aig=False`` matches the rtl_rewriter benchmark recipe).
    return get_yosys_metrics(m, via_aig=False)


def bench_seq_baseline() -> dict:
    _reset_seq()
    m, x, out, reg = _build_seq("seq_base")
    _seq_body(reg, out, x)
    return _seq_stats(m)


def bench_seq_encoding_only() -> dict:
    _reset_seq()
    m, x, out, reg = _build_seq("seq_enc")
    with optimized_encoding(Seq, module=m, objective="cells", search="swap"):
        _seq_body(reg, out, x)
    return _seq_stats(m)


def bench_seq_fsm_only() -> dict:
    _reset_seq()
    m, x, out, reg = _build_seq("seq_fsm")
    with optimized_fsm(reg, module=m, minimize=True, outputs=[out]):
        _seq_body(reg, out, x)
    return _seq_stats(m)


def bench_seq_both() -> dict:
    _reset_seq()
    m, x, out, reg = _build_seq("seq_both")
    with optimized_encoding(Seq, module=m, objective="cells", search="swap"):
        with optimized_fsm(reg, module=m, minimize=True, outputs=[out]):
            _seq_body(reg, out, x)
    return _seq_stats(m)


# ============================================================================
# Case 2 — 8-opcode control-word decoder (no FSM, pure combinational).
# A small CPU-style decoder: each output is a disjunction of opcode
# equalities. Encoding determines whether each disjunction collapses to
# a single bit-test (cheap) or stays a wide OR-of-equalities (expensive).
# ============================================================================

class Op(State, encoding=Encoding.BINARY):
    LOAD = state(); STORE = state()                # memory ops
    ADD  = state(); SUB   = state()                # arithmetic ops
    AND_ = state(); OR_   = state()                # bitwise ops
    JMP  = state(); BEQ   = state()                # branch ops


_OP_BASELINE = {n: i for i, n in enumerate(
    ["LOAD", "STORE", "ADD", "SUB", "AND_", "OR_", "JMP", "BEQ"])}


def _reset_op() -> None:
    restore_encoding(Op, _OP_BASELINE)


def _build_decoder(name: str):
    m = Netlist(name, with_clock=False, with_reset=False)
    op_in = m.input(Op.typ, "op")
    mem   = m.output(Bool(), "mem_en")      # LOAD | STORE
    we    = m.output(Bool(), "we")          # STORE
    alu   = m.output(Bool(), "alu_en")      # ADD | SUB | AND_ | OR_
    arith = m.output(Bool(), "is_arith")    # ADD | SUB
    br    = m.output(Bool(), "br_en")       # JMP | BEQ
    return m, op_in, mem, we, alu, arith, br


def _decoder_body(op_in, mem, we, alu, arith, br) -> None:
    mem   <<= (op_in == Op.LOAD) | (op_in == Op.STORE)
    we    <<= (op_in == Op.STORE)
    alu   <<= ((op_in == Op.ADD)  | (op_in == Op.SUB)
             | (op_in == Op.AND_) | (op_in == Op.OR_))
    arith <<= (op_in == Op.ADD)   | (op_in == Op.SUB)
    br    <<= (op_in == Op.JMP)   | (op_in == Op.BEQ)


def _decoder_stats(m) -> dict:
    return get_yosys_metrics(m, via_aig=False)


def bench_alu_baseline() -> dict:
    _reset_op()
    m, op_in, mem, we, alu, arith, br = _build_decoder("decode_base")
    _decoder_body(op_in, mem, we, alu, arith, br)
    return _decoder_stats(m)


def bench_alu_encoding() -> dict:
    _reset_op()
    m, op_in, mem, we, alu, arith, br = _build_decoder("decode_enc")
    with optimized_encoding(Op, module=m, objective="cells", search="swap"):
        _decoder_body(op_in, mem, we, alu, arith, br)
    return _decoder_stats(m)


# ============================================================================
# Driver
# ============================================================================

def _row_seq(label: str, stats: dict, baseline: dict | None = None) -> str:
    c = int(stats["num_cells"])
    t = int(stats["estimated_num_transistors"])
    if baseline is None:
        return f"{label:<32s} | cells={c:5d}  transistors={t:6d}"
    bc, bt = int(baseline["num_cells"]), int(baseline["estimated_num_transistors"])
    pc = (c - bc) / bc * 100.0 if bc else 0.0
    pt = (t - bt) / bt * 100.0 if bt else 0.0
    return (f"{label:<32s} | cells={c:5d} ({pc:+6.1f}%)  "
            f"transistors={t:6d} ({pt:+6.1f}%)")


def _row_alu(label: str, stats: dict, baseline: dict | None = None) -> str:
    c = int(stats["num_cells"])
    t = int(stats["estimated_num_transistors"])
    if baseline is None:
        return f"{label:<32s} | cells={c:5d}  transistors={t:6d}"
    bc, bt = int(baseline["num_cells"]), int(baseline["estimated_num_transistors"])
    pc = (c - bc) / bc * 100.0 if bc else 0.0
    pt = (t - bt) / bt * 100.0 if bt else 0.0
    return (f"{label:<32s} | cells={c:5d} ({pc:+6.1f}%)  "
            f"transistors={t:6d} ({pt:+6.1f}%)")


def main() -> None:
    print("=" * 70)
    print("Case 1 — 7-state sequence detector (sequential — yosys cells)")
    print("        Hopcroft merges 7 states into 4 equivalence classes.")
    print("=" * 70)
    t0 = time.time()
    base = bench_seq_baseline();              print(_row_seq("baseline",             base))
    enc  = bench_seq_encoding_only();         print(_row_seq("+ optimized_encoding", enc, base))
    fsm  = bench_seq_fsm_only();              print(_row_seq("+ optimized_fsm",      fsm, base))
    both = bench_seq_both();                  print(_row_seq("+ both (nested)",      both, base))
    print(f"[{time.time() - t0:.1f}s]")

    print()
    print("=" * 70)
    print("Case 2 — 8-opcode CPU control-word decoder (combinational)")
    print("        Each output is a disjunction of opcode-equality tests;")
    print("        encoding choice can collapse some ORs into single bit-tests.")
    print("=" * 70)
    t0 = time.time()
    base = bench_alu_baseline();              print(_row_alu("baseline",             base))
    enc  = bench_alu_encoding();              print(_row_alu("+ optimized_encoding", enc, base))
    print(f"[{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
