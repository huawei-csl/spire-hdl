"""Regression tests for the AIGER barrel shifter (``spirehdl_aiger``).

These tests were added after a subtle bug in ``bv_shift_left`` / ``bv_shift_right``:
the mux-based barrel shifter iterated over the shift-amount bits and simply
``break``-ed once a stage weight reached ``w_out``. Every higher shift bit was
then silently dropped, so an out-of-range shift aliased to a small in-range one.

Concretely, with ``x: UInt(16)``:

    x << 16   # AIG returned x      (should be 0)
    x >> 16   # AIG returned x      (should be 0)
    x << tgt  # for tgt >= 16, AIG used only tgt & 0xF

The Verilog ``<<`` / ``>>`` operators zero-fill an out-of-range shift natively,
so this was a silent divergence between the Verilog and AIGER back-ends: a
design correct in Verilog simulation was mis-compiled to AIG.

The fix keeps the barrel-shifter structure but OR-reduces the ignored high
shift bits into an ``overflow`` flag and forces the result to 0 when set. The
guard is synthesized *only* when the shift-amount operand is actually wide
enough to encode an out-of-range shift -- see
``test_overflow_guard_*`` below.
"""
from __future__ import annotations

import pytest

from spirehdl.spirehdl_aiger import _AIG, lit_const0
from spirehdl.spirehdl import UInt, Const, cat, reset_shared_cache
from spirehdl.spirehdl_module import Module
from spirehdl.helpers import refactor_module_to_aig, run_vectors


# ---------------------------------------------------------------------------
# Tiny combinational AIG evaluator
# ---------------------------------------------------------------------------
# ``_AIG.ands`` is built in topological order (``mk_and`` allocates a node only
# after its inputs exist), so a single forward pass evaluates the whole network.

def _eval_lits(aig: _AIG, result_lits, assignment) -> list[int]:
    """Evaluate ``result_lits`` for a given input ``assignment``.

    ``assignment`` maps input literals (even ints from ``aig.inputs``) to 0/1.
    """
    var_val = {0: 0}  # variable 0 is the constant-false node
    for lit in aig.inputs:
        var_val[lit >> 1] = assignment.get(lit, 0) & 1
    for lhs, r0, r1 in aig.ands:
        v0 = var_val[r0 >> 1] ^ (r0 & 1)
        v1 = var_val[r1 >> 1] ^ (r1 & 1)
        var_val[lhs >> 1] = v0 & v1
    return [var_val[l >> 1] ^ (l & 1) for l in result_lits]


def _lits_to_int(bits) -> int:
    return sum(b << i for i, b in enumerate(bits))


def _new_inputs(aig: _AIG, n: int) -> list[int]:
    lits = []
    for _ in range(n):
        l = aig._new_var()
        aig.inputs.append(l)
        lits.append(l)
    return lits


def _sample_values(width: int) -> list[int]:
    mask = (1 << width) - 1
    base = [0, 1, mask, 1 << (width - 1), 0xAAAAAAAA & mask, 0x55555555 & mask,
            0x1234 & mask, 0xFF & mask, 0xF0 & mask]
    walking = [1 << i for i in range(width)]
    return sorted(set(base + walking))


# ---------------------------------------------------------------------------
# Variable shifts (shift amount is a primary input)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("op", ["<<", ">>"])
@pytest.mark.parametrize("width", [8, 16])
def test_variable_shift_handles_overshift(op: str, width: int):
    """``x << s`` / ``x >> s`` with a variable amount ``s`` that can exceed the
    output width must zero-fill, not alias to ``s`` modulo a power of two."""
    reset_shared_cache()
    w_out = width                       # SpireHDL keeps source width for variable shifts
    mask = (1 << w_out) - 1

    aig = _AIG()
    a_lits = _new_inputs(aig, width)
    sh_lits = _new_inputs(aig, width)   # full-width shift amount -> can over-shift
    if op == "<<":
        res = aig.bv_shift_left(a_lits, sh_lits, w_out)
    else:
        res = aig.bv_shift_right(a_lits, sh_lits, w_out)

    for val in _sample_values(width):
        for sh in range(0, 2 * width + 8):   # sweeps well past w_out
            assign = {l: (val >> i) & 1 for i, l in enumerate(a_lits)}
            assign.update({l: (sh >> i) & 1 for i, l in enumerate(sh_lits)})
            got = _lits_to_int(_eval_lits(aig, res, assign))
            exp = ((val << sh) if op == "<<" else (val >> sh)) & mask
            assert got == exp, (
                f"{op} width={width}: 0x{val:X} {op} {sh} -> got 0x{got:X}, exp 0x{exp:X}"
            )


def test_cpu_pipe_alu_variable_left_shift_is_zero_past_width():
    """Minimal witness for the original bug: a 16-bit value left-shifted by 16
    (a 16-bit variable amount) must be 0, not the un-shifted value."""
    reset_shared_cache()
    aig = _AIG()
    a_lits = _new_inputs(aig, 16)
    sh_lits = _new_inputs(aig, 16)
    res = aig.bv_shift_left(a_lits, sh_lits, w_out=16)

    # src = 0x8000, shift amount = 16  ->  result must be 0x0000
    assign = {l: (0x8000 >> i) & 1 for i, l in enumerate(a_lits)}
    assign.update({l: (16 >> i) & 1 for i, l in enumerate(sh_lits)})
    assert _lits_to_int(_eval_lits(aig, res, assign)) == 0


# ---------------------------------------------------------------------------
# Constant shifts (shift amount arrives as a 0/1 literal vector, like
# ``visit_const`` produces via ``bv_from_int``)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("width", [8, 16])
def test_const_right_shift_overshift(width: int):
    """``x >> C`` with a constant ``C >= width``. ``op_shift`` keeps the source
    width for ``>>``, so a large constant used to fall off the end of the
    barrel shifter and wrongly return ``x``."""
    reset_shared_cache()
    mask = (1 << width) - 1
    for shift_amt in range(0, 2 * width + 4):
        aig = _AIG()
        a_lits = _new_inputs(aig, width)
        sh_lits = aig.bv_from_int(shift_amt, max(1, shift_amt.bit_length()))
        res = aig.bv_shift_right(a_lits, sh_lits, w_out=width)
        for val in _sample_values(width):
            assign = {l: (val >> i) & 1 for i, l in enumerate(a_lits)}
            got = _lits_to_int(_eval_lits(aig, res, assign))
            exp = (val >> shift_amt) & mask
            assert got == exp, (
                f">> width={width}: 0x{val:X} >> {shift_amt} -> got 0x{got:X}, exp 0x{exp:X}"
            )


@pytest.mark.parametrize("width", [8, 16])
def test_const_left_shift(width: int):
    """``x << C`` with a constant ``C``. ``op_shift`` widens the result to
    ``width + C`` for constant left shifts, so this case was already correct --
    locked in here as a non-regression guard."""
    reset_shared_cache()
    for shift_amt in range(0, 2 * width + 4):
        w_out = width + shift_amt        # matches op_shift() for constant "<<"
        mask = (1 << w_out) - 1
        aig = _AIG()
        a_lits = _new_inputs(aig, width)
        sh_lits = aig.bv_from_int(shift_amt, max(1, shift_amt.bit_length()))
        res = aig.bv_shift_left(a_lits, sh_lits, w_out=w_out)
        for val in _sample_values(width):
            assign = {l: (val >> i) & 1 for i, l in enumerate(a_lits)}
            got = _lits_to_int(_eval_lits(aig, res, assign))
            exp = (val << shift_amt) & mask
            assert got == exp, (
                f"<< width={width}: 0x{val:X} << {shift_amt} -> got 0x{got:X}, exp 0x{exp:X}"
            )


# ---------------------------------------------------------------------------
# The overflow guard is conditional: emitted only when the shift-amount
# operand is wide enough to encode an out-of-range shift.
# ---------------------------------------------------------------------------

def _left_shift_gate_count(value_w: int, shift_w: int, w_out: int) -> int:
    aig = _AIG()
    a = _new_inputs(aig, value_w)
    sh = _new_inputs(aig, shift_w)
    aig.bv_shift_left(a, sh, w_out)
    return len(aig.ands)


def test_overflow_guard_emitted_only_when_needed():
    """For ``w_out = 16`` there are 4 barrel stages (steps 1,2,4,8). A shift
    operand of <= 4 bits can express only 0..15, so no out-of-range shift is
    possible and no guard is emitted. The 5th selector bit is the first that
    can over-shift; from there on the guard's OR-reduction grows with every
    extra out-of-range bit."""
    width = 16   # active_n = (16 - 1).bit_length() == 4

    g4 = _left_shift_gate_count(width, 4, width)    # no guard
    g5 = _left_shift_gate_count(width, 5, width)    # guard appears
    g8 = _left_shift_gate_count(width, 8, width)
    g16 = _left_shift_gate_count(width, 16, width)

    assert g5 > g4, "the first out-of-range shift bit must synthesize the guard"
    assert g8 > g5 > g4, "guard cost must grow with the number of out-of-range bits"
    assert g16 > g8, "guard cost must keep growing for wider shift operands"


def test_overflow_guard_folds_away_for_in_range_amounts():
    """When the high shift bits cannot be set (constant 0 -- i.e. the operand
    width does not actually require a guard), structural simplification folds
    the guard away entirely: not a single extra gate is produced."""
    width = 16
    plain = _left_shift_gate_count(width, 4, width)  # 4 real bits, no high bits

    # 4 real selector bits + 12 constant-0 high bits: the OR-reduce collapses
    # to const0, so the guard mux is never built.
    aig = _AIG()
    a = _new_inputs(aig, width)
    sh = _new_inputs(aig, 4) + [lit_const0()] * 12
    aig.bv_shift_left(a, sh, width)
    assert len(aig.ands) == plain


# ---------------------------------------------------------------------------
# End-to-end: a module mirroring the cpu_pipe ALU shift, exported to AIG.
# ---------------------------------------------------------------------------

def test_shift_module_end_to_end():
    """Build a module with truncating (16-bit) and widened (32-bit, CPU-pipe
    style) shifts, export it through the AIGER back-end and check the bits.

    ``shl32`` reproduces the Verilog auto-widen idiom from the cpu_pipe ALU:
    ``reg [31:0] shl; shl <= src << tgt;`` -- in SpireHDL the source must be
    widened explicitly first via ``cat`` before the variable shift."""
    reset_shared_cache()
    m = Module("ShiftAlu", with_clock=False, with_reset=False)
    src = m.input(UInt(16), "src")
    tgt = m.input(UInt(16), "tgt")
    shl16 = m.output(UInt(16), "shl16")   # truncating 16-bit left shift
    shr16 = m.output(UInt(16), "shr16")   # 16-bit right shift
    shl32 = m.output(UInt(32), "shl32")   # widened, cpu_pipe-style left shift

    shl16 <<= src << tgt
    shr16 <<= src >> tgt
    src_32 = cat(src, Const(0, UInt(16)))  # LSB-first: src at [0:16], zeros at [16:32]
    shl32 <<= (src_32 << tgt)[0:32]

    vecs = []
    for sv in (0x8000, 0x0001, 0xFFFF, 0xABCD, 0x0000):
        for tv in (0, 1, 4, 15, 16, 17, 31, 32, 33, 64, 0xFFFF):
            vecs.append((
                f"src0x{sv:04X}_tgt{tv}",
                {"src": sv, "tgt": tv},
                {
                    "shl16": (sv << tv) & 0xFFFF,
                    "shr16": (sv >> tv) & 0xFFFF,
                    "shl32": (sv << tv) & 0xFFFFFFFF,
                },
            ))

    # 1) SpireHDL Simulator -- confirms the golden vectors match the DSL semantics.
    run_vectors(m, vecs)
    # 2) AIGER back-end -- the path the bug lived on.
    m_aig = refactor_module_to_aig(m, optimize=False)
    run_vectors(m_aig, vecs)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
