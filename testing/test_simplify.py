"""Tests for ``spire.simplify.apply_simplify`` — the opt_expr / opt_muxtree analogue.

Each test builds a small module, runs the pass directly (rather than going through ``to_verilog_lines``, which would
also run CSE), and checks the IR via either:

- ``_collect_ops(module)`` — walk every Signal's driver chain and return the set of Op1/Op2 op symbols still in use.
  If we simplified ``a | 0`` to ``a`` then `|` should no longer appear anywhere.
- ``_resolve(expr)`` — follow auto-generated wire indirections (created by ``_maybe_share``) to get at the underlying
  expression. Useful when we want to assert that the simplified driver IS a particular signal or constant instance.

A few end-to-end tests then go through ``to_verilog()`` (which also runs CSE afterward) and inspect the emitted
Verilog string.
"""

from __future__ import annotations

import re
from typing import Set

import pytest

from spire.expr import (
    Concat,
    Const,
    Expr,
    Op1,
    Op2,
    Resize,
    Signal,
    Slice,
    Ternary,
    UInt,
    Wire,
    mux,
    reset_shared_cache,
)
from spire.component import Netlist
from spire.simplify import apply_simplify


@pytest.fixture(autouse=True)
def _reset_cse():
    """Reset the auto-share UID counter between tests so signal names stay predictable."""
    reset_shared_cache()
    yield
    reset_shared_cache()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve(e: Expr) -> Expr:
    """Follow auto-generated wire drivers transitively to get the underlying expression.

    `_maybe_share` wraps every fresh Op-typed `Expr` in a `Wire` Signal with ``_auto_generated = True``. To check that
    simplification reduced the chain to ``a`` (an input Signal), we walk past those auto-wires.
    """
    while isinstance(e, Signal) and getattr(e, "_auto_generated", False) \
            and e._driver is not None:
        e = e._driver
    return e


def _collect_ops(module: Netlist) -> Set[str]:
    """Return the set of all Op1/Op2 operator symbols reachable from any Signal driver."""
    ops: Set[str] = set()
    seen: Set[int] = set()

    def walk(e: Expr) -> None:
        if id(e) in seen:
            return
        seen.add(id(e))
        if isinstance(e, Op1):
            ops.add(e.op)
            walk(e.a)
        elif isinstance(e, Op2):
            ops.add(e.op)
            walk(e.a)
            walk(e.b)
        elif isinstance(e, Ternary):
            walk(e.sel); walk(e.a); walk(e.b)
        elif isinstance(e, Concat):
            for p in e.parts:
                walk(p)
        elif isinstance(e, (Slice, Resize)):
            walk(e.a)
        elif isinstance(e, Signal):
            if e._driver is not None:
                walk(e._driver)

    for s in module._signals:
        if s._driver is not None:
            walk(s._driver)
    return ops


def _count_ternaries(module: Netlist) -> int:
    """Count distinct Ternary nodes reachable from user-declared (non-auto-shared) signals.

    Auto-shared wires created by ``_maybe_share`` can become orphaned after a guard-substitution rewrite (the wire is
    no longer referenced by any parent, but its driver — the now-dead Ternary — is still attached to the wire). Such
    dead Ternaries don't end up in the emitted Verilog because their wrapping wire emits as a standalone
    `assign sig_N = …;` line at most, and they don't contribute to the framework's cost metric either. So this helper
    deliberately ignores them.
    """
    n = 0
    seen: Set[int] = set()

    def walk(e: Expr) -> None:
        nonlocal n
        if id(e) in seen:
            return
        seen.add(id(e))
        if isinstance(e, Ternary):
            n += 1
            walk(e.sel); walk(e.a); walk(e.b)
        elif isinstance(e, Op1):
            walk(e.a)
        elif isinstance(e, Op2):
            walk(e.a); walk(e.b)
        elif isinstance(e, Concat):
            for p in e.parts:
                walk(p)
        elif isinstance(e, (Slice, Resize)):
            walk(e.a)
        elif isinstance(e, Signal) and e._driver is not None:
            walk(e._driver)

    for s in module._signals:
        # Only walk user-declared signals (inputs/outputs/named wires/regs).
        if getattr(s, "_auto_generated", False):
            continue
        if s._driver is not None:
            walk(s._driver)
    return n


# ---------------------------------------------------------------------------
# Constant folding
# ---------------------------------------------------------------------------

def test_const_fold_add():
    m = Netlist("t", with_clock=False, with_reset=False)
    y = m.output(UInt(8), "y")
    y <<= Const(5, UInt(8)) + Const(3, UInt(8))
    m.collect_signals()
    apply_simplify(m)
    r = _resolve(y._driver)
    assert isinstance(r, Const) and r.value == 8


def test_const_fold_bitwise_and():
    m = Netlist("t", with_clock=False, with_reset=False)
    y = m.output(UInt(8), "y")
    y <<= Const(0xF0, UInt(8)) & Const(0x33, UInt(8))
    m.collect_signals()
    apply_simplify(m)
    r = _resolve(y._driver)
    assert isinstance(r, Const) and r.value == 0x30


def test_const_fold_mul_wraps_to_width():
    m = Netlist("t", with_clock=False, with_reset=False)
    y = m.output(UInt(4), "y")
    # 5 * 5 = 25 ≡ 9 (mod 16). The result type for mul is sum of widths (8 bits), but the output is 4 bits so a Resize
    # trims it. Our pass folds the Resize of a Const back to a Const.
    y <<= Const(5, UInt(4)) * Const(5, UInt(4))
    m.collect_signals()
    apply_simplify(m)
    r = _resolve(y._driver)
    assert isinstance(r, Const) and r.value == 9


def test_const_fold_compare():
    m = Netlist("t", with_clock=False, with_reset=False)
    y = m.output(UInt(1), "y")
    y <<= (Const(5, UInt(8)) == Const(5, UInt(8)))
    m.collect_signals()
    apply_simplify(m)
    r = _resolve(y._driver)
    assert isinstance(r, Const) and r.value == 1


# ---------------------------------------------------------------------------
# Boolean / arithmetic identities
# ---------------------------------------------------------------------------

def test_or_with_zero_collapses_to_self():
    m = Netlist("t", with_clock=False, with_reset=False)
    a = m.input(UInt(8), "a")
    y = m.output(UInt(8), "y")
    y <<= a | Const(0, UInt(8))
    m.collect_signals()
    apply_simplify(m)
    assert _resolve(y._driver) is a
    assert "|" not in _collect_ops(m)


def test_and_with_zero_collapses_to_zero():
    m = Netlist("t", with_clock=False, with_reset=False)
    a = m.input(UInt(8), "a")
    y = m.output(UInt(8), "y")
    y <<= a & Const(0, UInt(8))
    m.collect_signals()
    apply_simplify(m)
    r = _resolve(y._driver)
    assert isinstance(r, Const) and r.value == 0
    assert "&" not in _collect_ops(m)


def test_xor_self_is_zero():
    m = Netlist("t", with_clock=False, with_reset=False)
    a = m.input(UInt(8), "a")
    y = m.output(UInt(8), "y")
    y <<= a ^ a
    m.collect_signals()
    apply_simplify(m)
    r = _resolve(y._driver)
    assert isinstance(r, Const) and r.value == 0


def test_or_self_is_self():
    m = Netlist("t", with_clock=False, with_reset=False)
    a = m.input(UInt(8), "a")
    y = m.output(UInt(8), "y")
    y <<= a | a
    m.collect_signals()
    apply_simplify(m)
    assert _resolve(y._driver) is a


def test_double_negation():
    m = Netlist("t", with_clock=False, with_reset=False)
    a = m.input(UInt(8), "a")
    y = m.output(UInt(8), "y")
    y <<= ~~a
    m.collect_signals()
    apply_simplify(m)
    assert _resolve(y._driver) is a


def test_negate_const():
    m = Netlist("t", with_clock=False, with_reset=False)
    y = m.output(UInt(8), "y")
    y <<= ~Const(0x55, UInt(8))
    m.collect_signals()
    apply_simplify(m)
    r = _resolve(y._driver)
    assert isinstance(r, Const) and r.value == 0xAA


# ---------------------------------------------------------------------------
# Trivial mux
# ---------------------------------------------------------------------------

def test_mux_const_true_selector():
    m = Netlist("t", with_clock=False, with_reset=False)
    a = m.input(UInt(8), "a")
    b = m.input(UInt(8), "b")
    y = m.output(UInt(8), "y")
    y <<= mux(Const(1, UInt(1)), a, b)
    m.collect_signals()
    apply_simplify(m)
    assert _resolve(y._driver) is a
    assert _count_ternaries(m) == 0


def test_mux_const_false_selector():
    m = Netlist("t", with_clock=False, with_reset=False)
    a = m.input(UInt(8), "a")
    b = m.input(UInt(8), "b")
    y = m.output(UInt(8), "y")
    y <<= mux(Const(0, UInt(1)), a, b)
    m.collect_signals()
    apply_simplify(m)
    assert _resolve(y._driver) is b
    assert _count_ternaries(m) == 0


def test_mux_equal_branches_collapses():
    m = Netlist("t", with_clock=False, with_reset=False)
    c = m.input(UInt(1), "c")
    a = m.input(UInt(8), "a")
    y = m.output(UInt(8), "y")
    # Same operand instance on both sides — structurally equal.
    y <<= mux(c, a, a)
    m.collect_signals()
    apply_simplify(m)
    assert _resolve(y._driver) is a
    assert _count_ternaries(m) == 0


def test_mux_structurally_equal_branches_collapses():
    m = Netlist("t", with_clock=False, with_reset=False)
    c = m.input(UInt(1), "c")
    a = m.input(UInt(8), "a")
    b = m.input(UInt(8), "b")
    y = m.output(UInt(8), "y")
    # Two distinct Op2 instances, same operands/op: should be recognised as equal by the canonical-key walker.
    y <<= mux(c, a + b, a + b)
    m.collect_signals()
    apply_simplify(m)
    # No Ternary should survive.
    assert _count_ternaries(m) == 0
    # The `+` should still be there — we collapsed the mux but kept the addition.
    assert "+" in _collect_ops(m)


# ---------------------------------------------------------------------------
# Mux-tree guard substitution
# ---------------------------------------------------------------------------

def test_mux_tree_outer_inner_same_guard_true_side():
    """mux(g, mux(g, A, B), F) → mux(g, A, F): one Ternary instead of two."""
    m = Netlist("t", with_clock=False, with_reset=False)
    g = m.input(UInt(1), "g")
    a = m.input(UInt(8), "a")
    b = m.input(UInt(8), "b")
    f = m.input(UInt(8), "f")
    y = m.output(UInt(8), "y")
    y <<= mux(g, mux(g, a, b), f)
    m.collect_signals()
    apply_simplify(m)
    assert _count_ternaries(m) == 1


def test_mux_tree_outer_inner_same_guard_false_side():
    """mux(g, T, mux(g, A, B)) → mux(g, T, B): one Ternary instead of two."""
    m = Netlist("t", with_clock=False, with_reset=False)
    g = m.input(UInt(1), "g")
    a = m.input(UInt(8), "a")
    b = m.input(UInt(8), "b")
    t = m.input(UInt(8), "t")
    y = m.output(UInt(8), "y")
    y <<= mux(g, t, mux(g, a, b))
    m.collect_signals()
    apply_simplify(m)
    assert _count_ternaries(m) == 1


def test_mux_tree_both_inner_share_outer_guard_collapses_when_branches_match():
    """mux(g, mux(g, A, _), mux(g, _, A)) — both sides simplify to A → fully collapses."""
    m = Netlist("t", with_clock=False, with_reset=False)
    g = m.input(UInt(1), "g")
    a = m.input(UInt(8), "a")
    x = m.input(UInt(8), "x")
    y_inp = m.input(UInt(8), "y_inp")
    out = m.output(UInt(8), "out")
    out <<= mux(g, mux(g, a, x), mux(g, y_inp, a))
    m.collect_signals()
    apply_simplify(m)
    assert _count_ternaries(m) == 0
    assert _resolve(out._driver) is a


# ---------------------------------------------------------------------------
# End-to-end Verilog emission (covers simplify + CSE interaction in to_verilog_lines)
# ---------------------------------------------------------------------------

def test_to_verilog_eliminates_redundant_addition():
    """A constant-zero addition should be gone from the emitted Verilog."""
    m = Netlist("t", with_clock=False, with_reset=False)
    a = m.input(UInt(8), "a")
    y = m.output(UInt(8), "y")
    y <<= a + Const(0, UInt(8))
    v = m.to_verilog(simplify=True)
    # No `+ ` operator should remain in the assignments (the input `a` is wired straight to `y` via a one-line assign).
    assert re.search(r"\+", v) is None, f"Expected `+` to be eliminated:\n{v}"


def test_to_verilog_collapses_constant_mux():
    """mux with a Const selector should leave no ternary `?` in the output."""
    m = Netlist("t", with_clock=False, with_reset=False)
    a = m.input(UInt(8), "a")
    b = m.input(UInt(8), "b")
    y = m.output(UInt(8), "y")
    y <<= mux(Const(1, UInt(1)), a, b)
    v = m.to_verilog(simplify=True)
    assert " ? " not in v, f"Expected ternary to be eliminated:\n{v}"


def test_to_verilog_preserves_correctness_for_normal_designs():
    """A normal design with no simplification opportunities should pass through unchanged."""
    m = Netlist("t", with_clock=False, with_reset=False)
    a = m.input(UInt(8), "a")
    b = m.input(UInt(8), "b")
    c = m.input(UInt(1), "c")
    y = m.output(UInt(8), "y")
    y <<= mux(c, a, b)
    v = m.to_verilog()
    assert "c" in v and "a" in v and "b" in v
    assert " ? " in v  # the ternary operator survives — it's an irreducible mux
