"""Walker (Step 4 of the FSM-encoding-search plan)."""
from __future__ import annotations

from spire.optimize.fsm._walker import (
    find_input_signals, find_state_consts, is_state_const, walk,
)
from spire.expr import Bool, Const, UInt, cat, mux
from spire.component import Netlist
from spire.state import Encoding, State, state


class S(State, encoding=Encoding.BINARY):
    A = state()
    B = state()
    C = state()
    D = state()


class T(State, encoding=Encoding.ONEHOT):
    X = state()
    Y = state()


def test_is_state_const_predicate():
    assert is_state_const(S.A, S)
    assert is_state_const(S.D, S)
    assert not is_state_const(S.A, T)      # cross-class rejection
    assert not is_state_const(Const(0, UInt(2)), S)
    assert not is_state_const(Const(1, UInt(3)), S)


def test_walker_finds_state_consts_in_mux_tree():
    """A typical FSM body: nested mux of state Consts under a sel."""
    m = Netlist("t", with_clock=False, with_reset=False)
    sel = m.input(Bool(), "sel")
    out = m.output(S.typ, "out")
    out <<= mux(sel, S.A, mux(sel, S.B, S.C))

    finder = walk([out._driver], S)
    found_ids = {id(c) for c in finder.found_consts}
    assert id(S.A) in found_ids
    assert id(S.B) in found_ids
    assert id(S.C) in found_ids


def test_walker_dedup_in_find_state_consts():
    """Find returns at most one entry per Const object even if referenced twice."""
    m = Netlist("t", with_clock=False, with_reset=False)
    sel = m.input(Bool(), "sel")
    # Reference S.A in both branches → same Const object referenced twice.
    out = m.output(S.typ, "out")
    out <<= mux(sel, S.A, S.A)

    consts = find_state_consts([out._driver], S)
    # Dedup by identity: only one entry.
    assert len(consts) == 1
    assert consts[0] is S.A


def test_walker_ignores_other_state_class():
    m = Netlist("t", with_clock=False, with_reset=False)
    sel = m.input(Bool(), "sel")
    out = m.output(S.typ, "out")
    # Mixed: S and T constants — wrong dimensions intentionally, just structurally.
    out <<= mux(sel, S.A, S.B)
    sig_other = m.wire(T.typ, "sig_other")
    sig_other <<= mux(sel, T.X, T.Y)

    found_s = find_state_consts([out._driver, sig_other._driver], S)
    found_t = find_state_consts([out._driver, sig_other._driver], T)
    s_ids = {id(c) for c in found_s}
    t_ids = {id(c) for c in found_t}
    assert id(S.A) in s_ids and id(S.B) in s_ids
    assert id(T.X) not in s_ids and id(T.Y) not in s_ids
    assert id(T.X) in t_ids and id(T.Y) in t_ids


def test_walker_finds_input_signals():
    m = Netlist("t", with_clock=False, with_reset=False)
    a = m.input(Bool(), "a")
    b = m.input(Bool(), "b")
    out = m.output(S.typ, "out")
    out <<= mux(a, S.A, mux(b, S.B, S.C))

    inputs = find_input_signals([out._driver], S)
    names = {s.name for s in inputs}
    # 'a' and 'b' are inputs; auto-shared CSE wires are skipped.
    assert "a" in names
    assert "b" in names
    # No state Const should be reported as a signal.
    for s in inputs:
        assert getattr(s, "_state_class", None) is None


def test_walker_excludes_passed_signals_from_inputs():
    m = Netlist("t", with_clock=False, with_reset=False)
    a = m.input(Bool(), "a")
    reg = m.reg(S.typ, "reg", init=S.A)
    out = m.output(S.typ, "out")
    out <<= mux(reg == S.B, S.C, S.D)

    inputs = find_input_signals([out._driver], S, exclude=[reg])
    names = {s.name for s in inputs}
    assert "reg" not in names
    assert "a" not in names    # 'a' isn't referenced by out._driver anyway


def test_walker_recurses_through_auto_shared_wires():
    """The walker follows auto-shared CSE wires via their _driver."""
    m = Netlist("t", with_clock=False, with_reset=False)
    sel = m.input(Bool(), "sel")
    # Force a deeper auto-shared chain by combining many Exprs.
    middle = mux(sel, S.A, S.B)              # → sig_N wraps this Ternary
    outer = mux(sel, middle, S.C)            # → another wrapper
    out = m.output(S.typ, "out")
    out <<= outer

    consts = find_state_consts([out._driver], S)
    found_ids = {id(c) for c in consts}
    assert id(S.A) in found_ids
    assert id(S.B) in found_ids
    assert id(S.C) in found_ids


def test_walker_const_only_root():
    """A root that's literally just a State Const should still report it."""
    consts = find_state_consts([S.A], S)
    assert consts == [S.A]


def test_walker_concat_index_state_const():
    """Concat parts can contain state Consts at various indices."""
    m = Netlist("t", with_clock=False, with_reset=False)
    bit = m.input(Bool(), "bit")
    # cat(...) places parts LSB-first; mix State Consts with arbitrary bits.
    packed = m.wire(UInt(4), "packed")
    packed <<= cat(bit, S.A)                 # 1-bit + 2-bit = 3-bit, framework may resize/Concat

    consts = find_state_consts([packed._driver], S)
    assert any(c is S.A for c in consts)
