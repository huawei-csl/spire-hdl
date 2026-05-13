"""apply_encoding (Step 10 of the FSM-encoding-search plan)."""
from __future__ import annotations

import pytest

from spirehdl.fsm._emit import apply_encoding, restore_encoding, snapshot_encoding
from spirehdl.spirehdl import Bool, mux
from spirehdl.spirehdl_module import Module
from spirehdl.spirehdl_state import Encoding, State, state


class S(State, encoding=Encoding.BINARY):
    A = state()
    B = state()
    C = state()
    D = state()


def setup_function(_fn):
    """Reset S to its declared encoding before each test (test isolation)."""
    restore_encoding(S, {"A": 0, "B": 1, "C": 2, "D": 3})


def test_snapshot_captures_current_values():
    snap = snapshot_encoding(S)
    assert snap == {"A": 0, "B": 1, "C": 2, "D": 3}


def test_apply_encoding_mutates_each_const_in_place():
    apply_encoding(S, {"A": 3, "B": 2, "C": 1, "D": 0})
    assert S.A.value == 3
    assert S.B.value == 2
    assert S.C.value == 1
    assert S.D.value == 0


def test_apply_encoding_updates_state_class_values_dict():
    apply_encoding(S, {"A": 3, "B": 2, "C": 1, "D": 0})
    assert S._values == {"A": 3, "B": 2, "C": 1, "D": 0}


def test_apply_encoding_propagates_to_existing_expression_tree():
    """The mutation should be visible to Exprs that referenced the Const
    objects *before* apply_encoding was called."""
    m = Module("t", with_clock=False, with_reset=False)
    sel = m.input(Bool(), "sel")
    out = m.output(S.typ, "out")
    # Capture the Expr now; verify post-apply that the Const values changed.
    out <<= mux(sel, S.A, S.C)

    apply_encoding(S, {"A": 3, "B": 2, "C": 1, "D": 0})

    # The Const objects in the driver tree are the SAME objects we mutated.
    from spirehdl.fsm._walker import find_state_consts
    consts = find_state_consts([out._driver], S)
    values = {c._state_name: c.value for c in consts}
    assert values["A"] == 3
    assert values["C"] == 1


def test_apply_then_restore_round_trips():
    snap = snapshot_encoding(S)
    apply_encoding(S, {"A": 7, "B": 6, "C": 5, "D": 4})
    assert S.A.value == 7
    restore_encoding(S, snap)
    assert S.A.value == 0
    assert S.B.value == 1
    assert S.C.value == 2
    assert S.D.value == 3


def test_apply_encoding_missing_state_raises():
    with pytest.raises(ValueError, match="missing state"):
        apply_encoding(S, {"A": 0, "B": 1})           # C and D missing


def test_apply_encoding_width_change_not_supported_yet():
    with pytest.raises(NotImplementedError):
        apply_encoding(S, {"A": 0, "B": 1, "C": 2, "D": 3}, width=4)
