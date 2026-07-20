"""Tests for spire.selection_emission — the `mux_emission` scope object
(region context-manager + decorator), the shape-based cascade rewrites
(chain / tournament / andor / bittree), the redundant-gating fix in
_SwitchState._claim_cases, and the opt-in whole-design auto pass."""

import contextlib
import random

import pytest

from spire import Simulator, SelectionEmissionConfig, mux_emission
from spire.component import Netlist
from spire.control_structures import case_, default, elif_, else_, if_, switch_
from spire.expr import Const, UInt, mux
from spire.selection_emission import apply_selection_emission, collect_chain
from spire.simulator import _sid


@pytest.fixture(autouse=True)
def _clear_pending_if_chain():
    """Several tests here deliberately end if_/elif_ chains without else_;
    clear the module-global pending-chain state so it can't leak into other
    test files (e.g. the scoping suite's no-pending-chain assertions)."""
    yield
    from spire.control_structures import _set_pending_chain
    _set_pending_chain(None)


def _region(mode):
    return mux_emission(mode) if mode else contextlib.nullcontext()


def _equiv(m_ref: Netlist, m_new: Netlist, input_gen, n=300, outs=("y",)):
    sr, sn = Simulator(m_ref), Simulator(m_new)
    rng = random.Random(1234)
    for _ in range(n):
        ins = input_gen(rng)
        for k, v in ins.items():
            sr.set(k, v)
            sn.set(k, v)
        sr.eval()
        sn.eval()
        for o in outs:
            assert sr.get(o) == sn.get(o), (ins, o, sr.get(o), sn.get(o))


# ---------------------------------------------------------------------------
# priority chains: decorator + region over hand-built chains
# ---------------------------------------------------------------------------

def _ff1_chain(x, n):
    chain = Const(0, UInt(5))
    for k in reversed(range(n)):
        chain = mux(x[k], Const(k, UInt(5)), chain)
    return chain


def _build_ff1(mode, via="region"):
    m = Netlist("FF1", with_clock=False, with_reset=False)
    x = m.input(UInt(16), "x")
    y = m.output(UInt(5), "y")
    if mode is None:
        y <<= _ff1_chain(x, 16)
    elif via == "region":
        with mux_emission(mode):
            y <<= _ff1_chain(x, 16)      # hand-built chain captured by region
    else:  # decorator
        deco = mux_emission(mode)(lambda bits: _ff1_chain(bits, 16))
        y <<= deco(x)
    return m


def test_tournament_region_hand_chain():
    m = _build_ff1("tournament", via="region")
    # eager: the graph itself is restructured, before any emission backend
    pairs, _ = collect_chain(m._signals_by_name()["y"]._driver) if hasattr(m, "_signals_by_name") else collect_chain(
        next(s for s in m._ports if s.name == "y")._driver)
    assert len(pairs) == 1  # tournament root, not a 16-deep chain
    _equiv(_build_ff1(None), m, lambda r: {"x": r.getrandbits(16)}, n=500)


def test_tournament_decorator():
    m = _build_ff1("tournament", via="decorator")
    _equiv(_build_ff1(None), m, lambda r: {"x": r.getrandbits(16)}, n=300)


def test_tournament_exhaustive_small():
    m_ref, m_t = _build_ff1(None), _build_ff1("tournament")
    sr, st = Simulator(m_ref), Simulator(m_t)
    for v in range(1 << 16):  # exhaustive over overlap patterns, sampled past low range
        if v % 37 and v > 4096:
            continue
        sr.set("x", v)
        st.set("x", v)
        sr.eval()
        st.eval()
        assert sr.get("y") == st.get("y"), v


def _build_if_chain(mode, conds="bits"):
    m = Netlist("IfC", with_clock=False, with_reset=False)
    y = m.output(UInt(8), "y")
    if conds == "bits":
        c = [m.input(UInt(1), f"c{i}") for i in range(18)]
        conds_exprs = c
    else:  # eq-const conditions on a shared selector
        s = m.input(UInt(5), "s")
        conds_exprs = [s == i for i in range(18)]
    y <<= 0xEE
    with _region(mode):
        with if_(conds_exprs[0]):
            y <<= 1
        for i in range(1, 18):
            with elif_(conds_exprs[i]):
                y <<= i + 1
        with else_():
            y <<= 0xAA
    return m


def test_if_chain_tournament_equivalence():
    _equiv(_build_if_chain(None), _build_if_chain("tournament"),
           lambda r: {f"c{i}": r.getrandbits(1) for i in range(18)}, n=600)


def test_if_chain_andor_when_eq_const():
    # the gating (`cond & ~covered`) is provably redundant for eq-const chains;
    # the classifier sees through it and andor applies to an if_ chain.
    ref = _build_if_chain(None, conds="eq")
    ao = _build_if_chain("andor", conds="eq")
    assert ao.to_verilog().count("?") == 0
    sr, sa = Simulator(ref), Simulator(ao)
    for v in range(32):
        sr.set("s", v)
        sa.set("s", v)
        sr.eval()
        sa.eval()
        assert sr.get("y") == sa.get("y"), v


def test_if_chain_eq_const_has_no_gating():
    # the unified _DisjointnessTracker: eq-const if_/elif_ arms skip the
    # `& ~covered` gating exactly like disjoint switch cases (no else_ here,
    # so `covered` is never consumed and must vanish entirely)
    m = Netlist("IfClean", with_clock=False, with_reset=False)
    s = m.input(UInt(4), "s")
    y = m.output(UInt(8), "y")
    y <<= 0
    with if_(s == 1):
        y <<= 0x11
    with elif_(s == 2):
        y <<= 0x22
    with elif_(s == 9):
        y <<= 0x33
    vl = m.to_verilog()
    assert "_covered" not in vl and "gated" not in vl


def test_if_chain_duplicate_condition_keeps_priority():
    # colliding eq-const arm gets gated (first match wins); later fresh arm
    # is provable again — mirrored from the switch behavior
    m = Netlist("IfDup", with_clock=False, with_reset=False)
    s = m.input(UInt(4), "s")
    y = m.output(UInt(8), "y")
    y <<= 0
    with if_(s == 1):
        y <<= 0x11
    with elif_(s == 1):   # duplicate: unreachable, must not shadow the first
        y <<= 0x99
    with elif_(s == 2):
        y <<= 0x22
    sim = Simulator(m)
    for v, expect in [(0, 0), (1, 0x11), (2, 0x22), (3, 0)]:
        sim.set("s", v)
        sim.eval()
        assert sim.get("y") == expect, v


def test_if_chain_andor_opaque_raises():
    m = Netlist("Bad", with_clock=False, with_reset=False)
    a = m.input(UInt(8), "a")
    y = m.output(UInt(4), "y")
    y <<= 0
    with pytest.raises(ValueError, match="disjoint"):
        with mux_emission("andor"):
            with if_(a < 10):
                y <<= 1
            with else_():
                y <<= 2


# ---------------------------------------------------------------------------
# switch_ under regions: all modes, defaults, partial assignment, registers
# ---------------------------------------------------------------------------

def _build_switch(mode, with_default=True, sparse=False, n_cases=16, selw=4):
    m = Netlist("Sw", with_clock=False, with_reset=False)
    s = m.input(UInt(selw), "s")
    vs = [m.input(UInt(8), f"v{i}") for i in range(n_cases)]
    y = m.output(UInt(8), "y")
    y <<= 0x5C  # pre-assignment: the fallthrough value when no arm matches
    labels = [i * 3 if sparse else i for i in range(n_cases)]
    with _region(mode):
        with switch_(s):
            for i in range(n_cases):
                with case_(labels[i]):
                    y <<= vs[i]
            if with_default:
                with default():
                    y <<= 0xD0
    return m


@pytest.mark.parametrize("mode", ["andor", "bittree", "tournament", "auto", "chain"])
@pytest.mark.parametrize("with_default", [True, False])
def test_switch_modes_equivalence(mode, with_default):
    ref = _build_switch(None, with_default)
    new = _build_switch(mode, with_default)
    _equiv(ref, new,
           lambda r: {"s": r.getrandbits(4),
                      **{f"v{i}": r.getrandbits(8) for i in range(16)}},
           n=400)


def test_switch_andor_sparse_labels_with_default():
    ref = _build_switch(None, True, sparse=True, n_cases=8, selw=5)
    new = _build_switch("andor", True, sparse=True, n_cases=8, selw=5)
    _equiv(ref, new,
           lambda r: {"s": r.getrandbits(5),
                      **{f"v{i}": r.getrandbits(8) for i in range(8)}},
           n=500)


def test_switch_andor_emits_no_ternary_and_no_covered():
    vl = _build_switch("andor", True).to_verilog()
    assert vl.count("?") == 0
    assert "_covered" not in vl


def test_disjoint_switch_has_no_gating_even_unhinted():
    # the _claim_cases fix: distinct constant labels -> no `& ~covered` chains
    vl = _build_switch(None, False).to_verilog()
    assert "_covered" not in vl


def test_overlapping_switch_keeps_priority():
    def build(mode):
        m = Netlist("Ov", with_clock=False, with_reset=False)
        s = m.input(UInt(2), "s")
        y = m.output(UInt(4), "y")
        y <<= 0
        with _region(mode):
            with switch_(s):
                with case_(1):
                    y <<= 1
                with case_(1, 2):  # overlaps: first match must win for s==1
                    y <<= 2
        return m

    for mode in (None, "tournament", "auto"):
        m = build(mode)
        sim = Simulator(m)
        for s_val, expect in [(0, 0), (1, 1), (2, 2), (3, 0)]:
            sim.set("s", s_val)
            sim.eval()
            assert sim.get("y") == expect, (mode, s_val)


def test_switch_register_hold_with_andor():
    m = Netlist("RegSw", with_clock=True, with_reset=False)
    s = m.input(UInt(4), "s")
    v = m.input(UInt(8), "v")
    reg = m.reg(UInt(8), "r")
    reg.set_init(0)
    with mux_emission("andor"):
        with switch_(s):
            for i in range(16):
                if i == 5:
                    continue  # s==5: no assignment -> register must hold
                with case_(i):
                    reg <<= (v + i)[0:8] if i else v

    sim = Simulator(m)
    sim.eval()
    sim.set("r", 0x77)
    sim.set("s", 5)
    sim.set("v", 0x11)
    assert sim._compute_next_state()[_sid(reg)] == 0x77  # hold
    sim.set("s", 3)
    assert sim._compute_next_state()[_sid(reg)] == 0x14  # 0x11 + 3


def test_nested_regions_inner_wins():
    def build(nested):
        m = Netlist("Nest", with_clock=False, with_reset=False)
        s = m.input(UInt(4), "s")
        vs = [m.input(UInt(8), f"v{i}") for i in range(16)]
        y = m.output(UInt(8), "y")
        y <<= 0
        outer = mux_emission("tournament") if nested else contextlib.nullcontext()
        with outer:
            with _region("andor" if nested else None):
                with switch_(s):
                    for i in range(16):
                        with case_(i):
                            y <<= vs[i]
        return m

    ref, new = build(False), build(True)
    assert new.to_verilog().count("?") == 0  # inner andor applied
    _equiv(ref, new,
           lambda r: {"s": r.getrandbits(4),
                      **{f"v{i}": r.getrandbits(8) for i in range(16)}},
           n=300)


def test_chain_may_not_straddle_region_boundary():
    m = Netlist("Straddle", with_clock=False, with_reset=False)
    c0 = m.input(UInt(1), "c0")
    c1 = m.input(UInt(1), "c1")
    y = m.output(UInt(4), "y")
    y <<= 0
    with mux_emission("tournament"):
        with if_(c0):
            y <<= 1
    with pytest.raises(RuntimeError, match="scope"):
        with elif_(c1):  # chain started inside the region, continued outside
            y <<= 2


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------

def test_andor_duplicate_label_raises_at_case():
    m = Netlist("Dup", with_clock=False, with_reset=False)
    s = m.input(UInt(2), "s")
    y = m.output(UInt(4), "y")
    y <<= 0
    with pytest.raises(ValueError, match="distinct"):
        with mux_emission("andor"):
            with switch_(s):
                with case_(1):
                    y <<= 1
                with case_(1):
                    y <<= 2


def test_andor_nonconst_label_raises_at_case():
    m = Netlist("NC", with_clock=False, with_reset=False)
    s = m.input(UInt(2), "s")
    t = m.input(UInt(2), "t")
    y = m.output(UInt(4), "y")
    y <<= 0
    with pytest.raises(ValueError, match="constant"):
        with mux_emission("andor"):
            with switch_(s):
                with case_(t):
                    y <<= 1


def test_bittree_missing_coverage_raises():
    m = Netlist("BT", with_clock=False, with_reset=False)
    s = m.input(UInt(3), "s")
    y = m.output(UInt(4), "y")
    y <<= 0
    with pytest.raises(ValueError, match="covering"):
        with mux_emission("bittree"):
            with switch_(s):
                for i in range(5):  # 5 of 8 labels, no default
                    with case_(i):
                        y <<= i


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="mode"):
        mux_emission("magic")


def test_region_skips_plain_assignments():
    # signals without a cascade are silently skipped, even under explicit modes
    m = Netlist("Plain", with_clock=False, with_reset=False)
    a = m.input(UInt(8), "a")
    y = m.output(UInt(8), "y")
    with mux_emission("andor"):
        y <<= a + 1  # no cascade — must not raise
    sim = Simulator(m)
    sim.set("a", 41)
    sim.eval()
    assert sim.get("y") == 42


# ---------------------------------------------------------------------------
# emission pass: whole-design auto-detection (opt-in)
# ---------------------------------------------------------------------------

def _hand_eq_chain_module():
    m = Netlist("Hand", with_clock=False, with_reset=False)
    s = m.input(UInt(4), "s")
    vs = [m.input(UInt(8), f"v{i}") for i in range(16)]
    y = m.output(UInt(8), "y")
    chain = Const(0, UInt(8))
    for i in reversed(range(16)):
        chain = mux(s == Const(i, UInt(4)), vs[i], chain)
    y <<= chain
    return m


def test_pass_auto_detects_hand_chain():
    m = _hand_eq_chain_module()
    m.collect_signals()
    n = apply_selection_emission(m, SelectionEmissionConfig(enabled=True))
    assert n == 1
    vl = m.to_verilog()
    assert vl.count("?") == 0  # disjoint eq-chain -> andor
    _equiv(_hand_eq_chain_module(), m,
           lambda r: {"s": r.getrandbits(4),
                      **{f"v{i}": r.getrandbits(8) for i in range(16)}},
           n=300)


def test_pass_disabled_is_noop():
    m = _hand_eq_chain_module()
    m.collect_signals()
    assert apply_selection_emission(m, SelectionEmissionConfig(enabled=False)) == 0


def test_selection_emission_kwarg_on_to_verilog():
    m = _hand_eq_chain_module()
    vl = m.to_verilog(selection_emission=True)
    assert vl.count("?") == 0
