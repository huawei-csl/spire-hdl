"""Simulation-verified examples for `spirehdl.spirehdl_state`.

The existing ``test_state_machine.py`` covers the encoding-width contract and
a 3-state demo. This file exercises richer FSM patterns end-to-end:

- a 4-state traffic-light Moore FSM with a `go` gate
- a Mealy edge-detector (output depends on state + input)
- a 4-bit serial sequence detector (matches the pattern "1101")
- a counter-style FSM with synchronous reset via a muxed next-state
- encoding-equivalence parity (BINARY / ONEHOT / GRAY produce identical
  cycle-accurate behaviour for the same transition table)

All tests use the in-tree Python simulator; no Verilator / Yosys required.
"""
from __future__ import annotations

import pytest

from spirehdl.spirehdl import Bool, UInt, mux
from spirehdl.spirehdl_control_structures import case_, default, if_, else_, switch_
from spirehdl.spirehdl_module import Module
from spirehdl.spirehdl_simulator import Simulator
from spirehdl.spirehdl_state import Encoding, State, state


# ---------------------------------------------------------------------------
# 1. Traffic-light Moore FSM
# ---------------------------------------------------------------------------

class Traffic(State, encoding=Encoding.BINARY):
    RED    = state()
    GREEN  = state()
    YELLOW = state()


def _build_traffic_fsm() -> tuple[Module, dict]:
    m = Module("traffic", with_clock=True, with_reset=False)
    go    = m.input(Bool(), "go")
    light = m.output(UInt(2), "light")
    state_reg = m.reg(Traffic.typ, "state_reg", init=Traffic.RED)

    light <<= state_reg                     # Moore: output = current state

    with switch_(state_reg):
        with case_(Traffic.RED):
            with if_(go):
                state_reg <<= Traffic.GREEN
        with case_(Traffic.GREEN):
            state_reg <<= Traffic.YELLOW
        with case_(Traffic.YELLOW):
            state_reg <<= Traffic.RED
        with default():
            state_reg <<= Traffic.RED
    return m, {"go": go, "light": light, "state_reg": state_reg}


def test_traffic_fsm_stays_red_when_go_low():
    m, sig = _build_traffic_fsm()
    sim = Simulator(m)
    sim.set(sig["go"], 0)
    sim.eval()
    assert sim.get(sig["state_reg"]) == Traffic.RED.value
    for _ in range(5):                       # 5 clock edges with go=0
        sim.step()
    assert sim.get(sig["state_reg"]) == Traffic.RED.value


def test_traffic_fsm_cycles_red_green_yellow():
    m, sig = _build_traffic_fsm()
    sim = Simulator(m)
    sim.set(sig["go"], 1)
    sim.eval()

    expected = [Traffic.RED, Traffic.GREEN, Traffic.YELLOW,
                Traffic.RED, Traffic.GREEN, Traffic.YELLOW]
    for step_idx, exp in enumerate(expected):
        assert sim.get(sig["state_reg"]) == exp.value, (
            f"step {step_idx}: want {exp.name}, got {sim.get(sig['state_reg'])}")
        assert sim.get(sig["light"]) == exp.value
        sim.step()


def test_traffic_fsm_default_recovers_from_garbage_state():
    """Forcing an out-of-range encoding hits `default()` and snaps back to RED."""
    m, sig = _build_traffic_fsm()
    sim = Simulator(m)
    sim.set(sig["go"], 0)
    sim.set(sig["state_reg"], 0b11)          # 3 is not a declared state
    sim.eval()
    sim.step()
    assert sim.get(sig["state_reg"]) == Traffic.RED.value


# ---------------------------------------------------------------------------
# 2. Mealy edge detector — output asserted on the rising edge of `in_bit`
# ---------------------------------------------------------------------------

class EdgeDetect(State, encoding=Encoding.BINARY):
    LOW  = state()
    HIGH = state()


def _build_edge_detector() -> tuple[Module, dict]:
    m = Module("edge_detect", with_clock=True, with_reset=False)
    in_bit = m.input(Bool(), "in_bit")
    rising = m.output(Bool(), "rising")
    st = m.reg(EdgeDetect.typ, "st", init=EdgeDetect.LOW)

    rising <<= 0                              # default

    with switch_(st):
        with case_(EdgeDetect.LOW):
            with if_(in_bit):
                rising <<= 1                  # Mealy: output asserts on
                st <<= EdgeDetect.HIGH        # the transition itself
        with case_(EdgeDetect.HIGH):
            with if_(~in_bit):
                st <<= EdgeDetect.LOW
        with default():
            st <<= EdgeDetect.LOW
    return m, {"in_bit": in_bit, "rising": rising, "st": st}


def test_edge_detector_asserts_only_on_rising_edge():
    m, sig = _build_edge_detector()
    sim = Simulator(m)

    # in_bit pattern: 0 0 1 1 1 0 1 1 0
    # rising         : 0 0 1 0 0 0 1 0 0
    pattern  = [0, 0, 1, 1, 1, 0, 1, 1, 0]
    expected = [0, 0, 1, 0, 0, 0, 1, 0, 0]

    for i, (b, exp) in enumerate(zip(pattern, expected)):
        sim.set(sig["in_bit"], b)
        sim.eval()
        assert sim.get(sig["rising"]) == exp, f"cycle {i}: want {exp}, got {sim.get(sig['rising'])}"
        sim.step()


# ---------------------------------------------------------------------------
# 3. Serial pattern detector — match "1101" on a 1-bit stream (overlap allowed)
# ---------------------------------------------------------------------------

class Match1101(State, encoding=Encoding.BINARY):
    S0    = state()   # waiting for 1
    S1    = state()   # saw 1
    S11   = state()   # saw 11
    S110  = state()   # saw 110
    # match completes when seeing the final 1; we emit `hit` on that cycle


def _build_pattern_detector() -> tuple[Module, dict]:
    m = Module("match1101", with_clock=True, with_reset=False)
    b   = m.input(Bool(), "b")
    hit = m.output(Bool(), "hit")
    st  = m.reg(Match1101.typ, "st", init=Match1101.S0)

    hit <<= 0

    with switch_(st):
        with case_(Match1101.S0):
            with if_(b):
                st <<= Match1101.S1
        with case_(Match1101.S1):
            with if_(b):
                st <<= Match1101.S11
            with else_():
                st <<= Match1101.S0
        with case_(Match1101.S11):
            with if_(b):
                st <<= Match1101.S11           # stay (extra 1's)
            with else_():
                st <<= Match1101.S110
        with case_(Match1101.S110):
            with if_(b):
                hit <<= 1                     # Mealy: assert on the match transition
                st <<= Match1101.S1           # overlap: '1' just consumed is start of next attempt
            with else_():
                st <<= Match1101.S0
        with default():
            st <<= Match1101.S0
    return m, {"b": b, "hit": hit, "st": st}


def _drive_stream(sim, bits, sig_b, sig_hit):
    """Feed bits MSB-first; return list of `hit` values per cycle."""
    out = []
    for bit in bits:
        sim.set(sig_b, bit)
        sim.eval()
        out.append(sim.get(sig_hit))
        sim.step()
    return out


def test_pattern_1101_detects_one_match():
    m, sig = _build_pattern_detector()
    sim = Simulator(m)
    # stream:  0 1 1 0 1 0 1 0
    # pattern matches on the 1101 ending at index 4 → hit asserts at that eval
    hits = _drive_stream(sim, [0, 1, 1, 0, 1, 0, 1, 0], sig["b"], sig["hit"])
    assert hits == [0, 0, 0, 0, 1, 0, 0, 0]


def test_pattern_1101_overlapping_matches():
    m, sig = _build_pattern_detector()
    sim = Simulator(m)
    # stream:  1 1 0 1 1 0 1 ...
    # matches at indices 3 and 6 (overlap: trailing '1' of first match is the
    # leading '1' of the second).
    hits = _drive_stream(sim, [1, 1, 0, 1, 1, 0, 1], sig["b"], sig["hit"])
    assert hits == [0, 0, 0, 1, 0, 0, 1]


def test_pattern_1101_no_match_on_short_stream():
    m, sig = _build_pattern_detector()
    sim = Simulator(m)
    hits = _drive_stream(sim, [1, 0, 1, 1], sig["b"], sig["hit"])
    assert hits == [0, 0, 0, 0]


# ---------------------------------------------------------------------------
# 4. Counter-FSM with synchronous reset (mux-based)
# ---------------------------------------------------------------------------

class CounterFSM(State, encoding=Encoding.BINARY):
    C0 = state(); C1 = state(); C2 = state(); C3 = state()


def test_counter_fsm_with_sync_reset():
    m = Module("counter_fsm", with_clock=True, with_reset=False)
    sync_rst = m.input(Bool(), "sync_rst")
    out = m.output(CounterFSM.typ, "out")
    st = m.reg(CounterFSM.typ, "st", init=CounterFSM.C0)

    out <<= st

    next_st = m.wire(CounterFSM.typ, "next_st")
    next_st <<= CounterFSM.C0                  # default driver (wire needs one)
    with switch_(st):
        with case_(CounterFSM.C0): next_st <<= CounterFSM.C1
        with case_(CounterFSM.C1): next_st <<= CounterFSM.C2
        with case_(CounterFSM.C2): next_st <<= CounterFSM.C3
        with case_(CounterFSM.C3): next_st <<= CounterFSM.C0
        with default():            next_st <<= CounterFSM.C0

    # Sync reset: when sync_rst is high, latch C0 next clock; otherwise advance.
    st <<= mux(sync_rst, CounterFSM.C0, next_st)

    sim = Simulator(m)
    sim.set(sync_rst, 0)
    sim.eval()

    # Free-run: C0 → C1 → C2 → C3 → C0 → C1
    seq = [CounterFSM.C0, CounterFSM.C1, CounterFSM.C2,
           CounterFSM.C3, CounterFSM.C0, CounterFSM.C1]
    for exp in seq:
        assert sim.get(st) == exp.value
        sim.step()

    # Assert sync_rst mid-run: state should latch C0 on the next edge,
    # regardless of where it was.
    sim.set(sync_rst, 1)
    sim.eval()
    sim.step()
    assert sim.get(st) == CounterFSM.C0.value
    # Release reset; counter advances again.
    sim.set(sync_rst, 0)
    sim.eval()
    sim.step()
    assert sim.get(st) == CounterFSM.C1.value


# ---------------------------------------------------------------------------
# 5. Encoding-equivalence parity
#
# The same logical 4-state FSM is built under BINARY, ONEHOT, and GRAY
# encodings. Driving them with the same input stream must produce identical
# observable behaviour on the cycle-accurate output (the state *value* may
# differ, but the mapped state name and the Moore output must match).
# ---------------------------------------------------------------------------

def _make_4state_fsm(encoding):
    class FSM(State, encoding=encoding):
        A = state(); B = state(); C = state(); D = state()
    return FSM


def _build_4state(encoding) -> tuple[Module, dict, type]:
    FSM = _make_4state_fsm(encoding)
    m = Module(f"fsm4_{encoding.value}", with_clock=True, with_reset=False)
    x = m.input(Bool(), "x")
    st = m.reg(FSM.typ, "st", init=FSM.A)
    # Output = a 2-bit "name index" so we can compare across encodings.
    name_idx = m.output(UInt(2), "name_idx")
    name_idx <<= 0                              # default driver (wire/output needs one)

    with switch_(st):
        with case_(FSM.A):
            name_idx <<= 0
            with if_(x): st <<= FSM.B
            with else_(): st <<= FSM.D
        with case_(FSM.B):
            name_idx <<= 1
            with if_(x): st <<= FSM.C
            with else_(): st <<= FSM.A
        with case_(FSM.C):
            name_idx <<= 2
            with if_(x): st <<= FSM.D
            with else_(): st <<= FSM.B
        with case_(FSM.D):
            name_idx <<= 3
            with if_(x): st <<= FSM.A
            with else_(): st <<= FSM.C
        with default():
            name_idx <<= 0
            st <<= FSM.A
    return m, {"x": x, "st": st, "name_idx": name_idx}, FSM


@pytest.mark.parametrize("encoding", [Encoding.BINARY, Encoding.ONEHOT, Encoding.GRAY])
def test_encoding_equivalence(encoding):
    """All three encodings yield identical cycle-accurate behaviour."""
    m, sig, FSM = _build_4state(encoding)
    sim = Simulator(m)

    # Drive a deterministic 10-cycle stream and record the per-cycle output.
    pattern = [1, 1, 0, 1, 0, 0, 1, 1, 1, 0]
    trace_names = []
    for x_val in pattern:
        sim.set(sig["x"], x_val)
        sim.eval()
        trace_names.append(sim.get(sig["name_idx"]))
        sim.step()

    # Reference trace computed in plain Python from the same transition table.
    # Avoids depending on the BINARY result — keeps the parity test honest.
    name_of = {0: "A", 1: "B", 2: "C", 3: "D"}
    next_state = {
        ("A", 1): "B", ("A", 0): "D",
        ("B", 1): "C", ("B", 0): "A",
        ("C", 1): "D", ("C", 0): "B",
        ("D", 1): "A", ("D", 0): "C",
    }
    cur = "A"
    expected_idx = []
    name_to_idx = {"A": 0, "B": 1, "C": 2, "D": 3}
    for x_val in pattern:
        expected_idx.append(name_to_idx[cur])
        cur = next_state[(cur, x_val)]

    assert trace_names == expected_idx, (
        f"encoding={encoding.value}: trace={trace_names}, expected={expected_idx}")


# ---------------------------------------------------------------------------
# 6. State-class introspection contract
# ---------------------------------------------------------------------------

def test_state_class_metadata():
    """Sanity-check the class-level metadata the State subclass exposes."""
    class S(State, encoding=Encoding.BINARY):
        ALPHA = state(); BETA = state(); GAMMA = state()

    assert S.names == ["ALPHA", "BETA", "GAMMA"]
    assert S._width == 2
    assert S.typ.width == 2
    assert S.ALPHA.value == 0
    assert S.BETA.value == 1
    assert S.GAMMA.value == 2
    # State Const constants are usable in equality / case_ — no extra wrapping needed.
    assert hasattr(S.ALPHA, "value")
    assert hasattr(S.ALPHA, "typ")


def test_empty_state_class_does_not_crash():
    """A `State` subclass with no `state()` entries is a no-op (allows base classes)."""
    class S(State, encoding=Encoding.BINARY):
        pass
    # Should not raise; class metadata simply not populated.
    assert not hasattr(S, "names") or S.names == []


def test_unknown_encoding_raises():
    with pytest.raises(ValueError, match="Unknown encoding"):
        class S(State, encoding="not_an_encoding"):  # type: ignore[arg-type]
            A = state()
