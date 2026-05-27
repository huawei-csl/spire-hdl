"""Tests for blackbox Components — Components with ``custom_verilog`` but empty ``elaborate``.

A blackbox is the limit case of the option-B custom-verilog feature: no Python sim model, only a Verilog string.
The collector seeds from peer IO wires when crossing a custom_verilog Component, so the parent's logic that
drives the blackbox's inputs is reachable even though the outputs have no driver chain back to them. The
simulator returns 0 for the blackbox's outputs as a stub (no Python model = no value to compute).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from spirehdl.spirehdl import (
    Signal,
    UInt,
    Wire,
    reset_shared_cache,
)
from spirehdl.spirehdl_module import Component
from spirehdl.spirehdl_simulator import Simulator


# ---------------------------------------------------------------------------
# Fixture: a blackbox adder — only custom_verilog, no elaborate logic
# ---------------------------------------------------------------------------

class BlackboxAdder(Component):
    """No Python model. The custom Verilog string is the only implementation."""

    def __init__(self):
        @dataclass
        class IO:
            a: Signal
            b: Signal
            sum: Signal
        self.io = IO(
            a   = Signal("a",   UInt(8), "input"),
            b   = Signal("b",   UInt(8), "input"),
            sum = Signal("sum", UInt(9), "output"),
        )
        self.elaborate()

    def elaborate(self):
        # Deliberately empty — this is a blackbox.
        pass

    def custom_verilog(self) -> str:
        return (
            f"  // --- blackbox adder ---\n"
            f"  assign {self.io.sum.name} = {self.io.a.name} + {self.io.b.name};"
        )


# ---------------------------------------------------------------------------
# Top-level blackbox
# ---------------------------------------------------------------------------

def test_top_level_blackbox_emits_custom_verilog():
    reset_shared_cache()
    bb = BlackboxAdder()
    m = bb.to_module(name="BlackboxAdder", with_clock=False, with_reset=False)
    v = m.to_verilog()

    # Custom block is the entire body — no auto-emit logic from elaborate.
    assert "// --- blackbox adder ---" in v
    assert "assign sum = a + b;" in v
    # Module has its ports declared but no internal wires/regs.
    assert "wire" not in re.sub(r"//.*", "", v)  # no wire decls (strip comments first)


def test_top_level_blackbox_simulator_returns_zero():
    reset_shared_cache()
    bb = BlackboxAdder()
    m = bb.to_module(name="BlackboxAdder", with_clock=False, with_reset=False)
    sim = Simulator(m)
    # Sim reads the output as 0 (stub) regardless of inputs — no Python model.
    sim.set("a", 5).set("b", 7).eval()
    assert sim.get("sum") == 0


# ---------------------------------------------------------------------------
# Embedded blackbox — the walker-reachability fix is the key thing being tested
# ---------------------------------------------------------------------------

class ParentWithBlackboxAndExtraLogic(Component):
    """Parent uses a blackbox AND has its own auto-emitted helper logic that
    drives the blackbox's inputs. Before fix 1 the helper logic would have
    been unreachable from any output cone (the blackbox's outputs have no
    driver chain back to the inputs) and the emitter would skip it.
    """

    def __init__(self):
        @dataclass
        class IO:
            x: Signal
            y: Signal
            result: Signal
        self.io = IO(
            x      = Signal("x",      UInt(8), "input"),
            y      = Signal("y",      UInt(8), "input"),
            result = Signal("result", UInt(9), "output"),
        )
        self.elaborate()

    def elaborate(self):
        # A helper expression that becomes a Wire because it's referenced twice
        # in driver chains; this Wire would NOT be reachable from `result` without
        # the peer-seeding fix in the collector.
        helper = Wire(UInt(8))
        helper <<= self.io.x ^ self.io.y

        bb = BlackboxAdder().make_internal()
        bb.io.a <<= helper      # parent drives blackbox.a from helper
        bb.io.b <<= self.io.y   # parent drives blackbox.b directly
        self.io.result <<= bb.io.sum


def test_embedded_blackbox_parent_helper_is_reachable():
    reset_shared_cache()
    top = ParentWithBlackboxAndExtraLogic()
    m = top.to_module(name="TopWithBlackbox", with_clock=False, with_reset=False)
    v = m.to_verilog()

    # The parent's helper Wire was reachable from the blackbox input — declared and driven.
    # (Driver may go via a CSE-shared `sig_N` wire — assert the (x ^ y) substring appears anywhere.)
    assert "wire [7:0] helper;" in v
    assert "(x ^ y)" in v

    # Blackbox IO wires are declared (parent uses them).
    assert "wire [8:0] sum;" in v
    assert "wire [7:0] a;" in v or "wire [7:0] a_1;" in v
    # Parent's assigns into the blackbox inputs are emitted.
    # (Names may be uniquified — `a` collides with the top's input `x` only if names overlap;
    # here `a` doesn't collide so it keeps its name.)
    assert re.search(r"assign a\b\s*=\s*helper;", v) is not None
    # Custom block emits.
    assert "// --- blackbox adder ---" in v


def test_embedded_blackbox_simulation_outputs_zero():
    reset_shared_cache()
    top = ParentWithBlackboxAndExtraLogic()
    m = top.to_module(name="TopWithBlackbox", with_clock=False, with_reset=False)
    sim = Simulator(m)
    # Result reads as 0 because the blackbox has no Python model — even though the parent's helper logic
    # computes a real value, the blackbox stops the eval chain and returns 0.
    sim.set("x", 0xAA).set("y", 0x55).eval()
    assert sim.get("result") == 0


# ---------------------------------------------------------------------------
# Multiple blackboxes — collector visits peers for each
# ---------------------------------------------------------------------------

def test_multiple_blackboxes_each_emit_and_stub():
    reset_shared_cache()

    class TwoBlackboxes(Component):
        def __init__(self):
            @dataclass
            class IO:
                a: Signal; b: Signal; c: Signal; d: Signal
                left_sum: Signal
                right_sum: Signal
            self.io = IO(
                a         = Signal("a",         UInt(8), "input"),
                b         = Signal("b",         UInt(8), "input"),
                c         = Signal("c",         UInt(8), "input"),
                d         = Signal("d",         UInt(8), "input"),
                left_sum  = Signal("left_sum",  UInt(9), "output"),
                right_sum = Signal("right_sum", UInt(9), "output"),
            )
            self.elaborate()
        def elaborate(self):
            left = BlackboxAdder().make_internal()
            left.io.a <<= self.io.a
            left.io.b <<= self.io.b
            self.io.left_sum <<= left.io.sum
            right = BlackboxAdder().make_internal()
            right.io.a <<= self.io.c
            right.io.b <<= self.io.d
            self.io.right_sum <<= right.io.sum

    m = TwoBlackboxes().to_module(name="TwoBlackboxes", with_clock=False, with_reset=False)
    v = m.to_verilog()
    # Two custom blocks emitted.
    assert v.count("// --- blackbox adder ---") == 2

    sim = Simulator(m)
    # Both outputs read as 0 (no Python model on either blackbox).
    sim.set("a", 1).set("b", 2).set("c", 3).set("d", 4).eval()
    assert sim.get("left_sum") == 0
    assert sim.get("right_sum") == 0
