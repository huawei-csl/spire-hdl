"""Tests for the per-signal-tag custom-Verilog escape hatch on Components.

A Component opts into custom emission by defining ``custom_verilog(self) -> str``. When the framework detects
this, the signals/expressions produced by ``elaborate()`` are tagged so the Verilog emitter skips them, and the
custom string is emitted alongside the rest of the module's auto-emitted Verilog. Python simulation still uses
``elaborate()``'s logic.

This is the option B / per-signal tagging design (see ``/workspaces/rtl_scout/component_custom_verilog.md``).
Blackboxes (Components with empty ``elaborate()``) are intentionally not supported by this implementation — the
walker still needs a sim model to evaluate.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from spirehdl.spirehdl import (
    Bool,
    Register,
    Signal,
    UInt,
    Wire,
    reset_shared_cache,
)
from spirehdl.spirehdl_module import Component, Module
from spirehdl.spirehdl_simulator import Simulator


# ---------------------------------------------------------------------------
# Fixture: a tiny adder that has both a Python sim model and a custom Verilog
# ---------------------------------------------------------------------------

class CustomAdder(Component):
    """8-bit + 8-bit -> 9-bit adder.

    ``elaborate()`` builds a non-trivial reference model that goes through an intermediate wire (``tmp_sum``) —
    that wire should disappear from the emitted Verilog when ``custom_verilog`` is in effect, but the simulator
    should still produce the correct sum via ``elaborate``.
    """

    def __init__(self):
        @dataclass
        class IO:
            a: Signal
            b: Signal
            sum: Signal
        self.io = IO(
            a   = Signal(typ=UInt(8), kind="input"),
            b   = Signal(typ=UInt(8), kind="input"),
            sum = Signal(typ=UInt(9), kind="output"),
        )
        self.elaborate()

    def elaborate(self):
        # Intermediate wire so a non-trivial chain exists; the tag-walker must reach and tag this too. If
        # `tmp_sum` ever appears in the emitted Verilog, the tagging is broken.
        tmp_sum = Wire(UInt(9))
        tmp_sum <<= self.io.a + self.io.b
        self.io.sum <<= tmp_sum

    def custom_verilog(self) -> str:
        # User-supplied implementation. Uses the IO Signal names so it picks up whatever uniquification did.
        return (
            f"  // --- custom adder (hand-written) ---\n"
            f"  assign {self.io.sum.name} = {self.io.a.name} + {self.io.b.name};"
        )


# ---------------------------------------------------------------------------
# Top-level Component with custom_verilog
# ---------------------------------------------------------------------------

def test_top_level_emits_custom_block_and_skips_sim_only_wires():
    reset_shared_cache()
    comp = CustomAdder()
    m = comp.to_module(name="CustomAdder", with_clock=False, with_reset=False)
    v = m.to_verilog()

    # Custom block landed in the output.
    assert "// --- custom adder (hand-written) ---" in v
    assert "assign sum = a + b;" in v

    # Sim-only intermediate wire was tagged and skipped.
    assert "tmp_sum" not in v
    # No auto-emitted assign for `sum` (only the custom one).
    assert v.count("assign sum =") == 1
    # And no `wire [8:0] tmp_sum;` declaration leaked through.
    assert "wire [8:0] tmp_sum" not in v


def test_top_level_simulator_uses_elaborate_logic():
    reset_shared_cache()
    comp = CustomAdder()
    m = comp.to_module(name="CustomAdder", with_clock=False, with_reset=False)

    sim = Simulator(m)
    sim.set("a", 5).set("b", 7).eval()
    assert sim.get("sum") == 12

    sim.set("a", 200).set("b", 100).eval()
    assert sim.get("sum") == 300  # 9-bit sum, doesn't overflow


# ---------------------------------------------------------------------------
# Component without custom_verilog — sanity check the path didn't regress
# ---------------------------------------------------------------------------

class PlainAdder(Component):
    """Same hardware as CustomAdder but without the custom_verilog method.
    Used to verify the existing auto-emit path is unaffected."""

    def __init__(self):
        @dataclass
        class IO:
            a: Signal
            b: Signal
            sum: Signal
        self.io = IO(
            a   = Signal(typ=UInt(8), kind="input"),
            b   = Signal(typ=UInt(8), kind="input"),
            sum = Signal(typ=UInt(9), kind="output"),
        )
        self.elaborate()

    def elaborate(self):
        tmp_sum = Wire(UInt(9))
        tmp_sum <<= self.io.a + self.io.b
        self.io.sum <<= tmp_sum


def test_plain_component_unaffected_by_feature():
    reset_shared_cache()
    plain = PlainAdder()
    m = plain.to_module(name="PlainAdder", with_clock=False, with_reset=False)
    v = m.to_verilog()
    # No custom block emitted (no Component opted in).
    assert "// Custom Verilog" not in v
    # The intermediate wire SHOULD appear in auto-emit mode (with whatever assigns the CSE share-wire path
    # produced — we just want to confirm it's not skipped).
    assert "tmp_sum" in v
    assert "assign sum = tmp_sum;" in v
    # The (a + b) expression appears somewhere in the auto-emit (might be via a CSE-shared `sig_N` wire, but
    # the substring is always present).
    assert "(a + b)" in v


# ---------------------------------------------------------------------------
# Embedded use: a CustomAdder nested inside a parent that adds an extra input
# ---------------------------------------------------------------------------

class TopWithCustomInner(Component):
    """parent: result = (a + b) + c, where the (a + b) is computed by an embedded CustomAdder. The parent's
    `+ c` stays auto-emitted; the inner block is replaced by its custom_verilog."""

    def __init__(self):
        @dataclass
        class IO:
            a: Signal
            b: Signal
            c: Signal
            result: Signal
        self.io = IO(
            a      = Signal(typ=UInt(8), kind="input"),
            b      = Signal(typ=UInt(8), kind="input"),
            c      = Signal(typ=UInt(8), kind="input"),
            result = Signal(typ=UInt(10), kind="output"),
        )
        self.elaborate()

    def elaborate(self):
        inner = CustomAdder().make_internal()
        inner.io.a <<= self.io.a
        inner.io.b <<= self.io.b
        self.io.result <<= inner.io.sum + self.io.c


def test_embedded_inner_emits_custom_block_and_parent_auto_emits():
    reset_shared_cache()
    top = TopWithCustomInner()
    m = top.to_module(name="Top", with_clock=False, with_reset=False)
    v = m.to_verilog()

    # Inner's custom block landed in the parent's module.
    assert "// --- custom adder (hand-written) ---" in v

    # Inner's elaborate-only wire is gone.
    assert "tmp_sum" not in v
    assert "wire [8:0] tmp_sum" not in v

    # Parent's `+ c` is still auto-emitted (combinational assign for `result`). The exact name of the inner's
    # `sum` wire might be uniquified, but the `+ c)` substring should always appear.
    assert "+ c)" in v

    # Inner's IO wires are emitted as declarations (parent uses them); but the assigns DRIVING the inner's
    # `sum` (from elaborate) are skipped.
    # Specifically, no `assign <inner_sum_wire> = (a + b);` line.
    assert "= (a + b);" not in v


def test_embedded_simulation_full_pipeline():
    reset_shared_cache()
    top = TopWithCustomInner()
    m = top.to_module(name="Top", with_clock=False, with_reset=False)
    sim = Simulator(m)
    sim.set("a", 5).set("b", 7).set("c", 3).eval()
    assert sim.get("result") == 15

    sim.set("a", 100).set("b", 50).set("c", 25).eval()
    assert sim.get("result") == 175


def test_embedded_multiple_instances():
    """Two CustomAdder instances inside one parent — both custom blocks should appear in the output, with names
    matching the (possibly-uniquified) IO wires of each instance."""

    class TopTwoAdders(Component):
        def __init__(self):
            @dataclass
            class IO:
                a: Signal; b: Signal; c: Signal; d: Signal
                ab_sum: Signal; cd_sum: Signal
            self.io = IO(
                a      = Signal(typ=UInt(8), kind="input"),
                b      = Signal(typ=UInt(8), kind="input"),
                c      = Signal(typ=UInt(8), kind="input"),
                d      = Signal(typ=UInt(8), kind="input"),
                ab_sum = Signal(typ=UInt(9), kind="output"),
                cd_sum = Signal(typ=UInt(9), kind="output"),
            )
            self.elaborate()
        def elaborate(self):
            ab = CustomAdder().make_internal()
            ab.io.a <<= self.io.a
            ab.io.b <<= self.io.b
            self.io.ab_sum <<= ab.io.sum
            cd = CustomAdder().make_internal()
            cd.io.a <<= self.io.c
            cd.io.b <<= self.io.d
            self.io.cd_sum <<= cd.io.sum

    reset_shared_cache()
    top = TopTwoAdders()
    m = top.to_module(name="TopTwo", with_clock=False, with_reset=False)
    v = m.to_verilog()
    # Two distinct custom blocks (one per instance).
    assert v.count("// --- custom adder (hand-written) ---") == 2

    sim = Simulator(m)
    sim.set("a", 1).set("b", 2).set("c", 3).set("d", 4).eval()
    assert sim.get("ab_sum") == 3
    assert sim.get("cd_sum") == 7


# ---------------------------------------------------------------------------
# Sequential logic inside the Component — Registers in elaborate are tagged
# ---------------------------------------------------------------------------

class CustomCounter(Component):
    """Accumulator with both Python sim (using a Register) and custom Verilog."""

    def __init__(self):
        @dataclass
        class IO:
            inc: Signal
            value: Signal
        self.io = IO(
            inc   = Signal(typ=UInt(8), kind="input"),
            value = Signal(typ=UInt(8), kind="output"),
        )
        self.elaborate()

    def elaborate(self):
        cnt = Register(UInt(8), init=0)
        cnt <<= cnt + self.io.inc
        self.io.value <<= cnt

    def custom_verilog(self) -> str:
        return (
            f"  // --- hand-tuned accumulator ---\n"
            f"  reg [7:0] hand_cnt;\n"
            f"  always @(posedge clk or posedge rst) begin\n"
            f"    if (rst) hand_cnt <= 8'd0;\n"
            f"    else     hand_cnt <= hand_cnt + {self.io.inc.name};\n"
            f"  end\n"
            f"  assign {self.io.value.name} = hand_cnt;"
        )


def test_register_in_elaborate_is_tagged_no_emit():
    reset_shared_cache()
    counter = CustomCounter()
    m = counter.to_module(name="CustomCounter", with_clock=True, with_reset=True)
    v = m.to_verilog()

    # The auto-emit Register from elaborate should NOT appear (auto-named "cnt").
    assert "reg [7:0] cnt;" not in v
    # The hand-tuned register from the custom block SHOULD appear.
    assert "reg [7:0] hand_cnt;" in v
    # The elaborate's assign for `value` should NOT appear (only the custom one).
    assert v.count("assign value =") == 1
    assert "assign value = hand_cnt;" in v
    # No auto-emit always block for cnt.
    assert "cnt <= (cnt + inc)" not in v


def test_register_in_elaborate_simulates_correctly():
    reset_shared_cache()
    counter = CustomCounter()
    m = counter.to_module(name="CustomCounter", with_clock=True, with_reset=True)
    sim = Simulator(m)
    sim.deassert_reset()
    sim.set("inc", 3).step()
    assert sim.get("value") == 3
    sim.step()
    assert sim.get("value") == 6
    sim.set("inc", 1).step()
    assert sim.get("value") == 7
