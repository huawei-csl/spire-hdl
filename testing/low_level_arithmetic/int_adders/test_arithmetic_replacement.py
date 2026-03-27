import pytest

from sprouthdl.arithmetic.int_arithmetic_config import ArithmeticConfig, replace_arithmetic_ops
from sprouthdl.arithmetic.int_multipliers.eval.multiplier_stage_options_demo_lib import FSAOption
from sprouthdl.arithmetic.int_multipliers.eval.testvector_generation import (
    AdderTestVectors,
    Encoding,
    MultiplierTestVectors,
    SubtractorTestVectors,
)
from sprouthdl.arithmetic.int_multipliers.multipliers.multiplier_stage_core import StageBasedMultiplierIO
from sprouthdl.helpers import run_vectors_on_simulator
from sprouthdl.sprouthdl import Signal, UInt, reset_shared_cache
from sprouthdl.sprouthdl_module import Component
from sprouthdl.sprouthdl_simulator import Simulator


# ---------------------------------------------------------------------------
# Simple components that use plain operators (+, -, *)
# ---------------------------------------------------------------------------

class _AdderOp(Component):
    def __init__(self, w: int):
        self.w = w
        self.io = StageBasedMultiplierIO(
            a=Signal(name="a", typ=UInt(w), kind="input"),
            b=Signal(name="b", typ=UInt(w), kind="input"),
            y=Signal(name="y", typ=UInt(w + 1), kind="output"),
        )
        self.elaborate()

    def elaborate(self):
        self.io.y <<= self.io.a + self.io.b


class _SubtractorOp(Component):
    def __init__(self, w: int):
        self.w = w
        self.io = StageBasedMultiplierIO(
            a=Signal(name="a", typ=UInt(w), kind="input"),
            b=Signal(name="b", typ=UInt(w), kind="input"),
            y=Signal(name="y", typ=UInt(w + 1), kind="output"),
        )
        self.elaborate()

    def elaborate(self):
        self.io.y <<= self.io.a - self.io.b


class _MultiplierOp(Component):
    def __init__(self, w: int):
        self.w = w
        self.io = StageBasedMultiplierIO(
            a=Signal(name="a", typ=UInt(w), kind="input"),
            b=Signal(name="b", typ=UInt(w), kind="input"),
            y=Signal(name="y", typ=UInt(2 * w), kind="output"),
        )
        self.elaborate()

    def elaborate(self):
        self.io.y <<= self.io.a * self.io.b


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

CONFIG = ArithmeticConfig(
    fsa_opt=FSAOption.PREFIX_BRENT_KUNG,
)

N_BITS = 8
N_VECS = 64


def test_replace_add():
    reset_shared_cache()

    # Before replacement: graph contains Op2<+>
    comp_before = _AdderOp(N_BITS)
    report_before = comp_before.to_module("BeforeAdder").module_analyze()
    assert report_before.by_class_incl_typ.get("Op2<+>", 0) > 0

    # After replacement: Op2<+> is gone, replaced by bitwise ops
    comp = _AdderOp(N_BITS)
    replace_arithmetic_ops(comp, CONFIG)
    module = comp.to_module("ReplacedAdder", with_clock=True, with_reset=True)

    report_after = module.module_analyze()
    assert report_after.by_class_incl_typ.get("Op2<+>", 0) == 0
    assert report_after.op_nodes > report_before.op_nodes

    vecs = AdderTestVectors(
        a_w=N_BITS, b_w=N_BITS, y_w=comp.io.y.typ.width,
        num_vectors=N_VECS, a_encoding=Encoding.unsigned,
        b_encoding=Encoding.unsigned, y_encoding=Encoding.unsigned,
    ).generate()

    sim = Simulator(module)
    run_vectors_on_simulator(sim, vecs, use_signed=False, with_clk=False)


def test_replace_sub():
    reset_shared_cache()

    # Before replacement: graph contains Op2<->
    comp_before = _SubtractorOp(N_BITS)
    report_before = comp_before.to_module("BeforeSub").module_analyze()
    assert report_before.by_class_incl_typ.get("Op2<->", 0) > 0

    # After replacement: Op2<-> is gone
    comp = _SubtractorOp(N_BITS)
    replace_arithmetic_ops(comp, CONFIG)
    module = comp.to_module("ReplacedSubtractor", with_clock=True, with_reset=True)

    report_after = module.module_analyze()
    assert report_after.by_class_incl_typ.get("Op2<->", 0) == 0
    assert report_after.op_nodes > report_before.op_nodes

    vecs = SubtractorTestVectors(
        a_w=N_BITS, b_w=N_BITS, y_w=comp.io.y.typ.width,
        num_vectors=N_VECS, a_encoding=Encoding.unsigned,
        b_encoding=Encoding.unsigned, y_encoding=Encoding.twos_complement,
    ).generate()

    sim = Simulator(module)
    run_vectors_on_simulator(sim, vecs, use_signed=False, with_clk=False)


def test_replace_mul():
    reset_shared_cache()

    # Before replacement: graph contains Op2<*>
    comp_before = _MultiplierOp(N_BITS)
    report_before = comp_before.to_module("BeforeMul").module_analyze()
    assert report_before.by_class_incl_typ.get("Op2<*>", 0) > 0

    # After replacement: Op2<*> is gone
    comp = _MultiplierOp(N_BITS)
    replace_arithmetic_ops(comp, CONFIG)
    module = comp.to_module("ReplacedMultiplier", with_clock=True, with_reset=True)

    report_after = module.module_analyze()
    assert report_after.by_class_incl_typ.get("Op2<*>", 0) == 0
    assert report_after.op_nodes > report_before.op_nodes

    vecs = MultiplierTestVectors(
        a_w=N_BITS, b_w=N_BITS, y_w=comp.io.y.typ.width,
        num_vectors=N_VECS, a_encoding=Encoding.unsigned,
        b_encoding=Encoding.unsigned, y_encoding=Encoding.unsigned,
    ).generate()

    sim = Simulator(module)
    run_vectors_on_simulator(sim, vecs, use_signed=False, with_clk=False)


def test_skip_different_width():
    """Op2 nodes with different-width operands should NOT be replaced."""
    reset_shared_cache()

    class _DiffWidthAdder(Component):
        def __init__(self):
            self.io = StageBasedMultiplierIO(
                a=Signal(name="a", typ=UInt(8), kind="input"),
                b=Signal(name="b", typ=UInt(4), kind="input"),
                y=Signal(name="y", typ=UInt(9), kind="output"),
            )
            self.elaborate()

        def elaborate(self):
            self.io.y <<= self.io.a + self.io.b

    comp = _DiffWidthAdder()

    report_before = comp.to_module("DiffWidthBefore").module_analyze()
    assert report_before.by_class_incl_typ.get("Op2<+>", 0) > 0

    # Re-create since to_module consumed the graph state
    comp = _DiffWidthAdder()
    replace_arithmetic_ops(comp, CONFIG)
    module = comp.to_module("DiffWidthAfter")

    report_after = module.module_analyze()
    # Op2<+> should still be present — widths differ, no replacement
    assert report_after.by_class_incl_typ.get("Op2<+>", 0) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
