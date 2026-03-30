import pytest
from dataclasses import dataclass

from sprouthdl.arithmetic.int_arithmetic_config import (
    ArithmeticAutoConfig,
    replace_arithmetic_ops,
)
from sprouthdl.arithmetic.int_multipliers.eval.testvector_generation import (
    AdderTestVectors,
    Encoding,
    MultiplierTestVectors,
    SubtractorTestVectors,
)
from sprouthdl.arithmetic.int_multipliers.multipliers.multiplier_stage_core import StageBasedMultiplierIO
from sprouthdl.helpers import get_yosys_metrics, get_aig_stats, run_vectors_on_simulator
from sprouthdl.sprouthdl import Signal, UInt, reset_shared_cache
from sprouthdl.sprouthdl_module import Component
from sprouthdl.sprouthdl_simulator import Simulator


# ---------------------------------------------------------------------------
# ALU component using plain operators
# ---------------------------------------------------------------------------

@dataclass
class _ALUIO:
    a: Signal
    b: Signal
    y_add: Signal
    y_sub: Signal
    y_mul: Signal


class _ALU(Component):
    def __init__(self, w: int):
        self.w = w
        self.io = _ALUIO(
            a=Signal(name="a", typ=UInt(w), kind="input"),
            b=Signal(name="b", typ=UInt(w), kind="input"),
            y_add=Signal(name="y_add", typ=UInt(w + 1), kind="output"),
            y_sub=Signal(name="y_sub", typ=UInt(w + 1), kind="output"),
            y_mul=Signal(name="y_mul", typ=UInt(2 * w), kind="output"),
        )
        self.elaborate()

    def elaborate(self):
        self.io.y_add <<= self.io.a + self.io.b
        self.io.y_sub <<= self.io.a - self.io.b
        self.io.y_mul <<= self.io.a * self.io.b


N_BITS = 8
N_VECS = 64
AUTO_CFG = ArithmeticAutoConfig(objective="area")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_auto_config_replaces_all_ops():
    """All Op2<+>, Op2<->, Op2<*> nodes must be gone after replacement."""
    reset_shared_cache()

    comp = _ALU(N_BITS)
    replace_arithmetic_ops(comp, AUTO_CFG)
    module = comp.to_module("AutoALU", with_clock=True, with_reset=True)

    report = module.module_analyze()
    assert report.by_class_incl_typ.get("Op2<+>", 0) == 0
    assert report.by_class_incl_typ.get("Op2<->", 0) == 0
    assert report.by_class_incl_typ.get("Op2<*>", 0) == 0


def test_auto_config_correctness_add():
    """Replaced ALU adder produces correct results."""
    reset_shared_cache()

    comp = _ALU(N_BITS)
    replace_arithmetic_ops(comp, AUTO_CFG)
    module = comp.to_module("AutoALU_add", with_clock=True, with_reset=True)

    vecs = AdderTestVectors(
        a_w=N_BITS, b_w=N_BITS, y_w=comp.io.y_add.typ.width,
        num_vectors=N_VECS,
        a_encoding=Encoding.unsigned, b_encoding=Encoding.unsigned,
        y_encoding=Encoding.unsigned,
    ).generate()

    sim = Simulator(module)
    remapped = []
    for name, inputs, expected in vecs:
        remapped.append((name, inputs, {"y_add": expected["y"]}))
    run_vectors_on_simulator(sim, remapped, use_signed=False, with_clk=False)


def test_auto_config_correctness_sub():
    """Replaced ALU subtractor produces correct results."""
    reset_shared_cache()

    comp = _ALU(N_BITS)
    replace_arithmetic_ops(comp, AUTO_CFG)
    module = comp.to_module("AutoALU_sub", with_clock=True, with_reset=True)

    vecs = SubtractorTestVectors(
        a_w=N_BITS, b_w=N_BITS, y_w=comp.io.y_sub.typ.width,
        num_vectors=N_VECS,
        a_encoding=Encoding.unsigned, b_encoding=Encoding.unsigned,
        y_encoding=Encoding.twos_complement,
    ).generate()

    sim = Simulator(module)
    remapped = []
    for name, inputs, expected in vecs:
        remapped.append((name, inputs, {"y_sub": expected["y"]}))
    run_vectors_on_simulator(sim, remapped, use_signed=False, with_clk=False)


def test_auto_config_correctness_mul():
    """Replaced ALU multiplier produces correct results."""
    reset_shared_cache()

    comp = _ALU(N_BITS)
    replace_arithmetic_ops(comp, AUTO_CFG)
    module = comp.to_module("AutoALU_mul", with_clock=True, with_reset=True)

    vecs = MultiplierTestVectors(
        a_w=N_BITS, b_w=N_BITS, y_w=comp.io.y_mul.typ.width,
        num_vectors=N_VECS,
        a_encoding=Encoding.unsigned, b_encoding=Encoding.unsigned,
        y_encoding=Encoding.unsigned,
    ).generate()

    sim = Simulator(module)
    remapped = []
    for name, inputs, expected in vecs:
        remapped.append((name, inputs, {"y_mul": expected["y"]}))
    run_vectors_on_simulator(sim, remapped, use_signed=False, with_clk=False)


def test_transistor_count_comparison():
    """Compare yosys transistor count and AIG depth across all objectives."""
    reset_shared_cache()

    # Unreplaced (operator-based, Yosys default synthesis)
    comp_plain = _ALU(N_BITS)
    mod_plain = comp_plain.to_module("ALU_plain")
    ym_plain = get_yosys_metrics(mod_plain)
    aig_plain = get_aig_stats(mod_plain)
    tc_plain = ym_plain["estimated_num_transistors"]
    depth_plain = aig_plain["depth"]

    results = [("plain (Yosys *)", tc_plain, depth_plain)]

    for objective in ("area", "delay", "adp"):
        reset_shared_cache()
        cfg = ArithmeticAutoConfig(objective=objective)
        comp = _ALU(N_BITS)
        replace_arithmetic_ops(comp, cfg)
        mod = comp.to_module(f"ALU_{objective}")
        ym = get_yosys_metrics(mod)
        aig = get_aig_stats(mod)
        results.append((objective, ym["estimated_num_transistors"], aig["depth"]))

    print(f"\n{'='*60}")
    print(f"  {N_BITS}-bit ALU (add + sub + mul) comparison")
    print(f"  {'Objective':<20s} {'Transistors':>12s} {'AIG Depth':>10s}")
    print(f"  {'-'*42}")
    for name, tc, depth in results:
        print(f"  {name:<20s} {tc:>12d} {depth:>10d}")
    print(f"{'='*60}")

    # Verify replacement happened
    reset_shared_cache()
    comp_check = _ALU(N_BITS)
    replace_arithmetic_ops(comp_check, AUTO_CFG)
    report = comp_check.to_module("ALU_check").module_analyze()
    assert report.by_class_incl_typ.get("Op2<+>", 0) == 0
    assert report.by_class_incl_typ.get("Op2<*>", 0) == 0


@pytest.mark.parametrize("objective", ["area", "delay", "adp"])
def test_all_objectives(objective):
    """Each objective should replace all ops and produce correct add results."""
    reset_shared_cache()

    cfg = ArithmeticAutoConfig(objective=objective)
    comp = _ALU(N_BITS)
    replace_arithmetic_ops(comp, cfg)
    module = comp.to_module(f"AutoALU_{objective}", with_clock=True, with_reset=True)

    report = module.module_analyze()
    assert report.by_class_incl_typ.get("Op2<+>", 0) == 0
    assert report.by_class_incl_typ.get("Op2<->", 0) == 0
    assert report.by_class_incl_typ.get("Op2<*>", 0) == 0

    # Quick functional check on adder
    vecs = AdderTestVectors(
        a_w=N_BITS, b_w=N_BITS, y_w=comp.io.y_add.typ.width,
        num_vectors=16,
        a_encoding=Encoding.unsigned, b_encoding=Encoding.unsigned,
        y_encoding=Encoding.unsigned,
    ).generate()
    sim = Simulator(module)
    remapped = [(n, i, {"y_add": e["y"]}) for n, i, e in vecs]
    run_vectors_on_simulator(sim, remapped, use_signed=False, with_clk=False)


def test_objectives_produce_different_configs():
    """Area vs delay objectives should potentially pick different configs for multipliers."""
    from sprouthdl.arithmetic.eval.auto_config import lookup_best_config

    # 16-bit unsigned multiplier has clear area vs delay tradeoff
    area_cfg, _ = lookup_best_config("*", 16, 16, signed=False, objective="area")
    delay_cfg, _ = lookup_best_config("*", 16, 16, signed=False, objective="delay")

    print(f"\n16-bit unsigned multiplier configs:")
    print(f"  area: tc={area_cfg['transistor_count']}, depth={area_cfg['aig_depth']}, "
          f"fsa={area_cfg['fsa_opt']}, ppg={area_cfg['ppg_opt']}, ppa={area_cfg['ppa_opt']}")
    print(f"  delay: tc={delay_cfg['transistor_count']}, depth={delay_cfg['aig_depth']}, "
          f"fsa={delay_cfg['fsa_opt']}, ppg={delay_cfg['ppg_opt']}, ppa={delay_cfg['ppa_opt']}")

    # Area-optimized should have fewer transistors
    assert area_cfg["transistor_count"] <= delay_cfg["transistor_count"]
    # Delay-optimized should have smaller or equal depth
    assert delay_cfg["aig_depth"] <= area_cfg["aig_depth"]


def test_swap_selection():
    """For asymmetric commutative ops, auto-config should pick the best orientation."""
    from sprouthdl.arithmetic.eval.auto_config import lookup_best_config

    # Check that both orientations are considered for 4x8 multiplier
    cfg_4x8, swap_4x8 = lookup_best_config("*", 4, 8, signed=False, objective="area")
    cfg_8x4, swap_8x4 = lookup_best_config("*", 8, 4, signed=False, objective="area")

    # Both lookups should find the same best config (since both orientations are checked)
    assert cfg_4x8["transistor_count"] == cfg_8x4["transistor_count"]
    # But one of them should have swap=True
    assert swap_4x8 != swap_8x4

    print(f"\n4x8 mul area: tc={cfg_4x8['transistor_count']}, swap={swap_4x8}")
    print(f"8x4 mul area: tc={cfg_8x4['transistor_count']}, swap={swap_8x4}")


# ---------------------------------------------------------------------------
# Asymmetric-width tests
# ---------------------------------------------------------------------------

A_W = 4
B_W = 8


@dataclass
class _AsymALUIO:
    a: Signal
    b: Signal
    y_add: Signal
    y_sub: Signal
    y_mul: Signal


class _AsymALU(Component):
    def __init__(self, a_w: int, b_w: int):
        self.a_w = a_w
        self.b_w = b_w
        max_w = max(a_w, b_w)
        self.io = _AsymALUIO(
            a=Signal(name="a", typ=UInt(a_w), kind="input"),
            b=Signal(name="b", typ=UInt(b_w), kind="input"),
            y_add=Signal(name="y_add", typ=UInt(max_w + 1), kind="output"),
            y_sub=Signal(name="y_sub", typ=UInt(max_w + 1), kind="output"),
            y_mul=Signal(name="y_mul", typ=UInt(a_w + b_w), kind="output"),
        )
        self.elaborate()

    def elaborate(self):
        self.io.y_add <<= self.io.a + self.io.b
        self.io.y_sub <<= self.io.a - self.io.b
        self.io.y_mul <<= self.io.a * self.io.b


def test_asymmetric_replaces_all_ops():
    """Asymmetric-width Op2 nodes must all be replaced."""
    reset_shared_cache()

    comp = _AsymALU(A_W, B_W)
    replace_arithmetic_ops(comp, AUTO_CFG)
    module = comp.to_module("AsymALU", with_clock=True, with_reset=True)

    report = module.module_analyze()
    assert report.by_class_incl_typ.get("Op2<+>", 0) == 0
    assert report.by_class_incl_typ.get("Op2<->", 0) == 0
    assert report.by_class_incl_typ.get("Op2<*>", 0) == 0


def test_asymmetric_correctness_add():
    """Replaced asymmetric adder (4+8 bit) produces correct results."""
    reset_shared_cache()

    comp = _AsymALU(A_W, B_W)
    replace_arithmetic_ops(comp, AUTO_CFG)
    module = comp.to_module("AsymALU_add", with_clock=True, with_reset=True)

    vecs = AdderTestVectors(
        a_w=A_W, b_w=B_W, y_w=comp.io.y_add.typ.width,
        num_vectors=N_VECS,
        a_encoding=Encoding.unsigned, b_encoding=Encoding.unsigned,
        y_encoding=Encoding.unsigned,
    ).generate()

    sim = Simulator(module)
    remapped = [(n, i, {"y_add": e["y"]}) for n, i, e in vecs]
    run_vectors_on_simulator(sim, remapped, use_signed=False, with_clk=False)


def test_asymmetric_correctness_sub():
    """Replaced asymmetric subtractor (4-8 bit) produces correct results."""
    reset_shared_cache()

    comp = _AsymALU(A_W, B_W)
    replace_arithmetic_ops(comp, AUTO_CFG)
    module = comp.to_module("AsymALU_sub", with_clock=True, with_reset=True)

    vecs = SubtractorTestVectors(
        a_w=A_W, b_w=B_W, y_w=comp.io.y_sub.typ.width,
        num_vectors=N_VECS,
        a_encoding=Encoding.unsigned, b_encoding=Encoding.unsigned,
        y_encoding=Encoding.twos_complement,
    ).generate()

    sim = Simulator(module)
    remapped = [(n, i, {"y_sub": e["y"]}) for n, i, e in vecs]
    run_vectors_on_simulator(sim, remapped, use_signed=False, with_clk=False)


def test_asymmetric_correctness_mul():
    """Replaced asymmetric multiplier (4*8 bit) produces correct results."""
    reset_shared_cache()

    comp = _AsymALU(A_W, B_W)
    replace_arithmetic_ops(comp, AUTO_CFG)
    module = comp.to_module("AsymALU_mul", with_clock=True, with_reset=True)

    vecs = MultiplierTestVectors(
        a_w=A_W, b_w=B_W, y_w=comp.io.y_mul.typ.width,
        num_vectors=N_VECS,
        a_encoding=Encoding.unsigned, b_encoding=Encoding.unsigned,
        y_encoding=Encoding.unsigned,
    ).generate()

    sim = Simulator(module)
    remapped = [(n, i, {"y_mul": e["y"]}) for n, i, e in vecs]
    run_vectors_on_simulator(sim, remapped, use_signed=False, with_clk=False)


def test_asymmetric_transistor_comparison():
    """Compare transistor count: plain vs auto-replaced for asymmetric ALU."""
    reset_shared_cache()

    comp_plain = _AsymALU(A_W, B_W)
    mod_plain = comp_plain.to_module("AsymALU_plain")
    ym_plain = get_yosys_metrics(mod_plain)
    tc_plain = ym_plain["estimated_num_transistors"]

    reset_shared_cache()

    comp_auto = _AsymALU(A_W, B_W)
    replace_arithmetic_ops(comp_auto, AUTO_CFG)
    mod_auto = comp_auto.to_module("AsymALU_auto")
    ym_auto = get_yosys_metrics(mod_auto)
    tc_auto = ym_auto["estimated_num_transistors"]

    print(f"\nAsymmetric ALU ({A_W}+{B_W} bit) transistor comparison:")
    print(f"  Plain: {tc_plain}")
    print(f"  Auto:  {tc_auto}")

    report = mod_auto.module_analyze()
    assert report.by_class_incl_typ.get("Op2<+>", 0) == 0
    assert report.by_class_incl_typ.get("Op2<*>", 0) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
