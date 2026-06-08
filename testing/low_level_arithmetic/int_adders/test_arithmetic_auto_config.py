import pytest
from dataclasses import dataclass

from spirehdl.arithmetic.int_arithmetic_config import (
    ArithmeticAutoConfig,
    replace_arithmetic_ops,
)
from spirehdl.arithmetic.int_multipliers.eval.testvector_generation import (
    AdderTestVectors,
    Encoding,
    MultiplierTestVectors,
    SubtractorTestVectors,
)
from spirehdl.arithmetic.int_multipliers.multipliers.multiplier_stage_core import StageBasedMultiplierIO
from spirehdl.helpers import get_yosys_metrics, get_aig_stats, run_vectors_on_simulator
from spirehdl.spirehdl import Signal, UInt, reset_shared_cache
from spirehdl.spirehdl_module import Component
from spirehdl.spirehdl_simulator import Simulator


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
    from spirehdl.arithmetic.eval.auto_config import lookup_best_config

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
    """Commutative ops: calling with (a,b) vs (b,a) must pick hardware that is equal
    on the *selection metric*, and for at least one asymmetric shape the swap flag must flip.

    The comparison uses ``transistor_count_heavy`` (== ``DEFAULT_LOOKUP_METRIC``), i.e. the
    column the lookup actually optimizes — NOT the secondary ``transistor_count`` (light)
    field. When the two orientations are co-optimal on the heavy metric, the lookup's
    tie-break may return configs whose light count differs by a couple transistors; that's an
    acceptable tie, not an asymmetry in what gets optimized.
    """
    from spirehdl.arithmetic.eval.auto_config import lookup_best_config

    from itertools import product

    widths = [1, 2, 4, 8, 16]
    flips = 0
    for op in ("*", "+"):
        for signed in (False, True):
            for objective in ("area", "delay", "adp"):
                for a_w, b_w in product(widths, repeat=2):
                    if a_w == b_w:
                        continue
                    cfg_ab, swap_ab = lookup_best_config(op, a_w, b_w, signed, objective)
                    cfg_ba, swap_ba = lookup_best_config(op, b_w, a_w, signed, objective)
                    if cfg_ab is None or cfg_ba is None:
                        continue
                    assert cfg_ab["transistor_count_heavy"] == cfg_ba["transistor_count_heavy"]
                    assert cfg_ab["aig_depth"] == cfg_ba["aig_depth"]
                    if swap_ab != swap_ba:
                        flips += 1
    assert flips > 0, "no observable swap flip across the full sweep"
    print(f"\nobserved {flips} swap flips across asymmetric commutative shapes")


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


# ---------------------------------------------------------------------------
# MAC (fused multiply-accumulate) tests
# ---------------------------------------------------------------------------

@dataclass
class _MacIO:
    a: Signal
    b: Signal
    c: Signal
    y: Signal


class _Mac(Component):
    def __init__(self, w: int):
        self.w = w
        self.io = _MacIO(
            a=Signal(name="a", typ=UInt(w), kind="input"),
            b=Signal(name="b", typ=UInt(w), kind="input"),
            c=Signal(name="c", typ=UInt(2 * w), kind="input"),
            y=Signal(name="y", typ=UInt(2 * w + 1), kind="output"),
        )
        self.elaborate()

    def elaborate(self):
        self.io.y <<= self.io.a * self.io.b + self.io.c


def test_mac_replaces_ops():
    """MAC pattern (a*b + c) should replace both * and + (fused or separate)."""
    reset_shared_cache()

    comp = _Mac(N_BITS)
    replace_arithmetic_ops(comp, AUTO_CFG)
    module = comp.to_module("MAC_replaced", with_clock=True, with_reset=True)

    report = module.module_analyze()
    # The plain + should be gone (either fused or replaced with prefix adder)
    assert report.by_class_incl_typ.get("Op2<+>", 0) == 0


def test_mac_correctness():
    """Replaced MAC produces correct y = a*b + c results."""
    reset_shared_cache()
    import random

    comp = _Mac(N_BITS)
    replace_arithmetic_ops(comp, AUTO_CFG)
    module = comp.to_module("MAC_correct", with_clock=True, with_reset=True)

    sim = Simulator(module)
    rng = random.Random(42)
    mask_ab = (1 << N_BITS) - 1
    mask_c = (1 << (2 * N_BITS)) - 1
    mask_y = (1 << (2 * N_BITS + 1)) - 1

    for _ in range(N_VECS):
        a = rng.randint(0, mask_ab)
        b = rng.randint(0, mask_ab)
        c = rng.randint(0, mask_c)
        expected = (a * b + c) & mask_y
        sim.set("a", a).set("b", b).set("c", c).eval()
        got = sim.get("y")
        assert got == expected, f"{a}*{b}+{c}: got={got} exp={expected}"


def test_mac_depth_improvement():
    """Auto-config MAC should have lower or equal depth than fixed config."""
    reset_shared_cache()

    comp_sep = _Mac(N_BITS)
    from spirehdl.arithmetic.int_arithmetic_config import ArithmeticConfig
    replace_arithmetic_ops(comp_sep, ArithmeticConfig())
    mod_sep = comp_sep.to_module("MAC_sep")
    aig_sep = get_aig_stats(mod_sep)

    reset_shared_cache()

    comp_auto = _Mac(N_BITS)
    replace_arithmetic_ops(comp_auto, ArithmeticAutoConfig(objective="delay"))
    mod_auto = comp_auto.to_module("MAC_auto_delay")
    aig_auto = get_aig_stats(mod_auto)

    print(f"\nMAC depth comparison ({N_BITS}-bit):")
    print(f"  Fixed config: depth={aig_sep['depth']}")
    print(f"  Auto (delay): depth={aig_auto['depth']}")

    assert aig_auto["depth"] <= aig_sep["depth"]


# ---------------------------------------------------------------------------
# Inner product tests
# ---------------------------------------------------------------------------

@dataclass
class _DotIO:
    x0: Signal; x1: Signal; x2: Signal; x3: Signal
    c0: Signal; c1: Signal; c2: Signal; c3: Signal
    y: Signal


class _Dot4(Component):
    """4-term inner product: y = c0*x0 + c1*x1 + c2*x2 + c3*x3"""
    def __init__(self, w: int):
        self.w = w
        self.io = _DotIO(
            x0=Signal(name="x0", typ=UInt(w), kind="input"),
            x1=Signal(name="x1", typ=UInt(w), kind="input"),
            x2=Signal(name="x2", typ=UInt(w), kind="input"),
            x3=Signal(name="x3", typ=UInt(w), kind="input"),
            c0=Signal(name="c0", typ=UInt(w), kind="input"),
            c1=Signal(name="c1", typ=UInt(w), kind="input"),
            c2=Signal(name="c2", typ=UInt(w), kind="input"),
            c3=Signal(name="c3", typ=UInt(w), kind="input"),
            y=Signal(name="y", typ=UInt(2 * w + 2), kind="output"),
        )
        self.elaborate()

    def elaborate(self):
        self.io.y <<= (self.io.c0 * self.io.x0
                      + self.io.c1 * self.io.x1
                      + self.io.c2 * self.io.x2
                      + self.io.c3 * self.io.x3)


def test_inner_product_replaces_ops():
    """Inner product chain should replace all * and + ops."""
    reset_shared_cache()

    comp = _Dot4(N_BITS)
    replace_arithmetic_ops(comp, AUTO_CFG)
    module = comp.to_module("Dot4_replaced", with_clock=True, with_reset=True)

    report = module.module_analyze()
    assert report.by_class_incl_typ.get("Op2<+>", 0) == 0
    assert report.by_class_incl_typ.get("Op2<*>", 0) == 0


def test_inner_product_correctness():
    """Replaced inner product produces correct results."""
    reset_shared_cache()
    import random

    comp = _Dot4(N_BITS)
    replace_arithmetic_ops(comp, AUTO_CFG)
    module = comp.to_module("Dot4_correct", with_clock=True, with_reset=True)

    sim = Simulator(module)
    rng = random.Random(123)
    mask = (1 << N_BITS) - 1
    mask_y = (1 << (2 * N_BITS + 2)) - 1

    for _ in range(N_VECS):
        vals = {f"x{i}": rng.randint(0, mask) for i in range(4)}
        vals.update({f"c{i}": rng.randint(0, mask) for i in range(4)})
        expected = sum(vals[f"c{i}"] * vals[f"x{i}"] for i in range(4)) & mask_y
        for k, v in vals.items():
            sim.set(k, v)
        sim.eval()
        got = sim.get("y")
        assert got == expected, f"dot4: got={got} exp={expected}"


def test_inner_product_depth():
    """Inner product with auto-config should improve depth vs fixed config."""
    reset_shared_cache()

    comp_fixed = _Dot4(N_BITS)
    from spirehdl.arithmetic.int_arithmetic_config import ArithmeticConfig
    replace_arithmetic_ops(comp_fixed, ArithmeticConfig())
    mod_fixed = comp_fixed.to_module("Dot4_fixed")
    aig_fixed = get_aig_stats(mod_fixed)

    reset_shared_cache()

    comp_auto = _Dot4(N_BITS)
    replace_arithmetic_ops(comp_auto, ArithmeticAutoConfig(objective="delay"))
    mod_auto = comp_auto.to_module("Dot4_auto_delay")
    aig_auto = get_aig_stats(mod_auto)

    print(f"\n4-term inner product depth ({N_BITS}-bit):")
    print(f"  Fixed config: depth={aig_fixed['depth']}")
    print(f"  Auto (delay): depth={aig_auto['depth']}")

    assert aig_auto["depth"] <= aig_fixed["depth"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
