# =========================
# Floating-point multiplier
# =========================
# Parameterizable by exponent width (EW) and fraction width (FW).
# - float16:  EW=5, FW=10   (total 16)
# - bfloat16: EW=8, FW=7    (total 16)

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
from spirehdl.helpers import run_vectors_on_simulator
from spirehdl.spirehdl import *
from spirehdl.spirehdl_module import Component, Module
from spirehdl.spirehdl_simulator import Simulator
from spirehdl.arithmetic.int_arithmetic_config import (
    MultiplierConfig,
    build_multiplier,
)

# Uses: Module, UInt, Bool, mux, cat (from your spire_hdl)


def _or_reduce_bits(vec_expr: Expr, hi: int, lo: int) -> Expr:
    """OR-reduce bits vec_expr[lo:hi+1] (inclusive). If hi < lo, returns 0."""
    if hi < lo:
        return 0
    acc = 0
    for i in range(lo, hi + 1):
        acc = acc | vec_expr[i]
    return acc


class FpMul(Component):

    def _extract_fields(self, a: Expr, b: Expr) -> Tuple[Expr, Expr, Expr, Expr, Expr, Expr]:
        W = self.W
        FW = self.FW
        sA = a[W - 1]  # sign
        sB = b[W - 1]

        eA = a[FW : W - 1]  # exponent EW bits
        eB = b[FW : W - 1]

        fA = a[0:FW]  # fraction FW bits
        fB = b[0:FW]
        return sA, sB, eA, eB, fA, fB

    def _classify_operands(self, eA: Expr, eB: Expr, fA: Expr, fB: Expr, exp_all_ones: int) -> Dict[str, Expr]:
        is_eA_zero = eA == 0
        is_eB_zero = eB == 0
        is_fA_zero = fA == 0
        is_fB_zero = fB == 0

        is_zeroA = is_eA_zero & is_fA_zero
        is_zeroB = is_eB_zero & is_fB_zero

        is_eA_all1 = eA == exp_all_ones
        is_eB_all1 = eB == exp_all_ones

        is_nanA = is_eA_all1 & (fA != 0)
        is_nanB = is_eB_all1 & (fB != 0)

        is_infA = is_eA_all1 & (fA == 0)
        is_infB = is_eB_all1 & (fB == 0)

        # NaN if: any NaN, or Inf * 0 (either way)
        is_nan_in = is_nanA | is_nanB | ((is_infA & is_zeroB) | (is_zeroA & is_infB))
        # Inf if: any Inf (and not NaN case)
        is_inf_in = is_infA | is_infB
        # Zero if: any zero (and not NaN or Inf cases)
        is_zero_in = is_zeroA | is_zeroB

        return {
            "is_eA_zero": is_eA_zero,
            "is_eB_zero": is_eB_zero,
            "is_zeroA": is_zeroA,
            "is_zeroB": is_zeroB,
            "is_nanA": is_nanA,
            "is_nanB": is_nanB,
            "is_infA": is_infA,
            "is_infB": is_infB,
            "is_nan_in": is_nan_in,
            "is_inf_in": is_inf_in,
            "is_zero_in": is_zero_in,
        }

    def _effective_mantissas(self, fA: Expr, fB: Expr, is_eA_zero: Expr, is_eB_zero: Expr) -> Tuple[Expr, Expr]:
        FW = self.FW
        one_and_frac_A = cat(fA, 1)  # width 1+FW
        one_and_frac_B = cat(fB, 1)

        mask_1pFW = (1 << (1 + FW)) - 1
        mskA = mux(is_eA_zero, 0, mask_1pFW)
        mskB = mux(is_eB_zero, 0, mask_1pFW)

        mA_eff = one_and_frac_A & mskA  # (1+FW) bits
        mB_eff = one_and_frac_B & mskB
        return mA_eff, mB_eff

    def _normalize_and_round(self, prod: Expr) -> Tuple[Expr, Expr, Expr, Expr]:
        FW = self.FW
        PROD_W = 2 + 2 * FW

        msb_high = prod[PROD_W - 1]  # 1 if product ≥ 2.0

        # Select normalized top (1+FW) bits (pre-round)
        top_hi = prod[FW + 1 : PROD_W]  # when msb_high=1 → [FW+1 : 2*FW+1]
        top_lo = prod[FW : PROD_W - 1]  # when msb_high=0 → [FW : 2*FW]
        mant_pre = mux(msb_high, top_hi, top_lo)  # (1+FW) bits

        # Rounding bits (GRS: guard/round/sticky)
        guard_hi = prod[FW]  # when msb_high=1 → bit FW
        guard_lo = prod[FW - 1]  # when msb_high=0 → bit FW-1
        guard = mux(msb_high, guard_hi, guard_lo)

        sticky_hi = _or_reduce_bits(prod, FW - 1, 0)  # when msb_high=1 → [0:FW]
        sticky_lo = _or_reduce_bits(prod, FW - 2, 0)  # when msb_high=0 → [0:FW-1]
        sticky = mux(msb_high, sticky_hi, sticky_lo)

        lsb = mant_pre[0]  # LSB of current mantissa
        round_up = guard & (sticky | lsb)  # round-to-nearest-even

        # Add rounding increment (width grows by 1)
        mant_round = mant_pre + mux(round_up, 1, 0)  # width (1+FW)+1 = FW+2
        carry = mant_round[FW + 1]  # did rounding overflow?
        mant_post = mux(carry, mant_round[1 : FW + 2], mant_round[0 : FW + 1])  # (1+FW) bits

        frac_norm = mant_post[0:FW]  # drop hidden 1 → FW bits
        return mant_post, frac_norm, msb_high, carry

    def _compute_exponent(self, eA: Expr, eB: Expr, msb_high: Expr, carry: Expr, BIAS: int, MAX_FINITE_E: int) -> Tuple[Expr, Expr, Expr]:
        EW = self.EW
        exp_sum = eA + eB  # width EW+1
        inc1 = mux(msb_high, 1, 0)
        inc2 = mux(carry, 1, 0)

        exp_sum_inc = (exp_sum + inc1) + inc2  # EW+2 bits
        bias_const = BIAS
        maxfinite_plus_bias = BIAS + MAX_FINITE_E

        # Underflow if exp_sum_inc <= BIAS (no subnormal output → zero)
        underflow = exp_sum_inc <= bias_const
        # Overflow if exp_sum_inc > BIAS + MAX_FINITE_E
        overflow = exp_sum_inc > maxfinite_plus_bias

        # Normal exponent field (EW bits): exp = exp_sum_inc - BIAS
        exp_unbiased = exp_sum_inc - bias_const  # width EW+2
        exp_norm = exp_unbiased[0:EW]  # truncate to EW
        return exp_norm, underflow, overflow

    def _pack_result(self, sY: Expr, exp_norm: Expr, frac_norm: Expr, is_nan_in: Expr, is_inf_in: Expr, is_zero_in: Expr) -> Tuple[Expr, Expr, Expr]:
        FW = self.FW
        EW = self.EW
        all1_E = (1 << EW) - 1
        qnan_payload = (1 << (FW - 1)) if FW > 0 else 1  # canonical quiet NaN

        # Priority: NaN > (Inf or overflow) > (Zero or underflow) > Normal
        is_nan = is_nan_in
        is_inf = (~is_nan) & is_inf_in
        is_zero = (~is_nan) & (~is_inf) & is_zero_in

        exp_field = mux(is_nan | is_inf, all1_E, mux(is_zero, 0, exp_norm))
        frac_field = mux(is_nan, qnan_payload, mux(is_inf | is_zero, 0, frac_norm))
        sign_field = mux(is_nan, 0, sY)  # sign is don't-care for NaN; choose 0
        return sign_field, exp_field, frac_field

    @dataclass
    class IO:
        a: Signal  # input
        b: Signal  # input
        y: Signal  # output

    def __init__(self, EW: int, FW: int, mult_cfg: Optional[MultiplierConfig] = None) -> None:
        self.EW = EW
        self.FW = FW
        self.W = 1 + EW + FW
        self.mult_cfg = mult_cfg

        self.io = self.IO(
            a=Signal(name="a", typ=UInt(self.W), kind="input"),
            b=Signal(name="b", typ=UInt(self.W), kind="input"),
            y=Signal(name="y", typ=UInt(self.W), kind="output"),
        )

        self.elaborate()

    def elaborate(self) -> None:
        EW = self.EW
        FW = self.FW
        W = self.W
        BIAS = (1 << (EW - 1)) - 1  # e.g. 15 for float16
        MAX_E = (1 << EW) - 1  # all-ones exponent
        MAX_FINITE_E = MAX_E - 1  # maximum finite exponent value

        a = self.io.a
        b = self.io.b
        y = self.io.y

        # ------------------
        # Field extraction
        # ------------------
        sA, sB, eA, eB, fA, fB = self._extract_fields(a, b)

        # ------------------
        # Classifiers
        # ------------------
        exp_all_ones = MAX_E
        cls = self._classify_operands(eA, eB, fA, fB, exp_all_ones)
        is_eA_zero = cls["is_eA_zero"]
        is_eB_zero = cls["is_eB_zero"]
        is_zeroA = cls["is_zeroA"]
        is_zeroB = cls["is_zeroB"]
        is_nan_in = cls["is_nan_in"]
        is_inf_in = cls["is_inf_in"]
        is_zero_in = cls["is_zero_in"]

        # ------------------
        # Sign
        # ------------------
        sY = sA ^ sB

        # ------------------
        # Effective mantissas with hidden '1' (mask out when exp==0 → treat as 0)
        # ------------------
        mA_eff, mB_eff = self._effective_mantissas(fA, fB, is_eA_zero, is_eB_zero)

        # ------------------
        # Multiply mantissas (unsigned product)
        # ------------------
        prod = build_multiplier(mA_eff, mB_eff, self.mult_cfg) if self.mult_cfg is not None else mA_eff * mB_eff  # width 2 + 2*FW
        PROD_W = 2 + 2 * FW

        # Multiply two (1+FW)‑bit unsigned integers:
        # Product width PROD_W = 2 + 2*FW
        # Since each is in [1.0, 2.0) (or 0), product is in [1.0, 4.0) (or 0)
        # Therefore the product’s top two bits determine normalization:
        # If product ≥ 2.0 → MSB=1: we right‑shift by 1 and add +1 to exponent
        # Else product in [1.0, 2.0) → use as is

        # Normalization: product ∈ [1.000..., 3.111...] range in fixed-point.
        # If MSB (bit PROD_W-1) is 1 (≥2.0), shift right by 1 and increment exponent.
        mant_post, frac_norm, msb_high, carry = self._normalize_and_round(prod)

        # ------------------
        # Exponent path (unsigned math)
        # ------------------
        exp_norm, underflow, overflow = self._compute_exponent(eA, eB, msb_high, carry, BIAS, MAX_FINITE_E)

        # ------------------
        # Special-case selection & packing
        # ------------------
        sign_field, exp_field, frac_field = self._pack_result(sY, exp_norm, frac_norm, is_nan_in, is_inf_in | overflow, is_zero_in | underflow)

        y <<= cat(frac_field, exp_field, sign_field)


def build_fp_mul(name: str, EW: int, FW: int) -> Module:
    comp = FpMul(EW, FW)
    return comp.to_module(name, with_clock=False, with_reset=False)


def build_f16_mul(name: str = "F16Mul") -> Module:
    # IEEE-754 binary16: exponent=5, fraction=10
    return build_fp_mul(name, EW=5, FW=10)


def build_bf16_mul(name: str = "BF16Mul") -> Module:
    # bfloat16: exponent=8, fraction=7
    return build_fp_mul(name, EW=8, FW=7)


# sanity_fp_mul_tests.py
# Assumes:
#   from spire_hdl import Simulator
#   from spire_hdl import build_f16_mul, build_bf16_mul
# Or adapt the imports to your paths (e.g., from spire_hdl import ...)


def half_to_float(h: int) -> float:
    s = (h >> 15) & 1
    e = (h >> 10) & 0x1F
    f = h & 0x3FF
    bias = 15
    if e == 0:
        if f == 0:
            return -0.0 if s else 0.0
        return ((-1.0) ** s) * (2 ** (1 - bias)) * (f / (1 << 10))
    if e == 0x1F:
        if f == 0:
            return float("-inf") if s else float("inf")
        return float("nan")
    return ((-1.0) ** s) * (2 ** (e - bias)) * (1.0 + f / (1 << 10))


def bf16_to_float(b: int) -> float:
    s = (b >> 15) & 1
    e = (b >> 7) & 0xFF
    f = b & 0x7F
    bias = 127
    if e == 0:
        if f == 0:
            return -0.0 if s else 0.0
        return ((-1.0) ** s) * (2 ** (1 - bias)) * (f / (1 << 7))
    if e == 0xFF:
        if f == 0:
            return float("-inf") if s else float("inf")
        return float("nan")
    return ((-1.0) ** s) * (2 ** (e - bias)) * (1.0 + f / (1 << 7))

VecLocal = Tuple[str, int, int, int]
VecGeneric = Tuple[str, Dict[str, int], Dict[str, int]]
def testvectors_aby_to_dict(vectors: List[VecLocal]) -> List[VecGeneric]:
    out: List[VecGeneric] = []
    for name, a_hex, b_hex, exp_hex in vectors:
        out.append((name, {"a": a_hex, "b": b_hex}, {"y": exp_hex}))
    return out

def run_vectors_aby(
    mod: Module,
    vectors: List[Tuple[str, int, int, int]],
    *,
    label: str = "",
    decoder: Optional[Callable[[int], float]] = None,
    use_signed: bool = False,
    raise_on_fail: bool = False,     # local used to never raise
    print_on_pass: bool = True,      # local used to print everything
    with_clk: bool = False,
) -> int:   
    
    sim = Simulator(mod)
    print(f"\n== {label} ==")

    generic = testvectors_aby_to_dict(vectors)
    
    # This prints PASS/FAIL, but note: it won't print a/b like your old local version.
    fails = run_vectors_on_simulator(
        sim,
        generic,
        decoder=decoder,
        use_signed=use_signed,
        raise_on_fail=raise_on_fail,
        print_on_pass=print_on_pass,
        with_clk=with_clk,
        test_name=label or None,
    )
    
    return fails==0

