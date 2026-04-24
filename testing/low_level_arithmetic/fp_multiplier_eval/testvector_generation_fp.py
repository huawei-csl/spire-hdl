from __future__ import annotations

import math
import random
from dataclasses import dataclass
from enum import Enum
from typing import Callable, List, Optional, Tuple, Union

# Encoding helpers for IEEE-like floats (arbitrary EW/FW)
from spirehdl.arithmetic.floating_point.fp_encoding import fp_encode, fp_limits
from spirehdl.arithmetic.floating_point.fp_encoding import fp_decode  # tie handling aid (IEEE only)

# HiFloat8 encode/decode will be imported lazily inside HiF8Format methods to
# avoid pulling optional heavy dependencies at import time.


# use for ieee float:  fp_encode(x: float, EW: int, FW: int, *, subnormals: bool = True) -> int
# use for hifloat:     float_to_hif8(value: float) -> int from spirehdl/floating_point/spire_hdl_hif8.py
# type (ieee, hifloat) shall be configurable; for IEEE also EW, FW and subnormals
# clamp values to the finite representable range before encoding
# sampling types supported:
# - uniform random (linear)
# - uniform random (log2 space)
# - normal distribution (linear) with std sigma
# - normal distribution (log2 space) with std sigma


class FPDist(Enum):
    UNIFORM_LINEAR = "uniform_linear"
    UNIFORM_LOG = "uniform_log"
    NORMAL_LINEAR = "normal_linear"
    NORMAL_LOG = "normal_log"


@dataclass(frozen=True)
class IEEEFormat:
    EW: int
    FW: int
    subnormals: bool = True

    @property
    def bit_width(self) -> int:
        return 1 + self.EW + self.FW

    def encode(self, x: float) -> int:
        return fp_encode(x, self.EW, self.FW, subnormals=self.subnormals)

    def limits(self) -> dict:
        return fp_limits(self.EW, self.FW)

    def min_pos(self) -> float:
        lim = self.limits()
        return lim["min_sub_pos"] if self.subnormals else lim["min_normal_pos"]

    def max_finite(self) -> float:
        return self.limits()["max_finite"]


@dataclass(frozen=True)
class HiF8Format:
    @property
    def bit_width(self) -> int:
        return 8

    def _import(self):
        # Local import to avoid optional dependencies during module import
        from spirehdl.arithmetic.floating_point.spire_hdl_hif8 import float_to_hif8, hif8_to_float
        return float_to_hif8, hif8_to_float

    def encode(self, x: float) -> int:
        float_to_hif8, _ = self._import()
        return float_to_hif8(x)

    def min_pos(self) -> float:
        # smallest positive finite value representable
        _, hif8_to_float = self._import()
        vals = [hif8_to_float(i) for i in range(256)]
        pos = [v for v in vals if math.isfinite(v) and v > 0.0]
        return min(pos) if pos else 0.0

    def max_finite(self) -> float:
        _, hif8_to_float = self._import()
        vals = [hif8_to_float(i) for i in range(256)]
        fin = [abs(v) for v in vals if math.isfinite(v)]
        return max(fin) if fin else 0.0


FPFormat = Union[IEEEFormat, HiF8Format]


def _log2(x: float) -> float:
    return math.log2(x) if x > 0.0 else float("-inf")


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(min(x, hi), lo)


class FPMultiplierTestVectors:
    def __init__(
        self,
        a_fmt: FPFormat,
        b_fmt: Optional[FPFormat] = None,
        y_fmt: Optional[FPFormat] = None,
        *,
        num_vectors: int = 64,
        tb_sigma: Optional[float] = None,
        dist: FPDist = FPDist.UNIFORM_LINEAR,
        clamp_to_finite: bool = True,
        rng: Optional[random.Random] = None,
    ) -> None:
        self.a_fmt = a_fmt
        self.b_fmt = b_fmt if b_fmt is not None else a_fmt
        self.y_fmt = y_fmt if y_fmt is not None else a_fmt
        self.num_vectors = int(num_vectors)
        self.tb_sigma = tb_sigma
        self.dist = dist
        self.clamp_to_finite = clamp_to_finite
        self.rng = rng or random.Random()

        # Derived bit widths (useful for downstream checks/logs)
        self.a_w = self.a_fmt.bit_width
        self.b_w = self.b_fmt.bit_width
        self.y_w = self.y_fmt.bit_width

        # Pre-compute ranges used for sampling/clamping
        self._a_min_pos = self.a_fmt.min_pos()
        self._b_min_pos = self.b_fmt.min_pos()
        self._y_min_pos = self.y_fmt.min_pos()
        self._a_max_fin = self.a_fmt.max_finite()
        self._b_max_fin = self.b_fmt.max_finite()
        self._y_max_fin = self.y_fmt.max_finite()

        # Overflow thresholds for IEEE; HiF8 handled by max_finite (no RNE threshold concept)
        self._y_pos_ovf = None
        self._y_neg_ovf = None
        if isinstance(self.y_fmt, IEEEFormat):
            lim = self.y_fmt.limits()
            self._y_pos_ovf = lim["pos_overflow_threshold"]
            self._y_neg_ovf = lim["neg_overflow_threshold"]

    # ---------------------- sampling helpers ----------------------
    def _uniform_linear(self, lo: float, hi: float) -> float:
        return self.rng.uniform(lo, hi)

    def _uniform_log2(self, lo_pos: float, hi_pos: float) -> float:
        # sample |x| with log2(|x|) uniform, then random sign
        if lo_pos <= 0.0:
            lo_pos = min(self._a_min_pos, self._b_min_pos, self._y_min_pos)
            if lo_pos <= 0.0:
                lo_pos = 2.0 ** -32
        z = self.rng.uniform(_log2(lo_pos), _log2(hi_pos))
        mag = 2.0 ** z
        sign = -1.0 if self.rng.random() < 0.5 else 1.0
        return sign * mag

    def _normal_linear(self, mean: float, sigma: float, lo: float, hi: float) -> float:
        x = self.rng.normalvariate(mean, sigma)
        return _clamp(x, lo, hi)

    def _normal_log2(self, mean_exp: float, sigma_exp: float, lo_pos: float, hi_pos: float) -> float:
        z = self.rng.normalvariate(mean_exp, sigma_exp)
        z = _clamp(z, _log2(lo_pos), _log2(hi_pos))
        mag = 2.0 ** z
        sign = -1.0 if self.rng.random() < 0.5 else 1.0
        return sign * mag

    def _sample_value(self, fmt: FPFormat, *, other_max: Optional[float] = None) -> float:
        """Sample a Python float according to selected distribution.
        other_max allows biasing range when sampling inputs jointly.
        """
        min_pos = fmt.min_pos()
        max_fin = fmt.max_finite()
        lo = -max_fin
        hi = max_fin

        if self.dist == FPDist.UNIFORM_LINEAR:
            x = self._uniform_linear(lo, hi)
        elif self.dist == FPDist.UNIFORM_LOG:
            x = self._uniform_log2(min_pos, max_fin)
        elif self.dist == FPDist.NORMAL_LINEAR:
            if self.tb_sigma is None:
                raise ValueError("tb_sigma must be set for NORMAL_LINEAR distribution")
            mean = 0.0
            x = self._normal_linear(mean, float(self.tb_sigma), lo, hi)
        elif self.dist == FPDist.NORMAL_LOG:
            if self.tb_sigma is None:
                raise ValueError("tb_sigma must be set for NORMAL_LOG distribution")
            mean_exp = 0.0  # center magnitude near 1.0
            x = self._normal_log2(mean_exp, float(self.tb_sigma), min_pos, max_fin)
        else:
            raise ValueError(f"Unsupported distribution: {self.dist}")

        if self.clamp_to_finite:
            x = _clamp(x, lo, hi)
        return x

    # ---------------------- public API ----------------------
    def generate(self) -> List[Tuple[str, dict, dict]]:
        vecs: List[Tuple[str, dict, dict]] = []
        for _ in range(self.num_vectors):
            a_val = self._sample_value(self.a_fmt)
            b_val = self._sample_value(self.b_fmt)

            # Compute product as Python float, then optionally clamp
            y_val = a_val * b_val

            # 2. update: on overflow encode as ±Inf (not max finite)
            # For IEEE, use exact overflow thresholds; for HiF8, use max_finite as cutoff.
            if math.isfinite(y_val):
                if isinstance(self.y_fmt, IEEEFormat) and self._y_pos_ovf is not None:
                    if y_val >= self._y_pos_ovf:
                        y_val = float("inf")
                    elif y_val <= self._y_neg_ovf:  # type: ignore[arg-type]
                        y_val = float("-inf")
                else:
                    # HiF8 or unknown: treat strictly beyond max_finite as overflow
                    if y_val > self._y_max_fin:
                        y_val = float("inf")
                    elif y_val < -self._y_max_fin:
                        y_val = float("-inf")

            # If still finite and clamping requested, clamp remaining to finite range
            if self.clamp_to_finite and math.isfinite(y_val):
                y_val = _clamp(y_val, -self._y_max_fin, self._y_max_fin)

            a_bits = self.a_fmt.encode(a_val)
            b_bits = self.b_fmt.encode(b_val)
            y_bits = self.y_fmt.encode(y_val)

            name = f"{a_val:.6g}*{b_val:.6g}"
            vecs.append((name, {"a": a_bits, "b": b_bits, "_a_val": a_val, "_b_val": b_val}, {"y": y_bits, "_y_val": y_val}))

        return vecs


# ---------------------- example usage ----------------------
def _demo() -> None:
    # Example 1: IEEE half (5,10), uniform in log2 domain
    f_ieee = IEEEFormat(EW=5, FW=10, subnormals=True)
    gen_ieee = FPMultiplierTestVectors(
        a_fmt=f_ieee,
        num_vectors=8,
        dist=FPDist.UNIFORM_LOG,
    )
    vecs_ieee = gen_ieee.generate()
    print("IEEE(5,10) example vectors (first 3):")
    for n, ins, outs in vecs_ieee[:3]:
        print(f"  {n} -> a=0x{ins['a']:04x}, b=0x{ins['b']:04x}, y=0x{outs['y']:04x}")

    # Example 2: HiFloat8, normal in log2 domain (sigma=2 exponents)
    try:
        f_hif8 = HiF8Format()
        gen_hif8 = FPMultiplierTestVectors(
            a_fmt=f_hif8,
            num_vectors=8,
            dist=FPDist.NORMAL_LOG,
            tb_sigma=2.0,
        )
        vecs_hif8 = gen_hif8.generate()
        print("HiF8 example vectors (first 3):")
        for n, ins, outs in vecs_hif8[:3]:
            print(f"  {n} -> a=0x{ins['a']:02x}, b=0x{ins['b']:02x}, y=0x{outs['y']:02x}")
    except ModuleNotFoundError as e:
        print("HiF8 example skipped (dependency missing):", e)


if __name__ == "__main__":
    _demo()