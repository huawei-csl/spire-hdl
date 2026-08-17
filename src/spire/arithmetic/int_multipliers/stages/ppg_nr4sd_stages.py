"""NR4SD- partial-product generator (Non-Redundant Radix-4 Signed-Digit).

See NR4SD.md (same directory) for the algorithm and references. The multiplier ``b`` is recoded into radix-4
signed digits d_j in {-2,-1,0,+1} (the MSB digit may also be +2) via a 1-bit carry recurrence; each digit
selects a partial product d_j * a placed at column 2j. The PP emission (magnitude select, conditional
negation, Booth sign-extension trick, +1 negation correction) is identical to Modified Booth — only the
*recoding* of b differs (2-bit pair + carry chain instead of Booth's overlapping 3-bit window).
"""
from collections import defaultdict
from typing import DefaultDict, List

from spire.arithmetic.int_multipliers.multipliers.multiplier_stage_core import (
    CompressorTreeAccumulator, PartialProductGeneratorBase, RippleCarryFinalAdder,
    StageBasedMultiplierBasic, StageBasedMultiplierIO,
)
from spire.expr import Bool, Const, Expr


class NR4SDPartialProductGenerator(PartialProductGeneratorBase):
    # Signed multiplier only (the NR4SD- MSB digit relies on the sign bit's negative weight). The multiplicand
    # ``a`` may be signed or unsigned — its sign is handled by the shared magnitude/sign-extension machinery.
    supported_signatures = (
        (True, True),
        (False, True),
    )
    # NR4SD- carry recurrence implementation: "ripple" (serial chain, fewest gates) or "prefix"
    # (Kogge-Stone, log depth — trades a few gates for much less depth at large widths). Both produce a
    # functionally identical multiplier. Default ripple; override per instance via ``carry_mode=`` arg.
    carry_mode = "ripple"

    def __init__(self, config, *, carry_mode=None):
        super().__init__(config)
        self.carry_mode = carry_mode if carry_mode is not None else type(self).carry_mode

    def _carries(self, b: Expr, ndig: int) -> List[Expr]:
        """Carry into each digit (c_0 = 0) from c_{j+1} = b1_j | (b0_j & c_j) over lower digits 0..ndig-2.

        ripple = serial chain. prefix = Kogge-Stone over (generate=b1, propagate=b0); c_j = G_{0:j-1}, the
        same boolean function, computed in log depth.
        """
        zero = Const(False, Bool())
        gp = [(b[2 * j + 1], b[2 * j]) for j in range(ndig - 1)]   # (generate, propagate) per lower digit
        if self.carry_mode == "ripple":
            carries, c = [zero], zero
            for g, p in gp:
                c = g | (p & c)
                carries.append(c)
            return carries
        if self.carry_mode == "prefix":
            m = len(gp)
            if m == 0:
                return [zero]
            cur, d = list(gp), 1
            while d < m:
                nxt = list(cur)
                for j in range(d, m):
                    g_hi, p_hi = cur[j]
                    g_lo, p_lo = cur[j - d]
                    nxt[j] = (g_hi | (p_hi & g_lo), p_hi & p_lo)
                cur, d = nxt, d * 2
            return [zero] + [cur[j][0] for j in range(m)]
        raise ValueError(f"unknown carry_mode {self.carry_mode!r}")

    def generate_columns(self, io: StageBasedMultiplierIO) -> DefaultDict[int, List[Expr]]:
        cols: DefaultDict[int, List[Expr]] = defaultdict(list)
        a, b = io.a, io.b
        wa, wb = a.typ.width, b.typ.width
        out_bits = io.y.typ.width
        a_signed = a.typ.signed
        if wb % 2 != 0:
            raise ValueError("NR4SD multiplier width must be even")

        # multiplicand multiples A and 2A (a sign-extended by 1, then shifted) — same as Booth
        a_ext = [a[i] for i in range(wa)] + [a[wa - 1] if a_signed else Const(False, Bool())]
        a2_ext = [Const(False, Bool())] + a_ext

        def get_se(bits: List[Expr], idx: int) -> Expr:
            return bits[idx] if idx < len(bits) else bits[-1]

        ndig = wb // 2
        carries = self._carries(b, ndig)        # carry into each digit (ripple or prefix; c_0 = 0)
        for j in range(ndig):
            b0, b1 = b[2 * j], b[2 * j + 1]
            c = carries[j]
            is_msb = (j == ndig - 1)            # MSB digit: b1 is the sign bit (negative weight)

            # --- NR4SD- recoder (equations verified against the truth tables in NR4SD.md) ---
            zero = (~b1 & ~b0 & ~c) | (b1 & b0 & c)        # d == 0
            two = (~b1 & b0 & c) | (b1 & ~b0 & ~c)         # |d| == 2
            use1 = ~zero & ~two                            # |d| == 1
            use2 = two
            neg = (b1 & ~(b0 & c)) if is_msb else (b1 ^ (b0 & c))   # d < 0

            # --- partial-product emission (identical to Modified Booth) ---
            base_w = 2 * j
            extend_bit: Expr = Const(False, Bool())
            for t in range(len(a2_ext) + 2):
                if t < len(a_ext):
                    mag = (get_se(a_ext, t) & use1) | (get_se(a2_ext, t) & use2)
                    emit_bit: Expr = mag ^ neg
                    extend_bit = ~emit_bit
                elif t == len(a_ext):
                    emit_bit = extend_bit if a_signed else ~neg
                elif t == len(a_ext) + 1:
                    emit_bit = Const(True, Bool())          # sign-extension-trick constant 1
                else:
                    continue
                weight = base_w + t
                if weight < out_bits:
                    cols[weight].append(emit_bit)

            if base_w < out_bits:
                cols[base_w].append(neg)                    # +1 two's-complement correction when negated
            if j == 0:
                corr = len(a_ext)
                if corr < out_bits:
                    cols[corr].append(Const(True, Bool()))

        return cols


def main() -> None:
    from spire.helpers import run_vectors
    from spire.arithmetic.int_multipliers.eval.testvector_generation import to_encoding, MultiplierTestVectors
    n = 8
    mult = StageBasedMultiplierBasic(
        a_w=n, b_w=n, signed_a=True, signed_b=True, optim_type="area",
        ppg_cls=NR4SDPartialProductGenerator, ppa_cls=CompressorTreeAccumulator, fsa_cls=RippleCarryFinalAdder,
    )
    module = mult.to_netlist(f"NR4SDMul{n}")
    vecs = MultiplierTestVectors(a_w=n, b_w=n, y_w=2 * n, num_vectors=64, tb_sigma=None,
                                 a_encoding=to_encoding(True), b_encoding=to_encoding(True),
                                 y_encoding=to_encoding(True)).generate()
    run_vectors(module, vecs, print_on_pass=True)


if __name__ == "__main__":
    main()
