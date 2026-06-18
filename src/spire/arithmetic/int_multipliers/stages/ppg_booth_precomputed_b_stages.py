from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import DefaultDict, List

from spire.arithmetic.int_multipliers.multipliers.multiplier_stage_core import (
    PartialProductGeneratorBase,
    StageBasedMultiplierIO,
    TwoInputAritConfig,
)
from spire.expr import Bool, Const, Expr


@dataclass
class BoothGroupDecode:
    """Precomputed Booth decode signals for a single b element, one Booth group.

    use1 = x ^ y        — select |a| (magnitude 1)
    use2 = pos2 | neg2  — select 2·|a| (magnitude 2)
    neg  = z            — sign: negate the partial product
    """
    use1: Expr
    use2: Expr
    neg:  Expr


def precompute_booth_b_decode(b_sig: Expr) -> List[BoothGroupDecode]:
    """Compute the Booth decode signals for all groups of a single b operand.

    Returns a list of BoothGroupDecode, one per Booth radix-2 group.
    These depend only on b, so they can be precomputed once and shared
    across multiple inner products that use the same b element.
    """
    wb = b_sig.typ.width
    b_signed = b_sig.typ.signed

    def bbit(k: int) -> Expr:
        if 0 <= k < wb:
            return b_sig[k]
        if k < 0:
            return Const(False, Bool())
        return b_sig[wb - 1] if b_signed else Const(False, Bool())

    extra_groups = 0 if b_signed else 1
    n_groups = (wb + 1 + extra_groups) // 2

    decode: List[BoothGroupDecode] = []
    for i in range(n_groups):
        x = bbit(2 * i - 1)
        y = bbit(2 * i)
        z = bbit(2 * i + 1)

        nz, ny, nx = ~z, ~y, ~x
        eq011 = nz & y & x
        eq100 = z & ny & nx

        pos2 = eq011
        neg2 = eq100

        use1 = x ^ y
        use2 = pos2 | neg2
        neg  = z

        decode.append(BoothGroupDecode(use1=use1, use2=use2, neg=neg))

    return decode


class BoothPrecomputedBPartialProductGenerator(PartialProductGeneratorBase):
    """Booth partial product generator that takes precomputed b-decode signals.

    Instead of deriving use1/use2/neg from b bits inside generate_columns,
    this generator's generate_columns_precomputed() method accepts a list of
    BoothGroupDecode (one per Booth group) that was precomputed from b.
    This allows the b-decode logic to be shared across multiple inner products
    that use the same b element (e.g. across rows in a matmul).

    The standard generate_columns() interface is also supported and simply
    calls precompute_booth_b_decode() internally, making this a drop-in
    replacement for BoothOptimizedPartialProductGenerator.
    """

    supported_signatures = (
        (False, False),
        (True, True),
    )

    def generate_columns(
        self, io: StageBasedMultiplierIO
    ) -> DefaultDict[int, List[Expr]]:
        decode = precompute_booth_b_decode(io.b)
        return self.generate_columns_precomputed(io.a, decode)

    def generate_columns_precomputed(
        self,
        a_sig: Expr,
        decode: List[BoothGroupDecode],
    ) -> DefaultDict[int, List[Expr]]:
        """Generate partial product columns using precomputed b-decode signals.

        Args:
            a_sig:  The a operand signal.
            decode: Precomputed Booth decode for b, one BoothGroupDecode per group.
                    Produced by precompute_booth_b_decode(b_sig).
        """
        cols: DefaultDict[int, List[Expr]] = defaultdict(list)

        wa = a_sig.typ.width
        a_signed = a_sig.typ.signed
        out_bits = self.config.out_width

        mag1 = [a_sig[i] for i in range(wa)] + [a_sig[wa - 1] if a_signed else Const(False, Bool())]
        a_ext = mag1
        a2_ext = [Const(False, Bool())] + a_ext

        def get_with_se(bits: List[Expr], idx: int) -> Expr:
            if idx < len(bits):
                return bits[idx]
            return bits[-1]

        for i, grp in enumerate(decode):
            use1, use2, neg = grp.use1, grp.use2, grp.neg
            base_w = 2 * i
            max_len = len(a2_ext) + 2
            extend_bit: Expr = Const(False, Bool())

            for t in range(max_len):
                if t < len(a_ext):
                    mag = (get_with_se(a_ext, t) & use1) | (get_with_se(a2_ext, t) & use2)
                    emit_bit = mag ^ neg
                    extend_bit = ~emit_bit
                elif t == len(a_ext):
                    emit_bit = extend_bit if a_signed else ~neg
                elif t == len(a_ext) + 1:
                    emit_bit = Const(True, Bool())
                else:
                    emit_bit = None

                if emit_bit is None:
                    continue

                weight = base_w + t
                if weight < out_bits:
                    cols[weight].append(emit_bit)

            if base_w < out_bits:
                cols[base_w].append(neg)

            if i == 0:
                correction_col = len(mag1)
                if correction_col < out_bits:
                    cols[correction_col].append(Const(True, Bool()))

        return cols
