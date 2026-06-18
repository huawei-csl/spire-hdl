from collections import defaultdict
from typing import DefaultDict, List

from spire.helpers import get_yosys_transistor_count, run_vectors
from spire.arithmetic.int_multipliers.multipliers.multiplier_stage_core import CompressorTreeAccumulator, FinalStageAdderBase, MultiplierTestVectorsInt, PartialProductAccumulatorBase, PartialProductGeneratorBase, RippleCarryFinalAdder, StageBasedMultiplierBasic, StageBasedMultiplierIO
from spire.expr import Bool, Const, Expr, SInt, fit_type
from spire.component import Module

class BoothOptimizedPartialProductGenerator(PartialProductGeneratorBase):
    supported_signatures = (
        (False, False),
    )

    def generate_columns(
        self, io: StageBasedMultiplierIO
    ) -> DefaultDict[int, List[Expr]]:
        cols: DefaultDict[int, List[Expr]] = defaultdict(list)

        use_diff_placement = False
        use_precompute = False
        use_precompute_v2 = False
        if not use_precompute:
            use_precompute_v2 = False

        a = io.a
        b = io.b
        wa = a.typ.width
        wb = b.typ.width
        out_bits = io.y.typ.width
        a_signed = a.typ.signed
        b_signed = b.typ.signed

        mag1 = [a[i] for i in range(wa)] + [Const(False, Bool())]
        mag2 = [Const(False, Bool())] + mag1[:-1]
        mag1_inv = [~bit for bit in mag1]
        mag2_inv = [~bit for bit in mag2]

        mag1_neg = mag1_inv
        mag2_neg = mag2_inv

        if use_precompute_v2:
            mag1_invs = -fit_type(a, SInt(wa + 1))
            mag2_invs = mag1_invs << 1
            mag1_neg = [mag1_invs[i] for i in range(wa + 1)]
            mag2_neg = [mag2_invs[i] for i in range(wa + 1)]

        a_ext = [a[i] for i in range(wa)] + [a[wa - 1] if a_signed else Const(False, Bool())]
        a2_ext = [Const(False, Bool())] + a_ext

        def bbit(k: int) -> Expr:
            if 0 <= k < wb:
                return b[k]
            return b[wb - 1] if b_signed else Const(False, Bool())

        n_groups = (wb + 2) // 2
        for group in range(n_groups):
            x = bbit(2 * group - 1)
            y = bbit(2 * group)
            z = bbit(2 * group + 1)

            nz, ny, nx = ~z, ~y, ~x
            eq000 = nz & ny & nx
            eq001 = nz & ny & x
            eq010 = nz & y & nx
            eq011 = nz & y & x
            eq100 = z & ny & nx
            eq101 = z & ny & x
            eq110 = z & y & nx
            eq111 = z & y & x

            pos1 = eq001 | eq010
            pos2 = eq011
            neg1 = eq101 | eq110
            neg2 = eq100
            neg = neg1 | neg2
            use1 = pos1 | neg1
            use2 = pos2 | neg2

            if not use_precompute:
                use1 = x ^ y
                neg = z

            base_w = 2 * group
            max_len = max(len(a_ext), len(a2_ext)) * 2
            for t in range(max_len):
                if t < wa + 1:
                    if use_precompute:
                        a_pos1 = mag1[t]
                        a_pos2 = mag2[t]
                        a_neg1 = mag1_neg[t]
                        a_neg2 = mag2_neg[t]

                        emit_bit = (
                            (a_pos1 & pos1)
                            | (a_pos2 & pos2)
                            | (a_neg1 & neg1)
                            | (a_neg2 & neg2)
                        )
                    else:
                        mag = (a_ext[t] & use1) | (a2_ext[t] & use2)
                        emit_bit = mag ^ neg
                elif t == wa + 1:
                    if group == 0 and use_diff_placement:
                        emit_bit = neg
                    else:
                        emit_bit = ~neg
                elif t == wa + 2:
                    if group == 0 and use_diff_placement:
                        emit_bit = neg
                    else:
                        emit_bit = Const(True, Bool())
                else:
                    emit_bit = None

                if emit_bit is None:
                    continue

                weight = base_w + t
                if weight < out_bits:
                    cols[weight].append(emit_bit)

            if group == 0 and not use_diff_placement:
                correction_col = len(mag1)
                if correction_col < out_bits:
                    cols[correction_col].append(Const(True, Bool()))

            if not use_precompute_v2:
                if base_w < out_bits:
                    cols[base_w].append(neg)

        total_bits = sum(len(v) for v in cols.values())

        return cols


class ConfiguredMultiplier(StageBasedMultiplierBasic):
    def __init__(
        self,
        a_w: int,
        b_w: int,
        *,
        signed_a: bool = False,
        signed_b: bool = False,
        optim_type: str = "area",
        ppg_cls: type[PartialProductGeneratorBase] = BoothOptimizedPartialProductGenerator,
        ppa_cls: type[PartialProductAccumulatorBase] = CompressorTreeAccumulator,
        fsa_cls: type[FinalStageAdderBase] = RippleCarryFinalAdder,
    ) -> None:
        super().__init__(
            a_w,
            b_w,
            signed_a=signed_a,
            signed_b=signed_b,
            optim_type=optim_type,
            ppg_cls=ppg_cls,
            ppa_cls=ppa_cls,
            fsa_cls=fsa_cls,
        )

    def elaborate(self) -> None:
        super().elaborate()
        cfg = self.config
        print(
            f"MultiplierCompressorTree (Booth optimised): {cfg.a_width}x{cfg.b_width} -> {cfg.out_width} bits"
        )


def gen_spire_module(mult: ConfiguredMultiplier) -> Module:
    return mult.to_module(f"Mul{mult.config.a_width}_ct_booth_opt")


def main() -> None:
    n_bits = 4
    num_vectors = 10
    signed = False

    mult = ConfiguredMultiplier(
        a_w=n_bits,
        b_w=n_bits,
        signed_a=signed,
        signed_b=signed,
        optim_type="area",
    )

    module = gen_spire_module(mult)
    transistor_count = get_yosys_transistor_count(module, n_iter_optimizations=10)
    print(f"Yosys-reported transistor count: {transistor_count}")

    specs, vecs, decoder = MultiplierTestVectorsInt(
        a_w=n_bits,
        b_w=n_bits,
        num_vectors=num_vectors,
        tb_sigma=None,
        signed_a=signed,
        signed_b=signed,
    ).generate()
    _ = specs
    run_vectors(module, vecs, decoder=decoder, use_signed=True, print_on_pass=True)


if __name__ == "__main__":
    main()