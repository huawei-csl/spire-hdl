from collections import defaultdict
from typing import DefaultDict, List

from spire.helpers import get_yosys_transistor_count, run_vectors
from spire.arithmetic.int_multipliers.multipliers.multiplier_stage_core import CompressorTreeAccumulator, FinalStageAdderBase, MultiplierTestVectorsInt, PartialProductAccumulatorBase, PartialProductGeneratorBase, RippleCarryFinalAdder, StageBasedMultiplierBasic, StageBasedMultiplierIO
from spire.expr import Concat, Expr
from spire.component import Module


# AND partial product generator (schoolbook method)

class AndPartialProductGenerator(PartialProductGeneratorBase):
    supported_signatures = (
        (False, False),
        (True, True),
        (True, False),
        (False, True),
    )

    def generate_columns(
        self, io: StageBasedMultiplierIO
    ) -> DefaultDict[int, List[Expr]]:
        cols: DefaultDict[int, List[Expr]] = defaultdict(list)

        a = io.a
        b = io.b
        a_vec: Expr = a
        b_vec: Expr = b

        out_bits = self.config.out_width #likley the same as io.y.typ.width

        # Signed operands are realised by sign-extending each operand and doing
        # an unsigned schoolbook multiply: the low ``out_bits`` of the product
        # then equal the true signed product mod 2**out_bits (which is exact, as
        # the signed result fits in out_bits). The sign run must reach every
        # output column, so extend to ``out_bits`` — not the historical ``2*width``.
        # For a standalone multiplier ``out_bits == 2*width`` so this is a no-op;
        # it only matters when the consumer asks for a wider result (e.g. a fused
        # MAC whose c-term makes out_bits = 2*n + 1), where stopping at 2*width
        # left the top column (the result's sign bit) uncorrected.
        if a.typ.signed:
            sign_bit = a[a.typ.width - 1]
            n_ext = max(a.typ.width, out_bits - a.typ.width)
            a_vec = Concat([a] + [sign_bit] * n_ext)
        if b.typ.signed:
            sign_bit = b[b.typ.width - 1]
            n_ext = max(b.typ.width, out_bits - b.typ.width)
            b_vec = Concat([b] + [sign_bit] * n_ext)

        for i in range(a_vec.typ.width):
            for j in range(b_vec.typ.width):
                weight = i + j
                if weight >= out_bits:
                    continue
                cols[weight].append(a_vec[i] & b_vec[j])

        total_bits = sum(len(v) for v in cols.values())

        return cols

def main() -> None:
    n_bits = 16
    signed = True
    
    mult = StageBasedMultiplierBasic(
        a_w=n_bits,
        b_w=n_bits,
        signed_a=signed,
        signed_b=signed,
        optim_type="area",
        ppg_cls=AndPartialProductGenerator,
        ppa_cls=CompressorTreeAccumulator,
        fsa_cls=RippleCarryFinalAdder,
    )

    module = mult.to_module(f"Mul{n_bits}")
    
    transistor_count = get_yosys_transistor_count(module, n_iter_optimizations=10)
    print(f"Yosys-reported transistor count: {transistor_count}")

    specs, vecs, decoder = MultiplierTestVectorsInt(
        a_w=n_bits,
        b_w=n_bits,
        num_vectors=16,
        tb_sigma=None,
        signed_a=signed,
        signed_b=signed,
    ).generate()
    _ = specs
    run_vectors(module, vecs, decoder=decoder, use_signed=True, print_on_pass=True)


if __name__ == "__main__":
    main()
