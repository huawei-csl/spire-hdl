from collections import defaultdict
from typing import DefaultDict, Dict, List

from sprouthdl.helpers import get_yosys_transistor_count, run_vectors
from sprouthdl.arithmetic.int_multipliers.multipliers.multiplier_stage_core import CompressorTreeAccumulator, FinalStageAdderBase, MultiplierTestVectorsInt, PartialProductAccumulatorBase, PartialProductGeneratorBase, RippleCarryFinalAdder, StageBasedMultiplierBasic, StageBasedMultiplierIO
from sprouthdl.sprouthdl import Bool, Concat, Const, Expr, SInt, Signal, UInt
from sprouthdl.sprouthdl_module import Module


class BaughWooleyPartialProductGenerator(PartialProductGeneratorBase):
    supported_signatures = ((True, True),)

    def __init__(self, config, upper_correction: bool = True) -> None:
        super().__init__(config)
        self.upper_correction = upper_correction

    def generate_columns(
        self, io: StageBasedMultiplierIO
    ) -> DefaultDict[int, List[Expr]]:
        cols: DefaultDict[int, List[Expr]] = defaultdict(list)

        a = io.a
        b = io.b
        wa = a.typ.width
        wb = b.typ.width
        out_bits = io.y.typ.width

        # ordinary partial products
        for i in range(wa - 1):
            for j in range(wb - 1):
                weight = i + j
                if weight >= out_bits:
                    continue
                cols[weight].append(a[i] & b[j])

        # sign row (i = wa-1)
        i = wa - 1
        for j in range(wb - 1):
            cols[i + j].append(~(a[i] & b[j]))

        # sign column (j = wb-1
        j = wb - 1
        for i in range(wa - 1):
            cols[i + j].append(~(a[i] & b[j]))

        # sign corner (i = wa-1, j = wb-1)
        cols[wa - 1 + wb - 1].append(a[wa - 1] & b[wb - 1])

        # correction bit upper
        if self.upper_correction:
            if out_bits == wa + wb: # default case
                cols[wa - 1 + wb - 1 + 1].append(Const(True, Bool()))
            else: # more general case where more output bits are present
                for i in range(wa - 1 + wb - 1 + 1, out_bits):
                    cols[i].append(Const(True, Bool()))

        # correction bits lower
        if wa == wb:
            cols[wa].append(Const(True, Bool()))
        else: # asymmetric case: correction = 2^(wa-1) + 2^(wb-1)
            cols[wa - 1].append(Const(True, Bool()))
            cols[wb - 1].append(Const(True, Bool()))

        total_bits = sum(len(v) for v in cols.values())

        return cols


def main() -> None:
    n_bits = 4
    signed = True

    mult = StageBasedMultiplierBasic(
        a_w=n_bits,
        b_w=n_bits,
        signed_a=signed,
        signed_b=signed,
        optim_type="area",
        ppg_cls=BaughWooleyPartialProductGenerator,
        ppa_cls=CompressorTreeAccumulator,
        fsa_cls=RippleCarryFinalAdder,
    )
    module = mult.to_module(f"Mul{n_bits}")

    transistor_count = get_yosys_transistor_count(module, n_iter_optimizations=10, deepsyn=False)
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
