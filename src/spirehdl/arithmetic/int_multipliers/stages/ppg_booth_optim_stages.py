from collections import defaultdict
from typing import DefaultDict, List

from spirehdl.arithmetic.int_multipliers.eval.testvector_generation import to_encoding, MultiplierTestVectors
from spirehdl.helpers import get_yosys_transistor_count, run_vectors
from spirehdl.arithmetic.int_multipliers.multipliers.multiplier_stage_core import CompressorTreeAccumulator, FinalStageAdderBase, PartialProductAccumulatorBase, PartialProductGeneratorBase, RippleCarryFinalAdder, StageBasedMultiplierBasic, StageBasedMultiplierIO
from spirehdl.spirehdl import Bool, Const, Expr
from spirehdl.spirehdl_module import Module

class BoothOptimizedPartialProductGenerator(PartialProductGeneratorBase):
    supported_signatures = (
        (False, False),
        (True, True),
    )

    def generate_columns(
        self, io: StageBasedMultiplierIO
    ) -> DefaultDict[int, List[Expr]]:
        cols: DefaultDict[int, List[Expr]] = defaultdict(list)

        a = io.a
        b = io.b
        wa = a.typ.width
        wb = b.typ.width
        out_bits = io.y.typ.width
        a_signed = a.typ.signed
        b_signed = b.typ.signed

        mag1 = [a[i] for i in range(wa)] + [a[wa - 1] if a_signed else Const(False, Bool())]
        a_ext = mag1
        a2_ext = [Const(False, Bool())] + a_ext

        # Helper function to get bits with sign extension
        def get_with_se(bits: List[Expr], idx: int) -> Expr:
            if idx < len(bits):
                return bits[idx]
            return bits[-1]

        # Helper function to get bits of b with sign extension with proper out-of-range behavior
        def bbit(k: int) -> Expr:
            if 0 <= k < wb:
                return b[k]
            if k < 0:
                return Const(False, Bool())
            return b[wb - 1] if b_signed else Const(False, Bool())

        extra_groups = 0 if b_signed else 1
        n_groups = (wb + 1 + extra_groups) // 2
        for i in range(n_groups):
            x = bbit(2 * i - 1)
            y = bbit(2 * i)
            z = bbit(2 * i + 1)

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

            use1 = x ^ y
            use2 = pos2 | neg2
            neg = z

            base_w = 2 * i
            max_len = len(a2_ext) + 2
            extend_bit: Expr = Const(False, Bool())
            for t in range(max_len):
                if t < len(a_ext):
                    mag = (get_with_se(a_ext, t) & use1) | (get_with_se(a2_ext, t) & use2)
                    emit_bit = mag ^ neg
                    extend_bit = ~emit_bit
                elif t == len(a_ext):
                    emit_bit = extend_bit if a_signed else ~neg # signed: extend, unsigned: 1 if neg, 0 if pos
                elif t == len(a_ext) + 1:
                    emit_bit = Const(True, Bool())  # fixed 1 correction at the block’s MSB together with ref 1
                else:
                    emit_bit = None

                if emit_bit is None:
                    continue

                weight = base_w + t
                if weight < out_bits:
                    cols[weight].append(emit_bit)

            if base_w < out_bits:
                # two's-complement +1 correction at the block’s LSB when negative (ref 1)
                cols[base_w].append(neg) 

            if i == 0:
                correction_col = len(mag1)
                if correction_col < out_bits:
                    cols[correction_col].append(Const(True, Bool()))

        total_bits = sum(len(v) for v in cols.values())

        return cols


def main() -> None:
    n_bits = 16
    signed = False

    mult = StageBasedMultiplierBasic(
        a_w=n_bits,
        b_w=n_bits,
        signed_a=signed,
        signed_b=signed,
        optim_type="area",
        ppg_cls=BoothOptimizedPartialProductGenerator,
        ppa_cls=CompressorTreeAccumulator,
        fsa_cls=RippleCarryFinalAdder,
    )
    module = mult.to_module(f"Mul{n_bits}")
    
    transistor_count = get_yosys_transistor_count(module, n_iter_optimizations=10)
    print(f"Yosys-reported transistor count: {transistor_count}")

    vecs = MultiplierTestVectors(
        a_w=n_bits,
        b_w=n_bits,
        y_w=2 * n_bits,
        num_vectors=16,
        tb_sigma=None,
        a_encoding=to_encoding(signed),
        b_encoding=to_encoding(signed),
        y_encoding=to_encoding(signed),
    ).generate()

    run_vectors(module, vecs, print_on_pass=True)

if __name__ == "__main__":
    main()
