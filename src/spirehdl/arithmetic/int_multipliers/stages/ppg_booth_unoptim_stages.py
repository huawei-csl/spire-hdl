from collections import defaultdict
from typing import DefaultDict, List

from spirehdl.arithmetic.int_multipliers.eval.testvector_generation import to_encoding, MultiplierTestVectors
from spirehdl.helpers import get_yosys_transistor_count, run_vectors
from spirehdl.arithmetic.int_multipliers.multipliers.multiplier_stage_core import CompressorTreeAccumulator, FinalStageAdderBase, PartialProductAccumulatorBase, PartialProductGeneratorBase, RippleCarryFinalAdder, StageBasedMultiplierBasic, StageBasedMultiplierIO
from spirehdl.spirehdl import Bool, Const, Expr
from spirehdl.spirehdl_module import Module


class BoothUnoptimizedPartialProductGenerator(PartialProductGeneratorBase):
    supported_signatures = (
        (False, False),
    )

    def generate_columns(
        self, io: StageBasedMultiplierIO
    ) -> DefaultDict[int, List[Expr]]:

        cols: DefaultDict[int, List[Expr]] = defaultdict(list)  # weight -> signal node indices

        a = io.a
        b = io.b
        wa = a.typ.width
        wb = b.typ.width
        out_bits = io.y.typ.width
        b_signed = b.typ.signed
        a_signed = a.typ.signed

        # --- helper to get multiplier bits with proper out-of-range behavior
        def bbit(k: int) -> Expr:
            if 0 <= k < wb:
                return b[k]
            # beyond MSB uses sign bit if signed, else 0
            return b[wb - 1] if b_signed else Const(False, Bool())

        # ---------- Precompute selectable versions of a ----------
        # |a| with 1-bit sign extension (for left-shift headroom)
        mag1: List[Expr] = [a[i] for i in range(wa)] + [a[wa - 1] if a_signed else Const(False, Bool())]  # len = wa+1
        # |2a| = mag1 << 1
        mag2: List[Expr] = [Const(False, Bool())] + mag1  # len = wa+2

        # Bitwise inverses for negative terms (two's complement done via +1 later)
        mag1_inv: List[Expr] = [~bit for bit in mag1]
        mag2_inv: List[Expr] = [~bit for bit in mag2]

        # Utility: sign-extend a vector "v" on reads beyond its length
        def get_se(v: List[Expr], t: int) -> Expr:
            return v[t] if t < len(v) else v[-1]  # extend with last (sign) bit

        # multiplicand extended by one sign bit for ±2a (shift-left) headroom
        a_ext: List[Expr] = [a[i] for i in range(wa)] + [a[wa - 1] if a_signed else Const(False, Bool())]  # len = wa+1
        a2_ext: List[Expr] = [Const(False, Bool())] + a_ext  # == (a_ext << 1), len = wa+2

        # Radix-4 Booth: one term per 2 multiplier bits
        n_groups = (wb+2) //2 #np.ceil( (wb + 1) / 2).astype(int)  # ceil(wb/2)
        for i in range(n_groups):
            x = bbit(2 * i - 1)  # low
            y = bbit(2 * i)      # mid
            z = bbit(2 * i + 1)  # high

            # Decode the 3-bit Booth code (z y x):
            # 000/111 -> 0; 001/010 -> +1; 011 -> +2; 100 -> -2; 101/110 -> -1
            nz, ny, nx = ~z, ~y, ~x

            eq000 = nz & ny & nx
            eq001 = nz & ny &  x
            eq010 = nz &  y & nx
            eq011 = nz &  y &  x
            eq100 =  z & ny & nx
            eq101 =  z & ny &  x
            eq110 =  z &  y & nx
            eq111 =  z &  y &  x

            pos1 = eq001 | eq010
            pos2 = eq011
            neg1 = eq101 | eq110
            neg2 = eq100
            neg  = neg1 | neg2
            use1 = pos1 | neg1         # select |a|
            use2 = pos2 | neg2         # select |2a|
            zero = eq000 | eq111        # zero term

            sel0      = eq000 | eq111             # 0
            sel_pos1  = eq001 | eq010             # +a
            sel_pos2  = eq011                     # +2a
            sel_neg1  = eq101 | eq110             # -a
            sel_neg2  = eq100                     # -2a

            # Emit magnitude bits (a_ext or a2_ext), conditionally inverted if neg
            # Then add +1 correction at the block LSB when neg (two's complement).
            # Place bits starting at column base_w = 2*i (radix-4 shift).
            base_w = 2 * i

            # Use the longer of the two candidates to cover both |a| and |2a|.
            max_len = max(len(a_ext), len(a2_ext)) *2
            for t in range(max_len):
                # pick bit t of |a| or |2a| guarded by enables; missing bits are 0
                uses_precomute = False
                if not uses_precomute:
                    # do sign extension for out-of-range bits
                    bit_a  = a_ext[t]  if t < len(a_ext)  else a_ext[-1]
                    bit_2a = a2_ext[t] if t < len(a2_ext) else a2_ext[-1]
                    mag    = (bit_a & use1) | (bit_2a & use2)  # selected magnitude bit

                    # conditional inversion for negative terms
                    emit_bit = (mag ^ neg)  # zero term forces 0 output
                else:
                    b_pos1 = get_se(mag1, t)
                    b_pos2 = get_se(mag2, t)
                    b_neg1 = get_se(mag1_inv, t)
                    b_neg2 = get_se(mag2_inv, t)

                    emit_bit = (b_pos1 & sel_pos1) | (b_pos2 & sel_pos2) | (b_neg1 & sel_neg1) | (b_neg2 & sel_neg2)

                w = base_w + t
                if w < out_bits:      # discard columns beyond output width
                    cols[w].append(emit_bit)

            # two's-complement +1 correction at the block’s LSB when neg
            if base_w < out_bits:
                cols[base_w].append(neg)

        total_bits = sum(len(v) for v in cols.values())

        return cols


def main() -> None:
    n_bits = 32
    signed = False

    mult = StageBasedMultiplierBasic(
        a_w=n_bits,
        b_w=n_bits,
        signed_a=signed,
        signed_b=signed,
        optim_type="area",
        ppg_cls=BoothUnoptimizedPartialProductGenerator,
        ppa_cls=CompressorTreeAccumulator,
        fsa_cls=RippleCarryFinalAdder,
    )
    module = mult.to_module(f"Mul{n_bits}")
    
    transistor_count = get_yosys_transistor_count(module, n_iter_optimizations=10)
    print(f"Yosys-reported transistor count: {transistor_count}")
    
    vecs = MultiplierTestVectors(
            a_w=n_bits,
            b_w=n_bits,
            y_w=2*n_bits,
            num_vectors=16,
            tb_sigma=None,
            a_encoding=to_encoding(signed),
            b_encoding=to_encoding(signed),
            y_encoding=to_encoding(signed),
        ).generate()
    
    
    run_vectors(module, vecs, print_on_pass=True)


if __name__ == "__main__":
    main()