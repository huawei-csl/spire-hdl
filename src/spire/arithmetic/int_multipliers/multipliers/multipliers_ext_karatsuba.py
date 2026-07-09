from typing import Callable, ClassVar, DefaultDict, Dict, Iterable, List, Literal, Optional, Tuple, Type


from spire.arithmetic.int_multipliers.multipliers.multiplier_stage_core import RippleCarryFinalAdder, StageBasedMultiplierIO
from spire.arithmetic.int_multipliers.multipliers.multipliers_ext import StageBasedMultiplierBase
from spire.arithmetic.int_multipliers.stages.ppa_fsa_util import OutputConfig, compressor_sum
from spire.arithmetic.int_multipliers.stages.ppa_stages import CarrySaveAccumulator, WallaceTreeAccumulator
from spire.arithmetic.int_multipliers.eval.testvector_generation import Encoding, MultiplierTestVectors, from_encoding, to_encoding

from spire.helpers import get_aig_stats, get_yosys_metrics, get_yosys_transistor_count, optimize_aag, run_vectors

from spire.expr import Bool, Concat, Const, Expr, Signal, SInt, UInt, mux, mux_if
from spire.io_record import Input, Output


from typing import List, Optional


class KaratsubaMultiplier(StageBasedMultiplierBase):

    """
    Unsigned n×n multiplier constructed recursively with a Karatsuba split and a
    configurable leaf multiplier.

    - Leaf stage:
        For small sub-problems (width <= 5 by default, or when recursion is
        limited through ``karatsuba_only_at_first_level``) the implementation
        instantiates ``multiplier_core_config`` directly.  When
        ``use_power_of_two_multipliers_only`` is set and the leaf width is odd,
        the helper ``_mul_nplus1_bit_from_n_bit_block`` fabricates the result
        out of the nearest power-of-two block.

    - Recursive stage:
        Let n = width, k = n // 2, a = a_hi * 2^k + a_lo, b = b_hi * 2^k + b_lo.

            p0 = a_lo * b_lo
            p2 = a_hi * b_hi
            p1 = (a_lo + a_hi) * (b_lo + b_hi)

            middle = p1 - p0 - p2

            y = p0 + (middle << k) + (p2 << (2k))

        All partial products are projected into 2*n bits, and shifts are modeled
        explicitly via ``Concat`` + slicing so the DSL keeps the bit width
        bounded at every step.
    """

    def __init__(
        self,
        *args,
        f_aag_lines: "Optional[Callable[..., List[str]]]" = None,
        use_compressor=False,
        use_power_of_two_multipliers_only=False,
        karatsuba_only_at_first_level=True,
        use_preoptimized_4bit_multiplier=False,
        **kwargs,
    ) -> None:
        super().__init__(*args, 
                         **kwargs)

        # Only unsigned supported for now
        assert (
            self.a_encoding == Encoding.unsigned and self.b_encoding == Encoding.unsigned
        ), "KaratsubaMultiplierFrom4BitBlocks supports only unsigned encoding"

        # For now: square multipliers only
        assert self.aw == self.bw, "Karatsuba implementation assumes aw == bw"
        assert self.aw >= 1, "Operand width must be at least 1 bit"

        # Handle f_aag_lines as class attribute or constructor argument, but not both
        if hasattr(self, "f_aag_lines"):
            if f_aag_lines is not None:
                raise ValueError("f_aag_lines class attribute exists, do not pass f_aag_lines as argument")
        else:
            self.f_aag_lines = f_aag_lines

        # I/O interface
        self.io: StageBasedMultiplierIO = StageBasedMultiplierIO(
            a=Input(UInt(self.aw)),
            b=Input(UInt(self.bw)),
            y=Output(UInt(self.aw + self.bw)),
        )

        self.use_compressor = use_compressor
        self.use_power_of_two_multipliers_only = use_power_of_two_multipliers_only
        self.karatsuba_only_at_first_level = karatsuba_only_at_first_level
        self.use_preoptimized_4bit_multiplier = use_preoptimized_4bit_multiplier

        from spire.arithmetic.int_multipliers.eval.multiplier_stage_options_demo_lib import ConfigItem, FSAOption, TwoInputAritEncodings, MultiplierOption, PPAOption, PPGOption

        if self.use_preoptimized_4bit_multiplier:

            # use optimized multiplier from 4-bit blocks
            self.multiplier_core_config =  ConfigItem(MultiplierOption.OPTIMIZED_MULTIPLIER_FROM_4BIT_BLOCKS_STRONG,
                                                    encodings=TwoInputAritEncodings.with_enc(Encoding.unsigned))
            self.use_power_of_two_multipliers_only = True  # needs to be true

        else:

            # use stage-based multiplier with specific options which give good results
            self.multiplier_core_config = ConfigItem(
                MultiplierOption.STAGE_BASED_MULTIPLIER,
                TwoInputAritEncodings.with_enc(Encoding.unsigned),
                PPGOption.BOOTH_OPTIMISED, # use the options from self
                PPAOption.CARRY_SAVE_TREE,
                FSAOption.PREFIX_RCA,
            )

        self.elaborate()

    def _mul_leaf(self, a_sig, b_sig, width):

        core = self.multiplier_core_config.multiplier_opt.value(
            a_w=width,
            b_w=width,
            a_encoding=self.multiplier_core_config.encodings.a,
            b_encoding=self.multiplier_core_config.encodings.b,
            ppg_cls=self.multiplier_core_config.ppg_opt.value if self.multiplier_core_config.ppg_opt else None,
            ppa_cls=self.multiplier_core_config.ppa_opt.value if self.multiplier_core_config.ppa_opt else None,
            fsa_cls=self.multiplier_core_config.fsa_opt.value if self.multiplier_core_config.fsa_opt else None,
            optim_type=self.optim_type,
        )

        core.io.a <<= a_sig
        core.io.b <<= b_sig

        return core.io.y

    # optional
    def _mul_nplus1_bit_from_n_bit_block(self, a5, b5, width=5):
        """
        Helper for odd widths when the leaf multiplier only supports power-of-two
        sizes.  Builds a ``width``×``width`` unsigned product by instantiating a
        ``(width-1)``×``(width-1)`` core for the low bits and adding explicit
        cross/high terms.  The returned ``Expr`` contains the lower ``2*width``
        bits of the product.
        """

        # naming is for width=5, but it works generically

        width_core = width-1

        # # Split into high / low parts
        a_lo = a5[0:width_core]  # lower width_core bits
        a_hi = a5[width_core]  # MSB of the width-bit operand
        b_lo = b5[0:width_core]
        b_hi = b5[width_core]

        p0_8 = self._mul_leaf(a_lo, b_lo, width_core)

        # Zero-extend so the (width-1)×(width-1) block sits in the low part of a 2*width-bit word
        t0 = Concat([p0_8, Const(0, UInt(2))])  # 2*width bits

        # 2) Cross terms: a_hi * b_lo and b_hi * a_lo  (each width_core bits)
        # Assuming scalar&vector broadcasting in your DSL; if not, replace with Mux.
        term_a = mux(a_hi, b_lo, Const(0, UInt(width_core)))  # b_lo & a_hi when a_hi == 1 else 0
        term_b = mux(b_hi, a_lo, Const(0, UInt(width_core)))  # a_lo & b_hi when b_hi == 1 else 0

        # Place the cross terms in the middle bits by shifting left width_core positions
        # and pad with zeros so every partial still spans exactly 2*width bits.
        t1 = Concat([Const(0, UInt(width_core)), term_a, Const(0, UInt(2))])  # 2*width bits
        t2 = Concat([Const(0, UInt(width_core)), term_b, Const(0, UInt(2))])  # 2*width bits

        # 3) High-high term: a_hi * b_hi lands near the MSB region of the 2*width-bit result
        term_hi = a_hi & b_hi  # 1 bit
        # Use zero padding to align it without exceeding the width budget.
        t3 = Concat([Const(0, UInt(2 * width_core)), term_hi, Const(0, UInt(1))])  # 2*width bits

        # 4) Final sum in a 2*width-bit ring (add may create a carry, slice back to 2*width)
        if not self.use_compressor:
            prod_11 = t0 + t1 + t2 + t3
        else:
            prod_11 = compressor_sum(
                config=OutputConfig(
                    # Local leaf width, not the top-level self.aw+self.bw — this helper runs at
                    # every recursion depth (functionally benign thanks to the slice below, but
                    # it oversized the leaf compressor/FSA at inner odd leaves).
                    out_width=2 * width,
                    optim_type=self.optim_type,
                ),
                partials=[t0, t1, t2, t3],
                ppa_cls=CarrySaveAccumulator,
                fsa_cls=RippleCarryFinalAdder,
            )

        return prod_11[0:2*width]

    # ------------------------------------------------------------------ #
    # Recursive Karatsuba construction (pure-combinational in this module)
    # ------------------------------------------------------------------ #
    def _build_karatsuba(self, a_sig, b_sig, width: int, level=0) -> Expr:
        """
        Recursively build a ``width``×``width`` unsigned multiplier with the
        Karatsuba split described in the class docstring.  Returns a value with
        exactly ``2*width`` bits.
        """
        assert width >= 1

        # Base case: use the configured leaf block (optionally padded for width)

        if width <=5 or (self.karatsuba_only_at_first_level and level == 1):
            if self.use_power_of_two_multipliers_only and width%2 == 1:
                return self._mul_nplus1_bit_from_n_bit_block(a_sig, b_sig, width)
            else:
                return self._mul_leaf(a_sig, b_sig, width)  

        level += 1  

        n = width
        total_w = 2 * n
        k = n // 2
        lo_w = k
        hi_w = n - k

        # Split operands: a = a_hi*2^k + a_lo
        a_lo = a_sig[0:lo_w]
        a_hi = a_sig[lo_w:n]
        b_lo = b_sig[0:lo_w]
        b_hi = b_sig[lo_w:n]

        # --- p0 = a_lo * b_lo  (lo_w x lo_w) ---------------------------
        p0 = self._build_karatsuba(a_lo, b_lo, lo_w, level)  # width: 2*lo_w

        # --- p2 = a_hi * b_hi  (hi_w x hi_w) ---------------------------
        p2 = self._build_karatsuba(a_hi, b_hi, hi_w, level)  # width: 2*hi_w

        # --- p1 = (a_lo + a_hi) * (b_lo + b_hi) -----------------------
        # Your DSL already widens the result by one bit to hold the carry,
        # so we don't need to manually extend the operands.
        a_sum = a_lo + a_hi  # width: sum_w = max(lo_w, hi_w) + 1
        b_sum = b_lo + b_hi
        # The DSL `+` always yields max(w)+1 bits, and for odd n the sum genuinely
        # needs hi_w+1 bits (e.g. n=7: a_lo<=7, a_hi<=15, sum<=22 > 15). Sizing the
        # sub-multiplier any narrower makes `core.io.a <<= a_sum` silently truncate
        # the carry MSB, corrupting products for all odd widths.
        sum_w = max(lo_w, hi_w) + 1

        p1 = self._build_karatsuba(a_sum, b_sum, sum_w, level)  # width: 2*sum_w

        middle = p1 - (p0 + p2)  # stays 2*n bits in your DSL

        # --- Shifts implemented via Concat + slicing -------------------
        # (x << k) mod 2^(2n) == take low 2n bits of Concat([zeros(k), x])

        # (middle << k)
        middle_shift = Concat([Const(0, UInt(k)), middle])  # width: 2*n + k

        # (p2 << 2k)
        p2_shift = Concat([Const(0, UInt(2 * k)), p2])  # width: 2*n + 2k

        # Final Karatsuba combination in 2*n-bit ring
        if not self.use_compressor:
            prod_2n = Concat([p0, p2]) + middle_shift
        else:

            # middle_shift_big_separated=[
            #     Concat([Const(0, UInt(k)), p1_2n]),
            #                        Concat([Const(0, UInt(k)), -p0_2n]),
            #                        Concat([Const(0, UInt(k)), -p2_2n]),
            #                         ]
            from spire.arithmetic.int_multipliers.eval.multiplier_stage_options_demo_lib import FSAOption, PPAOption

            prod_2n = compressor_sum(
                config=OutputConfig(
                    # Local width, not self.aw + self.bw: this runs at every recursion
                    # depth, so the top-level width would be wrong below level 1.
                    out_width=total_w,
                    optim_type=self.optim_type,
                ),
                partials=[p0, middle_shift, p2_shift],
                # partials= [p0_2n, p2_shift_big] + middle_shift_big_separated, # not used, gives larger circuits
                ppa_cls=PPAOption.CARRY_SAVE_TREE.value,
                fsa_cls=FSAOption.RIPPLE_CARRY.value,
            )

        # This function's contract is "exactly 2*width bits". The combine above is
        # wider (Concat + add carry / compressor slack); returning it un-sliced makes
        # the parent's Concat([p0, p2]) misplace p2's weight at every recursion
        # level below the top.
        return prod_2n[0:total_w]

    # ------------------------------------------------------------------ #
    # Top-level elaboration
    # ------------------------------------------------------------------ #
    def elaborate(self) -> None:
        n = self.aw
        assert n == self.bw, "Karatsuba implementation assumes aw == bw"

        prod = self._build_karatsuba(self.io.a, self.io.b, n)

        # Ensure we drive exactly 2*n bits (in case '+' returns an extra carry bit)
        self.io.y <<= prod[0 : 2 * n]


class KaratsubaMultiplierFromOptimized4BitBlocks(KaratsubaMultiplier):
    """
    Karatsuba multiplier specialized to use only 4-bit leaf multipliers.

    This class configures ``KaratsubaMultiplier`` with
    ``use_power_of_two_multipliers_only=True`` and a pre-defined
    ``multiplier_core_config`` that uses an optimized 4-bit block multiplier.
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            use_power_of_two_multipliers_only=True,
            use_preoptimized_4bit_multiplier=True,
            **kwargs,
        )


def test_multiplier() -> None:

    n_bits = 32
    signed = False

    c = KaratsubaMultiplier(
        a_w=n_bits,
        b_w=n_bits,
        a_encoding=to_encoding(signed),
        b_encoding=to_encoding(signed),
        ppg_cls=None,
        ppa_cls=None,
        fsa_cls=None,
        optim_type="area",
    )

    module = c.to_module("multiplier_ext_optimized")

    transistor_count = get_yosys_transistor_count(module, n_iter_optimizations=10)
    yosys_metrics = get_yosys_metrics(module)
    aig_gates = get_aig_stats(module)
    print(f"Yosys-reported transistor count: {transistor_count}")
    print(f"Yosys-reported metrics: {yosys_metrics}")
    print(f"AIG-reported gate count: {aig_gates}")

    # Run Simulation
    vecs = MultiplierTestVectors(
        a_w=n_bits,
        b_w=n_bits,
        y_w=2 * n_bits,
        num_vectors=16,
        tb_sigma=None,
        a_encoding=Encoding.unsigned,
        b_encoding=Encoding.unsigned,
        y_encoding=Encoding.unsigned,
    ).generate()
    run_vectors(module, vecs)


if __name__ == "__main__":
    test_multiplier()