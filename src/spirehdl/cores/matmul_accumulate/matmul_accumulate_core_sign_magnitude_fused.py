from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from math import log2
from typing import Callable, DefaultDict, Iterable, List, Literal, Optional

from spirehdl.aggregate.aggregate_array import Array
from spirehdl.arithmetic.int_multipliers.eval.multiplier_stage_options_demo_lib import FSAOption, PPAOption, PPGOption
from spirehdl.arithmetic.int_multipliers.eval.testvector_generation import Encoding, is_signed
from spirehdl.arithmetic.int_multipliers.multipliers.multiplier_stage_core import StageBasedMultiplierIO, TwoInputAritConfig

from spirehdl.cores.matmul_accumulate.matmul_accumulate_core_sign_magnitude import SignMagnitudeEncoderConfig
from spirehdl.spirehdl import Bool, Concat, Const, Expr, HDLType, SInt, Signal, UInt
from spirehdl.spirehdl_module import Component, Module
from spirehdl.arithmetic.int_arithmetic_config import (
    AdderConfig,
    MultiplierConfig as BaseMultiplierConfig,
    adder_tree,
    build_adder,
)
from spirehdl.cores.matmul_accumulate.matmul_accumulate_core import (
    MatmulAccumulateCore,
    MatmulAccumulateIO,
    MMAcDims,
    MMAcWidths,
)


@dataclass(frozen=True)
class StageConfig(TwoInputAritConfig):

    output_width: Optional[int] = None

    @property
    def out_width(self) -> int:
        if self.output_width is None:
            raise ValueError("output_width must be specified for StageConfig")
        return self.output_width

MultiplierConfig = BaseMultiplierConfig

@dataclass
class MMAcEncodedCfg:
    dims: MMAcDims
    widths: MMAcWidths
    mult_cfg: BaseMultiplierConfig
    add_cfg: AdderConfig
    encoding_cfg: SignMagnitudeEncoderConfig | None = field(default_factory=SignMagnitudeEncoderConfig)


def fused_inner_product(vec_a: Iterable[Expr], vec_b: Iterable[Expr], c_term: Expr, mult_cfg: MultiplierConfig, encoding: Encoding) -> Expr:
    a_list: List[Expr] = list(vec_a)
    b_list: List[Expr] = list(vec_b)
    if len(a_list) != len(b_list):
        raise ValueError("inner_product: length mismatch")
    if len(a_list) == 0:
        raise ValueError("inner_product: no operands provided")

    a_width = a_list[0].typ.width
    b_width = b_list[0].typ.width
    if any(sig.typ.width != a_width for sig in a_list):
        raise ValueError("inner_product: inconsistent widths in vector A")
    if any(sig.typ.width != b_width for sig in b_list):
        raise ValueError("inner_product: inconsistent widths in vector B")

    product_width = a_width + b_width
    max_product_sum = len(a_list) * ((1 << product_width) - 1)
    max_c = (1 << c_term.typ.width) - 1
    result_width = max(product_width, (max_product_sum + max_c).bit_length())

    stage_cfg = StageConfig(
        a_width=a_width,
        b_width=b_width,
        output_width=result_width,
        optim_type=mult_cfg.optim_type,
    )

    if mult_cfg.ppg_opt == PPGOption.BAUGH_WOOLEY:
        fused_upper_correction = True # True yields smaller circuit
        ppg = mult_cfg.ppg_opt.value(stage_cfg, upper_correction=not fused_upper_correction)
    else:
        fused_upper_correction = False
        ppg = mult_cfg.ppg_opt.value(stage_cfg)
    # todo: make sure ppa operates on unsigned values
    ppa = mult_cfg.ppa_opt.value(stage_cfg)
    fsa = mult_cfg.fsa_opt.value(stage_cfg)

    merged_cols_pos: DefaultDict[int, List[Expr]] = defaultdict(list)
    merged_cols_neg: DefaultDict[int, List[Expr]] = defaultdict(list)
    for idx, (a_sig, b_sig) in enumerate(zip(a_list, b_list)):
        # manual sign extension (not efficient)
        # a_sig_1 = cast(fit_width(a_sig, SInt(result_width)), UInt(result_width))
        # b_sig_1 = cast(fit_width(b_sig, SInt(result_width)), UInt(result_width))
        a_sig_1 = a_sig[:-1] # remove sign bit
        b_sig_1 = b_sig[:-1] # remove sign bit
        sign_a = a_sig[-1]
        sign_b = b_sig[-1]
        # compute sign bit of product
        sign_product = sign_a ^ sign_b
        io = StageBasedMultiplierIO(
            a=a_sig_1,
            b=b_sig_1,
            y=Signal(name=f"pp_{idx}", typ=UInt(result_width), kind="wire") # dummy, is not used
        )
        cols_tot = ppg.generate_columns(io)

        for weight, bits in cols_tot.items():
            if weight < result_width:
                bits_pos =[bit & ~sign_product for bit in bits]  # bits when product is positive
                bits_neg = [bit & sign_product for bit in bits]   # bits when product
                merged_cols_pos[weight].extend(bits_pos)
                merged_cols_neg[weight].extend(bits_neg)

    # add c term bits
    #if is_signed(encoding):
    #    c_term = s_ext(c_term, result_width) # sign-extend c_term to result_width, make sure source is SInt
    for bit_idx in range(min(result_width, c_term.typ.width-1)):
        merged_cols_pos[bit_idx].append(c_term[bit_idx] & ~c_term[-1])  # positive part of c_term
        merged_cols_neg[bit_idx].append(c_term[bit_idx] & c_term[-1])   # negative part of c_term

    reduced_cols_pos = ppa.accumulate(merged_cols_pos)
    reduced_cols_neg = ppa.accumulate(merged_cols_neg)
    filtered_cols_pos = {w: bits for w, bits in reduced_cols_pos.items() if w < result_width}
    filtered_cols_neg = {w: bits for w, bits in reduced_cols_neg.items() if w < result_width}
    result_bits_pos = fsa.resolve(filtered_cols_pos)
    result_bits_neg = fsa.resolve(filtered_cols_neg)
    result_bits = Concat(result_bits_pos) - Concat(result_bits_neg)
    return result_bits[:result_width]


class MatmulAccumulateComponent(MatmulAccumulateCore):
    """Reusable component for matrix multiply-accumulate."""

    def __init__(
        self,
        cfg: MMAcEncodedCfg,
        signed_io_type: bool = False,
    ):
        self.cfg = cfg
        self.io_hdl_type = SInt if (is_signed(self.cfg.add_cfg.encoding) and signed_io_type) else UInt

        def build_matrix(name: str, width: int, rows: int, cols: int, kind: str = "wire") -> Array:
            return Array(
                [
                    Array(
                        [
                            Signal(name=f"{name}_{i}_{j}", typ=self.io_hdl_type(width), kind=kind)
                            for j in range(cols)
                        ]
                    )
                    for i in range(rows)
                ]
            )

        self.A = build_matrix("a", self.cfg.widths.a_width, self.cfg.dims.dim_m, self.cfg.dims.dim_k, kind="input")
        self.B = build_matrix("b", self.cfg.widths.b_width, self.cfg.dims.dim_k, self.cfg.dims.dim_n, kind="input")
        self.C = build_matrix("c", self.cfg.widths.c_width, self.cfg.dims.dim_m, self.cfg.dims.dim_n, kind="input")

        self.elaborate()

        self.io = MatmulAccumulateIO(A=self.A, B=self.B, C=self.C, Y=self.Y)
        
    
    def _encode_matrix(self, matrix: Array) -> Array:
        enc_cfg = self.cfg.encoding_cfg
        if enc_cfg is None or enc_cfg.encoder_cls is None:
            return matrix
    
        encoded_rows = []
        rows = len(matrix)
        cols = len(matrix[0])
        for i in range(rows):
            encoded_row = []
            for j in range(cols):
                encoder = enc_cfg.encoder_cls(
                    width=matrix[i, j].typ.width,
                    clip_most_negative=enc_cfg.encoder_clip_most_negative,
                ).make_internal()
                encoder.io.i <<= matrix[i, j]
                encoded_row.append(encoder.io.o)
            encoded_rows.append(Array(encoded_row))
    
        return Array(encoded_rows)

    def elaborate(self):
        
        encoded_A = self._encode_matrix(self.A)
        encoded_B = self._encode_matrix(self.B)
        encoded_C = self._encode_matrix(self.C)
        
        rows = []
        for i in range(self.cfg.dims.dim_m):
            row = []
            a_row = encoded_A[i, :]
            for j in range(self.cfg.dims.dim_n):
                b_col = encoded_B[:, j]
                dot = fused_inner_product(a_row, b_col, encoded_C[i, j], self.cfg.mult_cfg, self.cfg.encoding_cfg)
                y_sig = Signal(name=f"y_{i}_{j}", typ=self.io_hdl_type(dot.typ.width), kind="output")
                y_sig <<= dot
                row.append(y_sig)
            rows.append(Array(row))
        self.Y = Array(rows)


@dataclass
class MatmulAccumulateBuildOut:
    component: MatmulAccumulateComponent
    module: Module
    A: Array
    B: Array
    C: Array
    Y: Array
