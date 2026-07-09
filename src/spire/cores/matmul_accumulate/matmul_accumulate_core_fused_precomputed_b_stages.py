from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import log2
from typing import Callable, DefaultDict, Iterable, List, Optional

from spire.composite.array import Array
from spire.arithmetic.int_multipliers.eval.multiplier_stage_options_demo_lib import FSAOption, PPAOption, PPGOption
from spire.arithmetic.int_multipliers.eval.testvector_generation import Encoding, is_signed
from spire.arithmetic.int_multipliers.multipliers.multiplier_stage_core import OptimType, StageBasedMultiplierIO, TwoInputAritConfig
from spire.arithmetic.int_multipliers.stages.ppg_booth_precomputed_b_stages import (
    BoothGroupDecode,
    BoothPrecomputedBPartialProductGenerator,
    precompute_booth_b_decode,
)
from spire.cores.matmul_accumulate.matmul_accumulate_core import MatmulAccumulateCore, MatmulAccumulateIO, MMAcDims, MMAcWidths
from spire.expr import Bool, Concat, Const, Expr, HDLType, SInt, Signal, UInt, fit_type, fit_width, s_ext
from spire.component import Component, Netlist


@dataclass
class MultiplierConfig:
    """Configuration for stage-based partial product flow."""

    ppg_opt: PPGOption
    ppa_opt: PPAOption
    fsa_opt: FSAOption
    optim_type: OptimType = "area"


@dataclass(frozen=True)
class StageConfig(TwoInputAritConfig):

    output_width: Optional[int] = None

    @property
    def out_width(self) -> int:
        if self.output_width is None:
            raise ValueError("output_width must be specified for StageConfig")
        return self.output_width


@dataclass
class MMAcFusedCfg:
    dims: MMAcDims
    widths: MMAcWidths
    mult_cfg: MultiplierConfig
    encoding: Encoding
    share_booth_b_precompute: bool = False


def fused_inner_product(
    vec_a: Iterable[Expr],
    vec_b: Iterable[Expr],
    c_term: Expr,
    mult_cfg: MultiplierConfig,
    encoding: Encoding,
    precomputed_b_decode: Optional[List[List[BoothGroupDecode]]] = None,
) -> Expr:
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

    use_precomputed_b = (
        precomputed_b_decode is not None
        and mult_cfg.ppg_opt == PPGOption.BOOTH_OPTIMISED_PRECOMPUTED_B
    )

    if mult_cfg.ppg_opt == PPGOption.BAUGH_WOOLEY:
        fused_upper_correction = True
        ppg = mult_cfg.ppg_opt.value(stage_cfg, upper_correction=not fused_upper_correction)
    elif use_precomputed_b:
        fused_upper_correction = False
        ppg = BoothPrecomputedBPartialProductGenerator(stage_cfg)
    else:
        fused_upper_correction = False
        ppg = mult_cfg.ppg_opt.value(stage_cfg)
    ppa = mult_cfg.ppa_opt.value(stage_cfg)
    fsa = mult_cfg.fsa_opt.value(stage_cfg)

    merged_cols: DefaultDict[int, List[Expr]] = defaultdict(list)
    for idx, (a_sig, b_sig) in enumerate(zip(a_list, b_list)):
        if use_precomputed_b:
            assert precomputed_b_decode is not None
            cols = ppg.generate_columns_precomputed(a_sig, precomputed_b_decode[idx])
        else:
            io = StageBasedMultiplierIO(
                a=a_sig,
                b=b_sig,
                y=Signal(name=f"pp_{idx}", typ=UInt(result_width), kind="wire")  # dummy
            )
            cols = ppg.generate_columns(io)
        for weight, bits in cols.items():
            if weight < result_width:
                merged_cols[weight].extend(bits)

    # common upper correction
    if fused_upper_correction:
        for i in range(a_width - 1 + b_width - 1 + 1 + int(log2(len(a_list))), result_width):
            merged_cols[i].append(Const(True, Bool()))

    # add c term bits
    if is_signed(encoding):
        c_term = s_ext(c_term, result_width)
    for bit_idx in range(min(result_width, c_term.typ.width)):
        merged_cols[bit_idx].append(c_term[bit_idx])

    reduced_cols = ppa.accumulate(merged_cols)
    filtered_cols = {w: bits for w, bits in reduced_cols.items() if w < result_width}
    result_bits = fsa.resolve(filtered_cols)
    return Concat(result_bits[:result_width])


class MatmulAccumulateComponent(MatmulAccumulateCore):
    """Reusable component for fused matrix multiply-accumulate."""

    def __init__(
        self,
        cfg: MMAcFusedCfg,
        signed_io_type: bool = False,
    ):
        self.cfg = cfg
        self.io_hdl_type = SInt if (is_signed(self.cfg.encoding) and signed_io_type) else UInt

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

    def elaborate(self):
        # Optionally precompute Booth decode signals for each B[k,j] element.
        # When enabled, use1/use2/neg for each Booth group of B[k,j] is computed
        # once and shared across all dim_m inner products in the same column j,
        # rather than being duplicated for each output row i.
        precomputed_b: Optional[List[List[List[BoothGroupDecode]]]] = None
        if (
            self.cfg.share_booth_b_precompute
            and self.cfg.mult_cfg.ppg_opt == PPGOption.BOOTH_OPTIMISED_PRECOMPUTED_B
        ):
            precomputed_b = [
                [precompute_booth_b_decode(self.B[k, j]) for j in range(self.cfg.dims.dim_n)]
                for k in range(self.cfg.dims.dim_k)
            ]

        rows = []
        for i in range(self.cfg.dims.dim_m):
            row = []
            a_row = self.A[i, :]
            for j in range(self.cfg.dims.dim_n):
                b_col = self.B[:, j]
                b_decode_for_j = (
                    [precomputed_b[k][j] for k in range(self.cfg.dims.dim_k)]
                    if precomputed_b is not None else None
                )
                dot = fused_inner_product(
                    a_row, b_col, self.C[i, j],
                    self.cfg.mult_cfg, self.cfg.encoding,
                    precomputed_b_decode=b_decode_for_j,
                )
                y_sig = Signal(name=f"y_{i}_{j}", typ=self.io_hdl_type(dot.typ.width), kind="output")
                y_sig <<= dot
                row.append(y_sig)
            rows.append(Array(row))
        self.Y = Array(rows)
