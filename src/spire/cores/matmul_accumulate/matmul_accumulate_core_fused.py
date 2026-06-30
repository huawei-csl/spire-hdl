from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import log2
from typing import Callable, DefaultDict, Iterable, List, Optional

from spire.composite.array import Array
from spire.arithmetic.int_multipliers.eval.multiplier_stage_options_demo_lib import FSAOption, PPAOption, PPGOption
from spire.arithmetic.int_multipliers.eval.testvector_generation import Encoding, is_signed
from spire.arithmetic.int_multipliers.multipliers.multiplier_stage_core import OptimType, SelectionMode, SplitMode, StageBasedMultiplierIO, TwoInputAritConfig
from spire.cores.matmul_accumulate.matmul_accumulate_core import MatmulAccumulateCore, MatmulAccumulateIO, MMAcDims, MMAcWidths
from spire.expr import Bool, Concat, Const, Expr, HDLType, SInt, Signal, UInt, fit_type, fit_width, reinterpret, s_ext
from spire.component import Component, Netlist


@dataclass
class MultiplierConfig:
    """Configuration for stage-based partial product flow."""

    ppg_opt: PPGOption
    ppa_opt: PPAOption
    fsa_opt: FSAOption
    optim_type: OptimType = "area"
    selection_mode: Optional[SelectionMode] = None
    split_mode: Optional[SplitMode] = None


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
        selection_mode=mult_cfg.selection_mode,
        split_mode=mult_cfg.split_mode,
    )

    if mult_cfg.ppg_opt == PPGOption.BAUGH_WOOLEY:
        # Lift BW's per-product constant corrections (upper, lower) and switch the
        # signed-C addition to BW's algebraic form. All three save area on the BW
        # baseline; see docs/booth_optim_fused_upper_correction.md and the handoff
        # doc for the algebra and measured gains.
        fused_upper_correction = True
        fused_lower_correction = True
        bw_style_c_term = True
        ppg = mult_cfg.ppg_opt.value(
            stage_cfg,
            upper_correction=not fused_upper_correction,
            lower_correction=not fused_lower_correction,
        )
    else:
        fused_upper_correction = False
        fused_lower_correction = False
        bw_style_c_term = False
        ppg = mult_cfg.ppg_opt.value(stage_cfg)
    ppa = mult_cfg.ppa_opt.value(stage_cfg)
    fsa = mult_cfg.fsa_opt.value(stage_cfg)

    # Booth PPGs select partial products based on operand signedness; when the
    # encoding is signed but the carrier signals happen to be UInt, reinterpret
    # them as SInt so the PPG sees the correct sign. BW reads bits directly and
    # is unaffected. Each PPG is responsible for emitting sign-extended columns
    # up to out_bits=result_width, so the merge below is a straight extend.
    def _to_signed(sig: Expr) -> Expr:
        if is_signed(encoding) and not sig.typ.signed:
            return reinterpret(sig, SInt(sig.typ.width))
        return sig

    merged_cols: DefaultDict[int, List[Expr]] = defaultdict(list)
    for idx, (a_sig, b_sig) in enumerate(zip(a_list, b_list)):
        io = StageBasedMultiplierIO(
            a=_to_signed(a_sig),
            b=_to_signed(b_sig),
            y=Signal(name=f"pp_{idx}", typ=UInt(result_width), kind="wire") # dummy, is not used
        )
        cols = ppg.generate_columns(io)
        for weight, bits in cols.items():
            if weight < result_width:
                merged_cols[weight].extend(bits)

    # common upper correction
    if fused_upper_correction:
        K = len(a_list)
        shift = a_width + b_width - 1  # column 2n-1 where the BW upper-correction run starts
        if K & (K - 1) == 0:
            # Power-of-2 K: the K per-product +1 runs collapse exactly into a single
            # run of +1s starting log2(K) columns higher. Easier-to-read form.
            for i in range(shift + int(log2(K)), result_width):
                merged_cols[i].append(Const(True, Bool()))
        else:
            # Arbitrary K: total upper correction equals -K * 2^(2n-1) mod 2^R.
            # Set a +1 in every column where that constant has a 1 bit.
            correction = (-K << shift) & ((1 << result_width) - 1)
            for i in range(result_width):
                if (correction >> i) & 1:
                    merged_cols[i].append(Const(True, Bool()))

    # common lower correction (BW): each per-product PPG would emit +1 at col wa
    # (symmetric) or +1 at col wa-1 plus +1 at col wb-1 (asymmetric). K copies
    # sum to a constant integer; emit only the bits of that sum, mod 2^R.
    if fused_lower_correction:
        K = len(a_list)
        if a_width == b_width:
            contribution = K << a_width
        else:
            contribution = (K << (a_width - 1)) + (K << (b_width - 1))
        correction = contribution & ((1 << result_width) - 1)
        for i in range(result_width):
            if (correction >> i) & 1:
                merged_cols[i].append(Const(True, Bool()))

    # add c term bits
    c_w = c_term.typ.width
    if bw_style_c_term and is_signed(encoding) and result_width > c_w:
        # BW-style sign extension of C: emit c[0..c_w-2] naturally, invert the
        # sign bit at col c_w-1, then add a constant +1 at every col in
        # [c_w-1, result_width). Algebraically equivalent to sign-replicating
        # c[c_w-1] across cols [c_w-1, R) but only emits one inverter and a run
        # of constants, instead of (R - c_w + 1) copies of the sign-bit signal.
        for k in range(c_w - 1):
            merged_cols[k].append(c_term[k])
        merged_cols[c_w - 1].append(~c_term[c_w - 1])
        for k in range(c_w - 1, result_width):
            merged_cols[k].append(Const(True, Bool()))
    else:
        if is_signed(encoding):
            c_term = s_ext(c_term, result_width) # sign-extend c_term to result_width, make sure source is SInt
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
        rows = []
        for i in range(self.cfg.dims.dim_m):
            row = []
            a_row = self.A[i, :]
            for j in range(self.cfg.dims.dim_n):
                b_col = self.B[:, j]
                dot = fused_inner_product(a_row, b_col, self.C[i, j], self.cfg.mult_cfg, self.cfg.encoding)
                y_sig = Signal(name=f"y_{i}_{j}", typ=self.io_hdl_type(dot.typ.width), kind="output")
                y_sig <<= dot
                row.append(y_sig)
            rows.append(Array(row))
        self.Y = Array(rows)


@dataclass
class MatmulAccumulateBuildOut:
    component: MatmulAccumulateComponent
    module: Netlist
    A: Array
    B: Array
    C: Array
    Y: Array
