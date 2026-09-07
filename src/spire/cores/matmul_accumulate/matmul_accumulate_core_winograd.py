from __future__ import annotations

from collections import namedtuple
from dataclasses import dataclass
from typing import Callable, Iterable, List, Literal, NamedTuple, Sequence

from spire.composite.array import Array
from spire.composite.record import CompositeRecord
from spire.arithmetic.int_arithmetic_config import (
    AdderConfig,
    MultiplierConfig,
    adder_tree,
    build_adder,
    build_multiplier,
)
from spire.arithmetic.int_multipliers.eval.testvector_generation import is_signed
from spire.expr import Expr, HDLType, SInt, Signal, UInt, fit_type
from spire.component import Component, Netlist


def inner_product(
    vec_a: Iterable[Expr], vec_b: Iterable[Expr], mult_cfg: MultiplierConfig, add_cfg: AdderConfig,
    alpha: Expr, beta: Expr
) -> Expr:
    a_list: List[Expr] = list(vec_a)
    b_list: List[Expr] = list(vec_b)
    if len(a_list) != len(b_list):
        raise ValueError("inner_product: length mismatch")
    if len(a_list) % 2 != 0:
        # The Winograd pairing below consumes elements two at a time; an odd tail element
        # would be silently dropped, producing wrong products.
        raise ValueError(f"inner_product (Winograd): vector length must be even, got {len(a_list)}")

    mult_k_list = []
    dim_k = len(a_list)
    for k in range(0, dim_k//2):  

        a_s0 = a_list[2*k]
        b_s0 = b_list[2*k+1]
        s0 = build_adder(a_s0, b_s0, add_cfg)
        a_s1 = a_list[2*k+1]
        b_s1 = b_list[2*k]
        s1 = build_adder(a_s1, b_s1, add_cfg)
        mult_k = build_multiplier(s0, s1, mult_cfg)
        mult_k_list.append(mult_k)

    # Negation depends on the ENCODING, not the carrier type: under signed encodings the UInt
    # carriers hold two's-complement patterns (same-width signed reinterpret is exact); under
    # unsigned encodings they hold plain values, which must WIDEN before the signed cast (a
    # same-width cast flips values >= 2^(w-1) negative — alpha 225 read as -31).
    def _neg(x):
        w = x.typ.width if is_signed(add_cfg.encoding) else x.typ.width + 1
        return -fit_type(x, SInt(w))

    if is_signed(add_cfg.encoding):
        summands = mult_k_list + [_neg(alpha), _neg(beta)]
        return adder_tree(summands, add_cfg)

    # Unsigned encodings: Winograd's -alpha/-beta terms are inherently signed, and unsigned
    # structural adders would zero-extend them. Accumulate through the IR's signed adds
    # (values stay exact; the caller wraps the pattern to the output width).
    acc = None
    for term in [*(fit_type(mk, SInt(mk.typ.width + 1)) for mk in mult_k_list), _neg(alpha), _neg(beta)]:
        acc = term if acc is None else acc + term
    return acc


# Shared IO/config dataclasses — the local redefinitions created distinct class identities
# from every other variant (drift risk; matmul_test_vectors type-hints the shared ones).
from spire.cores.matmul_accumulate.matmul_accumulate_core import (
    MatmulAccumulateCore, MatmulAccumulateIO, MMAcCfg, MMAcDims, MMAcWidths,
    _check_operator_mode_encodings,
)

class MatmulAccumulateComponent(MatmulAccumulateCore):
    """Reusable component for matrix multiply-accumulate."""

    def __init__(
        self,
        cfg: MMAcCfg,
        signed_io_type: bool = False,
    ):

        _check_operator_mode_encodings(cfg.mult_cfg, cfg.add_cfg.encoding,
                                       carriers_signed=False)  # pre-adder outputs are UInt-laundered
        self.cfg = cfg
        if cfg.dims.dim_k % 2 != 0:
            # The Winograd alpha/beta/pairing loops iterate range(dim_k // 2), silently dropping
            # the last K-term for odd dim_k and computing wrong results.
            raise ValueError(f"MatmulAccumulateComponent (Winograd): dim_k must be even, got {cfg.dims.dim_k}")
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

        self.A = build_matrix("a", self.cfg.widths.a_width, self.cfg.dims.dim_m, self.cfg.dims.dim_k, "input")
        self.B = build_matrix("b", self.cfg.widths.b_width, self.cfg.dims.dim_k, self.cfg.dims.dim_n, "input")
        self.C = build_matrix("c", self.cfg.widths.c_width, self.cfg.dims.dim_m, self.cfg.dims.dim_n, "input")

        self.elaborate()

        self.io: MatmulAccumulateIO = MatmulAccumulateIO(A=self.A, B=self.B, C=self.C, Y=self.Y)

    def elaborate(self):
        
        # Calculate alphas and betas
        alphas = []
        for i in range(self.cfg.dims.dim_m):
            alpha_ks = []
            for k in range(self.cfg.dims.dim_k//2):
                alpha_ks.append(build_multiplier(self.A[i, 2*k], self.A[i, 2*k + 1], self.cfg.mult_cfg))
            alpha_k = adder_tree(alpha_ks, self.cfg.add_cfg)
            alphas.append(alpha_k)
            
        betas = []
        for j in range(self.cfg.dims.dim_n):
            beta_ks = []
            for k in range(self.cfg.dims.dim_k//2):
                beta_ks.append(build_multiplier(self.B[2*k, j], self.B[2*k + 1, j], self.cfg.mult_cfg))
            beta_k = adder_tree(beta_ks, self.cfg.add_cfg)
            betas.append(beta_k)
            
        # same with Spire operators
        # alphas = []
        # for i in range(self.cfg.dims.dim_m):
        #     alphas.append(sum([self.A[i, 2*k] * self.A[i, 2*k + 1] for k in range(self.cfg.dims.dim_k//2)]))
        # betas = []
        # for j in range(self.cfg.dims.dim_n):
        #     betas.append(sum([self.B[2*k, j] * self.B[2*k + 1, j] for k in range(self.cfg.dims.dim_k//2)]))    
        
        # inner product and accumulation for each output element 
        rows = []
        for i in range(self.cfg.dims.dim_m):
            row = []
            a_row = self.A[i, :]
            for j in range(self.cfg.dims.dim_n):
                b_col = self.B[:, j]
                dot = inner_product(a_row, b_col, self.cfg.mult_cfg, self.cfg.add_cfg, alphas[i], betas[j])
                acc = build_adder(self.C[i, j], dot, self.cfg.add_cfg)
                y_sig = Signal(name=f"y_{i}_{j}", typ=self.io_hdl_type(acc.typ.width), kind="output")
                y_sig <<= acc
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


def build_matmul_accumulate(
    cfg: MMAcCfg,
    signed_io_type: bool = False,
) -> MatmulAccumulateBuildOut:
        
    component = MatmulAccumulateComponent(cfg, signed_io_type=signed_io_type)
    component_module = component.to_netlist("matmul_accumulate_core")
    A, B, C, Y = component.io.A, component.io.B, component.io.C, component.io.Y

    return MatmulAccumulateBuildOut(
        component=component,
        module=component_module,
        A=A,
        B=B,
        C=C,
        Y=Y,
    )


def ceil_log2(n: int) -> int:
    if n <= 1:
        return 0
    return (n - 1).bit_length()


def max_y_width_unsigned(
    a_width: int, b_width: int, dim_k: int, *, include_carry_from_add: bool = True
) -> int:
    carry = 1 if include_carry_from_add else 0
    return a_width + b_width + ceil_log2(dim_k) + carry
