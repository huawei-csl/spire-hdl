from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from sprouthdl.aggregate.aggregate_array import Array
from sprouthdl.arithmetic.int_arithmetic_config import (
    AdderConfig, MultiplierConfig, adder_tree, build_adder, build_multiplier,
)
from sprouthdl.arithmetic.int_multipliers.eval.testvector_generation import is_signed
from sprouthdl.cores.matmul_accumulate.matmul_accumulate_core import (
    MatmulAccumulateCore, MatmulAccumulateIO, MMAcCfg, MMAcDims, MMAcWidths,
)
from sprouthdl.sprouthdl import Expr, Register, SInt, Signal, UInt
from sprouthdl.sprouthdl_module import Module


def inner_product_pipelined(
    vec_a: Iterable[Expr], vec_b: Iterable[Expr],
    mult_cfg: MultiplierConfig, add_cfg: AdderConfig,
    name_prefix: str = "",
) -> Expr:
    """Inner product with one set of pipeline registers between multiplication and addition."""
    a_list: List[Expr] = list(vec_a)
    b_list: List[Expr] = list(vec_b)
    if len(a_list) != len(b_list):
        raise ValueError("inner_product_pipelined: length mismatch")

    products = [build_multiplier(a, b, mult_cfg) for a, b in zip(a_list, b_list)]

    pipeline_regs: List[Signal] = []
    for idx, p in enumerate(products):
        reg = Register(p.typ, init=0, name=f"{name_prefix}prod_reg_{idx}")
        reg <<= p
        pipeline_regs.append(reg)

    return adder_tree(pipeline_regs, add_cfg)


class MatmulAccumulatePipelinedComponent(MatmulAccumulateCore):
    """Matrix multiply-accumulate with one pipeline stage between products and the adder tree.

    Latency: outputs Y reflect the products of A,B from the previous cycle combined
    with the C value of the current cycle (C is added after the pipeline registers).
    """

    def __init__(self, cfg: MMAcCfg, signed_io_type: bool = False):
        self.cfg = cfg
        self.io_hdl_type = SInt if (is_signed(self.cfg.add_cfg.encoding) and signed_io_type) else UInt

        def build_matrix(name: str, width: int, rows: int, cols: int, kind: str = "wire") -> Array:
            return Array([
                Array([
                    Signal(name=f"{name}_{i}_{j}", typ=self.io_hdl_type(width), kind=kind) for j in range(cols)
                ])
                for i in range(rows)
            ])

        self.A = build_matrix("a", self.cfg.widths.a_width, self.cfg.dims.dim_m, self.cfg.dims.dim_k, "input")
        self.B = build_matrix("b", self.cfg.widths.b_width, self.cfg.dims.dim_k, self.cfg.dims.dim_n, "input")
        self.C = build_matrix("c", self.cfg.widths.c_width, self.cfg.dims.dim_m, self.cfg.dims.dim_n, "input")

        self.elaborate()

        self.io: MatmulAccumulateIO = MatmulAccumulateIO(A=self.A, B=self.B, C=self.C, Y=self.Y)

    def elaborate(self):
        rows = []
        for i in range(self.cfg.dims.dim_m):
            row = []
            a_row = self.A[i, :]
            for j in range(self.cfg.dims.dim_n):
                b_col = self.B[:, j]
                dot = inner_product_pipelined(
                    a_row, b_col, self.cfg.mult_cfg, self.cfg.add_cfg, name_prefix=f"r{i}_c{j}_",
                )
                acc = build_adder(self.C[i, j], dot, self.cfg.add_cfg)
                y_sig = Signal(name=f"y_{i}_{j}", typ=self.io_hdl_type(acc.typ.width), kind="output")
                y_sig <<= acc
                row.append(y_sig)
            rows.append(Array(row))
        self.Y = Array(rows)


@dataclass
class MatmulAccumulatePipelinedBuildOut:
    component: MatmulAccumulatePipelinedComponent
    module: Module
    A: Array
    B: Array
    C: Array
    Y: Array


def build_matmul_accumulate_pipelined(cfg: MMAcCfg, signed_io_type: bool = False) -> MatmulAccumulatePipelinedBuildOut:
    component = MatmulAccumulatePipelinedComponent(cfg, signed_io_type=signed_io_type)
    component_module = component.to_module("matmul_accumulate_core_pipelined", with_clock=True, with_reset=False)
    A, B, C, Y = component.io.A, component.io.B, component.io.C, component.io.Y

    return MatmulAccumulatePipelinedBuildOut(component=component, module=component_module, A=A, B=B, C=C, Y=Y)
