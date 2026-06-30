from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from spire.composite.array import Array
from spire.composite.floating_point import FloatingPoint, FloatingPointType
from spire.composite.record import CompositeRecord
from spire.arithmetic.int_arithmetic_config import AdderConfig, MultiplierConfig
from spire.expr import Signal, UInt
from spire.component import Component


@dataclass
class FpMMAcIO(CompositeRecord):
    A: Array  # dim_m x dim_k, FloatingPoint elements
    B: Array  # dim_k x dim_n, FloatingPoint elements
    C: Array  # dim_m x dim_n, FloatingPoint elements
    Y: Array  # dim_m x dim_n, FloatingPoint elements


@dataclass
class FpMMAcDims:
    dim_m: int
    dim_n: int
    dim_k: int


@dataclass
class FpMMAcCfg:
    dims: FpMMAcDims
    ftype: FloatingPointType
    adder_cfg: Optional[AdderConfig] = None
    mult_cfg: Optional[MultiplierConfig] = None


class FpMatmulAccumulateCore(Component):
    io: FpMMAcIO


class FpMatmulAccumulateComponent(FpMatmulAccumulateCore):

    def __init__(self, cfg: FpMMAcCfg) -> None:
        self.cfg = cfg
        ft = cfg.ftype
        W = ft.width_total
        dims = cfg.dims

        def _build_matrix(prefix: str, rows: int, cols: int, kind: str) -> Array:
            return Array([
                Array([
                    FloatingPoint(
                        ft,
                        bits=Signal(name=f"{prefix}_{i}_{j}", typ=UInt(W), kind=kind),
                        adder_cfg=cfg.adder_cfg,
                        mult_cfg=cfg.mult_cfg,
                    )
                    for j in range(cols)
                ])
                for i in range(rows)
            ])

        self.A = _build_matrix("a", dims.dim_m, dims.dim_k, "input")
        self.B = _build_matrix("b", dims.dim_k, dims.dim_n, "input")
        self.C = _build_matrix("c", dims.dim_m, dims.dim_n, "input")

        self.elaborate()
        self.io = FpMMAcIO(A=self.A, B=self.B, C=self.C, Y=self.Y)

    def elaborate(self) -> None:
        ft = self.cfg.ftype
        W = ft.width_total
        dims = self.cfg.dims

        rows = []
        for i in range(dims.dim_m):
            row = []
            for j in range(dims.dim_n):
                dot = None
                for k in range(dims.dim_k):
                    prod = self.A[i, k] * self.B[k, j]
                    dot = prod if dot is None else dot + prod
                acc = dot + self.C[i, j]
                y_sig = Signal(name=f"y_{i}_{j}", typ=UInt(W), kind="output")
                y_sig <<= acc.bits
                row.append(FloatingPoint(ft, bits=y_sig))
            rows.append(Array(row))
        self.Y = Array(rows)
