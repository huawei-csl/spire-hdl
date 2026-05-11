from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Literal, Sequence

import numpy as np

from spirehdl.aggregate.aggregate_array import Array
from spirehdl.arithmetic.int_multipliers.eval.multiplier_stage_options_demo_lib import (
    FSAOption,
    MultiplierOption,
    PPAOption,
    PPGOption,
    TwoInputAritEncodings,
)
from spirehdl.arithmetic.int_multipliers.eval.testvector_generation import Encoding
from spirehdl.arithmetic.prefix_adders.adders import StageBasedPrefixAdder
from spirehdl.arithmetic.int_multipliers.multipliers.multiplier_stage_core import (
    RippleCarryFinalAdder,
)
from spirehdl.helpers import get_yosys_metrics
from spirehdl.spirehdl import Expr, UInt
from spirehdl.spirehdl_module import Module
from spirehdl.spirehdl_simulator import Simulator


@dataclass
class MultiplierConfig:
    """Configuration for choosing between SpireHDL operator and explicit multiplier."""

    use_operator: bool = False
    multiplier_opt: MultiplierOption | None = None
    encodings: TwoInputAritEncodings | None = None
    ppg_opt: PPGOption | None = None
    ppa_opt: PPAOption | None = None
    fsa_opt: FSAOption | None = None
    optim_type: Literal["area", "speed"] = "area"

    def build(self, a: Expr, b: Expr) -> Expr:
        if self.use_operator:
            return a * b

        assert self.encodings is not None, "encodings must be provided for explicit multipliers"
        assert self.ppg_opt is not None and self.ppa_opt is not None and self.fsa_opt is not None

        multiplier = self.multiplier_opt.value(
            a_w=a.typ.width,
            b_w=b.typ.width,
            a_encoding=self.encodings.a,
            b_encoding=self.encodings.b,
            ppg_cls=self.ppg_opt.value,
            ppa_cls=self.ppa_opt.value,
            fsa_cls=self.fsa_opt.value,
            optim_type=self.optim_type,
        ).make_internal()
        multiplier.io.a <<= a
        multiplier.io.b <<= b
        return multiplier.io.y


@dataclass
class AdderConfig:
    """Configuration for choosing between SpireHDL operator and explicit adder."""

    use_operator: bool = False
    signed: bool = False
    optim_type: Literal["area", "speed"] = "area"
    fsa_opt: FSAOption | None = None
    full_output_bit: bool = True

    def build(self, a: Expr, b: Expr) -> Expr:
        if self.use_operator:
            return a + b

        adder = StageBasedPrefixAdder(
            a_w=a.typ.width,
            b_w=b.typ.width,
            signed_a=self.signed,
            signed_b=self.signed,
            optim_type=self.optim_type,
            fsa_cls=self.fsa_opt.value,
            full_output_bit=self.full_output_bit,
        ).make_internal()
        adder.io.a <<= a
        adder.io.b <<= b
        return adder.io.y


def adder_tree(values: Sequence[Expr], adder_cfg: AdderConfig) -> Expr:
    if len(values) == 0:
        raise ValueError("Adder tree requires at least one value")
    if len(values) == 1:
        return values[0]

    mid = len(values) // 2
    left = adder_tree(values[:mid], adder_cfg)
    right = adder_tree(values[mid:], adder_cfg)
    return adder_cfg.build(left, right)


def inner_product(
    vec_a: Iterable[Expr], vec_b: Iterable[Expr], mult_cfg: MultiplierConfig, add_cfg: AdderConfig
) -> Expr:
    a_list: List[Expr] = list(vec_a)
    b_list: List[Expr] = list(vec_b)
    if len(a_list) != len(b_list):
        raise ValueError("inner_product: length mismatch")

    products = [mult_cfg.build(a, b) for a, b in zip(a_list, b_list)]
    return adder_tree(products, add_cfg)


@dataclass
class MatmulAccumulateCore:
    module: Module
    A: Array
    B: Array
    C: Array
    Y: Array


def build_matmul_accumulate_core(
    dim: int,
    a_width: int,
    b_width: int,
    c_width: int,
    mult_cfg: MultiplierConfig,
    add_cfg: AdderConfig,
) -> MatmulAccumulateCore:
    if dim <= 0 or dim & (dim - 1) != 0:
        raise ValueError("Matrix dimension must be a power of two")

    m = Module("matmul_accumulate_core")

    # not for now, but in the future we would like a module, e.g. with IOs. But to convert to a module we need basic hdl types (signals with kind input and output)
    # maybe have an inner componenet with arrays as input and output (need input output kind for aggregate hdl type?), 
    # then wrap an outer comoponent or module with signals as input and output that connect to the inner component
    # io could be defined like this:
    # class MyRecord(AggregateRecord):
    #    a: Array = self.A
    #    b: Array = self.B
    #    c: Array = self.C
    #    y: Array = self.Y

    # self.io = MyRecord()

    def build_matrix(name: str, width: int) -> Array:
        return Array(
            [
                Array([m.input(UInt(width), f"{name}_{i}_{j}") for j in range(dim)])
                for i in range(dim)
            ]
        )

    A = build_matrix("a", a_width)
    B = build_matrix("b", b_width)
    C = build_matrix("c", c_width)

    rows = []
    for i in range(dim):
        row = []
        a_row = A[i, :]
        for j in range(dim):
            b_col = B[:, j]
            dot = inner_product(a_row, b_col, mult_cfg, add_cfg)
            acc = add_cfg.build(C[i, j], dot)
            y_sig = m.output(acc.typ, f"y_{i}_{j}")
            y_sig <<= acc
            row.append(y_sig)
        rows.append(Array(row))
    Y = Array(rows)

    return MatmulAccumulateCore(module=m, A=A, B=B, C=C, Y=Y)


def test_mmac_core_basic_simulation():
    dim = 4
    a_width = 8
    b_width = 8
    c_width = 20

    # use SpireHDL operators
    #mult_cfg = MultiplierConfig(use_operator=True)
    #add_cfg = AdderConfig(use_operator=True, full_output_bit=True)

    # use custom multiplier and adder
    mult_cfg = MultiplierConfig(use_operator=False,
                                multiplier_opt=MultiplierOption.STAGE_BASED_MULTIPLIER,
                                encodings=TwoInputAritEncodings.with_enc(Encoding.unsigned),
                                ppg_opt=PPGOption.AND, ppa_opt=PPAOption.WALLACE_TREE, fsa_opt=FSAOption.RIPPLE_CARRY)
    add_cfg = AdderConfig(use_operator=False, fsa_opt=FSAOption.RIPPLE_CARRY, full_output_bit=True)

    core = build_matmul_accumulate_core(dim, a_width, b_width, c_width, mult_cfg, add_cfg)

    print(f"Output matrix Y has shape: ({dim}, {dim}) with element width {core.Y[0,0].typ.width} bits")

    sim = Simulator(core.module)

    rng = np.random.default_rng(seed=42)
    a_vals = rng.integers(0, 2**a_width, size=(dim, dim), dtype=int)
    b_vals = rng.integers(0, 2**b_width, size=(dim, dim), dtype=int)
    c_vals = rng.integers(0, 2**c_width, size=(dim, dim), dtype=int)

    for i in range(dim):
        for j in range(dim):
            sim.set(core.A[i, j], int(a_vals[i, j]))
            sim.set(core.B[i, j], int(b_vals[i, j]))
            sim.set(core.C[i, j], int(c_vals[i, j]))

    sim.eval()

    y_hw = np.zeros((dim, dim), dtype=int)
    for i in range(dim):
        for j in range(dim):
            y_hw[i, j] = sim.get(core.Y[i, j])

    y_np = a_vals @ b_vals + c_vals
    assert np.array_equal(y_hw, y_np), "Simulation mismatch for matmul accumulate core"
    print("Matmul accumulate simulation passed. Y=\n", y_hw)

    # get yosys transistor count
    yosys_metrics = get_yosys_metrics(core.module)
    print(f"Yosys metrics: {yosys_metrics}")


if __name__ == "__main__":
    test_mmac_core_basic_simulation()
