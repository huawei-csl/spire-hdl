from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Literal, Sequence

import numpy as np

from spire.composite.array import Array
from spire.arithmetic.int_multipliers.eval.multiplier_stage_options_demo_lib import (
    FSAOption,
    MultiplierOption,
    PPAOption,
    PPGOption,
    TwoInputAritEncodings,
)
from spire.arithmetic.int_multipliers.eval.testvector_generation import Encoding, EncodingModel, is_signed
from spire.arithmetic.prefix_adders.adders import StageBasedPrefixAdder
from spire.cores.matmul_accumulate.matmul_accumulate_core_winograd import AdderConfig, MMAcCfg, MMAcDims, MMAcWidths, MatmulAccumulateComponent, MultiplierConfig, build_matmul_accumulate, max_y_width_unsigned
from spire.helpers import get_yosys_metrics
from spire.expr import Expr, SInt, Signal, UInt
from spire.component import Component, Netlist
from spire.simulator import Simulator


def test_mmac_core_basic_simulation():
    dim_m = 4
    dim_n = 4
    dim_k = 4
    a_width = 4
    b_width = 4
    c_width = max_y_width_unsigned(a_width, b_width, dim_k, include_carry_from_add=False)
    encoding = Encoding.twos_complement
    signed_io_type = True
    n_iter_optimizations = 0  # none for default

    # use Spire operators
    # mult_cfg = MultiplierConfig(use_operator=True)
    # add_cfg = AdderConfig(use_operator=True, full_output_bit=True)

    # use custom multiplier and adder
    mult_cfg = MultiplierConfig(
        use_operator=False,
        multiplier_opt=MultiplierOption.STAGE_BASED_MULTIPLIER,
        encodings=TwoInputAritEncodings.with_enc(encoding),
        ppg_opt=PPGOption.BAUGH_WOOLEY if is_signed(encoding) else PPGOption.AND,
        ppa_opt=PPAOption.WALLACE_TREE,
        fsa_opt=FSAOption.RIPPLE_CARRY,
    )
    add_cfg = AdderConfig(use_operator=False, fsa_opt=FSAOption.RIPPLE_CARRY, full_output_bit=True, encoding=encoding)

    core_config = MMAcCfg(
        dims=MMAcDims(dim_m=dim_m, dim_n=dim_n, dim_k=dim_k),
        widths=MMAcWidths(a_width=a_width, b_width=b_width, c_width=c_width),
        mult_cfg=mult_cfg,
        add_cfg=add_cfg,
    )

    core = MatmulAccumulateComponent(core_config, signed_io_type=signed_io_type)

    print(
        f"Output matrix Y has shape: ({core_config.dims.dim_m}, {core_config.dims.dim_n}) with element width {core.io.Y[0,0].typ.width} bits"
    )

    # For simulation, operate on the module built directly from the reusable component.
    module = core.to_module("matmul_accumulate_core")

    print("Starting simulation...")
    sim = Simulator(module)

    # either use this
    # a_min, a_max = EncodingModel(encoding).value_range(a_width)
    # b_min, b_max = EncodingModel(encoding).value_range(b_width)
    # c_min, c_max = EncodingModel(encoding).value_range(c_width)
    # rng = np.random.default_rng(seed=42)
    # a_vals = rng.integers(a_min, a_max, size=(dim_m, dim_k), dtype=int)
    # b_vals = rng.integers(b_min, b_max, size=(dim_k, dim_n), dtype=int)
    # c_vals = rng.integers(c_min, c_max, size=(dim_m, dim_n), dtype=int)

    # or
    a_vals = EncodingModel(encoding).get_uniform_sample_np(a_width, size=(dim_m, dim_k))
    b_vals = EncodingModel(encoding).get_uniform_sample_np(b_width, size=(dim_k, dim_n))
    c_vals = EncodingModel(encoding).get_uniform_sample_np(c_width, size=(dim_m, dim_n))

    set_matrix(sim, core.io.A, a_vals)
    set_matrix(sim, core.io.B, b_vals)
    set_matrix(sim, core.io.C, c_vals)

    sim.eval()

    y_hw = np.zeros((dim_m, dim_n), dtype=int)
    for i in range(dim_m):
        for j in range(dim_n):
            y_hw[i, j] = sim.get(core.io.Y[i, j])

    y_np = a_vals @ b_vals + c_vals
    if signed_io_type:
        assert np.array_equal(y_hw, y_np), "Simulation mismatch for matmul accumulate core"
    else:
        # encode each element according to the encoding
        y_np_encoded = np.vectorize(lambda x: EncodingModel(encoding).encode_value(int(x), core.io.Y[0, 0].typ.width))(y_np)
        assert np.array_equal(y_hw, y_np_encoded), "Simulation mismatch for matmul accumulate core"

    # get yosys transistor count
    yosys_metrics = get_yosys_metrics(module, n_iter_optimizations=n_iter_optimizations)
    print(f"Yosys metrics: {yosys_metrics}")


def set_matrix(sim: Simulator, port2d, vals):
    for i, row in enumerate(vals):
        for j, v in enumerate(row):
            sim.set(port2d[i, j], int(v))


if __name__ == "__main__":
    test_mmac_core_basic_simulation()
