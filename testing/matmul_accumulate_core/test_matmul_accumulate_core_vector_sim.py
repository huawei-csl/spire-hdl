from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from spire.arithmetic.int_multipliers.eval.multiplier_stage_options_demo_lib import (
    FSAOption,
    MultiplierOption,
    PPAOption,
    PPGOption,
    TwoInputAritEncodings,
)
from spire.arithmetic.int_multipliers.eval.testvector_generation import Encoding, EncodingModel, TestVectors
from spire.helpers import refactor_module_to_aig, run_vectors_on_simulator, sim_and_switch_count
from spire.expr import Op2
from spire.component import Netlist
from spire.simulator import Simulator
from spire.verilog_testbench import TestbenchGenSimulator, write_vector_data_file
from spire.arithmetic.int_arithmetic_config import AdderConfig, MultiplierConfig
from spire.cores.matmul_accumulate.matmul_accumulate_core import (
    MMAcCfg,
    MMAcDims,
    MMAcWidths,
    build_matmul_accumulate,
    max_y_width_unsigned,
)


def _shape2d(arr) -> tuple[int, int]:
    return len(arr), len(arr[0])


def _rand_mat(rng: np.random.Generator, arr, width: int) -> np.ndarray:
    rows, cols = _shape2d(arr)
    return rng.integers(0, 2**width, size=(rows, cols), dtype=int)


def _build_vectors(core, num_vectors: int) -> list[tuple[str, dict[str, int], dict[str, int]]]:
    a_width = core.A[0, 0].typ.width
    b_width = core.B[0, 0].typ.width
    c_width = core.C[0, 0].typ.width

    a_rows, a_cols = _shape2d(core.A)  # A: (m, k)
    b_rows, b_cols = _shape2d(core.B)  # B: (k, n)
    c_rows, c_cols = _shape2d(core.C)  # C/Y: (m, n)

    rng = np.random.default_rng(seed=42)
    vectors: List[Tuple[str, Dict[str, int], Dict[str, int]]] = []

    for idx in range(num_vectors):
        a_vals = _rand_mat(rng, core.A, a_width)
        b_vals = _rand_mat(rng, core.B, b_width)
        c_vals = _rand_mat(rng, core.C, c_width)
        y_vals = a_vals @ b_vals + c_vals

        ins: Dict[str, int] = {}
        outs: Dict[str, int] = {}

        # A: (m, k)
        for i in range(a_rows):
            for k in range(a_cols):
                ins[core.A[i, k].name] = int(a_vals[i, k])

        # B: (k, n)
        for k in range(b_rows):
            for j in range(b_cols):
                ins[core.B[k, j].name] = int(b_vals[k, j])

        # C/Y: (m, n)
        for i in range(c_rows):
            for j in range(c_cols):
                ins[core.C[i, j].name] = int(c_vals[i, j])
                outs[core.Y[i, j].name] = int(y_vals[i, j])

        vectors.append((f"vec_{idx}", ins, outs))

    return vectors


def _build_vectors_encoding(
    core, encoding: Encoding, num_vectors: int
) -> TestVectors:
    a_width = core.A[0, 0].typ.width
    b_width = core.B[0, 0].typ.width
    c_width = core.C[0, 0].typ.width

    a_rows, a_cols = _shape2d(core.A)  # A: (m, k)
    b_rows, b_cols = _shape2d(core.B)  # B: (k, n)
    c_rows, c_cols = _shape2d(core.C)  # C/Y: (m, n)

    enc_model = EncodingModel(encoding)
    vectors: TestVectors = []

    for idx in range(num_vectors):
        a_vals = enc_model.get_uniform_sample_np(a_width, size=(a_rows, a_cols))
        b_vals = enc_model.get_uniform_sample_np(b_width, size=(b_rows, b_cols))
        c_vals = enc_model.get_uniform_sample_np(c_width, size=(c_rows, c_cols))
        y_vals = a_vals @ b_vals + c_vals
        
        # encode the values according to the encoding
        a_vals = np.vectorize(lambda x: enc_model.encode_value(int(x), a_width))(a_vals)
        b_vals = np.vectorize(lambda x: enc_model.encode_value(int(x), b_width))(b_vals)
        c_vals = np.vectorize(lambda x: enc_model.encode_value(int(x), c_width))(c_vals)
        y_vals = np.vectorize(lambda x: enc_model.encode_value(int(x), core.Y[0, 0].typ.width))(y_vals)

        ins: Dict[str, int] = {}
        outs: Dict[str, int] = {}

        # A: (m, k)
        for i in range(a_rows):
            for k in range(a_cols):
                ins[core.A[i, k].name] = int(a_vals[i, k])

        # B: (k, n)
        for k in range(b_rows):
            for j in range(b_cols):
                ins[core.B[k, j].name] = int(b_vals[k, j])

        # C/Y: (m, n)
        for i in range(c_rows):
            for j in range(c_cols):
                ins[core.C[i, j].name] = int(c_vals[i, j])
                outs[core.Y[i, j].name] = int(y_vals[i, j])

        vectors.append((f"vec_{idx}", ins, outs))

    return vectors

def test_mmac_core_vector_simulation():
    dim_m = 4
    dim_n = 4
    dim_k = 4
    a_width = 4
    b_width = 4
    c_width = max_y_width_unsigned(a_width, b_width, dim_k, include_carry_from_add=False)

    signed_io_type = False

    encoding = Encoding.unsigned

    mult_cfg = MultiplierConfig(
        use_operator=False,
        multiplier_opt=MultiplierOption.STAGE_BASED_MULTIPLIER,
        encodings=TwoInputAritEncodings.with_enc(encoding),
        ppg_opt=PPGOption.AND,
        ppa_opt=PPAOption.WALLACE_TREE,
        fsa_opt=FSAOption.RIPPLE_CARRY,
    )
    add_cfg = AdderConfig(use_operator=False, fsa_opt=FSAOption.RIPPLE_CARRY, full_output_bit=True)

    core_cfg = MMAcCfg(
        dims=MMAcDims(dim_m=dim_m, dim_n=dim_n, dim_k=dim_k),
        widths=MMAcWidths(a_width=a_width, b_width=b_width, c_width=c_width),
        mult_cfg=mult_cfg,
        add_cfg=add_cfg,
    )

    core = build_matmul_accumulate(cfg=core_cfg, signed_io_type=signed_io_type)

    core.module = refactor_module_to_aig(core.module)

    # vectors = _build_vectors(core, num_vectors=5)
    vectors = _build_vectors_encoding(core, encoding=encoding, num_vectors=50)

    sim = Simulator(core.module)
    run_vectors_on_simulator(
        sim,
        vectors,
        decoder=None,
        print_on_pass=False,
        test_name="Spire Simulation, MMAC vectors",
    )

    sim_tb = TestbenchGenSimulator(core.module)

    run_vectors_on_simulator(
        sim_tb, vectors, decoder=None, print_on_pass=False, test_name="TbGen Simulation"
    )

    write_vector_data_file(vectors, "mmac_core_vectors.dat")
    sim_tb.to_data_driver_testbench_file(filepath="mmac_core_tb_sim.v", data_file="mmac_core_vectors.dat")

    # also save verilog file
    core.module.to_verilog_file("mmac_core.v")

    # swact simulation -------------------------------
    switches = sim_and_switch_count(core.module, vectors)
    print(f"Average AND switches: {switches}")


if __name__ == "__main__":
    test_mmac_core_vector_simulation()
