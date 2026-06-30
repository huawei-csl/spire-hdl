from __future__ import annotations

import numpy as np

from spire.composite.floating_point import FloatingPointType
from spire.arithmetic.int_arithmetic_config import AdderConfig, MultiplierConfig
from spire.arithmetic.int_multipliers.eval.multiplier_stage_options_demo_lib import (
    FSAOption,
    MultiplierOption,
    PPAOption,
    PPGOption,
    TwoInputAritEncodings,
)
from spire.arithmetic.int_multipliers.eval.testvector_generation import Encoding
from spire.cores.matmul_accumulate.matmul_accumulate_core_float import (
    FpMMAcCfg,
    FpMMAcDims,
    FpMatmulAccumulateComponent,
)
from spire.arithmetic.floating_point.fp_encoding import fp_decode, fp_encode
from spire.cores.matmul_accumulate.matmul_test_vectors import generate_fp_matmul_vectors
from spire.helpers import get_yosys_metrics, run_vectors_on_simulator
from spire.simulator import Simulator


def _float16_matmul_accumulate(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Reference accumulation using numpy float16 arithmetic."""
    dim_m, dim_k = a.shape
    _, dim_n = b.shape
    y = np.empty((dim_m, dim_n), dtype=np.float16)
    for i in range(dim_m):
        for j in range(dim_n):
            acc = np.float16(c[i, j])
            for k in range(dim_k):
                acc = np.float16(acc + np.float16(a[i, k] * b[k, j]))
            y[i, j] = acc
    return y


def test_fp_matmul_accumulate_simulation():
    dim = 4
    ft = FloatingPointType(exponent_width=5, fraction_width=8)

    cfg = FpMMAcCfg(dims=FpMMAcDims(dim_m=dim, dim_n=dim, dim_k=dim), ftype=ft)
    core = FpMatmulAccumulateComponent(cfg)
    module = core.to_module("fp_matmul_accumulate")

    print(f"Output matrix Y has shape: ({dim}, {dim}) with element width {core.io.Y[0, 0].bits.typ.width} bits")

    vectors = generate_fp_matmul_vectors(core, num_vectors=128)

    sim = Simulator(module)
    failures = run_vectors_on_simulator(sim, vectors, use_signed=False, raise_on_fail=True, print_on_pass=False)
    print(f"Simulation complete: {failures} failures")

    # yosys_metrics = get_yosys_metrics(module)
    # print(f"Yosys metrics: {yosys_metrics}")


def test_fp_matmul_accumulate_simulation_approx():
    """Approximate simulation test using numpy float16 as reference (np.isclose)."""
    dim = 4
    ft = FloatingPointType(exponent_width=5, fraction_width=8)
    ew, fw = ft.exponent_width, ft.fraction_width

    cfg = FpMMAcCfg(dims=FpMMAcDims(dim_m=dim, dim_n=dim, dim_k=dim), ftype=ft)
    core = FpMatmulAccumulateComponent(cfg)
    module = core.to_module("fp_matmul_accumulate_approx")

    sim = Simulator(module)

    rng = np.random.default_rng(seed=123)
    a_vals = rng.uniform(0.1, 2.0, size=(dim, dim))
    b_vals = rng.uniform(0.1, 2.0, size=(dim, dim))
    c_vals = rng.uniform(0.1, 2.0, size=(dim, dim))

    def _encode(x: float) -> int:
        return fp_encode(float(x), ew, fw)

    def _decode(b: int) -> float:
        return fp_decode(int(b), ew, fw)

    a_bits = np.vectorize(_encode)(a_vals)
    b_bits = np.vectorize(_encode)(b_vals)
    c_bits = np.vectorize(_encode)(c_vals)

    # Quantise inputs to the target FP format before passing to the reference.
    a_q = np.vectorize(_decode)(a_bits)
    b_q = np.vectorize(_decode)(b_bits)
    c_q = np.vectorize(_decode)(c_bits)

    for i in range(dim):
        for k in range(dim):
            sim.set(core.io.A[i, k].bits, int(a_bits[i, k]))
    for k in range(dim):
        for j in range(dim):
            sim.set(core.io.B[k, j].bits, int(b_bits[k, j]))
    for i in range(dim):
        for j in range(dim):
            sim.set(core.io.C[i, j].bits, int(c_bits[i, j]))

    sim.eval()

    y_hw = np.vectorize(_decode)(
        np.array([[sim.get(core.io.Y[i, j].bits) for j in range(dim)] for i in range(dim)])
    )
    y_expected = _float16_matmul_accumulate(a_q, b_q, c_q)

    assert np.isclose(y_hw, y_expected, rtol=1e-2, atol=1e-2).all(), (
        f"Mismatch:\nhw      = {y_hw}\nexpected= {y_expected}"
    )


def test_fp_matmul_accumulate_simulation_with_cfg():
    """Bit-exact simulation with explicit stage-based multiplier and ripple-carry adder configs."""
    dim = 4
    ft = FloatingPointType(exponent_width=5, fraction_width=8)

    mult_cfg = MultiplierConfig(
        use_operator=False,
        multiplier_opt=MultiplierOption.STAGE_BASED_MULTIPLIER,
        encodings=TwoInputAritEncodings.with_enc(Encoding.unsigned),
        ppg_opt=PPGOption.AND,
        ppa_opt=PPAOption.WALLACE_TREE,
        fsa_opt=FSAOption.RIPPLE_CARRY,
    )
    adder_cfg = AdderConfig(
        use_operator=False,
        fsa_opt=FSAOption.RIPPLE_CARRY,
        full_output_bit=True,
        encoding=Encoding.unsigned,
    )

    cfg = FpMMAcCfg(
        dims=FpMMAcDims(dim_m=dim, dim_n=dim, dim_k=dim),
        ftype=ft,
        adder_cfg=adder_cfg,
        mult_cfg=mult_cfg,
    )
    core = FpMatmulAccumulateComponent(cfg)
    module = core.to_module("fp_matmul_accumulate_cfg")

    vectors = generate_fp_matmul_vectors(core, num_vectors=32)

    sim = Simulator(module)
    failures = run_vectors_on_simulator(sim, vectors, use_signed=False, raise_on_fail=True, print_on_pass=False)
    print(f"Simulation complete (with cfg): {failures} failures")


if __name__ == "__main__":
    test_fp_matmul_accumulate_simulation()
    test_fp_matmul_accumulate_simulation_approx()
    test_fp_matmul_accumulate_simulation_with_cfg()
