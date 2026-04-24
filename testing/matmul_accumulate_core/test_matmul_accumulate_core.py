from __future__ import annotations

from spirehdl.arithmetic.int_multipliers.eval.multiplier_stage_options_demo_lib import (
    FSAOption,
    MultiplierOption,
    PPAOption,
    PPGOption,
    TwoInputAritEncodings,
)
from spirehdl.arithmetic.int_multipliers.eval.testvector_generation import Encoding, is_signed
from spirehdl.arithmetic.int_arithmetic_config import AdderConfig, MultiplierConfig
from spirehdl.cores.matmul_accumulate.matmul_accumulate_core import (
    MMAcCfg,
    MMAcDims,
    MMAcWidths,
    MatmulAccumulateComponent,
    max_y_width_unsigned,
)
from spirehdl.cores.matmul_accumulate.matmul_test_vectors import generate_matmul_vectors
from spirehdl.helpers import get_yosys_metrics, run_vectors_on_simulator
from spirehdl.spirehdl_simulator import Simulator


def test_mmac_core_basic_simulation():
    dim_m = 4
    dim_n = 4
    dim_k = 4
    a_width = 4
    b_width = 4
    c_width = max_y_width_unsigned(a_width, b_width, dim_k, include_carry_from_add=False)
    encoding = Encoding.twos_complement
    n_iter_optimizations = 0

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

    core = MatmulAccumulateComponent(core_config)
    module = core.to_module("matmul_accumulate_core")

    print(f"Output matrix Y has shape: ({dim_m}, {dim_n}) with element width {core.io.Y[0, 0].typ.width} bits")

    vectors = generate_matmul_vectors(core, encoding=encoding, num_vectors=16)

    print("Starting simulation...")
    sim = Simulator(module)
    failures = run_vectors_on_simulator(sim, vectors, use_signed=False, raise_on_fail=True, print_on_pass=False)
    print(f"Simulation complete: {failures} failures")

    print("Getting Yosys metrics...")
    yosys_metrics = get_yosys_metrics(module, n_iter_optimizations=n_iter_optimizations)
    print(f"Yosys metrics: {yosys_metrics}")
    
    # just to show how to export verilog:
    #module.to_verilog("matmul_accumulate_core.v")


if __name__ == "__main__":
    test_mmac_core_basic_simulation()
