from __future__ import annotations

from spirehdl.arithmetic.int_multipliers.eval.multiplier_stage_options_demo_lib import FSAOption, PPAOption, PPGOption
from spirehdl.arithmetic.int_multipliers.eval.testvector_generation import Encoding, is_signed
from spirehdl.cores.matmul_accumulate.matmul_accumulate_core import MMAcDims, MMAcWidths, max_y_width_unsigned
from spirehdl.cores.matmul_accumulate.matmul_accumulate_core_fused import MMAcFusedCfg, MatmulAccumulateComponent, MultiplierConfig
from spirehdl.cores.matmul_accumulate.matmul_test_vectors import generate_matmul_vectors
from spirehdl.helpers import get_yosys_metrics, run_vectors_on_simulator
from spirehdl.spirehdl_simulator import Simulator


def test_mmac_core_basic_simulation():
    dim = 4
    a_width = 8
    b_width = 8
    c_width = max_y_width_unsigned(a_width, b_width, dim, include_carry_from_add=False)
    encoding = Encoding.twos_complement

    mult_cfg = MultiplierConfig(
        ppg_opt=PPGOption.BAUGH_WOOLEY if is_signed(encoding) else PPGOption.AND,
        ppa_opt=PPAOption.WALLACE_TREE,
        fsa_opt=FSAOption.RIPPLE_CARRY,
    )

    dims = MMAcDims(dim_m=dim, dim_n=dim, dim_k=dim)
    widths = MMAcWidths(a_width=a_width, b_width=b_width, c_width=c_width)
    cfg = MMAcFusedCfg(dims=dims, widths=widths, mult_cfg=mult_cfg, encoding=encoding)

    core = MatmulAccumulateComponent(cfg)
    module = core.to_module("matmul_accumulate_core_fused")
    print(f"Output matrix Y has shape: ({dim}, {dim}) with element width {core.io.Y[0, 0].typ.width} bits")

    vectors = generate_matmul_vectors(core, encoding=encoding, num_vectors=16)

    sim = Simulator(module)
    failures = run_vectors_on_simulator(sim, vectors, use_signed=False, raise_on_fail=True, print_on_pass=False)
    print(f"Simulation complete: {failures} failures")

    yosys_metrics = get_yosys_metrics(module)
    print(f"Yosys metrics: {yosys_metrics}")


if __name__ == "__main__":
    test_mmac_core_basic_simulation()
