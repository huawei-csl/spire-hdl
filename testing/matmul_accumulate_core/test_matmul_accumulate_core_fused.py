from __future__ import annotations

import pytest

from spire.arithmetic.int_multipliers.eval.multiplier_stage_options_demo_lib import FSAOption, PPAOption, PPGOption
from spire.arithmetic.int_multipliers.eval.testvector_generation import Encoding, is_signed
from spire.cores.matmul_accumulate.matmul_accumulate_core import MMAcDims, MMAcWidths, max_y_width_unsigned
from spire.cores.matmul_accumulate.matmul_accumulate_core_fused import MMAcFusedCfg, MatmulAccumulateComponent, MultiplierConfig
from spire.cores.matmul_accumulate.matmul_test_vectors import generate_matmul_vectors
from spire.helpers import get_yosys_metrics, run_vectors_on_simulator
from spire.simulator import Simulator


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


# Regression sweep covering:
#   - BAUGH_WOOLEY fused upper correction for arbitrary (incl. non-power-of-2) dim_k
#   - BOOTH_UNOPTIMISED in the fused path (bbit(-1) and sign-extension fixes)
#   - BOOTH_OPTIMISED in the fused path (known broken at non-power-of-2 dim_k --
#     its per-term BW-style trick is calibrated for the natural width 2n
#     wraparound and emits a signal-dependent bias when out_bits > 2n that cannot
#     be cancelled by a constant fused correction; those cases are xfailed below).
# The full cross-product checks each PPG against both signed and unsigned encodings
# (where the PPG supports them) and the signed_io_type=False/True input typings,
# across power-of-2 and non-power-of-2 dim_k.
@pytest.mark.parametrize("ppg_opt", [
    PPGOption.BAUGH_WOOLEY,
    PPGOption.BOOTH_UNOPTIMISED,
    PPGOption.BOOTH_OPTIMISED,
])
@pytest.mark.parametrize("encoding", [Encoding.twos_complement, Encoding.unsigned])
@pytest.mark.parametrize("signed_io_type", [False, True])
@pytest.mark.parametrize("dim_k", [2, 3, 4, 5, 7, 8])
def test_fused_matmul_ppg_dim_k_sweep(ppg_opt, encoding, signed_io_type, dim_k, request):
    if ppg_opt == PPGOption.BAUGH_WOOLEY and encoding == Encoding.unsigned:
        pytest.skip("BAUGH_WOOLEY only supports signed encodings")
    if ppg_opt == PPGOption.BOOTH_OPTIMISED:
        # BOOTH_OPTIMISED's per-term BW trick is calibrated for out_bits == wa+wb.
        # In the fused matmul out_bits = result_width > wa+wb, which exposes a
        # signal-dependent bias the natural-width algebra cannot cancel. See
        # docs/booth_optim_fused_upper_correction.md.
        request.applymarker(pytest.mark.xfail(
            reason="BOOTH_OPTIMISED unreliable in fused path for out_bits > wa+wb",
            strict=False,
        ))

    mult_cfg = MultiplierConfig(
        ppg_opt=ppg_opt,
        ppa_opt=PPAOption.WALLACE_TREE,
        fsa_opt=FSAOption.RIPPLE_CARRY,
    )
    dims = MMAcDims(dim_m=2, dim_n=2, dim_k=dim_k)
    widths = MMAcWidths(a_width=4, b_width=4, c_width=10)
    cfg = MMAcFusedCfg(dims=dims, widths=widths, mult_cfg=mult_cfg, encoding=encoding)

    core = MatmulAccumulateComponent(cfg, signed_io_type=signed_io_type)
    module = core.to_module(f"mm_{ppg_opt.name}_{encoding.name}_sio{int(signed_io_type)}_k{dim_k}")
    vectors = generate_matmul_vectors(core, encoding=encoding, num_vectors=8)
    sim = Simulator(module)
    run_vectors_on_simulator(sim, vectors, use_signed=False, raise_on_fail=True, print_on_pass=False)


if __name__ == "__main__":
    test_mmac_core_basic_simulation()
