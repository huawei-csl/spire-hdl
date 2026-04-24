from __future__ import annotations

from sprouthdl.arithmetic.int_arithmetic_config import AdderConfig, MultiplierConfig
from sprouthdl.arithmetic.int_multipliers.eval.multiplier_stage_options_demo_lib import (
    FSAOption, MultiplierOption, PPAOption, PPGOption, TwoInputAritEncodings,
)
from sprouthdl.arithmetic.int_multipliers.eval.testvector_generation import Encoding, is_signed
from sprouthdl.cores.matmul_accumulate.matmul_accumulate_core import MMAcCfg, MMAcDims, MMAcWidths, max_y_width_unsigned
from sprouthdl.cores.matmul_accumulate.matmul_accumulate_core_pipelined import MatmulAccumulatePipelinedComponent
from sprouthdl.cores.matmul_accumulate.matmul_test_vectors import generate_matmul_vectors
from sprouthdl.helpers import get_yosys_metrics, run_vectors_on_simulator
from sprouthdl.sprouthdl_simulator import Simulator


def _build_pipelined_core():
    dim_m = 4
    dim_n = 4
    dim_k = 4
    a_width = 4
    b_width = 4
    c_width = max_y_width_unsigned(a_width, b_width, dim_k, include_carry_from_add=False)
    encoding = Encoding.twos_complement

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

    core = MatmulAccumulatePipelinedComponent(core_config)
    module = core.to_module("matmul_accumulate_core_pipelined", with_clock=True, with_reset=False)
    return core, module, encoding


def test_mmac_core_pipelined_basic_simulation():
    core, module, encoding = _build_pipelined_core()

    print(f"Output matrix Y has shape: ({len(core.io.Y)}, {len(core.io.Y[0])}) "
          f"with element width {core.io.Y[0, 0].typ.width} bits")

    vectors = generate_matmul_vectors(core, encoding=encoding, num_vectors=16)

    print("Starting simulation...")
    sim = Simulator(module)
    failures = run_vectors_on_simulator(
        sim, vectors, use_signed=False, raise_on_fail=True, print_on_pass=False, with_clk=True,
    )
    print(f"Simulation complete: {failures} failures")

    print("Getting Yosys metrics...")
    yosys_metrics = get_yosys_metrics(module, n_iter_optimizations=0)
    print(f"Yosys metrics: {yosys_metrics}")


def test_mmac_core_pipelined_has_pipeline_registers():
    """Smoke-check that pipeline registers actually exist between mult and add."""
    core, module, _ = _build_pipelined_core()

    reg_signals = [s for s in module._signals if s.kind == "reg"]
    assert len(reg_signals) > 0, "expected pipeline registers in module"

    # One register per (i,j,k) product term: dim_m * dim_n * dim_k.
    expected = core.cfg.dims.dim_m * core.cfg.dims.dim_n * core.cfg.dims.dim_k
    prod_regs = [s for s in reg_signals if "prod_reg_" in s.name]
    assert len(prod_regs) == expected, f"expected {expected} product pipeline registers, got {len(prod_regs)}"


def test_mmac_core_pipelined_latency_behavior():
    """Verify the pipeline registers introduce one cycle of delay on the A*B path.

    With pipeline regs: at cycle n, regs hold A_{n}*B_{n} captured during the step (the simulator samples current
    inputs into the registers when advancing the clock). If after that step we change A,B but only call eval() (no
    clock edge), the products in the regs must NOT update — they are sequential state, only the C path is
    combinational.
    """
    core, module, encoding = _build_pipelined_core()
    vectors = generate_matmul_vectors(core, encoding=encoding, num_vectors=2)

    sim = Simulator(module)

    # Cycle 1: load vector 0, step, eval -> Y matches vector 0's expected.
    name0, ins0, outs0 = vectors[0]
    for k, v in ins0.items():
        if k[0] != "_":
            sim.set(k, v)
    sim.step()
    sim.eval()
    for oname, exp in outs0.items():
        if oname[0] == "_":
            continue
        got = sim.peek(oname)
        assert got == exp, f"cycle 1 mismatch on {oname}: got=0x{got:X} exp=0x{exp:X}"

    # Now change A,B,C to vector 1 *without* stepping the clock. Pipeline regs still hold vector 0's products. The
    # output Y is (registered v0 products) + (current C from v1) — which differs from both v0's and v1's expected Y.
    _, ins1, outs1 = vectors[1]
    for k, v in ins1.items():
        if k[0] != "_":
            sim.set(k, v)
    sim.eval()

    a_name = core.io.A[0, 0].name
    b_name = core.io.B[0, 0].name
    if ins0[a_name] != ins1[a_name] or ins0[b_name] != ins1[b_name]:
        # If A or B changed, at least one Y output must differ from v1's expected (regs still hold v0 products).
        any_diff = any(sim.peek(oname) != exp for oname, exp in outs1.items() if oname[0] != "_")
        assert any_diff, "pipeline appears to be transparent — registers may be missing"


if __name__ == "__main__":
    test_mmac_core_pipelined_basic_simulation()
    test_mmac_core_pipelined_has_pipeline_registers()
    test_mmac_core_pipelined_latency_behavior()
