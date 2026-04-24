"""Test correctness and transistor counts for Booth-B precompute sharing vs. baseline."""
from __future__ import annotations

import numpy as np

from spirehdl.arithmetic.int_multipliers.eval.multiplier_stage_options_demo_lib import FSAOption, PPAOption, PPGOption
from spirehdl.arithmetic.int_multipliers.eval.testvector_generation import Encoding, EncodingModel
from spirehdl.cores.matmul_accumulate.matmul_accumulate_core import MMAcDims, MMAcWidths, max_y_width_unsigned
from spirehdl.cores.matmul_accumulate.matmul_accumulate_core_fused_precomputed_b_stages import MMAcFusedCfg, MatmulAccumulateComponent, MultiplierConfig
from spirehdl.helpers import get_yosys_metrics
from spirehdl.spirehdl_analyzer import _Analyzer
from spirehdl.spirehdl_simulator import Simulator


def run_test(dim: int, share: bool, seed: int = 42):
    a_width = 8
    b_width = 8
    encoding = Encoding.unsigned
    c_width = max_y_width_unsigned(a_width, b_width, dim, include_carry_from_add=False)

    ppg_opt = PPGOption.BOOTH_OPTIMISED_PRECOMPUTED_B if share else PPGOption.BOOTH_OPTIMISED
    mult_cfg = MultiplierConfig(
        ppg_opt=ppg_opt,
        ppa_opt=PPAOption.WALLACE_TREE,
        fsa_opt=FSAOption.RIPPLE_CARRY,
    )
    dims = MMAcDims(dim_m=dim, dim_n=dim, dim_k=dim)
    widths = MMAcWidths(a_width=a_width, b_width=b_width, c_width=c_width)
    cfg = MMAcFusedCfg(dims=dims, widths=widths, mult_cfg=mult_cfg, encoding=encoding,
                       share_booth_b_precompute=share)

    core = MatmulAccumulateComponent(cfg)
    module_name = f"mmac_booth_{'share' if share else 'no_share'}_{dim}x{dim}"
    module = core.to_module(module_name)

    # Analyzer (fast, no external tools)
    report = _Analyzer(include_wiring=False, include_consts=False, include_reg_cones=False).run(module)

    # Simulation
    rng = np.random.default_rng(seed=seed)
    enc = EncodingModel(encoding)
    a_vals = rng.integers(0, 2**a_width, size=(dim, dim), dtype=np.int64)
    b_vals = rng.integers(0, 2**b_width, size=(dim, dim), dtype=np.int64)
    c_vals = rng.integers(0, 2**c_width, size=(dim, dim), dtype=np.int64)

    sim = Simulator(module)
    for i in range(dim):
        for j in range(dim):
            sim.set(core.io.A[i, j], int(a_vals[i, j]))
            sim.set(core.io.B[i, j], int(b_vals[i, j]))
            sim.set(core.io.C[i, j], int(c_vals[i, j]))
    sim.eval()

    y_hw = np.zeros((dim, dim), dtype=np.int64)
    for i in range(dim):
        for j in range(dim):
            y_hw[i, j] = sim.get(core.io.Y[i, j])

    y_width = core.io.Y[0, 0].typ.width
    y_np = a_vals @ b_vals + c_vals
    y_np_encoded = np.vectorize(lambda x: enc.encode_value(int(x), y_width))(y_np)
    sim_ok = bool(np.array_equal(y_hw, y_np_encoded))

    # Yosys
    metrics = get_yosys_metrics(module)
    transistors = metrics['estimated_num_transistors']

    return sim_ok, transistors, report


if __name__ == "__main__":
    results = {}
    for dim in [4]:
        for share in [False, True]:
            label = f"{dim}x{dim} {'share' if share else 'no_share'}"
            sim_ok, transistors, report = run_test(dim, share)
            results[label] = dict(sim_ok=sim_ok, transistors=transistors, report=report)

    # Print everything at the end
    col_w = 20
    print(f"\n{'Config':<{col_w}} {'sim_ok':<8} {'transistors':>12}  {'op_nodes':>10}  {'Op1':>7}  {'Op2':>8}  {'max_depth':>10}")
    print("-" * 85)
    for label, r in results.items():
        rep = r['report']
        print(
            f"{label:<{col_w}} {str(r['sim_ok']):<8} {r['transistors']:>12,}"
            f"  {rep.op_nodes:>10,}  {rep.by_class.get('Op1', 0):>7,}  {rep.by_class.get('Op2', 0):>8,}  {rep.max_depth:>10}"
        )

# Results (4x4, 8-bit unsigned, Wallace tree + ripple carry):
#
#   Config               sim_ok   transistors    op_nodes      Op1       Op2  max_depth
#   4x4 no_share         True         248,896      43,440    1,280    42,160         41
#   4x4 share            True         248,896      41,280      560    40,720         41
#
# Conclusion:
#   Precomputing the Booth b-decode (use1/use2/neg per group) for each B[k,j] and
#   sharing it across all dim_m rows reduces the sprout IR operator count by ~5%
#   (−2,160 op nodes) and cuts Op1 (NOT/INV) nodes by 56% (1,280 → 560), since the
#   decode expressions are counted once instead of dim_m times.
#
#   However, the Yosys transistor count is identical (248,896) in both cases.
#   Yosys performs its own global CSE during synthesis and already merges the
#   structurally identical b-decode subgraphs regardless of whether they are shared
#   at the IR level. The sharing therefore does not translate to a smaller netlist.
