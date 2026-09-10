"""Matmul-accumulate candidates for the spire.design_db.lean and design_db Lean-tier tests: Spire's balanced
`MatmulAccumulateComponent` (golden) and a left-deep rewrite, 2x2x4 signed 8-bit, C 20-bit.
"""
from __future__ import annotations
import random
from typing import Callable, Dict

from spire.arithmetic.int_arithmetic_config import AdderConfig, MultiplierConfig, build_adder, build_multiplier
from spire.arithmetic.int_multipliers.eval.multiplier_stage_options_demo_lib import TwoInputAritEncodings
from spire.arithmetic.int_multipliers.eval.testvector_generation import Encoding
from spire.composite.array import Array
from spire.cores.matmul_accumulate.matmul_accumulate_core import MMAcCfg, MMAcDims, MMAcWidths, MatmulAccumulateComponent
from spire.expr import Op2, Signal
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from lean_matmul_helper import MatmulContract
from spire.simulator import Simulator
from spire.visitor import expr_children

ENC = Encoding.twos_complement
CFG = MMAcCfg(
    dims=MMAcDims(dim_m=2, dim_n=2, dim_k=4),
    widths=MMAcWidths(a_width=8, b_width=8, c_width=20),
    mult_cfg=MultiplierConfig(use_operator=True, encodings=TwoInputAritEncodings.with_enc(ENC)),
    add_cfg=AdderConfig(use_operator=True, encoding=ENC, full_output_bit=True),
)
CONTRACT = MatmulContract(m=2, n=2, k=4, a_width=8, b_width=8, c_width=20, y_width=21)


class LeftDeepMatmulAccumulate(MatmulAccumulateComponent):
    """Architecture rewrite: replace Spire's balanced adder_tree by a left-deep chain."""

    def elaborate(self):
        rows = []
        for i in range(self.cfg.dims.dim_m):
            row = []
            for j in range(self.cfg.dims.dim_n):
                products = [build_multiplier(self.A[i, k], self.B[k, j], self.cfg.mult_cfg)
                            for k in range(self.cfg.dims.dim_k)]
                dot = products[0]
                for p in products[1:]:
                    dot = build_adder(dot, p, self.cfg.add_cfg)
                acc = build_adder(self.C[i, j], dot, self.cfg.add_cfg)
                y = Signal(name=f'y_{i}_{j}', typ=self.io_hdl_type(acc.typ.width), kind='output')
                y <<= acc
                row.append(y)
            rows.append(Array(row))
        self.Y = Array(rows)


class CommutedMulMatmulAccumulate(LeftDeepMatmulAccumulate):
    """Left-deep with every multiplier's operands swapped (B*A): a local delta vs LeftDeepMatmulAccumulate
    that yields a different netlist/AIG (a commuted addition would dedup to the same structure)."""

    def elaborate(self):
        rows = []
        for i in range(self.cfg.dims.dim_m):
            row = []
            for j in range(self.cfg.dims.dim_n):
                products = [build_multiplier(self.B[k, j], self.A[i, k], self.cfg.mult_cfg)
                            for k in range(self.cfg.dims.dim_k)]
                dot = products[0]
                for p in products[1:]:
                    dot = build_adder(dot, p, self.cfg.add_cfg)
                acc = build_adder(self.C[i, j], dot, self.cfg.add_cfg)
                y = Signal(name=f'y_{i}_{j}', typ=self.io_hdl_type(acc.typ.width), kind='output')
                y <<= acc
                row.append(y)
            rows.append(Array(row))
        self.Y = Array(rows)


# candidate name -> constructor.
CANDIDATES: Dict[str, Callable[[], MatmulAccumulateComponent]] = {
    'BalancedMmac': lambda: MatmulAccumulateComponent(CFG, signed_io_type=True),
    'LeftDeepMmac': lambda: LeftDeepMatmulAccumulate(CFG, signed_io_type=True),
    'CommutedMulMmac': lambda: CommutedMulMatmulAccumulate(CFG, signed_io_type=True),
}



def add_depth(net) -> int:
    memo = {}
    def depth(e):
        if e is None: return 0
        if id(e) in memo: return memo[id(e)]
        child = max([depth(c) for c in expr_children(e)] or [0])
        d = child + (1 if isinstance(e, Op2) and e.op == '+' else 0)
        memo[id(e)] = d
        return d
    return max(depth(p) for p in net._ports if p.kind == 'output')


def run_sim(core, vectors: int = 10000, seed: int = 17) -> int:
    """Simulate `vectors` random full-range cases against C + A*B; raises on the first mismatch."""
    d = core.cfg.dims
    net = core.to_netlist('sim')
    sim = Simulator(net); rng = random.Random(seed)
    cmin, cmax = -(1 << (CFG.widths.c_width - 1)), (1 << (CFG.widths.c_width - 1)) - 1
    for _ in range(vectors):
        A = [[rng.randint(-128, 127) for _ in range(d.dim_k)] for _ in range(d.dim_m)]
        B = [[rng.randint(-128, 127) for _ in range(d.dim_n)] for _ in range(d.dim_k)]
        C = [[rng.randint(cmin, cmax) for _ in range(d.dim_n)] for _ in range(d.dim_m)]
        for i in range(d.dim_m):
            for k in range(d.dim_k): sim.set(core.io.A[i, k], A[i][k])
        for k in range(d.dim_k):
            for j in range(d.dim_n): sim.set(core.io.B[k, j], B[k][j])
        for i in range(d.dim_m):
            for j in range(d.dim_n): sim.set(core.io.C[i, j], C[i][j])
        sim.eval()
        for i in range(d.dim_m):
            for j in range(d.dim_n):
                ref = C[i][j] + sum(A[i][k] * B[k][j] for k in range(d.dim_k))
                got = sim.get(core.io.Y[i, j])
                if got != ref:
                    raise AssertionError((i, j, A, B, C, got, ref))
    return vectors
