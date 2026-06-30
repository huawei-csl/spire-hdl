"""Output-stationary tiled signed matmul accelerator (spire.cores.output_stationary_matmul).

Drives the accelerator end-to-end through its `mode` + RAM-access interface: load A/B tiles into
the input RAM (access mode), run the matmul (compute mode), read C tiles back (access mode), and
compare to a signed reference. Also checks external RAM access, accumulator-width validation, the
host pack/unpack helpers, and that the full 64×64 instance elaborates.
"""
import random

import pytest

from spire import Simulator
from spire.cores.output_stationary_matmul import (
    TiledMatmul, TiledMatmulConfig, pack_tile, unpack_tile, _ceil_log2,
)


def _reference(A, B, M, N, K):
    """Signed integer C = A · B."""
    return [[sum(A[i][k] * B[k][j] for k in range(K)) for j in range(N)] for i in range(M)]


def _rand_matrix(rows, cols, in_w, rng):
    lo, hi = -(1 << (in_w - 1)), (1 << (in_w - 1)) - 1
    return [[rng.randint(lo, hi) for _ in range(cols)] for _ in range(rows)]


def _load_input(sim, dut, A, B):
    """Write all A/B tiles into the input RAM via the external access port (mode=0)."""
    cfg, T, iw = dut.cfg, dut.cfg.T, dut.cfg.in_width
    sim.set("mode", 0)
    for ti in range(cfg.MT):
        for tk in range(cfg.KT):
            tile = [[A[ti * T + r][tk * T + c] for c in range(T)] for r in range(T)]
            sim.set("in_addr", dut.a_tile_addr(ti, tk)).set("in_wdata", pack_tile(tile, T, iw)).set("in_wen", 1).step()
    for tk in range(cfg.KT):
        for tj in range(cfg.NT):
            tile = [[B[tk * T + r][tj * T + c] for c in range(T)] for r in range(T)]
            sim.set("in_addr", dut.b_tile_addr(tk, tj)).set("in_wdata", pack_tile(tile, T, iw)).set("in_wen", 1).step()
    sim.set("in_wen", 0)


def _compute(sim, dut):
    """Pulse start in compute mode and step until busy clears; returns the cycle count."""
    cfg = dut.cfg
    sim.set("mode", 1).set("start", 1).step()
    sim.set("start", 0)
    cyc, bound = 0, cfg.MT * cfg.NT * cfg.KT + 50
    while sim.peek("busy") == 1 and cyc < bound:
        sim.step()
        cyc += 1
    assert sim.peek("busy") == 0, "compute did not finish within the cycle bound"
    return cyc


def _read_output(sim, dut):
    """Read all C tiles back through the external access port (mode=0)."""
    cfg, T, aw = dut.cfg, dut.cfg.T, dut.cfg.acc_width
    C = [[0] * cfg.N for _ in range(cfg.M)]
    sim.set("mode", 0)
    for ti in range(cfg.MT):
        for tj in range(cfg.NT):
            sim.set("out_addr", dut.c_tile_addr(ti, tj)).eval()
            tile = unpack_tile(sim.peek("out_rdata"), T, aw)
            for r in range(T):
                for c in range(T):
                    C[ti * T + r][tj * T + c] = tile[r][c]
    return C


def _run(cfg, seed=0):
    """Full flow: build, load, compute, read back; return (C_hw, C_ref, cycles)."""
    dut = TiledMatmul(cfg)
    sim = Simulator(dut)
    rng = random.Random(seed)
    A = _rand_matrix(cfg.M, cfg.K, cfg.in_width, rng)
    B = _rand_matrix(cfg.K, cfg.N, cfg.in_width, rng)
    _load_input(sim, dut, A, B)
    cycles = _compute(sim, dut)
    return _read_output(sim, dut), _reference(A, B, cfg.M, cfg.N, cfg.K), cycles


# ---------------------------------------------------------------------------
# Correctness — full simulation vs signed reference, several parametrizations
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("M,N,K,T,w", [
    (8, 8, 8, 4, 8),       # square, 2×2×2 tiles
    (16, 16, 16, 4, 8),    # square, 4×4×4 tiles
    (8, 12, 16, 4, 6),     # non-square dims, narrower inputs
    (4, 4, 4, 4, 8),       # exactly one tile (single core op)
    (12, 8, 4, 2, 5),      # T=2 tiles
])
def test_matmul_correct(M, N, K, T, w):
    cfg = TiledMatmulConfig(M=M, N=N, K=K, T=T, in_width=w)
    C_hw, C_ref, cycles = _run(cfg, seed=M * 7 + N * 3 + K)
    assert C_hw == C_ref
    assert cycles == cfg.MT * cfg.NT * cfg.KT     # output-stationary: KT cycles per output tile


def test_matmul_handles_negative_values():
    # Force a tile of strongly-negative operands so two's-complement products/sums are exercised.
    cfg = TiledMatmulConfig(M=8, N=8, K=8, T=4, in_width=8)
    dut = TiledMatmul(cfg)
    sim = Simulator(dut)
    A = [[-128 if (i + j) % 2 == 0 else 127 for j in range(8)] for i in range(8)]
    B = [[-128 if (i * j) % 3 == 0 else -7 for j in range(8)] for i in range(8)]
    _load_input(sim, dut, A, B)
    _compute(sim, dut)
    C_hw = _read_output(sim, dut)
    assert C_hw == _reference(A, B, 8, 8, 8)
    assert any(v < 0 for row in C_hw for v in row)   # negatives actually present


# ---------------------------------------------------------------------------
# External RAM access (mode = 0)
# ---------------------------------------------------------------------------

def test_input_ram_external_readback():
    cfg = TiledMatmulConfig(M=8, N=8, K=8, T=4, in_width=8)
    dut = TiledMatmul(cfg)
    sim = Simulator(dut)
    sim.set("mode", 0)
    word = pack_tile([[1, -2, 3, -4], [5, 6, -7, 8], [-9, 10, 11, -12], [13, -14, 15, 16]], 4, 8)
    sim.set("in_addr", 0).set("in_wdata", word).set("in_wen", 1).step()
    sim.set("in_wen", 0)
    sim.set("in_addr", 0).eval()
    assert sim.peek("in_rdata") == word      # combinational read returns what we wrote


def test_compute_does_not_disturb_during_access_mode():
    # With mode=0, start has no effect (compute and access are mutually exclusive).
    cfg = TiledMatmulConfig(M=4, N=4, K=4, T=4, in_width=8)
    dut = TiledMatmul(cfg)
    sim = Simulator(dut)
    sim.set("mode", 0).set("start", 1).step().step()
    assert sim.peek("busy") == 0              # never starts while in access mode


# ---------------------------------------------------------------------------
# Config validation + host helpers
# ---------------------------------------------------------------------------

def test_default_acc_width_is_safe_for_64():
    cfg = TiledMatmulConfig()    # 64×64×64, 8-bit
    assert cfg.acc_width == 2 * 8 + _ceil_log2(64) + 1     # 23
    assert cfg.acc_width >= 22   # signed worst case |sum| = 64*2^14 = 2^20 needs 22 bits


def test_acc_width_too_small_raises():
    with pytest.raises(ValueError):
        TiledMatmulConfig(M=64, N=64, K=64, T=4, in_width=8, acc_width=16)


def test_dims_must_be_multiple_of_tile():
    with pytest.raises(ValueError):
        TiledMatmulConfig(M=10, N=8, K=8, T=4)


def test_pack_unpack_roundtrip():
    tile = [[1, -2, 127], [-128, 0, 63], [-1, 64, -64]]
    word = pack_tile(tile, 3, 8)
    assert unpack_tile(word, 3, 8) == tile


# ---------------------------------------------------------------------------
# The full 64×64 instance elaborates (build / structural check; the full
# ~4096-cycle simulation lives in a slow opt-in test below).
# ---------------------------------------------------------------------------

def test_build_64x64():
    cfg = TiledMatmulConfig()    # 64×64×64, T=4
    dut = TiledMatmul(cfg)
    assert (cfg.MT, cfg.NT, cfg.KT) == (16, 16, 16)
    assert dut.in_elem_w == 4 * 4 * 8           # one A/B tile word
    assert dut.out_elem_w == 4 * 4 * cfg.acc_width
    assert dut.in_depth == 16 * 16 + 16 * 16    # A region + B region
    assert dut.out_depth == 16 * 16
    nl = dut.to_netlist("tiled_matmul_64")      # lowers/elaborates the whole accelerator
    names = {p.name for p in nl._ports}
    assert {"mode", "start", "busy", "in_addr", "in_wdata", "in_wen", "in_rdata",
            "out_addr", "out_rdata"} <= names


def test_matmul_64x64_full_sim():
    # The headline case: full 64×64 · 64×64, simulated end-to-end (~4096 compute cycles, a few seconds).
    cfg = TiledMatmulConfig()    # 64×64×64, T=4
    C_hw, C_ref, cycles = _run(cfg, seed=7)
    assert C_hw == C_ref
    assert cycles == 16 * 16 * 16


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
