# Output-stationary tiled matmul accelerator

A parametrizable signed (two's-complement) matrix-multiply accelerator that computes `C = A · B`
(`A`: M×K, `B`: K×N) by streaming `T×T` tiles through **one** combinational `T×T×T` tile engine and
accumulating each output tile *in place* over the K dimension (output-stationary). Operands and
results live in on-chip RAMs; a `mode` input selects between external RAM access and internal
compute — the two never run at once, so the top-level interface is just `mode` + the RAM ports.

Lives in [`spire/cores/output_stationary_matmul.py`](../src/spire/cores/output_stationary_matmul.py)
and reuses the existing
[`MatmulAccumulateComponent`](../src/spire/cores/matmul_accumulate/matmul_accumulate_core.py)
(`Y = A·B + C`, combinational) as the tile engine.

```python
from spire.cores.output_stationary_matmul import (
    TiledMatmul, TiledMatmulConfig, pack_tile, unpack_tile)
```

## How it works

For each output tile `(ti, tj)` the engine streams `tk = 0 … KT-1`, feeding the core `C = 0` on the
first K-tile and the **running accumulator** afterwards; on the last K-tile it writes the finished
tile to the output RAM. RAM reads are combinational, so it sustains **one K-step per cycle** —
`MT·NT·KT` cycles total (e.g. `16·16·16 = 4096` for 64×64×64 with `T=4`).

```
for ti in 0..MT:                       # output tile row
  for tj in 0..NT:                     # output tile col      ── one output tile is "stationary":
    acc = 0                            #                          it stays in a register while…
    for tk in 0..KT:                   # K-tile               ── …the K dimension streams through
      acc = core(A[ti,tk], B[tk,tj], acc)    # Y = A·B + acc  (the existing 4×4×4 core)
    out_ram[ti,tj] = acc
```

## Memory layout

Tiles are packed **row-major, element 0 = LSBs** (matching `Array.to_bits()` and the core's leaf
order); `pack_tile` / `unpack_tile` are the exact host-side inverses. Addressing is pure index
arithmetic — A is stored row-major and **B column-major** so the K-tiles one output column needs are
contiguous:

```
 input RAM (word = one packed T×T tile, T·T·in_width bits, depth MT·KT + KT·NT)
 ┌──────────────────────────────┬──────────────────────────────────────────┐
 │ A tiles  (row-major)         │ B tiles  (column-major)                  │
 │ addr = ti·KT + tk            │ addr = MT·KT + tj·KT + tk                │
 └──────────────────────────────┴──────────────────────────────────────────┘

 output RAM (word = one packed T×T accumulator tile, T·T·acc_width bits, depth MT·NT)
 ┌──────────────────────────────┐
 │ C tiles  (row-major)         │   addr = ti·NT + tj
 └──────────────────────────────┘
```

The host helpers `dut.a_tile_addr(ti, tk)`, `dut.b_tile_addr(tk, tj)`, `dut.c_tile_addr(ti, tj)`
return these addresses.

## Top-level interface

| Port | Dir | Meaning |
|---|---|---|
| `mode` | in | `0` = external RAM access, `1` = compute |
| `start` | in | pulse (with `mode=1`) to begin a matmul |
| `busy` | out | high while computing |
| `in_addr` / `in_wdata` / `in_wen` | in | write a tile word into the input RAM (access mode) |
| `in_rdata` | out | read a tile word back from the input RAM |
| `out_addr` | in | output-RAM tile address |
| `out_rdata` | out | read a C tile word back |

`mode` muxes every RAM port between the external agent and the internal controller, so compute and
external access are mutually exclusive (a `start` while `mode=0` is ignored).

## Configuration

```python
TiledMatmulConfig(M=64, N=64, K=64, T=4, in_width=8, acc_width=None)
```

- `M, N, K` — matrix dims (each a multiple of `T`).
- `T` — tile size (the engine is `T×T×T`; `4` reuses the 4×4×4 core).
- `in_width` — signed element width of A and B.
- `acc_width` — accumulator/output width. **Defaults to `2·in_width + ⌈log₂K⌉ + 1`**, the smallest
  width that cannot overflow the signed K-reduction (`|Σ| ≤ K·2^(2·in_width−2)`); a smaller override
  raises. The output RAM stores C at full `acc_width`.

Everything is signed two's complement end-to-end (the core is built with `Encoding.twos_complement`
and `signed_io_type=True`).

## Usage

```python
from spire import Simulator
from spire.cores.output_stationary_matmul import (
    TiledMatmul, TiledMatmulConfig, pack_tile, unpack_tile)

cfg = TiledMatmulConfig(M=64, N=64, K=64, T=4, in_width=8)   # acc_width -> 23
dut = TiledMatmul(cfg)
sim = Simulator(dut)
T = cfg.T

# 1) ACCESS mode — load A and B tiles into the input RAM
sim.set("mode", 0)
for ti in range(cfg.MT):
    for tk in range(cfg.KT):
        a = [[A[ti*T+r][tk*T+c] for c in range(T)] for r in range(T)]
        sim.set("in_addr", dut.a_tile_addr(ti, tk)) \
           .set("in_wdata", pack_tile(a, T, cfg.in_width)).set("in_wen", 1).step()
for tk in range(cfg.KT):
    for tj in range(cfg.NT):
        b = [[B[tk*T+r][tj*T+c] for c in range(T)] for r in range(T)]
        sim.set("in_addr", dut.b_tile_addr(tk, tj)) \
           .set("in_wdata", pack_tile(b, T, cfg.in_width)).set("in_wen", 1).step()
sim.set("in_wen", 0)

# 2) COMPUTE mode — pulse start, run until done
sim.set("mode", 1).set("start", 1).step()
sim.set("start", 0)
while sim.peek("busy"):
    sim.step()

# 3) ACCESS mode — read the C tiles back
sim.set("mode", 0)
C = [[0] * cfg.N for _ in range(cfg.M)]
for ti in range(cfg.MT):
    for tj in range(cfg.NT):
        sim.set("out_addr", dut.c_tile_addr(ti, tj)).eval()
        tile = unpack_tile(sim.peek("out_rdata"), T, cfg.acc_width)
        for r in range(T):
            for c in range(T):
                C[ti*T+r][tj*T+c] = tile[r][c]
# C == A @ B   (signed)
```

## See also

- End-to-end tests (correctness across parametrizations + the full 64×64 sim):
  [`testing/test_output_stationary_matmul.py`](../testing/test_output_stationary_matmul.py)
- The tile engine: [`matmul_accumulate_core.py`](../src/spire/cores/matmul_accumulate/matmul_accumulate_core.py)
- RAM primitive: [`README_memories.md`](README_memories.md)
