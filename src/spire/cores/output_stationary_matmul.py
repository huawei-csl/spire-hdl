"""Output-stationary tiled integer matmul accelerator (signed / two's complement).

Computes ``C = A · B`` for ``A`` (M×K), ``B`` (K×N) using a single **combinational** T×T×T
:class:`~spire.cores.matmul_accumulate.matmul_accumulate_core.MatmulAccumulateComponent` as the tile
engine, accumulating each T×T output tile *in place* over the K dimension (output-stationary).
Operands and results live in on-chip RAMs; a ``mode`` input selects between **external RAM access**
and **internal compute** — the two never run at the same time, so the top-level IO is just
``mode`` + the RAM access ports.

Heavily parametrizable via :class:`TiledMatmulConfig` (matrix dims ``M,N,K``, tile size ``T``,
input element width, accumulator width). All arithmetic is signed two's complement.

Memory layout — chosen so tile addressing is pure index arithmetic (no division):

* **one input RAM**, word = one packed T×T tile (``T*T*in_width`` bits):
    - A tiles **row-major**:    ``addr = ti*KT + tk``                  (ti∈[0,MT), tk∈[0,KT))
    - B tiles **column-major**: ``addr = MT*KT + tj*KT + tk``          (tj∈[0,NT), tk∈[0,KT))
  Storing B column-major makes the KT tiles that one output column consumes contiguous, matching
  the output-stationary inner loop over k.
* **one output RAM**, word = one packed T×T accumulator tile (``T*T*acc_width`` bits):
    - C tiles **row-major**:    ``addr = ti*NT + tj``

Tiles are packed **row-major, element 0 = LSBs** (matches ``Array.to_bits()`` and the core's leaf
order), so :func:`pack_tile` / :func:`unpack_tile` are exact inverses of the hardware packing.

Schedule: one K-step per cycle (RAM reads are combinational). For output tile ``(ti,tj)`` the engine
streams ``tk = 0..KT-1``, feeding ``C = 0`` on the first K-tile and the running accumulator after,
and writes the completed tile to the output RAM on the last K-tile. Total ≈ ``MT*NT*KT`` cycles.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from spire.component import Component
from spire.io_record import IORecord, Input, Output
from spire.expr import Bool, Register, SInt, UInt, fit_width, mux
from spire.control_structures import if_, elif_
from spire.composite.array import Array
from spire.primitives.primitive_ram import RamPrimitive
from spire.cores.matmul_accumulate.matmul_accumulate_core import (
    MatmulAccumulateComponent, MMAcCfg, MMAcDims, MMAcWidths,
)
from spire.arithmetic.int_arithmetic_config import MultiplierConfig, AdderConfig
from spire.arithmetic.int_multipliers.eval.testvector_generation import Encoding


def _ceil_log2(n: int) -> int:
    return 0 if n <= 1 else (n - 1).bit_length()


@dataclass
class TiledMatmulConfig:
    """Configuration for :class:`TiledMatmul`. All sizes in elements; widths in bits."""
    M: int = 64
    N: int = 64
    K: int = 64
    T: int = 4
    in_width: int = 8
    acc_width: Optional[int] = None     # default: large enough for the signed K-reduction

    def __post_init__(self):
        if self.T <= 0:
            raise ValueError("T must be > 0")
        for d, nm in ((self.M, "M"), (self.N, "N"), (self.K, "K")):
            if d <= 0 or d % self.T != 0:
                raise ValueError(f"{nm}={d} must be a positive multiple of T={self.T}")
        self.MT = self.M // self.T
        self.NT = self.N // self.T
        self.KT = self.K // self.T
        # Signed worst case: |a·b| ≤ 2^(2w-2); the full-K sum needs 2w + ceil_log2(K) bits.
        # +1 of headroom. This is the minimum legal accumulator width.
        safe = 2 * self.in_width + _ceil_log2(self.K) + 1
        if self.acc_width is None:
            self.acc_width = safe
        elif self.acc_width < safe:
            raise ValueError(
                f"acc_width={self.acc_width} too small; need ≥ {safe} bits so the signed "
                f"K={self.K} reduction of {self.in_width}-bit inputs cannot overflow"
            )


# ---------------------------------------------------------------------------
# Host-side tile (un)packing — exact inverse of the hardware Array.to_bits() order
# (row-major, element 0 = LSBs), with two's-complement element encoding.
# ---------------------------------------------------------------------------

def pack_tile(tile: List[List[int]], T: int, bits: int) -> int:
    """Pack a T×T row-major matrix of signed ints into one RAM word (two's complement)."""
    mask = (1 << bits) - 1
    word = 0
    for r in range(T):
        for c in range(T):
            word |= (tile[r][c] & mask) << ((r * T + c) * bits)
    return word


def unpack_tile(word: int, T: int, bits: int) -> List[List[int]]:
    """Unpack one RAM word into a T×T row-major matrix of signed ints."""
    mask = (1 << bits) - 1
    half = 1 << (bits - 1)
    out = []
    for r in range(T):
        row = []
        for c in range(T):
            v = (word >> ((r * T + c) * bits)) & mask
            row.append(v - (1 << bits) if v >= half else v)
        out.append(row)
    return out


class TiledMatmul(Component):
    """Output-stationary tiled signed matmul; see module docstring."""

    def __init__(self, cfg: Optional[TiledMatmulConfig] = None):
        self.cfg = cfg or TiledMatmulConfig()
        c = self.cfg
        self.in_elem_w = c.T * c.T * c.in_width
        self.out_elem_w = c.T * c.T * c.acc_width
        self.in_depth = c.MT * c.KT + c.KT * c.NT      # A region then B region
        self.out_depth = c.MT * c.NT
        self.in_addr_w = max(1, (self.in_depth - 1).bit_length())
        self.out_addr_w = max(1, (self.out_depth - 1).bit_length())

        self.io = IORecord(
            mode=Input(Bool()),          # 0 = external RAM access, 1 = compute
            start=Input(Bool()),         # pulse (with mode=1) to begin a matmul
            busy=Output(Bool()),         # high while computing
            # external access to the input RAM (tile-granular words)
            in_addr=Input(UInt(self.in_addr_w)),
            in_wdata=Input(UInt(self.in_elem_w)),
            in_wen=Input(Bool()),
            in_rdata=Output(UInt(self.in_elem_w)),
            # external access to the output RAM (read C back)
            out_addr=Input(UInt(self.out_addr_w)),
            out_rdata=Output(UInt(self.out_elem_w)),
        )
        self.elaborate()

    # -- host-side address helpers (the layout described in the module docstring) --
    def a_tile_addr(self, ti: int, tk: int) -> int:
        return ti * self.cfg.KT + tk

    def b_tile_addr(self, tk: int, tj: int) -> int:
        return self.cfg.MT * self.cfg.KT + tj * self.cfg.KT + tk

    def c_tile_addr(self, ti: int, tj: int) -> int:
        return ti * self.cfg.NT + tj

    def elaborate(self):
        c = self.cfg
        T = c.T
        io = self.io

        # ---- on-chip RAMs (combinational reads, clocked writes) ----
        in_ram = RamPrimitive(UInt(self.in_elem_w), self.in_depth,
                              num_read_ports=2, num_write_ports=1, name="in_ram")
        out_ram = RamPrimitive(UInt(self.out_elem_w), self.out_depth,
                               num_read_ports=1, num_write_ports=1, name="out_ram")

        # ---- control registers (seed to idle; Simulator inits them at cycle 0) ----
        busy = Register(Bool(), init=0, name="busy_r")   # distinct from the `busy` output port
        i = Register(UInt(max(1, c.MT.bit_length())), init=0, name="ti")     # output tile row
        j = Register(UInt(max(1, c.NT.bit_length())), init=0, name="tj")     # output tile col
        k = Register(UInt(max(1, c.KT.bit_length())), init=0, name="tk")     # K-tile index
        acc = Register(UInt(self.out_elem_w), init=0, name="acc")            # packed T×T accumulator

        start_compute = io.mode & io.start & ~busy
        k_last = (k == (c.KT - 1))
        j_last = (j == (c.NT - 1))
        i_last = (i == (c.MT - 1))
        done_now = busy & k_last & j_last & i_last

        # ---- address generation (combinational; fit to RAM addr width on assign) ----
        addr_A = i * c.KT + k
        addr_B = (c.MT * c.KT) + j * c.KT + k
        addr_C = i * c.NT + j

        # ---- input RAM wiring + mode mux ----
        # read port 0: A tile (compute) or external readback (access)
        in_ram.io.r0_addr <<= mux(io.mode, addr_A, io.in_addr)
        in_ram.io.r1_addr <<= addr_B                      # read port 1: B tile (compute only)
        io.in_rdata <<= in_ram.io.r0_data
        a_word = in_ram.io.r0_data
        b_word = in_ram.io.r1_data
        # write port 0: external loads, access mode only
        in_ram.io.w0_addr <<= io.in_addr
        in_ram.io.w0_data <<= io.in_wdata
        in_ram.io.w0_en <<= io.in_wen & ~io.mode

        # ---- T×T×T tile engine (signed) ----
        mult_cfg = MultiplierConfig(use_operator=True)
        add_cfg = AdderConfig(use_operator=True, full_output_bit=True, encoding=Encoding.twos_complement)
        core = MatmulAccumulateComponent(
            MMAcCfg(MMAcDims(T, T, T), MMAcWidths(c.in_width, c.in_width, c.acc_width), mult_cfg, add_cfg),
            signed_io_type=True,
        )
        core.io.A <<= a_word                              # bulk-unpack the packed tile words
        core.io.B <<= b_word
        core.io.C <<= mux(k == 0, 0, acc)                 # output-stationary: 0 on first K-tile, else acc
        # Y is acc_width+1 wide; fit each element back to acc_width (lossless — acc_width covers full K)
        y_word = Array([Array([fit_width(core.io.Y[r, cc], SInt(c.acc_width)) for cc in range(T)])
                        for r in range(T)]).to_bits()

        with if_(busy):
            acc <<= y_word

        # ---- output RAM: controller writes the completed tile on the last K-step ----
        out_ram.io.w0_addr <<= addr_C
        out_ram.io.w0_data <<= y_word
        out_ram.io.w0_en <<= io.mode & busy & k_last
        out_ram.io.r0_addr <<= io.out_addr                # external reads C (access mode)
        io.out_rdata <<= out_ram.io.r0_data

        # ---- control: nested counters (k fastest, then j, then i) ----
        io.busy <<= busy
        k_next = mux(k_last, 0, k + 1)
        j_next = mux(k_last, mux(j_last, 0, j + 1), j)
        i_next = mux(k_last & j_last, mux(i_last, i, i + 1), i)
        with if_(start_compute):
            busy <<= 1
            i <<= 0
            j <<= 0
            k <<= 0
        with elif_(busy):
            busy <<= ~done_now
            k <<= k_next
            j <<= j_next
            i <<= i_next
