from __future__ import annotations

import heapq
from collections import defaultdict
from math import floor
from typing import ClassVar, DefaultDict, Dict, List, Optional, Tuple

from sprouthdl.arithmetic.int_multipliers.multipliers.multiplier_stage_core import (
    SelectionMode,
    StageMultiplierConfig,
    PartialProductAccumulatorBase,
    _LeveledBit,
    half_adder,
    full_adder_fast,
    full_adder_low_area,
)
from sprouthdl.sprouthdl import Bool, Const, Expr


class WallaceTreeAccumulator(PartialProductAccumulatorBase):
    """Classic Wallace tree reduction of partial-product columns."""

    # Wallace is a tie between LIFO and canonical at widths 6-16 and a
    # small canonical win at width 4 (-1 depth, -1 gate). Default to
    # canonical because it matches the scripted-policy action sequence
    # that every downstream ml_ppa study assumes.
    default_selection_mode: ClassVar[SelectionMode] = "canonical"

    def __init__(
        self,
        config: StageMultiplierConfig,
        *,
        selection_mode: Optional[SelectionMode] = None,
    ) -> None:
        super().__init__(config, selection_mode=selection_mode)
        self._full_adder = (
            full_adder_low_area
            if self.config.optim_type == "area"
            else full_adder_fast
        )

    def accumulate(self, columns: Dict[int, List[Expr]]) -> DefaultDict[int, List[Expr]]:
        if self.selection_mode == "canonical":
            return self._accumulate_canonical(columns)
        return self._accumulate_fifo_lifo(columns)

    def _accumulate_fifo_lifo(
        self, columns: Dict[int, List[Expr]]
    ) -> DefaultDict[int, List[Expr]]:
        """FIFO/LIFO schedule: next_cols buffer-swap pattern.

        Each pass copies the column, processes all bits via FA/HA using
        LIFO pop, puts results in a separate next_cols buffer, then swaps.
        """
        cols: DefaultDict[int, List[Expr]] = defaultdict(list)
        for weight, bits in columns.items():
            cols[weight].extend(bits)

        while True:
            next_cols: DefaultDict[int, List[Expr]] = defaultdict(list)
            progress = False

            for weight in sorted(cols.keys()):
                bits = list(cols[weight])
                orig_height = len(bits)

                while len(bits) >= 3:
                    x, y, z = bits.pop(), bits.pop(), bits.pop()
                    s, c = self._full_adder(x, y, z)
                    next_cols[weight].append(s)
                    next_cols[weight + 1].append(c)
                    progress = True

                if len(bits) == 2 and orig_height > 2:
                    s, c = half_adder(bits.pop(), bits.pop())
                    next_cols[weight].append(s)
                    next_cols[weight + 1].append(c)
                    progress = True
                else:
                    next_cols[weight].extend(bits)

            if not progress:
                return cols

            cols = next_cols

    def _accumulate_canonical(
        self, columns: Dict[int, List[Expr]]
    ) -> DefaultDict[int, List[Expr]]:
        """Canonical schedule: snapshot-based in-place reduction.

        Each outer iteration snapshots column heights, then for every
        column issues ``n_fa = h0 // 3`` FAs plus a final HA if exactly
        2 bits remain. The height snapshot prevents the same iteration
        from reacting to carries dropped by an earlier weight's FA.
        """
        cols = self._wrap_columns(columns)
        while True:
            snapshot = self._column_heights(cols)
            progress = False
            for weight, h0 in sorted(snapshot.items()):
                n_fa = h0 // 3
                rem = h0 - n_fa * 3
                for _ in range(n_fa):
                    if len(cols[weight]) >= 3:
                        self._apply_fa(
                            cols[weight], cols[weight + 1], self._full_adder
                        )
                        progress = True
                if rem == 2 and h0 > 2 and len(cols[weight]) >= 2:
                    self._apply_ha(cols[weight], cols[weight + 1])
                    progress = True
            if not progress:
                return self._unwrap_columns(cols)


class BalancedDelayWallaceAccumulator(WallaceTreeAccumulator):
    """Wallace reduction with cross-column priority-queue scheduling.

    Only supports canonical mode — the priority-queue scheduling
    requires arrival-level tracking which is specific to canonical
    bit selection.
    """

    default_selection_mode: ClassVar[SelectionMode] = "canonical"

    @staticmethod
    def _fa_output_level(bits: List[_LeveledBit]) -> Optional[int]:
        """Return the output level if an FA were applied to the three
        earliest-arriving bits of ``bits`` right now, or ``None`` if
        the column has fewer than 3 bits."""
        if len(bits) < 3:
            return None
        levels = sorted(b.level for b in bits)
        return levels[2] + 2

    def _accumulate_canonical(
        self, columns: Dict[int, List[Expr]]
    ) -> DefaultDict[int, List[Expr]]:
        cols = self._wrap_columns(columns)

        heap: List[Tuple[int, int, int]] = []

        def push(weight: int) -> None:
            lvl = self._fa_output_level(cols[weight])
            if lvl is not None:
                heapq.heappush(heap, (lvl, weight, len(cols[weight])))

        for weight in sorted(cols.keys()):
            push(weight)

        while heap:
            lvl, weight, stamp = heapq.heappop(heap)
            if stamp != len(cols[weight]):
                continue
            if self._fa_output_level(cols[weight]) != lvl:
                continue
            if len(cols[weight]) < 3:
                continue
            self._apply_fa(
                cols[weight], cols[weight + 1], self._full_adder
            )
            push(weight)
            push(weight + 1)

        # Cleanup: drain any column still > 2 bits with FA/HA moves.
        while True:
            any_tall = False
            for weight in sorted(list(cols.keys())):
                while len(cols[weight]) > 2:
                    any_tall = True
                    if len(cols[weight]) >= 3:
                        self._apply_fa(
                            cols[weight], cols[weight + 1], self._full_adder
                        )
                    else:
                        self._apply_ha(
                            cols[weight], cols[weight + 1]
                        )
            if not any_tall:
                break
        return self._unwrap_columns(cols)


class EagerWallaceAccumulator(WallaceTreeAccumulator):
    """Wallace reduction with live column heights instead of a
    per-iteration snapshot.

    Only supports canonical mode.
    """

    default_selection_mode: ClassVar[SelectionMode] = "canonical"

    def _accumulate_canonical(
        self, columns: Dict[int, List[Expr]]
    ) -> DefaultDict[int, List[Expr]]:
        cols = self._wrap_columns(columns)
        while True:
            progress = False
            for weight in sorted(cols.keys()):
                h0 = len(cols[weight])  # live, not a pre-iteration snapshot
                n_fa = h0 // 3
                rem = h0 - n_fa * 3
                for _ in range(n_fa):
                    if len(cols[weight]) >= 3:
                        self._apply_fa(
                            cols[weight], cols[weight + 1], self._full_adder
                        )
                        progress = True
                if rem == 2 and h0 > 2 and len(cols[weight]) >= 2:
                    self._apply_ha(cols[weight], cols[weight + 1])
                    progress = True
            if not progress:
                return self._unwrap_columns(cols)


class DaddaTreeAccumulator(PartialProductAccumulatorBase):
    """Dadda tree reduction using progressively tighter column height
    thresholds.

    Uses a single ``accumulate()`` method for all selection modes
    because the loop structure (threshold-based in-place reduction) is
    identical — only the bit-picking rule differs.
    """

    # Dadda is where canonical selection shines: depth drops by 60% on
    # average. Default to canonical.
    default_selection_mode: ClassVar[SelectionMode] = "canonical"

    def __init__(
        self,
        config: StageMultiplierConfig,
        *,
        selection_mode: Optional[SelectionMode] = None,
    ) -> None:
        super().__init__(config, selection_mode=selection_mode)
        self._full_adder = (
            full_adder_low_area
            if self.config.optim_type == "area"
            else full_adder_fast
        )

    def accumulate(self, columns: Dict[int, List[Expr]]) -> DefaultDict[int, List[Expr]]:
        cols = self._wrap_columns(columns)
        max_height = max((len(bits) for bits in cols.values()), default=0)
        if max_height <= 2:
            return self._unwrap_columns(cols)

        thresholds = self._build_thresholds(max_height)
        stage_limits = list(reversed(thresholds))[1:]  # skip the largest value

        for target in stage_limits:
            for weight in sorted(list(cols.keys())):
                while len(cols[weight]) > target:
                    h = len(cols[weight])
                    if h >= 3 and (h - 2) >= target:
                        self._apply_fa(
                            cols[weight], cols[weight + 1], self._full_adder
                        )
                    elif h >= 2:
                        self._apply_ha(cols[weight], cols[weight + 1])
                    else:
                        break
        return self._unwrap_columns(cols)

    @staticmethod
    def _build_thresholds(max_height: int) -> List[int]:
        thresholds = [2]
        while thresholds[-1] < max_height:
            next_val = floor(3 * thresholds[-1] / 2)
            if next_val == thresholds[-1]:
                next_val += 1
            thresholds.append(next_val)
        return thresholds


class CarrySaveAccumulator(PartialProductAccumulatorBase):
    """Iterative carry-save reduction using only full adders."""

    # CarrySave is the one PPA where canonical *loses* (+2 to +3 depth,
    # 0 to +2 gates across widths 4-16). The LIFO order already produces
    # a well-shaped tree. Default to LIFO.
    default_selection_mode: ClassVar[SelectionMode] = "lifo"

    def __init__(
        self,
        config: StageMultiplierConfig,
        *,
        selection_mode: Optional[SelectionMode] = None,
    ) -> None:
        super().__init__(config, selection_mode=selection_mode)
        self._full_adder = (
            full_adder_low_area
            if self.config.optim_type == "area"
            else full_adder_fast
        )

    def accumulate(self, columns: Dict[int, List[Expr]]) -> DefaultDict[int, List[Expr]]:
        if self.selection_mode == "canonical":
            return self._accumulate_canonical(columns)
        return self._accumulate_fifo_lifo(columns)

    def _accumulate_fifo_lifo(
        self, columns: Dict[int, List[Expr]]
    ) -> DefaultDict[int, List[Expr]]:
        """FIFO/LIFO schedule: next_cols buffer-swap, FA-only."""
        cols: DefaultDict[int, List[Expr]] = defaultdict(list)
        for weight, bits in columns.items():
            cols[weight].extend(bits)

        while True:
            next_cols: DefaultDict[int, List[Expr]] = defaultdict(list)
            progress = False

            for weight in sorted(cols.keys()):
                bits = list(cols[weight])
                while len(bits) >= 3:
                    x, y, z = bits.pop(), bits.pop(), bits.pop()
                    s, c = self._full_adder(x, y, z)
                    next_cols[weight].append(s)
                    next_cols[weight + 1].append(c)
                    progress = True
                if bits:
                    next_cols[weight].extend(bits)

            if not progress:
                return cols

            cols = next_cols

    def _accumulate_canonical(
        self, columns: Dict[int, List[Expr]]
    ) -> DefaultDict[int, List[Expr]]:
        """Canonical schedule: snapshot + in-place + cleanup."""
        cols = self._wrap_columns(columns)
        while True:
            snapshot = self._column_heights(cols)
            progress = False
            for weight, h0 in sorted(snapshot.items()):
                n_fa = h0 // 3
                for _ in range(n_fa):
                    if len(cols[weight]) >= 3:
                        self._apply_fa(
                            cols[weight], cols[weight + 1], self._full_adder
                        )
                        progress = True
            if not progress:
                break
        # Final cleanup: bring any column still > 2 down with FA/HA.
        while True:
            any_tall = False
            for weight in sorted(list(cols.keys())):
                while len(cols[weight]) > 2:
                    any_tall = True
                    if len(cols[weight]) >= 3:
                        self._apply_fa(
                            cols[weight], cols[weight + 1], self._full_adder
                        )
                    else:
                        self._apply_ha(
                            cols[weight], cols[weight + 1]
                        )
            if not any_tall:
                break
        return self._unwrap_columns(cols)


class FourTwoCompressorAccumulator(PartialProductAccumulatorBase):
    """Reduction based on 4:2 compressors backed by chained full adders."""

    # FourTwoCompressor canonical is a Pareto improvement at every
    # tested width. Default to canonical.
    default_selection_mode: ClassVar[SelectionMode] = "canonical"

    def __init__(
        self,
        config: StageMultiplierConfig,
        *,
        selection_mode: Optional[SelectionMode] = None,
    ) -> None:
        super().__init__(config, selection_mode=selection_mode)
        self._full_adder = (
            full_adder_low_area
            if self.config.optim_type == "area"
            else full_adder_fast
        )
        self._zero = Const(False, Bool())

    def accumulate(self, columns: Dict[int, List[Expr]]) -> DefaultDict[int, List[Expr]]:
        if self.selection_mode == "canonical":
            return self._accumulate_canonical(columns)
        return self._accumulate_fifo_lifo(columns)

    def _accumulate_fifo_lifo(
        self, columns: Dict[int, List[Expr]]
    ) -> DefaultDict[int, List[Expr]]:
        """FIFO/LIFO schedule: next_cols buffer-swap with 4:2/FA/HA."""
        cols: DefaultDict[int, List[Expr]] = defaultdict(list)
        for weight, bits in columns.items():
            cols[weight].extend(bits)

        while True:
            next_cols: DefaultDict[int, List[Expr]] = defaultdict(list)
            progress = False

            for weight in sorted(cols.keys()):
                bits = list(cols[weight])
                orig_height = len(bits)

                while len(bits) >= 4:
                    a = bits.pop()
                    b = bits.pop()
                    c = bits.pop()
                    d = bits.pop()
                    sum_bit, carry_low, carry_high = self._compress_4_2(a, b, c, d)
                    next_cols[weight].append(sum_bit)
                    next_cols[weight + 1].extend((carry_low, carry_high))
                    progress = True

                if len(bits) == 3:
                    x, y, z = bits.pop(), bits.pop(), bits.pop()
                    s, c = self._full_adder(x, y, z)
                    next_cols[weight].append(s)
                    next_cols[weight + 1].append(c)
                    progress = True
                elif len(bits) == 2 and orig_height > 2:
                    s, c = half_adder(bits.pop(), bits.pop())
                    next_cols[weight].append(s)
                    next_cols[weight + 1].append(c)
                    progress = True
                else:
                    next_cols[weight].extend(bits)

            if not progress:
                return cols

            cols = next_cols

    def _accumulate_canonical(
        self, columns: Dict[int, List[Expr]]
    ) -> DefaultDict[int, List[Expr]]:
        """Canonical schedule: snapshot-based with 4:2/FA/HA."""
        cols = self._wrap_columns(columns)
        while True:
            snapshot = self._column_heights(cols)
            progress = False
            for weight, h0 in sorted(snapshot.items()):
                n_c42 = h0 // 4
                rem = h0 - n_c42 * 4
                for _ in range(n_c42):
                    if len(cols[weight]) >= 4:
                        self._apply_c42(
                            cols[weight], cols[weight + 1],
                            self._full_adder, self._zero,
                        )
                        progress = True
                if rem == 3 and len(cols[weight]) >= 3:
                    self._apply_fa(
                        cols[weight], cols[weight + 1], self._full_adder
                    )
                    progress = True
                elif rem == 2 and h0 > 2 and len(cols[weight]) >= 2:
                    self._apply_ha(cols[weight], cols[weight + 1])
                    progress = True
            if not progress:
                return self._unwrap_columns(cols)

    def _compress_4_2(self, a: Expr, b: Expr, c: Expr, d: Expr) -> Tuple[Expr, Expr, Expr]:
        s1, c1 = self._full_adder(a, b, c)
        s2, c2 = self._full_adder(s1, d, self._zero)
        return s2, c1, c2


class FourTwoCompressorParallelAccumulator(FourTwoCompressorAccumulator):
    """Variant of :class:`FourTwoCompressorAccumulator` that uses the
    "true 4:2 compressor" gate pattern (parallel XOR + majority carry
    with an explicit horizontal carry-in) in place of the default pair
    of cascaded full adders.

    Forces LIFO because the canonical path uses ``_apply_c42`` which
    hard-codes cascaded FAs and would bypass this override.
    """

    default_selection_mode: ClassVar[SelectionMode] = "lifo"

    def _compress_4_2(
        self,
        a: Expr,
        b: Expr,
        c: Expr,
        d: Expr,
        carry_in: Optional[Expr] = None,
    ) -> Tuple[Expr, Expr, Expr]:
        if carry_in is None:
            carry_in = self._zero
        parity_abc = a ^ b ^ c
        carry_chain_out = (a & b) | (a & c) | (b & c)
        sum_bit = parity_abc ^ d ^ carry_in
        carry_bit = (parity_abc & d) | (parity_abc & carry_in) | (d & carry_in)
        return sum_bit, carry_bit, carry_chain_out


class FiveTwoCompressorAccumulator(PartialProductAccumulatorBase):
    """Reduction based on 5-input / 3-output compressors, built as two
    cascaded full adders.

    Falls back to cascaded 4:2 / FA / HA compressions for columns
    whose height is not a multiple of five.
    """

    default_selection_mode: ClassVar[SelectionMode] = "lifo"

    def __init__(
        self,
        config: StageMultiplierConfig,
        *,
        selection_mode: Optional[SelectionMode] = None,
    ) -> None:
        super().__init__(config, selection_mode=selection_mode)
        self._full_adder = (
            full_adder_low_area
            if self.config.optim_type == "area"
            else full_adder_fast
        )
        self._zero = Const(False, Bool())

    def accumulate(self, columns: Dict[int, List[Expr]]) -> DefaultDict[int, List[Expr]]:
        return self._accumulate_fifo_lifo(columns)

    def _accumulate_fifo_lifo(
        self, columns: Dict[int, List[Expr]]
    ) -> DefaultDict[int, List[Expr]]:
        """FIFO/LIFO schedule: next_cols buffer-swap with 5:2/4:2/FA/HA."""
        cols: DefaultDict[int, List[Expr]] = defaultdict(list)
        for weight, bits in columns.items():
            cols[weight].extend(bits)

        while True:
            next_cols: DefaultDict[int, List[Expr]] = defaultdict(list)
            progress = False

            for weight in sorted(cols.keys()):
                bits = list(cols[weight])
                orig_height = len(bits)

                while len(bits) >= 5:
                    a = bits.pop()
                    b = bits.pop()
                    c = bits.pop()
                    d = bits.pop()
                    e = bits.pop()
                    sum_bit, carry_low, carry_high = self._compress_5_2(a, b, c, d, e)
                    next_cols[weight].append(sum_bit)
                    next_cols[weight + 1].extend((carry_low, carry_high))
                    progress = True

                if len(bits) == 4:
                    a = bits.pop()
                    b = bits.pop()
                    c = bits.pop()
                    d = bits.pop()
                    s1, c1 = self._full_adder(a, b, c)
                    s2, c2 = self._full_adder(s1, d, self._zero)
                    next_cols[weight].append(s2)
                    next_cols[weight + 1].extend((c1, c2))
                    progress = True
                elif len(bits) == 3:
                    x, y, z = bits.pop(), bits.pop(), bits.pop()
                    s, c = self._full_adder(x, y, z)
                    next_cols[weight].append(s)
                    next_cols[weight + 1].append(c)
                    progress = True
                elif len(bits) == 2 and orig_height > 2:
                    s, c = half_adder(bits.pop(), bits.pop())
                    next_cols[weight].append(s)
                    next_cols[weight + 1].append(c)
                    progress = True
                else:
                    next_cols[weight].extend(bits)

            if not progress:
                return cols

            cols = next_cols

    def _compress_5_2(
        self, a: Expr, b: Expr, c: Expr, d: Expr, e: Expr
    ) -> Tuple[Expr, Expr, Expr]:
        s1, c1 = self._full_adder(a, b, c)
        s2, c2 = self._full_adder(s1, d, e)
        return s2, c1, c2


class FiveTwoCompressorParallelAccumulator(FiveTwoCompressorAccumulator):
    """Variant of :class:`FiveTwoCompressorAccumulator` whose inner
    5-input compressor is expressed as two stacked parallel
    XOR + majority cells rather than two cascaded full adders.
    """

    def _compress_5_2(
        self, a: Expr, b: Expr, c: Expr, d: Expr, e: Expr
    ) -> Tuple[Expr, Expr, Expr]:
        # First "FA" on (a, b, c) in parallel gates:
        parity_abc = a ^ b ^ c
        carry_abc = (a & b) | (a & c) | (b & c)
        # Second "FA" on (parity_abc, d, e) in parallel gates:
        sum_bit = parity_abc ^ d ^ e
        carry_de = (parity_abc & d) | (parity_abc & e) | (d & e)
        return sum_bit, carry_de, carry_abc
