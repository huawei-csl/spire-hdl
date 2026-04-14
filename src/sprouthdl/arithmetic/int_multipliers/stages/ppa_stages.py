from __future__ import annotations

import heapq
from collections import defaultdict
from math import floor
from typing import ClassVar, DefaultDict, Dict, List, Optional, Tuple

from sprouthdl.arithmetic.int_multipliers.multipliers.multiplier_stage_core import StageMultiplierConfig, PartialProductAccumulatorBase, _LeveledBit, half_adder, full_adder_fast, full_adder_low_area
from sprouthdl.sprouthdl import Bool, Const, Expr


class WallaceTreeAccumulator(PartialProductAccumulatorBase):
    """Classic Wallace tree reduction of partial-product columns."""

    # Wallace is a tie between legacy and canonical at widths 6-16 and a
    # small canonical win at width 4 (-1 depth, -1 gate). Default to
    # canonical because it matches the scripted-policy action sequence
    # that every downstream ml_ppa study assumes.
    canonical_bit_selection: ClassVar[bool] = True

    def __init__(
        self,
        config: StageMultiplierConfig,
        *,
        canonical_bit_selection: Optional[bool] = None,
    ) -> None:
        super().__init__(config, canonical_bit_selection=canonical_bit_selection)
        self._full_adder = (
            full_adder_low_area
            if self.config.optim_type == "area"
            else full_adder_fast
        )

    def accumulate(self, columns: Dict[int, List[Expr]]) -> DefaultDict[int, List[Expr]]:
        if self.canonical_bit_selection:
            return self._accumulate_canonical(columns)
        return self._accumulate_legacy(columns)

    def _accumulate_legacy(
        self, columns: Dict[int, List[Expr]]
    ) -> DefaultDict[int, List[Expr]]:
        # copy columns
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
        """Mirrors ``wallace_policy`` from scripted_policies.py: each
        outer iteration snapshots column heights, then for every column
        issues ``n_fa = h0 // 3`` FAs (using canonical earliest-first
        bits) plus a final HA if exactly 2 bits remain. Repeats until
        quiescent. The height snapshot is critical — it prevents the
        same iteration from reacting to carries dropped into the column
        by an earlier weight's FA.
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
                        self._apply_fa_canonical(
                            cols[weight], cols[weight + 1], self._full_adder
                        )
                        progress = True
                if rem == 2 and h0 > 2 and len(cols[weight]) >= 2:
                    self._apply_ha_canonical(cols[weight], cols[weight + 1])
                    progress = True
            if not progress:
                return self._unwrap_columns(cols)


class BalancedDelayWallaceAccumulator(WallaceTreeAccumulator):
    """Wallace reduction with cross-column priority-queue scheduling.

    Canonical Wallace (both the snapshot and eager variants) picks
    full-adder targets one *column* at a time in weight order. The
    per-FA bit selection inside a column is already delay-optimal —
    :meth:`_take_earliest` picks the three smallest-level bits, which
    is equivalent to the greedy Balanced Delay Tree rule of Oklobdzija
    et al. — but the *column* choice is not.

    This variant maintains a min-heap keyed by
    ``(min_fa_output_level, weight)`` where
    ``min_fa_output_level = sorted_levels(cols[w])[2] + 2`` is the
    output level of the next FA applied to column ``w`` if we took
    its three earliest bits right now. At each step we pop the column
    whose next FA lands at the lowest level, apply one FA there, then
    re-push both the source column and the carry-receiving column
    (``w + 1``) with their updated priorities.

    The intuition: reducing the shallowest-output FA first keeps all
    newly-generated carries at the lowest possible level, so when a
    deeper column finally has to consume an upstream carry it sees
    the *earliest* possible arrival time. Columns compete on merit
    instead of following a fixed spatial order.

    After the priority loop drains (all remaining columns have
    fewer than 3 bits and would not benefit from an FA), a cleanup
    pass issues half-adders on any column still above height 2,
    mirroring :class:`CarrySaveAccumulator`'s tail loop.
    """

    canonical_bit_selection: ClassVar[bool] = True

    @staticmethod
    def _fa_output_level(bits: List[_LeveledBit]) -> Optional[int]:
        """Return the output level if an FA were applied to the three
        earliest-arriving bits of ``bits`` right now, or ``None`` if
        the column has fewer than 3 bits."""
        if len(bits) < 3:
            return None
        # Third-smallest level determines the FA output level because
        # the FA sum/carry lands at ``max(picked) + 2`` and the three
        # picked bits are the smallest-level ones.
        levels = sorted(b.level for b in bits)
        return levels[2] + 2

    def _accumulate_canonical(
        self, columns: Dict[int, List[Expr]]
    ) -> DefaultDict[int, List[Expr]]:
        cols = self._wrap_columns(columns)

        # Seed the priority queue with every column that can fire an FA.
        # Heap entries are ``(fa_output_level, weight, stale_guard)`` so
        # popped entries whose priority no longer matches the column's
        # current state are discarded rather than re-validated.
        heap: List[Tuple[int, int, int]] = []

        def push(weight: int) -> None:
            lvl = self._fa_output_level(cols[weight])
            if lvl is not None:
                heapq.heappush(heap, (lvl, weight, len(cols[weight])))

        for weight in sorted(cols.keys()):
            push(weight)

        while heap:
            lvl, weight, stamp = heapq.heappop(heap)
            # Stale entry: column has changed since this was pushed.
            if stamp != len(cols[weight]):
                continue
            if self._fa_output_level(cols[weight]) != lvl:
                continue
            if len(cols[weight]) < 3:
                continue
            self._apply_fa_canonical(
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
                        self._apply_fa_canonical(
                            cols[weight], cols[weight + 1], self._full_adder
                        )
                    else:
                        self._apply_ha_canonical(
                            cols[weight], cols[weight + 1]
                        )
            if not any_tall:
                break
        return self._unwrap_columns(cols)


class EagerWallaceAccumulator(WallaceTreeAccumulator):
    """Wallace reduction with live column heights instead of a
    per-iteration snapshot.

    Identical to :class:`WallaceTreeAccumulator` except that
    :meth:`_accumulate_canonical` reads ``h0 = len(cols[weight])`` on
    the fly inside the column loop. Carries dropped into column
    ``w + 1`` earlier in the same iteration are therefore visible when
    we reach ``w + 1``, eliminating the one-iteration stall that the
    snapshot version inherits from ``wallace_policy``.

    Correctness is preserved by the ``_LeveledBit`` arrival-level
    tracking: a freshly-dropped carry sits above the level-1 primary
    bits in :meth:`_take_earliest`'s ordering and will only be picked
    once the lower-level bits in the column are exhausted, so no bit
    is ever consumed before it has actually arrived.

    Motivated by the fact that at 8×8 the tallest PP column stalls for
    one full iteration waiting for its ``w + 1`` neighbour to catch
    the carries it just dropped; letting ``w + 1`` swallow them in the
    same pass compresses one outer iteration out of the critical path.
    """

    canonical_bit_selection: ClassVar[bool] = True

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
                        self._apply_fa_canonical(
                            cols[weight], cols[weight + 1], self._full_adder
                        )
                        progress = True
                if rem == 2 and h0 > 2 and len(cols[weight]) >= 2:
                    self._apply_ha_canonical(cols[weight], cols[weight + 1])
                    progress = True
            if not progress:
                return self._unwrap_columns(cols)


class DaddaTreeAccumulator(PartialProductAccumulatorBase):
    """Dadda tree reduction using progressively tighter column height thresholds."""

    # Dadda is where canonical selection shines: depth drops by 60% on
    # average (Dadda 8x8 goes 76 -> 27 AIG levels, Dadda 12x12 goes
    # 160 -> 29, Dadda 16x16 goes 257 -> 37). The legacy LIFO strands
    # primary PP bits near the top of the column that then cascade
    # through later FAs, serialising the reduction. Gate cost peaks
    # at +16% (w=12); most widths are +0 or small. Default to canonical.
    canonical_bit_selection: ClassVar[bool] = True

    def __init__(
        self,
        config: StageMultiplierConfig,
        *,
        canonical_bit_selection: Optional[bool] = None,
    ) -> None:
        super().__init__(config, canonical_bit_selection=canonical_bit_selection)
        self._full_adder = (
            full_adder_low_area
            if self.config.optim_type == "area"
            else full_adder_fast
        )

    def accumulate(self, columns: Dict[int, List[Expr]]) -> DefaultDict[int, List[Expr]]:
        if self.canonical_bit_selection:
            return self._accumulate_canonical(columns)
        return self._accumulate_legacy(columns)

    def _accumulate_legacy(
        self, columns: Dict[int, List[Expr]]
    ) -> DefaultDict[int, List[Expr]]:
        cols: DefaultDict[int, List[Expr]] = defaultdict(list)
        for weight, bits in columns.items():
            cols[weight].extend(bits)

        max_height = max((len(bits) for bits in cols.values()), default=0)
        if max_height <= 2:
            return cols

        thresholds = self._build_thresholds(max_height)
        stage_limits = list(reversed(thresholds))[1:]  # skip the largest value

        for target in stage_limits:
            for weight in sorted(cols.keys()):
                reduced, carries = self._reduce_column_to_target(cols[weight], target)
                cols[weight] = reduced
                cols[weight + 1].extend(carries)

        return cols

    def _accumulate_canonical(
        self, columns: Dict[int, List[Expr]]
    ) -> DefaultDict[int, List[Expr]]:
        """Mirrors ``dadda_policy`` from scripted_policies.py: builds the
        standard 3n/2 threshold sequence, then for each descending target
        reduces every column down to that target via FA (where safe —
        i.e. ``cur - 2 >= target``) or HA otherwise.
        """
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
                        self._apply_fa_canonical(
                            cols[weight], cols[weight + 1], self._full_adder
                        )
                    elif h >= 2:
                        self._apply_ha_canonical(cols[weight], cols[weight + 1])
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

    def _reduce_column_to_target(
        self, bits: List[Expr], target: int
    ) -> Tuple[List[Expr], List[Expr]]:
        working = list(bits)
        carries: List[Expr] = []

        while len(working) > target:
            if len(working) >= 3 and len(working) - 2 >= target:
                x = working.pop()
                y = working.pop()
                z = working.pop()
                s, c = self._full_adder(x, y, z)
                working.append(s)
                carries.append(c)
            else:
                x = working.pop()
                y = working.pop()
                s, c = half_adder(x, y)
                working.append(s)
                carries.append(c)

        return working, carries


class CarrySaveAccumulator(PartialProductAccumulatorBase):
    """Iterative carry-save reduction using only full adders."""

    # CarrySave is the one PPA where canonical *loses* (+2 to +3 depth,
    # 0 to +2 gates across widths 4-16). The legacy LIFO happens to
    # already produce a well-shaped tree for the FA-only reduction. We
    # default to legacy and expose the flag only for completeness.
    canonical_bit_selection: ClassVar[bool] = False

    def __init__(
        self,
        config: StageMultiplierConfig,
        *,
        canonical_bit_selection: Optional[bool] = None,
    ) -> None:
        super().__init__(config, canonical_bit_selection=canonical_bit_selection)
        self._full_adder = (
            full_adder_low_area
            if self.config.optim_type == "area"
            else full_adder_fast
        )

    def accumulate(self, columns: Dict[int, List[Expr]]) -> DefaultDict[int, List[Expr]]:
        if self.canonical_bit_selection:
            return self._accumulate_canonical(columns)
        return self._accumulate_legacy(columns)

    def _accumulate_legacy(
        self, columns: Dict[int, List[Expr]]
    ) -> DefaultDict[int, List[Expr]]:
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
        """Mirrors ``carry_save_policy`` from scripted_policies.py:
        iterate snapshot -> issue ``h0 // 3`` FAs per column (FA-only),
        repeat until quiescent; then a final cleanup sweep with FA+HA
        to bring any still-above-height-2 column down to <= 2.
        """
        cols = self._wrap_columns(columns)
        while True:
            snapshot = self._column_heights(cols)
            progress = False
            for weight, h0 in sorted(snapshot.items()):
                n_fa = h0 // 3
                for _ in range(n_fa):
                    if len(cols[weight]) >= 3:
                        self._apply_fa_canonical(
                            cols[weight], cols[weight + 1], self._full_adder
                        )
                        progress = True
            if not progress:
                break
        # Final cleanup: any column with height > 2 gets reduced by
        # FA (if possible) or HA, matching the policy's tail loop.
        while True:
            any_tall = False
            for weight in sorted(list(cols.keys())):
                while len(cols[weight]) > 2:
                    any_tall = True
                    if len(cols[weight]) >= 3:
                        self._apply_fa_canonical(
                            cols[weight], cols[weight + 1], self._full_adder
                        )
                    else:
                        self._apply_ha_canonical(
                            cols[weight], cols[weight + 1]
                        )
            if not any_tall:
                break
        return self._unwrap_columns(cols)


class FourTwoCompressorAccumulator(PartialProductAccumulatorBase):
    """Reduction based on 4:2 compressors backed by chained full adders."""

    # FourTwoCompressor canonical is a Pareto improvement at every
    # tested width: -2 to -13 depth AND -11 to -210 gates. No reason
    # to keep legacy as default.
    canonical_bit_selection: ClassVar[bool] = True

    def __init__(
        self,
        config: StageMultiplierConfig,
        *,
        canonical_bit_selection: Optional[bool] = None,
    ) -> None:
        super().__init__(config, canonical_bit_selection=canonical_bit_selection)
        self._full_adder = (
            full_adder_low_area
            if self.config.optim_type == "area"
            else full_adder_fast
        )
        self._zero = Const(False, Bool())

    def accumulate(self, columns: Dict[int, List[Expr]]) -> DefaultDict[int, List[Expr]]:
        if self.canonical_bit_selection:
            return self._accumulate_canonical(columns)
        return self._accumulate_legacy(columns)

    def _accumulate_legacy(
        self, columns: Dict[int, List[Expr]]
    ) -> DefaultDict[int, List[Expr]]:
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
        """Mirrors ``four_two_policy`` from scripted_policies.py: each
        iteration snapshots column heights, then per column issues
        ``n_c42 = h0 // 4`` 4:2 compressors followed by one FA (rem==3)
        or HA (rem==2). Repeats until no column changes height.
        """
        cols = self._wrap_columns(columns)
        while True:
            snapshot = self._column_heights(cols)
            progress = False
            for weight, h0 in sorted(snapshot.items()):
                n_c42 = h0 // 4
                rem = h0 - n_c42 * 4
                for _ in range(n_c42):
                    if len(cols[weight]) >= 4:
                        self._apply_c42_canonical(
                            cols[weight], cols[weight + 1],
                            self._full_adder, self._zero,
                        )
                        progress = True
                if rem == 3 and len(cols[weight]) >= 3:
                    self._apply_fa_canonical(
                        cols[weight], cols[weight + 1], self._full_adder
                    )
                    progress = True
                elif rem == 2 and h0 > 2 and len(cols[weight]) >= 2:
                    self._apply_ha_canonical(cols[weight], cols[weight + 1])
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

    With ``carry_in = 0`` — the value used when the inherited
    reduction policy calls this method with four inputs — the output
    bits are logically equivalent to the base class' cascaded-FA
    implementation; only the gate structure differs. Keeping the
    ``carry_in`` parameter on the signature leaves room for a future
    reduction policy to thread a horizontal carry chain between
    compressors in the same column, turning the 4-input / 3-output
    structure into a true 4:2 compressor.

    Forces the legacy reduction path because the canonical path has
    its own hard-coded cascaded-FA cell in
    :meth:`PartialProductAccumulatorBase._apply_c42_canonical` and
    would bypass this override otherwise.
    """

    canonical_bit_selection: ClassVar[bool] = False

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

    Commonly called a "5:2 compressor" in prior art — a true 5:2
    compressor would take 5 data inputs plus 2 horizontal carry-ins
    and produce 2 local outputs plus 2 horizontal carry-outs (three
    cascaded full adders). This implementation drops the horizontal
    chain, so it is really a 5:3 compressor locally: 5 bits in column
    k reduce to 1 sum in column k and 2 carries in column k+1 — one
    more input bit reduced per two-FA cost than the classic "4:2"
    (which is a 4:3) cell.

    Falls back to cascaded 4:2 / FA / HA compressions for columns
    whose height is not a multiple of five.
    """

    canonical_bit_selection: ClassVar[bool] = False

    def __init__(
        self,
        config: StageMultiplierConfig,
        *,
        canonical_bit_selection: Optional[bool] = None,
    ) -> None:
        super().__init__(config, canonical_bit_selection=canonical_bit_selection)
        self._full_adder = (
            full_adder_low_area
            if self.config.optim_type == "area"
            else full_adder_fast
        )
        self._zero = Const(False, Bool())

    def accumulate(self, columns: Dict[int, List[Expr]]) -> DefaultDict[int, List[Expr]]:
        return self._accumulate_legacy(columns)

    def _accumulate_legacy(
        self, columns: Dict[int, List[Expr]]
    ) -> DefaultDict[int, List[Expr]]:
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

    Logically equivalent to the base class (same three output bits
    for every input combination) — only the gate structure differs,
    which the downstream synthesis tool may optimize differently.
    Written out explicitly for symmetry with
    :class:`FourTwoCompressorParallelAccumulator`.
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