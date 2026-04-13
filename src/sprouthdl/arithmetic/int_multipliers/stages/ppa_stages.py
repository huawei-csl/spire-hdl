from __future__ import annotations

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

                while len(bits) >= 3:
                    x, y, z = bits.pop(), bits.pop(), bits.pop()
                    s, c = self._full_adder(x, y, z)
                    next_cols[weight].append(s)
                    next_cols[weight + 1].append(c)
                    progress = True

                if len(bits) == 2:
                    s, c = half_adder(bits.pop(), bits.pop())
                    next_cols[weight].append(s)
                    next_cols[weight + 1].append(c)
                    progress = True
                elif len(bits) == 1:
                    next_cols[weight].append(bits.pop())

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
                if rem == 2 and len(cols[weight]) >= 2:
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
                elif len(bits) == 2:
                    s, c = half_adder(bits.pop(), bits.pop())
                    next_cols[weight].append(s)
                    next_cols[weight + 1].append(c)
                    progress = True
                elif len(bits) == 1:
                    next_cols[weight].append(bits.pop())

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
                elif rem == 2 and len(cols[weight]) >= 2:
                    self._apply_ha_canonical(cols[weight], cols[weight + 1])
                    progress = True
            if not progress:
                return self._unwrap_columns(cols)

    def _compress_4_2(self, a: Expr, b: Expr, c: Expr, d: Expr) -> Tuple[Expr, Expr, Expr]:
        s1, c1 = self._full_adder(a, b, c)
        s2, c2 = self._full_adder(s1, d, self._zero)
        return s2, c1, c2