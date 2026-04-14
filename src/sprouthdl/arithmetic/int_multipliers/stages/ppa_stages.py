
from __future__ import annotations

from math import floor
from typing import ClassVar, DefaultDict, Dict, List, Optional

from sprouthdl.arithmetic.int_multipliers.multipliers.multiplier_stage_core import (
    StageMultiplierConfig,
    PartialProductAccumulatorBase,
    full_adder_fast,
    full_adder_low_area,
)
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
        """Legacy and canonical share the same Wallace *schedule*.

        The only difference is which physical bits are consumed by each
        FA/HA:
        - legacy: historical LIFO ``pop()``
        - canonical: earliest-arrival-first
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
                        self._apply_fa(cols[weight], cols[weight + 1], self._full_adder)
                        progress = True
                if rem == 2 and h0 > 2 and len(cols[weight]) >= 2:
                    self._apply_ha(cols[weight], cols[weight + 1])
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
        """Legacy and canonical share the same Dadda stage schedule."""
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
                        self._apply_fa(cols[weight], cols[weight + 1], self._full_adder)
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
        """Legacy and canonical share the same FA-only pass schedule."""
        cols = self._wrap_columns(columns)
        while True:
            snapshot = self._column_heights(cols)
            progress = False
            for weight, h0 in sorted(snapshot.items()):
                for _ in range(h0 // 3):
                    if len(cols[weight]) >= 3:
                        self._apply_fa(cols[weight], cols[weight + 1], self._full_adder)
                        progress = True
            if not progress:
                break

        # No cleanup pass is needed: once a snapshot produces no FA, all
        # columns are already at height <= 2.
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
        """Legacy and canonical share the same per-pass 4:2/FA/HA schedule."""
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
                            cols[weight],
                            cols[weight + 1],
                            self._full_adder,
                            self._zero,
                        )
                        progress = True
                if rem == 3 and len(cols[weight]) >= 3:
                    self._apply_fa(cols[weight], cols[weight + 1], self._full_adder)
                    progress = True
                elif rem == 2 and h0 > 2 and len(cols[weight]) >= 2:
                    self._apply_ha(cols[weight], cols[weight + 1])
                    progress = True
            if not progress:
                return self._unwrap_columns(cols)
