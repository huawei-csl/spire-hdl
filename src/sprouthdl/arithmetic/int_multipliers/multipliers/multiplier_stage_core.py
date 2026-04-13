from __future__ import annotations

import abc
from collections import defaultdict
from dataclasses import dataclass
from typing import ClassVar, DefaultDict, Dict, List, Literal, Optional, Tuple, Type

import numpy as np

from sprouthdl.arithmetic.int_multipliers.eval.testvector_generation import Encoding
from sprouthdl.sprouthdl_module import Component
from sprouthdl.sprouthdl import Bool, Concat, Const, Expr, Signal, SInt, UInt, mux


# ---- common arithmetic helpers -------------------------------------------------

def half_adder(x: Expr, y: Expr) -> Tuple[Expr, Expr]:
    return x ^ y, x & y  # sum, carry


def full_adder_fast(x: Expr, y: Expr, z: Expr) -> Tuple[Expr, Expr]:
    s1 = x ^ y
    return s1 ^ z, (s1 & z) | (x & y)


def full_adder_low_area(x: Expr, y: Expr, z: Expr) -> Tuple[Expr, Expr]:
    s = x ^ y ^ z
    return s, (x & y) | (y & z) | (z & x)


class _LeveledBit:
    """A partial-product-tree bit annotated with its symbolic arrival level
    and a stable insertion order.

    Used exclusively by the canonical-bit-selection PPA path (enabled via
    ``PartialProductAccumulatorBase.canonical_bit_selection``). The idea
    mirrors :class:`sprouthdl.ml_ppa.environment.PPAEnv`'s ``BitMeta``:
    before each FA/HA/4:2 reduction we sort a column's bits by
    ``(level, ord_)`` and consume the earliest-arriving k. This
    earliest-first rule collapses the critical path aggressively.
    """

    __slots__ = ("expr", "level", "ord_")

    def __init__(self, expr: Expr, level: int, ord_: int) -> None:
        self.expr = expr
        self.level = level
        self.ord_ = ord_


# ---- abstract component/stage definitions --------------------------------------
@dataclass(frozen=True)
class TwoInputAritConfig:
    a_width: int
    b_width: int
    signed_a: bool = False
    signed_b: bool = False
    optim_type: Literal["area", "speed"] = "area"

    @property
    def out_width(self) -> int:
        raise NotImplementedError


@dataclass(frozen=True)
class StageMultiplierConfig(TwoInputAritConfig):  # might be renamed to StageMultiplierConfig
    a_width: int
    b_width: int
    signed_a: bool
    signed_b: bool
    optim_type: Literal["area", "speed"]

    @property
    def out_width(self) -> int:
        return self.a_width + self.b_width


class StageBase(abc.ABC):

    def __init__(self, config: TwoInputAritConfig) -> None:
        self.config = config


class PartialProductGeneratorBase(StageBase, abc.ABC):
    supported_signatures: ClassVar[Optional[Tuple[Tuple[bool, bool], ...]]] = None

    @abc.abstractmethod
    def generate_columns(self, io: "StageBasedMultiplierIO") -> DefaultDict[int, List[Expr]]:
        raise NotImplementedError


class PartialProductAccumulatorBase(StageBase, abc.ABC):
    # Default bit-selection rule for the column reduction loop.
    #
    # When False (legacy), each concrete PPA picks bits from its column
    # lists with its historical FIFO/LIFO rule — whatever pattern the
    # accumulate() implementation uses via list.pop() / slicing. When
    # True, the accumulate() implementation follows the *canonical*
    # earliest-arrival rule: sort the column by (arrival_level, ord_)
    # and consume the first k bits, matching
    # sprouthdl.ml_ppa.environment.PPAEnv._canonical_bits. The action
    # sequence is otherwise identical — only which physical bits get
    # fed into each FA/HA/4:2 application differs.
    #
    # Per-PPA defaults are set on the concrete subclasses based on the
    # benchmark numbers in docs/ml_ppa_results.md §"canonicalisation
    # surprise" and the follow-up comparison study: Dadda, Wallace and
    # FourTwoCompressor default to True (canonical strictly wins or is
    # Pareto-neutral); CarrySave and CompressorTree default to False
    # (legacy wins on depth, which is their usual use case). Users
    # can override per-instance via the constructor argument.
    canonical_bit_selection: ClassVar[bool] = False

    def __init__(
        self,
        config: TwoInputAritConfig,
        *,
        canonical_bit_selection: Optional[bool] = None,
    ) -> None:
        super().__init__(config)
        if canonical_bit_selection is not None:
            self.canonical_bit_selection = canonical_bit_selection

    @abc.abstractmethod
    def accumulate(self, columns: Dict[int, List[Expr]]) -> DefaultDict[int, List[Expr]]:
        raise NotImplementedError

    # ---- canonical-bit-selection helpers ---------------------------------
    #
    # The helpers below are used by concrete PPAs whenever
    # self.canonical_bit_selection is True. They maintain a parallel
    # column representation where each entry is a _LeveledBit carrying
    # the symbolic arrival level and a stable insertion index, so the
    # bit-picking step can always pop the k earliest-arriving bits from
    # a column. Leave self.canonical_bit_selection = False to bypass
    # them entirely and keep the historical behaviour.

    def _wrap_columns(
        self, columns: Dict[int, List[Expr]]
    ) -> DefaultDict[int, List[_LeveledBit]]:
        """Turn a plain ``columns`` dict into level-tracked lists.

        Every PP bit enters at level 1 (a single AND of two inputs) and
        is stamped with a unique, monotonically increasing ``ord_``. The
        counter is stored on the instance so later FA/HA applications
        can keep issuing fresh insertion indices.
        """
        wrapped: DefaultDict[int, List[_LeveledBit]] = defaultdict(list)
        ord_counter = 0
        for weight in sorted(columns.keys()):
            for e in columns[weight]:
                wrapped[weight].append(_LeveledBit(e, level=1, ord_=ord_counter))
                ord_counter += 1
        self._ord_counter = ord_counter
        return wrapped

    def _unwrap_columns(
        self, wrapped: DefaultDict[int, List[_LeveledBit]]
    ) -> DefaultDict[int, List[Expr]]:
        """Strip the level/order metadata and return a plain columns dict."""
        out: DefaultDict[int, List[Expr]] = defaultdict(list)
        for weight, bits in wrapped.items():
            out[weight].extend(b.expr for b in bits)
        return out

    def _take_earliest(
        self, bits: List[_LeveledBit], k: int
    ) -> List[_LeveledBit]:
        """Pop the ``k`` earliest-arrival bits from ``bits`` in place.

        Sort key is ``(arrival_level, ord_)`` — lowest level wins,
        ``ord_`` breaks ties by insertion order. Mirrors
        :meth:`sprouthdl.ml_ppa.environment.PPAEnv._canonical_bits`
        exactly so replayed scripted-policy action logs reproduce
        the same circuit.
        """
        if len(bits) < k:
            raise ValueError(f"column has {len(bits)} bits, need {k}")
        ranked = sorted(range(len(bits)), key=lambda i: (bits[i].level, bits[i].ord_))
        take_idx = sorted(ranked[:k])  # sort by index for stable in-place deletion
        taken = [bits[i] for i in take_idx]
        for i in reversed(take_idx):
            del bits[i]
        return taken

    def _apply_fa_canonical(
        self,
        col_lower: List[_LeveledBit],
        col_upper: List[_LeveledBit],
        full_adder,
    ) -> None:
        """Consume 3 canonical-earliest bits from ``col_lower``, append
        the sum back to the same column and the carry to ``col_upper``,
        both at level ``max(input_levels) + 2`` (symbolic FA depth)."""
        taken = self._take_earliest(col_lower, 3)
        s, c = full_adder(taken[0].expr, taken[1].expr, taken[2].expr)
        new_lvl = max(t.level for t in taken) + 2
        col_lower.append(_LeveledBit(s, new_lvl, self._ord_counter))
        col_upper.append(_LeveledBit(c, new_lvl, self._ord_counter + 1))
        self._ord_counter += 2

    def _apply_ha_canonical(
        self,
        col_lower: List[_LeveledBit],
        col_upper: List[_LeveledBit],
    ) -> None:
        """Half-adder counterpart to :meth:`_apply_fa_canonical`.

        Consumes 2 earliest bits; outputs land at ``max+1``."""
        taken = self._take_earliest(col_lower, 2)
        s, c = half_adder(taken[0].expr, taken[1].expr)
        new_lvl = max(t.level for t in taken) + 1
        col_lower.append(_LeveledBit(s, new_lvl, self._ord_counter))
        col_upper.append(_LeveledBit(c, new_lvl, self._ord_counter + 1))
        self._ord_counter += 2

    def _apply_c42_canonical(
        self,
        col_lower: List[_LeveledBit],
        col_upper: List[_LeveledBit],
        full_adder,
        zero: Expr,
    ) -> None:
        """4:2 compressor implemented as two cascaded FAs, matching
        :meth:`FourTwoCompressorAccumulator._compress_4_2`.

        ``s1, c1 = FA(a, b, c)`` at ``l_inner = max(la, lb, lc) + 2``;
        ``s2, c2 = FA(s1, d, 0)`` at ``l_outer = max(l_inner, ld) + 2``.
        ``c1`` sits at the shallower ``l_inner`` level, ``s2`` and
        ``c2`` at ``l_outer``.
        """
        taken = self._take_earliest(col_lower, 4)
        a, b, c, d = (t.expr for t in taken)
        s1, c1 = full_adder(a, b, c)
        s2, c2 = full_adder(s1, d, zero)
        l_inner = max(taken[0].level, taken[1].level, taken[2].level) + 2
        l_outer = max(l_inner, taken[3].level) + 2
        col_lower.append(_LeveledBit(s2, l_outer, self._ord_counter))
        col_upper.append(_LeveledBit(c1, l_inner, self._ord_counter + 1))
        col_upper.append(_LeveledBit(c2, l_outer, self._ord_counter + 2))
        self._ord_counter += 3

    @staticmethod
    def _column_heights(cols: DefaultDict[int, List[_LeveledBit]]) -> Dict[int, int]:
        """Return a snapshot of ``{weight: height}`` for every column."""
        return {w: len(bits) for w, bits in cols.items()}


class FinalStageAdderBase(StageBase, abc.ABC):
    @abc.abstractmethod
    def resolve(self, columns: Dict[int, List[Expr]], carry_in: Optional[Expr] = None) -> List[Expr]:
        raise NotImplementedError


class CompressorTreeAccumulator(PartialProductAccumulatorBase):
    # Default: legacy FIFO reduction. At widths 12-16 the legacy path
    # is ~37% shallower than canonical on this PPA, and canonical only
    # wins gates by 5-9%. CompressorTree is typically chosen when
    # delay matters, so we keep the original as the default and let
    # users opt into canonical for area-first flows.
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
            reduced = True
            for weight in sorted(cols.keys()):
                bits = cols[weight]
                if len(bits) > 2:
                    reduced = False
                    sum_bits, carry_bits = self._compress_column(bits)
                    next_cols[weight].extend(sum_bits)
                    next_cols[weight + 1].extend(carry_bits)
                else:
                    next_cols[weight].extend(bits)
            cols = next_cols
            if reduced:
                break

        return cols

    def _accumulate_canonical(
        self, columns: Dict[int, List[Expr]]
    ) -> DefaultDict[int, List[Expr]]:
        """Mirrors ``sprouthdl.ml_ppa.methods.scripted_policies.compressor_tree_policy``:
        outer loop reduces each column in ascending weight order using
        per-step earliest-arrival FAs, repeating until every column has
        at most 2 bits. Sums from an FA are appended back to the *same*
        column and become available for the next FA in the same pass.
        """
        cols = self._wrap_columns(columns)
        changed = True
        while changed:
            changed = False
            for weight in sorted(list(cols.keys())):
                while len(cols[weight]) > 2:
                    self._apply_fa_canonical(
                        cols[weight], cols[weight + 1], self._full_adder
                    )
                    changed = True
        return self._unwrap_columns(cols)

    def _compress_column(self, bits: List[Expr]) -> Tuple[List[Expr], List[Expr]]:
        sum_bits: List[Expr] = []
        carry_bits: List[Expr] = []
        work_bits = list(bits)
        while len(work_bits) >= 3:
            x, y, z = work_bits[:3]
            work_bits = work_bits[3:]
            s, c = self._full_adder(x, y, z)
            sum_bits.append(s)
            carry_bits.append(c)
        if len(work_bits) == 2:
            s, c = half_adder(work_bits[0], work_bits[1])
            sum_bits.append(s)
            carry_bits.append(c)
        elif len(work_bits) == 1:
            sum_bits.append(work_bits[0])
        return sum_bits, carry_bits


class RippleCarryFinalAdder(FinalStageAdderBase):
    def resolve(self, columns: Dict[int, List[Expr]], carry_in: Optional[Expr] = None) -> List[Expr]:
        max_weight = self.config.out_width
        result_bits: List[Expr] = []
        carry: Optional[Expr] = carry_in

        for weight in range(max_weight):
            bits = list(columns.get(weight, []))
            if carry is not None:
                bits.append(carry)

            if len(bits) == 0:
                result_bits.append(Const(False, Bool()))
                carry = None
            elif len(bits) == 1:
                result_bits.append(bits[0])
                carry = None
            elif len(bits) == 2:
                s, carry = half_adder(bits[0], bits[1])
                result_bits.append(s)
            elif len(bits) == 3:
                s, carry = full_adder_fast(bits[0], bits[1], bits[2])
                result_bits.append(s)
            else:
                raise ValueError(
                    f"Unexpected number of bits ({len(bits)}) in column {weight} during final addition"
                )

        if carry is not None:
            result_bits.append(carry)

        return result_bits


@dataclass
class StageBasedMultiplierIO:
    a: Signal
    b: Signal
    y: Signal


class StageBasedMultiplierBasic(Component):

    def __init__(
        self,
        a_w: int,
        b_w: int,
        *,
        signed_a: bool = False,
        signed_b: bool = False,
        optim_type: Literal["area", "speed"] = "area",
        ppg_cls: Type[PartialProductGeneratorBase],
        ppa_cls: Type[PartialProductAccumulatorBase] = CompressorTreeAccumulator,
        fsa_cls: Type[FinalStageAdderBase] = RippleCarryFinalAdder,
    ) -> None:
        self.config = StageMultiplierConfig(a_w, b_w, signed_a, signed_b, optim_type)

        supported = ppg_cls.supported_signatures
        if supported is not None and (signed_a, signed_b) not in supported:
            raise ValueError(
                f"{ppg_cls.__name__} does not support signed_a={signed_a}, signed_b={signed_b}"
            )

        base_typ_a = SInt if signed_a else UInt
        base_typ_b = SInt if signed_b else UInt
        base_type_y = SInt if (signed_a or signed_b) else UInt

        self.io : StageBasedMultiplierIO = StageBasedMultiplierIO(
            a=Signal(name="a", typ=base_typ_a(a_w), kind="input"),
            b=Signal(name="b", typ=base_typ_b(b_w), kind="input"),
            y=Signal(name="y", typ=base_type_y(self.config.out_width), kind="output"),
        )

        self.ppg = ppg_cls(self.config)
        self.ppa = ppa_cls(self.config)
        self.fsa = fsa_cls(self.config)

        self.elaborate()

    def elaborate(self) -> None:
        columns = self.ppg.generate_columns(self.io)
        reduced_columns = self.ppa.accumulate(columns)
        if max(reduced_columns.keys()) >= self.config.out_width:
            reduced_columns = {k: v for k, v in reduced_columns.items() if k < self.config.out_width}
        result_bits = self.fsa.resolve(reduced_columns)
        self.io.y <<= Concat(result_bits[:self.config.out_width])

        # debugging
        self.colums = columns
        self.reduced_columns = reduced_columns

@dataclass
class MultiplierTestVectorsInt:
    a_w: int
    b_w: int
    num_vectors: int = 64
    tb_sigma: Optional[float] = None
    signed_a: bool = False
    signed_b: bool = False

    def generate(self) -> Tuple[Dict[str, UInt], List[Tuple[str, Dict[str, int], Dict[str, int]]], None]:
        vecs: List[Tuple[str, Dict[str, int], Dict[str, int]]] = []

        for _ in range(self.num_vectors):
            if self.tb_sigma is not None:

                def rand_unsigned(width: int) -> int:
                    return int(np.round(np.random.normal((1 << (width - 1)), self.tb_sigma)))

                def rand_signed(width: int) -> int:
                    return int(np.round(np.random.normal(0, self.tb_sigma)))

                va = rand_signed(self.a_w) if self.signed_a else rand_unsigned(self.a_w)
                vb = rand_signed(self.b_w) if self.signed_b else rand_unsigned(self.b_w)

                def clamp_unsigned(value: int, width: int) -> int:
                    return max(min(value, (1 << width) - 1), 0)

                def clamp_signed(value: int, width: int) -> int:
                    return max(min(value, (1 << (width - 1)) - 1), -(1 << (width - 1)))

                va = clamp_signed(va, self.a_w) if self.signed_a else clamp_unsigned(va, self.a_w)
                vb = clamp_signed(vb, self.b_w) if self.signed_b else clamp_unsigned(vb, self.b_w)
            else:
                if self.signed_a:
                    va = np.random.randint(-(1 << (self.a_w - 1)), 1 << (self.a_w - 1))
                else:
                    va = np.random.randint(0, 1 << self.a_w)
                if self.signed_b:
                    vb = np.random.randint(-(1 << (self.b_w - 1)), 1 << (self.b_w - 1))
                else:
                    vb = np.random.randint(0, 1 << self.b_w)

            vecs.append((f"{va}*{vb}", {"a": va, "b": vb}, {"y": va * vb}))

        spec = {"a": UInt(self.a_w), "b": UInt(self.b_w), "y": UInt(self.a_w + self.b_w)}
        return spec, vecs, None
