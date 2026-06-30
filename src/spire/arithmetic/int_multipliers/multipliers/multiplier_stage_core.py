from __future__ import annotations

import abc
from collections import defaultdict
from dataclasses import dataclass
from typing import ClassVar, DefaultDict, Dict, List, Literal, Optional, Tuple, Type

import numpy as np

from spire.arithmetic.int_multipliers.eval.testvector_generation import Encoding
from spire.component import Component
from spire.expr import Bool, Concat, Const, Expr, Signal, SInt, UInt, mux
from spire.io_record import IORecord, Input, Output


# ---- common arithmetic helpers -------------------------------------------------

def half_adder(x: Expr, y: Expr) -> Tuple[Expr, Expr]:
    return x ^ y, x & y  # sum, carry


def full_adder_fast(x: Expr, y: Expr, z: Expr) -> Tuple[Expr, Expr]:
    """Shared-XOR full adder: 5 gates, partial-propagate `s1=x^y` reused
    in sum and carry. OR operand order has the `(s1&z)` term first."""
    s1 = x ^ y
    return s1 ^ z, (s1 & z) | (x & y)


def full_adder_fast2(x: Expr, y: Expr, z: Expr) -> Tuple[Expr, Expr]:
    """Logically identical to `full_adder_fast` (same 5-gate shared-XOR
    structure), but emits the carry-out with the `(x&y)` term first
    (`(x & y) | (z & s1)` instead of `(s1 & z) | (x & y)`).

    The two forms compute the same boolean function, but the operand
    order affects the emitted Verilog wire ordering and thus can impact
    downstream techmap optimizations.
    """
    s1 = x ^ y
    return s1 ^ z, (x & y) | (z & s1)


def full_adder_low_area(x: Expr, y: Expr, z: Expr) -> Tuple[Expr, Expr]:
    """Textbook majority full adder: 7 gates, no sharing. Sometimes wins
    when downstream techmap can re-factor the symmetric majority form."""
    s = x ^ y ^ z
    return s, (x & y) | (y & z) | (z & x)


OptimType = Literal["area", "speed"]
SelectionMode = Literal["fifo", "lifo", "earliest"]
# Prefix-FSA carry-tree split strategy (consumed in fsa_stages.py). Defined here, next to SelectionMode, so config
# objects can carry it without importing the FSA module (which would be an import cycle).
SplitMode = Literal["min_depth_splits", "first_split"]


class _LeveledBit:
    """A partial-product-tree bit annotated with symbolic arrival level
    and stable insertion order.

    Used by all PPA selection modes. The *schedule* remains
    algorithm-specific; only the bit-picking rule changes:

    - ``fifo``: left-to-right consumption (historical CompressorTree)
    - ``lifo``: stack-style ``pop()`` consumption (historical Wallace/Dadda/CarrySave/FourTwo)
    - ``earliest``: earliest-arrival-first by ``(level, ord_)``
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
    optim_type: OptimType = "area"
    selection_mode: Optional[SelectionMode] = None
    split_mode: Optional[SplitMode] = None

    @property
    def out_width(self) -> int:
        raise NotImplementedError


@dataclass(frozen=True)
class StageMultiplierConfig(TwoInputAritConfig):  # might be renamed to StageMultiplierConfig
    a_width: int
    b_width: int
    signed_a: bool
    signed_b: bool
    optim_type: OptimType

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
    # Default bit-selection mode for the column reduction loop.
    #
    # Three modes are available:
    #   "fifo"      — consume bits from the front of the column (FIFO)
    #   "lifo"      — consume bits from the back of the column (LIFO / pop)
    #   "earliest"  — consume the k earliest-arriving bits by (level, ord_)
    #
    # Per-PPA defaults are set on concrete subclasses. Users can
    # override per-instance via the ``selection_mode`` constructor arg.
    default_selection_mode: ClassVar[SelectionMode] = "lifo"

    def __init__(
        self,
        config: TwoInputAritConfig,
        *,
        selection_mode: Optional[SelectionMode] = None,
    ) -> None:
        super().__init__(config)
        # Resolution order: explicit arg -> config.selection_mode -> this class's default.
        mode = selection_mode if selection_mode is not None else getattr(config, "selection_mode", None)
        self.selection_mode: SelectionMode = mode if mode is not None else self.__class__.default_selection_mode
        self._ord_counter = 0

    @abc.abstractmethod
    def accumulate(self, columns: Dict[int, List[Expr]]) -> DefaultDict[int, List[Expr]]:
        raise NotImplementedError

    # ---- _LeveledBit column helpers -------------------------------------

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

    # ---- bit-selection dispatch -----------------------------------------

    def _take_bits(
        self, bits: List[_LeveledBit], k: int,
    ) -> List[_LeveledBit]:
        """Pop ``k`` bits from ``bits`` using ``self.selection_mode``."""
        if self.selection_mode == "earliest":
            return self._take_earliest(bits, k)
        if self.selection_mode == "lifo":
            return self._take_lifo(bits, k)
        if self.selection_mode == "fifo":
            return self._take_fifo(bits, k)
        raise ValueError(f"unsupported selection mode: {self.selection_mode}")

    @staticmethod
    def _take_lifo(bits: List[_LeveledBit], k: int) -> List[_LeveledBit]:
        if len(bits) < k:
            raise ValueError(f"column has {len(bits)} bits, need {k}")
        return [bits.pop() for _ in range(k)]

    @staticmethod
    def _take_fifo(bits: List[_LeveledBit], k: int) -> List[_LeveledBit]:
        if len(bits) < k:
            raise ValueError(f"column has {len(bits)} bits, need {k}")
        return [bits.pop(0) for _ in range(k)]

    def _take_earliest(
        self, bits: List[_LeveledBit], k: int
    ) -> List[_LeveledBit]:
        """Pop the ``k`` earliest-arrival bits from ``bits`` in place.

        Sort key is ``(arrival_level, ord_)`` — lowest level wins,
        ``ord_`` breaks ties by insertion order.
        """
        if len(bits) < k:
            raise ValueError(f"column has {len(bits)} bits, need {k}")
        ranked = sorted(range(len(bits)), key=lambda i: (bits[i].level, bits[i].ord_))
        take_idx = sorted(ranked[:k])  # sort by index for stable in-place deletion
        taken = [bits[i] for i in take_idx]
        for i in reversed(take_idx):
            del bits[i]
        return taken

    # ---- unified FA / HA / 4:2 helpers ----------------------------------
    #
    # These use _take_bits() which dispatches to the active selection
    # mode. ``col_lower`` is both the source of input bits and the
    # destination for the sum; ``col_upper`` receives carries. For
    # next_cols-style schedules, pass the working copy as col_lower
    # and next_cols[weight+1] as col_upper.

    def _apply_fa(
        self,
        col_lower: List[_LeveledBit],
        col_upper: List[_LeveledBit],
        full_adder,
    ) -> None:
        """Consume 3 bits from ``col_lower``, produce sum + carry."""
        taken = self._take_bits(col_lower, 3)
        s, c = full_adder(taken[0].expr, taken[1].expr, taken[2].expr)
        new_lvl = max(t.level for t in taken) + 2
        col_lower.append(_LeveledBit(s, new_lvl, self._ord_counter))
        col_upper.append(_LeveledBit(c, new_lvl, self._ord_counter + 1))
        self._ord_counter += 2

    def _apply_ha(
        self,
        col_lower: List[_LeveledBit],
        col_upper: List[_LeveledBit],
    ) -> None:
        """Consume 2 bits from ``col_lower``, produce sum + carry."""
        taken = self._take_bits(col_lower, 2)
        s, c = half_adder(taken[0].expr, taken[1].expr)
        new_lvl = max(t.level for t in taken) + 1
        col_lower.append(_LeveledBit(s, new_lvl, self._ord_counter))
        col_upper.append(_LeveledBit(c, new_lvl, self._ord_counter + 1))
        self._ord_counter += 2

    def _apply_c42(
        self,
        col_lower: List[_LeveledBit],
        col_upper: List[_LeveledBit],
        full_adder,
        zero: Expr,
    ) -> None:
        """4:2 compressor: two cascaded FAs consuming 4 bits."""
        taken = self._take_bits(col_lower, 4)
        a, b, c, d = (t.expr for t in taken)
        s1, c1 = full_adder(a, b, c)
        s2, c2 = full_adder(s1, d, zero)
        l_inner = max(taken[0].level, taken[1].level, taken[2].level) + 2
        l_outer = max(l_inner, taken[3].level) + 2
        col_lower.append(_LeveledBit(s2, l_outer, self._ord_counter))
        col_upper.append(_LeveledBit(c1, l_inner, self._ord_counter + 1))
        col_upper.append(_LeveledBit(c2, l_outer, self._ord_counter + 2))
        self._ord_counter += 3

    def _pop(self, bits: List[Expr], n: int = 1) -> List[Expr]:
        """Pop ``n`` raw bits from a column honoring fifo (front) / lifo (back). Used by the next_cols buffer-swap
        schedules; the earliest schedule routes through ``_take_bits`` instead and never reaches this."""
        end = 0 if self.selection_mode == "fifo" else -1
        return [bits.pop(end) for _ in range(n)]

    def _apply_compress(self, col_lower: List[_LeveledBit], col_upper: List[_LeveledBit], k: int, compress_fn) -> None:
        """Earliest-schedule k:2 / k:3 compressor. ``compress_fn(*bits) -> (sum, *carries)`` is taken off ``self`` so a
        subclass gate override (e.g. the parallel compressors) is honored; the k input bits follow the active mode."""
        taken = self._take_bits(col_lower, k)
        sum_expr, *carries = compress_fn(*[t.expr for t in taken])
        lvl = max(t.level for t in taken) + 2
        col_lower.append(_LeveledBit(sum_expr, lvl, self._ord_counter))
        self._ord_counter += 1
        for carry in carries:
            col_upper.append(_LeveledBit(carry, lvl, self._ord_counter))
            self._ord_counter += 1

    @staticmethod
    def _column_heights(cols: DefaultDict[int, List[_LeveledBit]]) -> Dict[int, int]:
        """Return a snapshot of ``{weight: height}`` for every column."""
        return {w: len(bits) for w, bits in cols.items()}


class FinalStageAdderBase(StageBase, abc.ABC):
    @abc.abstractmethod
    def resolve(self, columns: Dict[int, List[Expr]], carry_in: Optional[Expr] = None) -> List[Expr]:
        raise NotImplementedError


class CompressorTreeAccumulator(PartialProductAccumulatorBase):
    # Default: FIFO reduction. At widths 12-16 the FIFO path is ~37%
    # shallower than earliest, and earliest only wins gates by 5-9%.
    # CompressorTree is typically chosen when delay matters, so we keep
    # FIFO as the default and let users opt into earliest for area-first
    # flows.
    default_selection_mode: ClassVar[SelectionMode] = "fifo"

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
        if self.selection_mode == "earliest":
            return self._accumulate_earliest(columns)
        return self._accumulate_fifo_lifo(columns)

    def _accumulate_fifo_lifo(
        self, columns: Dict[int, List[Expr]]
    ) -> DefaultDict[int, List[Expr]]:
        """FIFO/LIFO schedule: each pass isolates results in a ``next_cols``
        buffer. Columns are compressed in ascending weight order; sums
        and carries land in the next pass's buffers, not the current one.
        """
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

    def _accumulate_earliest(
        self, columns: Dict[int, List[Expr]]
    ) -> DefaultDict[int, List[Expr]]:
        """Earliest schedule: in-place greedy reduction. Each column is
        reduced until height <= 2 before moving to the next weight.
        """
        cols = self._wrap_columns(columns)
        changed = True
        while changed:
            changed = False
            for weight in sorted(list(cols.keys())):
                while len(cols[weight]) > 2:
                    self._apply_fa(
                        cols[weight], cols[weight + 1], self._full_adder
                    )
                    changed = True
        return self._unwrap_columns(cols)

    def _compress_column(self, bits: List[Expr]) -> Tuple[List[Expr], List[Expr]]:
        sum_bits: List[Expr] = []
        carry_bits: List[Expr] = []
        # fifo consumes from the front (default); lifo consumes from the back. earliest never routes here.
        work_bits = list(bits) if self.selection_mode != "lifo" else list(reversed(bits))
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
        # Pick FA form per optim_type so the eval DB can record real
        # area-vs-speed deltas. Previously this hardcoded full_adder_fast
        # (shared XOR), making optim_type a no-op for this FSA.
        _fa = (
            full_adder_low_area
            if self.config.optim_type == "area"
            else full_adder_fast
        )

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
                s, carry = _fa(bits[0], bits[1], bits[2])
                result_bits.append(s)
            else:
                raise ValueError(
                    f"Unexpected number of bits ({len(bits)}) in column {weight} during final addition"
                )

        if carry is not None:
            result_bits.append(carry)

        return result_bits


class StageBasedMultiplierIO(IORecord):
    """IO for stage-based multipliers: operands ``a``, ``b`` and product ``y``."""


class StageBasedMultiplierBasic(Component):

    def __init__(
        self,
        a_w: int,
        b_w: int,
        *,
        signed_a: bool = False,
        signed_b: bool = False,
        optim_type: OptimType = "area",
        ppg_cls: Type[PartialProductGeneratorBase],
        ppa_cls: Type[PartialProductAccumulatorBase] = CompressorTreeAccumulator,
        fsa_cls: Type[FinalStageAdderBase] = RippleCarryFinalAdder,
        selection_mode: Optional[SelectionMode] = None,
        split_mode: Optional[SplitMode] = None,
    ) -> None:
        self.config = StageMultiplierConfig(a_w, b_w, signed_a, signed_b, optim_type,
                                            selection_mode=selection_mode, split_mode=split_mode)

        supported = ppg_cls.supported_signatures
        if supported is not None and (signed_a, signed_b) not in supported:
            raise ValueError(
                f"{ppg_cls.__name__} does not support signed_a={signed_a}, signed_b={signed_b}"
            )

        base_typ_a = SInt if signed_a else UInt
        base_typ_b = SInt if signed_b else UInt
        base_type_y = SInt if (signed_a or signed_b) else UInt

        self.io : StageBasedMultiplierIO = StageBasedMultiplierIO(
            a=Input(base_typ_a(a_w)),
            b=Input(base_typ_b(b_w)),
            y=Output(base_type_y(self.config.out_width)),
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
