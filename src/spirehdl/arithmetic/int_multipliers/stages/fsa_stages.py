from __future__ import annotations

from typing import Callable, ClassVar, Dict, List, Optional, Set, Tuple

from spirehdl.arithmetic.int_multipliers.multipliers.multiplier_stage_core import FinalStageAdderBase
from spirehdl.arithmetic.prefix_adders.prefix_adder_topologies import P_brent_kung, P_han_carlson, P_kogge_stone, P_ladner_fischer, P_ripple_carry, P_sklansky, P_sparse_kogge_stone_2, P_sparse_kogge_stone_4, Pair, analyze_prefix_matrix, legalize_P
from spirehdl.arithmetic.prefix_adders.prefix_adder_specials import ZCG_n, multi_scan_n
from spirehdl.spirehdl import Bool, Concat, Const, Expr, UInt, fit_type


def _exists(nodes: Set[Pair], i: int, j: int) -> bool:
    return (i == j) or ((i, j) in nodes)


def _find_split(nodes: Set[Pair], i: int, j: int) -> int | None:
    for k in range(i - 1, j - 1, -1):
        if _exists(nodes, i, k + 1) and _exists(nodes, k, j):
            return k
    return None

class PlusOperatorAdderFinalStage(FinalStageAdderBase):

    def resolve(self, columns: Dict[int, List[Expr]], carry_in: Optional[Expr] = None) -> List[Expr]:

        # columns to UInt inputs
        width = self.config.out_width
        max_col = max(columns.keys(), default=width - 1)
        working_width = max(width, max_col + 1)
        row_a: List[Expr] = []
        row_b: List[Expr] = []
        zero = Const(False, Bool())
        for idx in range(working_width):
            bits = list(columns.get(idx, []))
            if len(bits) > 2:
                raise ValueError(
                    f"Column {idx} has {len(bits)} bits; expected at most 2 before addition"
                )
            if len(bits) == 0:
                row_a.append(zero)
                row_b.append(zero)
            elif len(bits) == 1:
                row_a.append(bits[0])
                row_b.append(zero)
            else:  # len == 2
                row_a.append(bits[0])
                row_b.append(bits[1])

        a_unit = fit_type(Concat(row_a), UInt(working_width))
        b_unit = fit_type(Concat(row_b), UInt(working_width))
        sum_unit = a_unit + b_unit
        if carry_in is not None:
            cin = fit_type(Concat([carry_in]), UInt(working_width))
            sum_unit = sum_unit + cin
        return [sum_unit[i] for i in range(sum_unit.typ.width)]


class NaiveRippleCarryFinalStage(FinalStageAdderBase):
    """Textbook ripple-carry adder final stage, emitted as a direct chain.

    NAIVE_RIPPLE_CARRY: textbook a^b^c / (a&b)|(c&(a^b)) recurrence, emitted
    as a direct chain. Beats every other FSA at width 8-16 when the design
    is synthesised via the yosys read_verilog path that downstream consumers
    use (e.g. vendor tools).
    
    Implementation note: we deliberately stop the FA chain at column
    ``out_width - 1`` and only produce the final carry-out if
    ``full_output_bit=True``. Truncating the chain (vs computing the MSB
    carry and relying on yosys DCE) lets yosys see a tighter logic cone
    and produces noticeably smaller netlists.
    """

    def resolve(self, columns: Dict[int, List[Expr]], carry_in: Optional[Expr] = None) -> List[Expr]:
        out_w = self.config.out_width
        zero = Const(False, Bool())
        carry: Expr = carry_in if carry_in is not None else zero
        sum_bits: List[Expr] = []

        for i in range(out_w):
            bits = list(columns.get(i, []))
            if len(bits) > 2:
                raise ValueError(
                    f"NaiveRippleCarryFinalStage expects ≤2 bits/column, "
                    f"got {len(bits)} at column {i}"
                )
            if len(bits) == 0:
                # Empty spill column (only happens at column out_w-1 when
                # full_output_bit=True for an unsigned adder). Just propagate
                # the incoming carry as the final sum bit.
                sum_bits.append(carry)
                carry = zero
            elif len(bits) == 1:
                # Effectively a half-adder of (lone_bit, carry).
                a = bits[0]
                sum_bits.append(a ^ carry)
                carry = a & carry
            else:  # len(bits) == 2
                # Textbook FA recurrence with the (a^b) partial propagate
                # shared between the sum and carry expressions. This mirrors
                # ``full_adder_fast`` but is inlined here to avoid the
                # per-column ``half_adder`` + ``full_adder_fast`` split that
                # ``RippleCarryFinalAdder`` uses; the flatter chain maps to
                # a cheaper cell mix through yosys ``read_verilog`` tech map.
                a, b = bits[0], bits[1]
                p = a ^ b
                sum_bits.append(p ^ carry)
                carry = (a & b) | (carry & p)

        return sum_bits


class PrefixAdderFinalStage(FinalStageAdderBase):
    """Final stage adder that realises a chosen prefix network."""

    prefix_matrix_builder: ClassVar[Callable[[int], Set[Pair]]] = staticmethod(
        None # to be overridden by subclasses
    )
    depth_optimize: ClassVar[bool] = True

    def resolve(self, columns: Dict[int, List[Expr]], carry_in: Optional[Expr] = None) -> List[Expr]:
        """Collapse <=2 bits/column into a final sum using a prefix carry network.

        `columns` is expected to come from the compressor tree, so each column
        must contain at most two bits (carry-save form). The method:
        1) normalizes sparse columns into two dense operand rows,
        2) computes bitwise propagate/generate signals,
        3) evaluates group (G, P) terms defined by the selected prefix topology,
        4) derives carries and returns sum bits plus the final carry-out bit.
        """
        # Include any populated spill column beyond the configured output width.
        width = self.config.out_width
        max_col = max(columns.keys(), default=width - 1)
        working_width = max(width, max_col + 1)

        zero = Const(False, Bool())
        row_a: List[Expr] = []
        row_b: List[Expr] = []

        # Convert sparse per-column bit lists into two aligned addend rows.
        for idx in range(working_width):
            bits = list(columns.get(idx, []))
            if len(bits) > 2:
                raise ValueError(
                    f"Column {idx} has {len(bits)} bits; expected at most 2 before prefix addition"
                )
            if len(bits) == 0:
                row_a.append(zero)
                row_b.append(zero)
            elif len(bits) == 1:
                row_a.append(bits[0])
                row_b.append(zero)
            else:  # len == 2
                row_a.append(bits[0])
                row_b.append(bits[1])

        # Per-bit propagate/generate for the two-row representation.
        propagates = [a ^ b for a, b in zip(row_a, row_b)]
        generates = [a & b for a, b in zip(row_a, row_b)]

        # Build and sanitize the chosen prefix topology for this width.
        raw_nodes = set(self.prefix_matrix_builder(working_width))
        nodes = legalize_P(working_width, raw_nodes)

        best_k: Dict[Pair, int] = {}
        if self.depth_optimize and nodes:
            # Precompute split points that minimize recursion depth.
            _, best_k, _, _ = analyze_prefix_matrix(working_width, nodes)

        # Cache group (G, P) results for node intervals (i, j).
        gp_cache: Dict[Pair, Tuple[Expr, Expr]] = {}

        def gp(i: int, j: int) -> Tuple[Expr, Expr]:
            key = (i, j)
            if key in gp_cache:
                return gp_cache[key]
            if i == j:
                result = (generates[i], propagates[i])
                gp_cache[key] = result
                return result
            if key not in nodes:
                raise ValueError(
                    f"Prefix matrix missing node {(i, j)} required for carry propagation"
                )
            if self.depth_optimize:
                k = best_k[key]
            else:
                k = _find_split(nodes, i, j)
                if k is None:
                    raise ValueError(f"No legal split for prefix node {(i, j)}")
            g_left, p_left = gp(i, k + 1)
            g_right, p_right = gp(k, j)
            # Prefix combine: (G,P) = (G_hi | (P_hi & G_lo), P_hi & P_lo).
            result = (g_left | (p_left & g_right), p_left & p_right)
            gp_cache[key] = result
            return result

        # c[0] is Cin; c[i+1] is carry into bit i+1.
        carries: List[Expr] = [zero] * (working_width + 1)
        if carry_in is not None:
            carries[0] = carry_in
        for idx in range(working_width):
            g_prefix, p_prefix = gp(idx, 0)
            carries[idx + 1] = g_prefix | (p_prefix & carries[0])

        # Sum bits and append final carry-out.
        sums = [propagates[idx] ^ carries[idx] for idx in range(working_width)]
        result_bits: List[Expr] = list(sums)
        result_bits.append(carries[working_width])
        return result_bits


class KoggeStonePrefixFinalStage(PrefixAdderFinalStage):
    prefix_matrix_builder: ClassVar[Callable[[int], Set[Pair]]] = staticmethod(P_kogge_stone)

class BrentKungPrefixFinalStage(PrefixAdderFinalStage):
    prefix_matrix_builder: ClassVar[Callable[[int], Set[Pair]]] = staticmethod(P_brent_kung)

class SklanskyPrefixFinalStage(PrefixAdderFinalStage):
    prefix_matrix_builder: ClassVar[Callable[[int], Set[Pair]]] = staticmethod(P_sklansky)

class RipplePrefixFinalStage(PrefixAdderFinalStage):
    prefix_matrix_builder: ClassVar[Callable[[int], Set[Pair]]] = staticmethod(P_ripple_carry)
    depth_optimize: ClassVar[bool] = False

class HanCarlsonPrefixFinalStage(PrefixAdderFinalStage):
    prefix_matrix_builder: ClassVar[Callable[[int], Set[Pair]]] = staticmethod(P_han_carlson)

class LadnerFischerPrefixFinalStage(PrefixAdderFinalStage):
    prefix_matrix_builder: ClassVar[Callable[[int], Set[Pair]]] = staticmethod(P_ladner_fischer)

class SparseKoggeStone2PrefixFinalStage(PrefixAdderFinalStage):
    prefix_matrix_builder: ClassVar[Callable[[int], Set[Pair]]] = staticmethod(P_sparse_kogge_stone_2)

class SparseKoggeStone4PrefixFinalStage(PrefixAdderFinalStage):
    prefix_matrix_builder: ClassVar[Callable[[int], Set[Pair]]] = staticmethod(P_sparse_kogge_stone_4)

class MultiScanPrefixFinalStage(PrefixAdderFinalStage):
    prefix_matrix_builder: ClassVar[Callable[[int], Set[Pair]]] = staticmethod(multi_scan_n)

class ZCGPrefixFinalStage(PrefixAdderFinalStage):
    prefix_matrix_builder: ClassVar[Callable[[int], Set[Pair]]] = staticmethod(ZCG_n)
