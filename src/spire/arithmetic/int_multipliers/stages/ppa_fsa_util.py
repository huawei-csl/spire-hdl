from dataclasses import dataclass
from typing import DefaultDict, List, Optional, Tuple, Type

from spire.arithmetic.int_multipliers.multipliers.multiplier_stage_core import OptimType, FinalStageAdderBase, SelectionMode, SplitMode, StageMultiplierConfig, PartialProductAccumulatorBase
from spire.expr import Bool, Concat, Const, Expr


@dataclass
class OutputConfig:
    out_width: int
    optim_type: OptimType
    selection_mode: Optional[SelectionMode] = None
    split_mode: Optional[SplitMode] = None

def compressor_sum(
    config: StageMultiplierConfig | OutputConfig,
    partials: List[Tuple[Expr, int] | Expr],
    ppa_cls: Type[PartialProductAccumulatorBase],
    fsa_cls: Type[FinalStageAdderBase],
) -> Expr:
    """
    Build a compressor tree from a set of partial products.

    Args:
        config:   MultiplierConfig or OutputConfig for the multiplier.
        partials: list of (signal, lsb_offset) tuples.
                  Each signal is a multi-bit Expr; lsb_offset is the bit weight
                  of signal[0] in the final sum.
                  OR
                  listof multi-bit Exprs (equivalent to lsb_offset == 0).
        
        ppg_cls:  partial-product accumulator / compressor-tree class
                  (e.g. CompressorTreeAccumulator, CarrySaveAccumulator).
        fsa_cls:  final-stage adder class (e.g. RippleCarryFinalAdder).

    Returns:
        Expr for the final sum.
    """
    from collections import defaultdict

    # Build Dict[int, List[Expr]]: column -> list of bits
    cols: DefaultDict[int, List["Expr"]] = defaultdict(list)

    def unpack_partials(partials: List[Tuple[Expr, int] | Expr]) -> List[Tuple[Expr, int]]:
        """
        Unpack partial products, expanding Concats into individual bits with offsets.
        """
        partials_unpacked: List[Tuple[Expr, int]] = []
        for sig_offset in partials:

            if isinstance(sig_offset, tuple):
                item, offset = sig_offset
            else:
                item = sig_offset
                offset = 0

            index = 0
            if isinstance(item, Concat):
                for part in item.parts:
                    
                    # if part is Const, skip it
                    if isinstance(part, Const) and getattr(part, "value", None) == 0:
                        index += part.typ.width
                        continue
                    
                    partials_unpacked.append((part, index + offset))
                    index += part.typ.width
            else:
                partials_unpacked.append((item, offset))
        return partials_unpacked

    partials_unpacked = unpack_partials(partials)

    for sig, offset in partials_unpacked:

        width = sig.typ.width
        for i in range(width):
            bit = sig[i]
            # Skip literal zero bits so they don't bloat the tree
            if isinstance(bit, Const) and getattr(bit, "value", None) == 0:
                continue
            cols[i + offset].append(bit)

    return compressor_sum_columns(config, cols, ppa_cls, fsa_cls)


def sign_extension_columns(operands: List[Expr], out_w: int) -> DefaultDict[int, List[Expr]]:
    """Weight->bits columns for `sum(operands)` with signed operands in sign-extension-compression
    form: a signed operand of width w contributes its low bits, its INVERTED sign bit at column
    w-1, and the constant -2^(w-1); the constants of all operands fold into one constant row
    (mod 2^out_w). Unsigned operands contribute their bits as-is (zero-extension is implicit).

    Shared by `build_multi_input_add` (the builder) and `eval_mia` (the sweep), so the DB rows
    measure exactly the circuit the builder produces."""
    from collections import defaultdict

    cols: DefaultDict[int, List[Expr]] = defaultdict(list)
    const_total = 0
    for op in operands:
        w = op.typ.width
        if op.typ.signed:
            for k in range(w - 1):
                cols[k].append(op[k])
            cols[w - 1].append(~op[w - 1])
            const_total -= 1 << (w - 1)
        else:
            for k in range(w):
                cols[k].append(op[k])
    correction = const_total & ((1 << out_w) - 1)
    for k in range(out_w):
        if (correction >> k) & 1:
            cols[k].append(Const(True, Bool()))
    return cols


def compressor_sum_columns(
    config: StageMultiplierConfig | OutputConfig,
    cols: DefaultDict[int, List[Expr]],
    ppa_cls: Type[PartialProductAccumulatorBase],
    fsa_cls: Type[FinalStageAdderBase],
) -> Expr:
    """Reduce prebuilt weight->bits columns via a compressor tree and a final-stage adder.

    The column-level core of `compressor_sum`, for callers that assemble the columns
    themselves (e.g. the sign-extension handling in the signed multi-input adder).
    """
    # Partial product accumulator / compressor tree
    ppa = ppa_cls(config=config)
    ppa_cols = ppa.accumulate(cols)

    # Final stage adder
    fsa = fsa_cls(config=config)
    fsa_bits = fsa.resolve(ppa_cols)  # list of bits, LSB first

    return Concat(fsa_bits)