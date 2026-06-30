from dataclasses import dataclass
from typing import DefaultDict, List, Optional, Tuple, Type

from spire.arithmetic.int_multipliers.multipliers.multiplier_stage_core import OptimType, FinalStageAdderBase, SelectionMode, SplitMode, StageMultiplierConfig, PartialProductAccumulatorBase
from spire.expr import Concat, Const, Expr


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

    # Partial product accumulator / compressor tree
    ppa = ppa_cls(config=config)
    ppa_cols = ppa.accumulate(cols)

    # Final stage adder
    fsa = fsa_cls(config=config)
    fsa_bits = fsa.resolve(ppa_cols)  # list of bits, LSB first

    return Concat(fsa_bits)