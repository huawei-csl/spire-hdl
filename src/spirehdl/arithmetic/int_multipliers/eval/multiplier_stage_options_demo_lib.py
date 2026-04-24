from enum import Enum
from typing import List, NamedTuple, Self, Tuple, Type, TypeVar, Optional

from spirehdl.arithmetic.int_multipliers.multipliers.multipliers_ext_karatsuba import KaratsubaMultiplier, KaratsubaMultiplierFromOptimized4BitBlocks
from spirehdl.arithmetic.int_multipliers.multipliers.multipliers_ext_optimized import OptimizedMultiplierFrom4BitBlocks, OptimizedMultiplierFrom4BitBlocksStrong, OptimizedMultiplier, OptimizedSignMagnitudeMultiplier
from spirehdl.arithmetic.int_multipliers.multipliers.mutipliers_ext import StageBasedMultiplierBase, StageBasedMultiplier, StageBasedSignMagnitudeExtMultiplier, StageBasedSignMagnitudeExtToTwosComplementMultiplier, StageBasedSignMagnitudeExtToTwosComplementUpperMultiplier, StageBasedSignMagnitudeExtUpMultiplier, StageBasedSignMagnitudeMultiplier, StageBasedSignMagnitudeToTwosComplementMultiplier, StarMultiplier
from spirehdl.arithmetic.int_multipliers.stages.ppa_stages import BalancedDelayWallaceAccumulator, CarrySaveAccumulator, DaddaTreeAccumulator, EagerWallaceAccumulator, FiveTwoCompressorAccumulator, FiveTwoCompressorParallelAccumulator, FourTwoCompressorAccumulator, FourTwoCompressorParallelAccumulator, WallaceTreeAccumulator
from spirehdl.arithmetic.int_multipliers.eval.testvector_generation import Encoding, MultiplierTestVectors, to_encoding
from spirehdl.arithmetic.int_multipliers.stages.ppg_baugh_wooley_stages import BaughWooleyPartialProductGenerator
from spirehdl.arithmetic.int_multipliers.stages.ppg_and_stages import AndPartialProductGenerator
from spirehdl.arithmetic.int_multipliers.stages.ppg_booth_optim_stages import BoothOptimizedPartialProductGenerator
from spirehdl.arithmetic.int_multipliers.stages.ppg_booth_precomputed_b_stages import BoothPrecomputedBPartialProductGenerator
from spirehdl.arithmetic.int_multipliers.stages.ppg_booth_unoptim_stages import BoothUnoptimizedPartialProductGenerator
from spirehdl.arithmetic.int_multipliers.multipliers.multiplier_stage_core import CompressorTreeAccumulator, RippleCarryFinalAdder, StageBasedMultiplierBasic
from spirehdl.arithmetic.int_multipliers.stages.fsa_stages import BrentKungPrefixFinalStage, HanCarlsonPrefixFinalStage, KoggeStonePrefixFinalStage, LadnerFischerPrefixFinalStage, MultiScanPrefixFinalStage, NaiveRippleCarryFinalStage, PlusOperatorAdderFinalStage, PrefixAdderFinalStage, RipplePrefixFinalStage, SklanskyPrefixFinalStage, SparseKoggeStone2PrefixFinalStage, SparseKoggeStone4PrefixFinalStage, ZCGPrefixFinalStage


# Options for each stage

# Partial Product Generator option values have type Type[PartialProductGeneratorBase]
class PPGOption(Enum):
    AND = AndPartialProductGenerator # and partial products
    BAUGH_WOOLEY = BaughWooleyPartialProductGenerator
    BOOTH_UNOPTIMISED = BoothUnoptimizedPartialProductGenerator
    BOOTH_OPTIMISED = BoothOptimizedPartialProductGenerator
    #BOOTH_OPTIMISED_PRECOMPUTED_B = BoothPrecomputedBPartialProductGenerator # doesnt really help compared to BOOTH_OPTIMISED, but included for completeness
    NONE = None


# Partial Product Accumulator option values have type Type[PartialProductAccumulatorBase]
class PPAOption(Enum):
    ACCUMULATOR_TREE = CompressorTreeAccumulator
    WALLACE_TREE = WallaceTreeAccumulator
    EAGER_WALLACE_TREE = EagerWallaceAccumulator # was not really in the pareto front
    BDT_WALLACE_TREE = BalancedDelayWallaceAccumulator # sometimes beats wallace tree for signed mult
    DADDA_TREE = DaddaTreeAccumulator
    CARRY_SAVE_TREE = CarrySaveAccumulator
    FOUR_TWO_COMPRESSOR = FourTwoCompressorAccumulator
    FOUR_TWO_COMPRESSOR_PARALLEL = FourTwoCompressorParallelAccumulator # seemed worse than the non-parallel version
    FIVE_TWO_COMPRESSOR = FiveTwoCompressorAccumulator # worse than 4:2
    FIVE_TWO_COMPRESSOR_PARALLEL = FiveTwoCompressorParallelAccumulator # worse than 4:2
    NONE = None

# Final Stage Adder option values have type Type[FinalStageAdderBase]
class FSAOption(Enum):
    RIPPLE_CARRY = RippleCarryFinalAdder
    NAIVE_RIPPLE_CARRY = NaiveRippleCarryFinalStage
    PREFIX_KOGGE_STONE = KoggeStonePrefixFinalStage
    PREFIX_BRENT_KUNG = BrentKungPrefixFinalStage
    PREFIX_SKLANSKY = SklanskyPrefixFinalStage
    PREFIX_LADNER_FISCHER = LadnerFischerPrefixFinalStage
    PREFIX_SPARSE_KOGGE_STONE_2 = SparseKoggeStone2PrefixFinalStage
    PREFIX_SPARSE_KOGGE_STONE_4 = SparseKoggeStone4PrefixFinalStage
    PREFIX_RCA = RipplePrefixFinalStage
    PREFIX_MULTI_SCAN = MultiScanPrefixFinalStage
    PREFIX_ZCG = ZCGPrefixFinalStage
    PREFIX_HAN_CARLSON = HanCarlsonPrefixFinalStage
    PLUS_OPERATOR = PlusOperatorAdderFinalStage
    NONE = None

# Multiplier option values have type Type[StageBasedMultiplierBase]
class MultiplierOption(Enum):
    STAGE_BASED_MULTIPLIER = StageBasedMultiplier
    STAGE_BASED_SIGN_MAGNITUDE_MULTIPLIER = StageBasedSignMagnitudeMultiplier
    STAGE_BASED_SIGN_MAGNITUDE_EXT_MULTIPLIER = StageBasedSignMagnitudeExtMultiplier
    STAGE_BASED_SIGN_MAGNITUDE_EXT_UP_MULTIPLIER = StageBasedSignMagnitudeExtUpMultiplier
    STAGE_BASED_SIGN_MAGNITUDE_TO_TWOS_COMPLEMENT_MULTIPLIER = StageBasedSignMagnitudeToTwosComplementMultiplier
    STAGE_BASED_SIGN_MAGNITUDE_EXT_TO_TWOS_COMPLEMENT_MULTIPLIER = StageBasedSignMagnitudeExtToTwosComplementMultiplier
    STAGE_BASED_SIGN_MAGNITUDE_EXT_TO_TWOS_COMPLEMENT_UPPER_MULTIPLIER = StageBasedSignMagnitudeExtToTwosComplementUpperMultiplier
    STAR_MULTIPLIER = StarMultiplier
    OPTIMIZED_MULTIPLIER = OptimizedMultiplier
    OPTIMIZED_SIGN_MAGNITUDE_MULTIPLIER = OptimizedSignMagnitudeMultiplier
    OPTIMIZED_MULTIPLIER_FROM_4BIT_BLOCKS = OptimizedMultiplierFrom4BitBlocks
    OPTIMIZED_MULTIPLIER_FROM_4BIT_BLOCKS_STRONG = OptimizedMultiplierFrom4BitBlocksStrong
    KARATSUBA_MULTIPLIER = KaratsubaMultiplier
    KARATSUBA_MULTIPLIER_FROM_OPTIMIZED_4BIT_BLOCKS = KaratsubaMultiplierFromOptimized4BitBlocks

def supports_stages(multiplier_option: MultiplierOption) -> bool:
    stages_not_supported = [
        MultiplierOption.STAR_MULTIPLIER,
        MultiplierOption.OPTIMIZED_MULTIPLIER,
        MultiplierOption.OPTIMIZED_SIGN_MAGNITUDE_MULTIPLIER,
        MultiplierOption.OPTIMIZED_MULTIPLIER_FROM_4BIT_BLOCKS,
    ]
    return not (multiplier_option in stages_not_supported)


E = TypeVar("E", bound=Enum)
def get_list_from_enum(enum_cls: Type[E]) -> list[E]:
    result = []
    for entry in enum_cls:
        result += [entry]
    return result


class TwoInputAritEncodings(NamedTuple):
    a: Encoding
    b: Encoding
    y: Encoding

    def with_(self, **changes) -> Self:
        return self._replace(**changes)

    def set_inputs(self, enc: Encoding) -> Self:
        return self.with_(a=enc, b=enc)

    def set_output(self, enc: Encoding) -> Self:
        return self.with_(y=enc)

    def set_all(self, enc: Encoding) -> Self:
        return self.with_(a=enc, b=enc, y=enc)

    @classmethod
    def with_enc(cls, enc: Encoding) -> Self:
        return cls(a=enc, b=enc, y=enc)

class ConfigItem(NamedTuple):
    multiplier_opt: MultiplierOption
    encodings: TwoInputAritEncodings
    ppg_opt: Optional[PPGOption] = None
    ppa_opt: Optional[PPAOption] = None
    fsa_opt: Optional[FSAOption] = None
    all_sigma: bool = True


def encoding_for_multiplier(multiplier_cls: type[StageBasedMultiplierBase]) -> List[TwoInputAritEncodings]:
    if multiplier_cls == StageBasedSignMagnitudeMultiplier:
        return [TwoInputAritEncodings.with_enc(Encoding.sign_magnitude)]
    elif multiplier_cls == StageBasedSignMagnitudeExtMultiplier:
        return [TwoInputAritEncodings.with_enc(Encoding.sign_magnitude_ext)]
    elif multiplier_cls == StageBasedSignMagnitudeExtUpMultiplier:
        return [TwoInputAritEncodings.with_enc(Encoding.sign_magnitude_ext).set_output(Encoding.sign_magnitude_ext_up)]
    elif multiplier_cls == StageBasedMultiplier:
        return [TwoInputAritEncodings.with_enc(Encoding.unsigned), TwoInputAritEncodings.with_enc(Encoding.twos_complement)]
    elif multiplier_cls == StageBasedSignMagnitudeToTwosComplementMultiplier:
        return [TwoInputAritEncodings.with_enc(Encoding.sign_magnitude).set_output(Encoding.twos_complement)]
    elif multiplier_cls == StageBasedSignMagnitudeExtToTwosComplementMultiplier:
        return [TwoInputAritEncodings.with_enc(Encoding.sign_magnitude_ext).set_output(Encoding.twos_complement)]
    elif multiplier_cls == StageBasedSignMagnitudeExtToTwosComplementUpperMultiplier:
        return [TwoInputAritEncodings.with_enc(Encoding.sign_magnitude_ext).set_output(Encoding.twos_complement_upper)]
    elif multiplier_cls == StarMultiplier:
        return [TwoInputAritEncodings.with_enc(Encoding.unsigned), TwoInputAritEncodings.with_enc(Encoding.twos_complement)]
    elif multiplier_cls == OptimizedMultiplier:
        return [TwoInputAritEncodings.with_enc(Encoding.unsigned), TwoInputAritEncodings.with_enc(Encoding.twos_complement)]
    elif multiplier_cls == OptimizedSignMagnitudeMultiplier:
        return [TwoInputAritEncodings.with_enc(Encoding.sign_magnitude)]
    elif multiplier_cls == OptimizedMultiplierFrom4BitBlocks:
        return [TwoInputAritEncodings.with_enc(Encoding.unsigned)]
    elif multiplier_cls == OptimizedMultiplierFrom4BitBlocksStrong:
        return [TwoInputAritEncodings.with_enc(Encoding.unsigned)]
    elif multiplier_cls == KaratsubaMultiplier:
        return [TwoInputAritEncodings.with_enc(Encoding.unsigned), TwoInputAritEncodings.with_enc(Encoding.twos_complement)]
    elif multiplier_cls == KaratsubaMultiplierFromOptimized4BitBlocks:
        return [TwoInputAritEncodings.with_enc(Encoding.unsigned)]
    else:
        raise ValueError(f"Unknown multiplier class: {multiplier_cls}")
