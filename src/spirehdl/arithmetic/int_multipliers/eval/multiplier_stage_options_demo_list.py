from typing import List, Tuple
from spirehdl.arithmetic.int_multipliers.eval.multiplier_stage_options_demo_lib import ConfigItem, FSAOption, TwoInputAritEncodings, PPAOption, PPGOption, MultiplierOption, encoding_for_multiplier, get_list_from_enum, supports_stages
from spirehdl.arithmetic.int_multipliers.multipliers.multipliers_ext_optimized import OptimizedMultiplierFrom4BitBlocks
from spirehdl.arithmetic.int_multipliers.eval.testvector_generation import Encoding, to_encoding
from spirehdl.aig.aig_to_aag import aig_file_to_aag_lines


demos1: list[ConfigItem] = [
    ConfigItem(
        MultiplierOption.STAGE_BASED_SIGN_MAGNITUDE_MULTIPLIER,
        TwoInputAritEncodings.with_enc(Encoding.sign_magnitude),
        PPGOption.AND,
        PPAOption.WALLACE_TREE,
        FSAOption.RIPPLE_CARRY,
    ),
    ConfigItem(
        MultiplierOption.STAGE_BASED_SIGN_MAGNITUDE_EXT_MULTIPLIER,
        TwoInputAritEncodings.with_enc(Encoding.sign_magnitude_ext),
        PPGOption.AND,
        PPAOption.WALLACE_TREE,
        FSAOption.RIPPLE_CARRY,
    ),
    ConfigItem(
        MultiplierOption.STAGE_BASED_SIGN_MAGNITUDE_EXT_UP_MULTIPLIER,
        TwoInputAritEncodings.with_enc(Encoding.sign_magnitude_ext).set_output(Encoding.sign_magnitude_ext_up),
        PPGOption.AND,
        PPAOption.WALLACE_TREE,
        FSAOption.RIPPLE_CARRY,
    ),
    ConfigItem(
        MultiplierOption.STAGE_BASED_MULTIPLIER,
        TwoInputAritEncodings.with_enc(Encoding.unsigned),
        PPGOption.AND,
        PPAOption.WALLACE_TREE,
        FSAOption.RIPPLE_CARRY,
    ),
    ConfigItem(
        MultiplierOption.STAGE_BASED_SIGN_MAGNITUDE_TO_TWOS_COMPLEMENT_MULTIPLIER,
        TwoInputAritEncodings.with_enc(Encoding.sign_magnitude).set_output(Encoding.twos_complement),
        PPGOption.AND,
        PPAOption.WALLACE_TREE,
        FSAOption.RIPPLE_CARRY,
    ),
    ConfigItem(
        MultiplierOption.STAGE_BASED_SIGN_MAGNITUDE_EXT_TO_TWOS_COMPLEMENT_MULTIPLIER,
        TwoInputAritEncodings.with_enc(Encoding.sign_magnitude_ext).set_output(Encoding.twos_complement),
        PPGOption.AND,
        PPAOption.WALLACE_TREE,
        FSAOption.RIPPLE_CARRY,
    ),
    ConfigItem(
        MultiplierOption.STAGE_BASED_SIGN_MAGNITUDE_EXT_TO_TWOS_COMPLEMENT_UPPER_MULTIPLIER,
        TwoInputAritEncodings.with_enc(Encoding.sign_magnitude_ext).set_output(Encoding.twos_complement_upper),
        PPGOption.AND,
        PPAOption.WALLACE_TREE,
        FSAOption.RIPPLE_CARRY,
    ),
    ConfigItem(
        MultiplierOption.STAR_MULTIPLIER,
        TwoInputAritEncodings.with_enc(Encoding.twos_complement),
        PPGOption.NONE,
        PPAOption.NONE,
        FSAOption.NONE,
    ),
    ConfigItem(
        MultiplierOption.STAR_MULTIPLIER,
        TwoInputAritEncodings.with_enc(Encoding.unsigned),
        PPGOption.NONE,
        PPAOption.NONE,
        FSAOption.NONE,
    ),
]


def get_selection1_list(large_sweep: bool = True, multiplier_option_sigma_sweep: bool = False, stages_sigmas_sweep: bool = False) -> List[ConfigItem]:

    config_items : list[ConfigItem] = []

    ppg_0 = PPGOption.AND
    ppa_0 = PPAOption.WALLACE_TREE
    fsa_0 = FSAOption.PREFIX_BRENT_KUNG

    # vary multiplier options
    for sm in get_list_from_enum(MultiplierOption):  # StageMultiplier.get_list_with_all():
        for encoding in encoding_for_multiplier(sm.value):
            if not supports_stages(sm):
                config_items.append(ConfigItem(sm, encoding, PPGOption.NONE, PPAOption.NONE, FSAOption.NONE, all_sigma=multiplier_option_sigma_sweep))
            else:
                config_items.append(ConfigItem(sm, encoding, ppg_0, ppa_0, fsa_0, all_sigma=multiplier_option_sigma_sweep))

    # # vary PPG options
    # for ppg in get_list_from_enum(PPGOption):
    #     if ppg == PPGOption.NONE:
    #         continue
    #     for signed_a, signed_b in ppg.value.supported_signatures: # or ((False, False),):
    #         if signed_a != signed_b:
    #             continue
    #         encoding = to_encoding(signed_a)
    #         config_items.append(ConfigItem(MultiplierOption.STAGE_BASED_MULTIPLIER_BASIC, MultiplierEncodings.with_enc(encoding), ppg, ppa_0, fsa_0))

    # # vary PPA options
    # for ppa in get_list_from_enum(PPAOption):
    #     if ppa == PPAOption.NONE:
    #         continue
    #     config_items.append(ConfigItem(MultiplierOption.STAGE_BASED_MULTIPLIER_BASIC, MultiplierEncodings.with_enc(Encoding.unsigned), ppg_0, ppa, fsa_0))
    #     config_items.append(ConfigItem(MultiplierOption.STAGE_BASED_MULTIPLIER_BASIC, MultiplierEncodings.with_enc(Encoding.twos_complement), ppg_0, ppa, fsa_0))

    # # vary FSA options
    # for fsa in get_list_from_enum(FSAOption):
    #     if fsa == FSAOption.NONE:
    #         continue
    #     config_items.append(ConfigItem(MultiplierOption.STAGE_BASED_MULTIPLIER_BASIC, MultiplierEncodings.with_enc(Encoding.unsigned), ppg_0, ppa_0, fsa))
    #     config_items.append(ConfigItem(MultiplierOption.STAGE_BASED_MULTIPLIER_BASIC, MultiplierEncodings.with_enc(Encoding.twos_complement), ppg_0, ppa_0, fsa))

    # add supposedly "best" option
    # config_items.append(ConfigItem(MultiplierOption.STAGE_BASED_MULTIPLIER_BASIC, MultiplierEncodings.with_enc(Encoding.unsigned), PPGOption.BOOTH_OPTIMISED, PPAOption.DADDA_TREE, FSAOption.PREFIX_SKLANSKY))
    # config_items.append(ConfigItem(multiplier_opt=MultiplierOption.STAGE_BASED_MULTIPLIER_BASIC, encodings=MultiplierEncodings.with_enc(Encoding.twos_complement), ppg_opt=PPGOption.BOOTH_OPTIMISED, ppa_opt=PPAOption.DADDA_TREE, fsa_opt=FSAOption.PREFIX_SKLANSKY))

    # now all combinations of PPG, PPA, FSA for basic multiplier with unsigned and twos_complement
    # with all_sigma = False to make it quicker
    
    if not large_sweep:
        return config_items
    
    from itertools import product
    for sm, ppg, ppa, fsa in product([MultiplierOption.STAGE_BASED_MULTIPLIER], get_list_from_enum(PPGOption), get_list_from_enum(PPAOption), get_list_from_enum(FSAOption)):
        if supports_stages(sm):
            if ppg == PPGOption.NONE or ppa == PPAOption.NONE or fsa == FSAOption.NONE:
                continue
        if not supports_stages(sm):
            if ppg != PPGOption.NONE or ppa != PPAOption.NONE or fsa != FSAOption.NONE:
                continue

        for encodings in encoding_for_multiplier(sm.value):
            
            config_item = ConfigItem(sm, encodings, ppg, ppa, fsa, all_sigma=stages_sigmas_sweep)

            if config_item in config_items:
                continue

            # check
            signature_signed = (encodings.a == Encoding.twos_complement, encodings.b == Encoding.twos_complement)
            if  signature_signed not in ppg.value.supported_signatures:
                continue

            config_items.append(config_item)

    return config_items


def get_selection1_list_optimized() -> List[ConfigItem]:
        
    config_items = [
        ConfigItem(
            MultiplierOption.OPTIMIZED_MULTIPLIER_FROM_4BIT_BLOCKS_STRONG,
            TwoInputAritEncodings.with_enc(Encoding.unsigned),
            PPGOption.NONE,
            PPAOption.NONE,
            FSAOption.NONE,
        ),
    ]
    return config_items
    


if __name__ == "__main__":
    demos1 = get_selection1_list(large_sweep=True, multiplier_option_sigma_sweep=False, stages_sigmas_sweep=False)
    print(f"Defined {len(demos1)} demo configurations to try")