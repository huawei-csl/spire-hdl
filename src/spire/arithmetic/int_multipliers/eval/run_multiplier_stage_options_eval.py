"""Demonstrate how to mix and match multiplier stages."""

from __future__ import annotations

from typing import Tuple

from spire.arithmetic.int_multipliers.eval.multiplier_stage_options_demo_lib import FSAOption, PPAOption, PPGOption
from spire.arithmetic.int_multipliers.eval.testvector_generation import Encoding, MultiplierTestVectors, to_encoding
from spire.arithmetic.int_multipliers.multipliers.multiplier_stage_core import StageBasedMultiplierBasic
from spire.helpers import run_vectors
from spire.expr import reset_shared_cache

def run_stage_multiplier_demo() -> None:  # pragma: no cover - demonstration only

    # define some demo combinations to try
    demos: Tuple[Tuple[PPGOption, PPAOption, FSAOption], ...] = (
        (PPGOption.AND, PPAOption.WALLACE_TREE, FSAOption.RIPPLE_CARRY),
        (PPGOption.AND, PPAOption.CARRY_SAVE_TREE, FSAOption.PREFIX_KOGGE_STONE),
        (PPGOption.AND, PPAOption.FOUR_TWO_COMPRESSOR, FSAOption.PREFIX_BRENT_KUNG),
        (PPGOption.BAUGH_WOOLEY, PPAOption.ACCUMULATOR_TREE, FSAOption.PREFIX_KOGGE_STONE),
        (PPGOption.BOOTH_UNOPTIMISED, PPAOption.DADDA_TREE, FSAOption.PREFIX_RCA),
        (PPGOption.BOOTH_OPTIMISED, PPAOption.ACCUMULATOR_TREE, FSAOption.PREFIX_BRENT_KUNG)
    )

    num_vectors = 100
    completed_demo_runs = 0

    for width in (4, 8, 16):

        for ppg_opt, ppa_opt, fsa_opt in demos:
            for signed_a, signed_b in ppg_opt.value.supported_signatures or ((False, False),):
                print(f"Building multiplier with width={width}, signed_a={signed_a}, signed_b={signed_b}, PPG={ppg_opt.name}, PPA={ppa_opt.name}, FSA={fsa_opt.name}")
                reset_shared_cache()
                
                if not signed_a:
                    continue # debug only

                multiplier = StageBasedMultiplierBasic(
                    a_w=width,
                    b_w=width,
                    signed_a=signed_a,
                    signed_b=signed_b,
                    ppg_cls=ppg_opt.value,
                    ppa_cls=ppa_opt.value,
                    fsa_cls=fsa_opt.value,
                    optim_type="area",
                )
                
                module = multiplier.to_module(
                    f"demo_{ppg_opt.name.lower()}_{signed_a}_{signed_b}_{fsa_opt.name.lower()}"
                )
                print(
                    f"Built module '{module.name}' using PPG={ppg_opt.name}, PPA={ppa_opt.name}, FSA={fsa_opt.name}"
                )
                

                vecs = MultiplierTestVectors(
                    a_w=width,
                    b_w=width,
                    num_vectors=num_vectors,
                    tb_sigma=None,
                    a_encoding=to_encoding(signed_a),
                    b_encoding=to_encoding(signed_b),
                    y_encoding=to_encoding(signed_a or signed_b),
                ).generate()

                run_vectors(module, vecs)

                completed_demo_runs += 1
                print(f"Completed {completed_demo_runs} multiplier demos.")
                gr = module.module_analyze()
                print(f"Graph report: {gr}")


if __name__ == "__main__":
    run_stage_multiplier_demo()