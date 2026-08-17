"""Demonstrate how to mix and match multiplier stages."""

from __future__ import annotations

from enum import Enum
from typing import NamedTuple, Self, Tuple, Type

from spire.arithmetic.int_multipliers.eval.testvector_generation import Encoding, MultiplierTestVectors
from spire.helpers import run_vectors
from spire.expr import reset_shared_cache
from spire.arithmetic.int_multipliers.eval.multiplier_stage_options_demo_list import demos1


def run_stage_multiplier_ext_demo() -> None:  # pragma: no cover - demonstration only

    config_items = demos1


    completed_demo_runs = 0

    num_vectors = 100
    bitwidths = [4, 8, 16]

    for width in bitwidths:

        for config_item in config_items:

            reset_shared_cache()

            multiplier = config_item.multiplier_opt.value(
                a_w=width,
                b_w=width,
                a_encoding=config_item.encodings.a,
                b_encoding=config_item.encodings.b,
                ppg_cls=config_item.ppg_opt.value,
                ppa_cls=config_item.ppa_opt.value,
                fsa_cls=config_item.fsa_opt.value,
                optim_type="area",
            )

            module = multiplier.to_netlist(f"demo_{config_item.ppg_opt.name.lower()}_{config_item.encodings.a.name.lower()}_{config_item.encodings.b.name.lower()}_{config_item.fsa_opt.name.lower()}")
            print(f"Built module '{module.name}' using PPG={config_item.ppg_opt.name}, PPA={config_item.ppa_opt.name}, FSA={config_item.fsa_opt.name}")

            vecs = MultiplierTestVectors(
                a_w=width,
                b_w=width,
                y_w=multiplier.io.y.typ.width,
                num_vectors=num_vectors,
                tb_sigma=None,
                a_encoding=config_item.encodings.a,
                b_encoding=config_item.encodings.b,
                y_encoding=config_item.encodings.y,
            ).generate()

            run_vectors(module, vecs)

            completed_demo_runs += 1
            print(f"Completed {completed_demo_runs} multiplier demos.")
            gr = module.module_analyze()
            print(f"Graph report: {gr}")


if __name__ == "__main__":
    run_stage_multiplier_ext_demo()