"""Demonstrate how to mix and match multiplier stages."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from enum import Enum
import sys
import time
from typing import NamedTuple, Self, Tuple, Type
import uuid

import numpy as np
from tqdm import tqdm


from spirehdl.arithmetic.int_multipliers.eval.multiplier_stage_options_demo_lib import ConfigItem, TwoInputAritEncodings, MultiplierOption, PPAOption, PPGOption
from spirehdl.arithmetic.int_multipliers.eval.multiplier_stage_options_demo_lib import FSAOption
from spirehdl.arithmetic.int_multipliers.eval.multiplier_stage_options_demo_ext_stat_helper import MultiplierRow, ParquetCollector, _flatten_op_nodes
from spirehdl.arithmetic.int_multipliers.multipliers.mutipliers_ext import StageBasedMultiplierBase
from spirehdl.arithmetic.int_multipliers.eval.testvector_generation import Encoding, MultiplierTestVectors, to_encoding

from spirehdl.helpers import get_aig_stats, get_switch_count, get_yosys_metrics, get_yosys_transistor_count, refactor_module_to_aig, run_vectors
from spirehdl.spirehdl import Op2, reset_shared_cache
from spirehdl.arithmetic.int_multipliers.eval.multiplier_stage_options_demo_list import demos1, get_selection1_list, get_selection1_list, get_selection1_list_optimized

def get_target_sigma_index(sigmas: list, n_bits: int) -> int:

    def target_sigma_for(n_bits: int) -> float:
        """σ_target = 3/16 * 2**n_bits."""
        return (3.0 / 16.0) * (2 ** int(n_bits))

    tgt = target_sigma_for(n_bits)

    sigmas_dist = np.abs(sigmas.astype(float) - tgt)
    idx = np.argmin(sigmas_dist)
    return int(idx)


def run_configuration(
    multiplier_opt: MultiplierOption,
    encodings: TwoInputAritEncodings,
    ppg_opt: PPGOption,
    ppa_opt: PPAOption,
    fsa_opt: FSAOption,
    n_bits: int,
    num_vectors: int,
    sigmas: list,
    all_sigmas: bool = True,
) -> None:
    reset_shared_cache()

    multiplier = multiplier_opt.value(
        a_w=n_bits,
        b_w=n_bits,
        a_encoding=encodings.a,
        b_encoding=encodings.b,
        ppg_cls=ppg_opt.value,
        ppa_cls=ppa_opt.value,
        fsa_cls=fsa_opt.value,
        optim_type="area",
    )

    module = multiplier.to_module(f"mult_{ppg_opt.name.lower()}_{encodings.a.name.lower()}_{encodings.b.name.lower()}_{fsa_opt.name.lower()}")
    print(f"Built module '{module.name}' using PPG={ppg_opt.name}, PPA={ppa_opt.name}, FSA={fsa_opt.name}")

    vecs = MultiplierTestVectors(
                a_w=n_bits,
                b_w=n_bits,
                y_w=multiplier.io.y.typ.width,
                num_vectors=num_vectors,
                tb_sigma=None,
                a_encoding=encodings.a,
                b_encoding=encodings.b,
                y_encoding=encodings.y,
            ).generate()

    run_vectors(module, vecs[0:min(16, len(vecs))])  # smoke test

    # -- swact --
    m_aig = refactor_module_to_aig(module)

    # AIG network test sim
    print("Sim (AIG) …")
    run_vectors(m_aig, vecs[0:min(16, len(vecs))])  # smoke test

    exprs = m_aig.all_exprs()
    all_ands = [e for e in exprs if isinstance(e, Op2) and e.op == "&"]

    def run_and_count(vecs_run) -> int:
        states_list = run_vectors(m_aig, vecs_run, exprs=all_ands)
        return get_switch_count(states_list)             

    # switches = run_and_count(vecs)
    # print(f"Average AND switches: {switches}")

    switches = []
    if not all_sigmas:
        sigmas = [sigmas[get_target_sigma_index(np.array(sigmas), n_bits)]]
    for sigma in sigmas:
        vecs = MultiplierTestVectors(
                      a_w=n_bits,
                      b_w=n_bits,
                      y_w=multiplier.io.y.typ.width,
                      num_vectors=num_vectors,
                      tb_sigma=sigma,
                      a_encoding=encodings.a,
                      b_encoding=encodings.b,
                      y_encoding=encodings.y,
                  ).generate()
        switches.append(run_and_count(vecs))
        print(f"Average AND switches (sigma={sigma}): {switches[-1]}")

    gr = m_aig.module_analyze()
    tc = get_yosys_transistor_count(m_aig)
    ym = get_yosys_metrics(m_aig)
    num_aig_gates = len(all_ands) # aig.gates()
    aig_stats = get_aig_stats(m_aig)

    run_id = uuid.uuid4().hex
    t_now = time.time()

    rows = []
    for sigma_val, sw_val in zip(sigmas, switches):
        row = MultiplierRow(
                    run_id=run_id,
                    timestamp=t_now,
                    module_name=module.name,
            
                    n_bits=int(n_bits),
                    multiplier_opt=multiplier_opt.name,
                    ppg_opt=ppg_opt.name,
                    ppa_opt=ppa_opt.name,
                    fsa_opt=fsa_opt.name,
                    a_enc=encodings.a.name,
                    b_enc=encodings.b.name,
                    y_enc=encodings.y.name,
                    a_w=int(multiplier.io.a.typ.width),
                    b_w=int(multiplier.io.b.typ.width),
                    y_w=int(multiplier.io.y.typ.width),
                    num_vectors=int(num_vectors),
            
                    sigma=float(sigma_val),
                    switches=int(sw_val),
                    n_sigmas=len(sigmas),
                    multiple_sigmas=len(sigmas) > 1,
            
                    total_expr_nodes=int(gr.total_expr_nodes),
                    max_depth=int(gr.max_depth),
                    depth_y=int(gr.output_depth["y"]),
                    op_nodes_json=_flatten_op_nodes(gr.op_nodes),
            
                    num_wires=int(ym["num_wires"]),
                    num_cells=int(ym["num_cells"]),
                    estimated_num_transistors=int(ym["estimated_num_transistors"]),
                    transistor_count=int(tc),
                    
                    num_aig_gates=aig_stats["num_gates"],
                    aig_depth=aig_stats["depth"],
                    
                    optimization_effort = 2 if multiplier_opt == MultiplierOption.OPTIMIZED_MULTIPLIER else 1,
                )
        rows.append(row)
    return rows

def run_stage_multiplier_ext_demo(config_items: list[ConfigItem]) -> None:  # pragma: no cover - demonstration only

    num_vectors = 2000//100
    bitwidths = [4, 8, 16, 24, 32] #[4, 8, 16, 24, 32]
    sigma_factor = 0.5
    n_steps_sigma = 8
    parallel = True
    max_workers = 80

    sys.setrecursionlimit(10000)

    # datecode
    run_id = time.strftime("%Y%m%d_%H%M%S")
    out_file = f"data/multiplier_runs_{run_id}.parquet"
    collector = ParquetCollector(out_file)

    if parallel:
        bitwidths = list(reversed(bitwidths))  # start with big ones first to get better load balancing

    results = []
    errors = []
    for n_bits in bitwidths:

        sigma_max = 2**n_bits * sigma_factor
        sigma_start = sigma_max / n_steps_sigma        
        sigmas = np.linspace(sigma_start, sigma_max, n_steps_sigma)

        if not parallel:

            for config_item in config_items:                
                rows = run_configuration(config_item.multiplier_opt, config_item.encodings, config_item.ppg_opt, config_item.ppa_opt, config_item.fsa_opt, n_bits, num_vectors, sigmas, all_sigmas=config_item.all_sigma)
                collector.extend(rows)

        else:              

            _worker = run_configuration

            with ProcessPoolExecutor(max_workers=max_workers) as ex:
                futures = []
                for config_item in config_items: 
                    futures.append(
                        ex.submit(_worker, config_item.multiplier_opt, config_item.encodings, config_item.ppg_opt, config_item.ppa_opt, config_item.fsa_opt, n_bits, num_vectors, sigmas, all_sigmas=config_item.all_sigma)
                    )
                for fut in tqdm(as_completed(futures), total=len(futures), desc="Running demos", unit="demo"):
                    try:
                        rec = fut.result()
                        # Safely aggregate on the main process:
                        if collector is not None and rec is not None:
                            collector.extend(rec)   # or whatever your API is
                            results.extend(rec)
                    except Exception as e:
                        errors.append(f"Error in demo: {str(e)}")
                        print(f"Error in demo: {str(e)}")

    if errors:
        print(f"\nEncountered {len(errors)} errors:")
        for i, error in enumerate(errors, 1):
            print(f"{i}. {error}")
        print(f"\n(Encountered {len(errors)} errors)")

    # get number of rows added
    print(f"Total number of configurations run: {len(config_items) * len(bitwidths) * n_steps_sigma}")
    print(f"Total rows collected: {collector.n_rows()}")
    print(f"Total number of demos: {len(config_items)}")
    print(f"Total number of errors: {len(errors)}")

    # save to file
    collector.save(append=True)
    print(f"Saved to '{out_file}'")

if __name__ == "__main__":
    # run_stage_multiplier_ext_demo(config_items=demos1)
    run_stage_multiplier_ext_demo(config_items=get_selection1_list(large_sweep=True, stages_sigmas_sweep=False))
    #run_stage_multiplier_ext_demo(config_items=get_selection1_list_optimized())