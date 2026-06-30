"""Floating-point multiplier evaluation runner.

Adopted from the integer-stage demo. This version:
- Builds either IEEE-like (EW/FW) or HiFloat8 FP multipliers
- Generates FP test vectors using low_level_arithmetic/fp_multiplier_eval/testvector_generation_fp.py
- Optionally sweeps sigma values (for normal distributions) and collects switching/activity + synthesis metrics

See run_stage_multiplier_ext_demo at bottom for a minimal example.
"""

from __future__ import annotations

from testing.floating_point.synthesise_fp2 import flowy_optimize

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
import functools
import sys
import time
from typing import NamedTuple, Self, Tuple, Type
import os
import math
import uuid

from spire.arithmetic.floating_point.hif8 import hif8_to_float
from spire.arithmetic.floating_point.fp_encoding import fp_decode


try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency for sigma linspace
    np = None  # type: ignore
from tqdm import tqdm


from spire.arithmetic.int_multipliers.eval.multiplier_stage_options_demo_ext_stat_helper import ParquetCollector, _flatten_op_nodes

from spire.helpers import get_aig_stats, get_switch_count, get_yosys_metrics, get_yosys_transistor_count, refactor_module_to_aig, run_vectors
from spire.expr import Op2, reset_shared_cache
from spire.aiger import AigerExporter, AigerImporter
from spire.component import IOCollector
import matplotlib
matplotlib.use("Agg")  # headless plotting
import matplotlib.pyplot as plt

# FP module builders
from spire.arithmetic.floating_point.float_mult_sn import build_fp_mul_sn

# FP testvector generator
from testing.low_level_arithmetic.fp_multiplier_eval.testvector_generation_fp import FPMultiplierTestVectors, IEEEFormat, HiF8Format, FPDist


# ---------------------- config and rows ----------------------


class FPFormatKind(Enum):
    IEEE = "ieee"
    HIF8 = "hif8"


@dataclass
class FPConfig:
    name: str
    kind: FPFormatKind
    # IEEE params (used when kind==IEEE)
    EW: int | None = None
    FW: int | None = None
    subnormals: bool = True
    # Testvector generation
    dist: FPDist = FPDist.UNIFORM_LINEAR
    all_sigma: bool = True
    
    def get_n_bits(self) -> int:
        if self.kind == FPFormatKind.IEEE:
            assert self.EW is not None and self.FW is not None, "EW/FW must be set for IEEE format"
            return 1 + int(self.EW) + int(self.FW)
        elif self.kind == FPFormatKind.HIF8:
            return 8
        else:
            raise ValueError(f"Unsupported format kind: {self.kind}")


@dataclass
class FPMultiplierRow:
    # IDs / meta
    run_id: str
    timestamp: float
    module_name: str

    # Config
    fmt_kind: str
    EW: int | None
    FW: int | None
    subnormals: bool
    a_w: int
    b_w: int
    y_w: int
    num_vectors: int
    dist: str

    # Sweep / result
    sigma: float
    switches: float
    n_sigmas: int
    multiple_sigmas: bool

    # Graph report
    total_expr_nodes: int
    max_depth: int
    depth_y: int
    op_nodes_json: str

    # Yosys
    num_wires: int
    num_cells: int
    estimated_num_transistors: int
    transistor_count: int
    
    # AIG
    num_aig_gates: int
    aig_depth: int

def _choose_sigma_index(sigmas: list[float], bitwidth: int) -> int:
    if np is None:
        return 0
    def target_sigma_for(n_bits: int) -> float:
        return (3.0 / 16.0) * (2 ** int(n_bits))
    tgt = target_sigma_for(bitwidth)
    arr = np.array(sigmas, dtype=float)
    idx = int(np.argmin(np.abs(arr - tgt)))
    return idx


def get_module(cfg: FPConfig):
    """Build FP multiplier module according to config."""
    if cfg.kind == FPFormatKind.IEEE:
        assert cfg.EW is not None and cfg.FW is not None, "EW/FW must be set for IEEE format"
        module = build_fp_mul_sn(f"F{1+cfg.EW+cfg.FW}Mul", EW=int(cfg.EW), FW=int(cfg.FW), subnormals=bool(cfg.subnormals))
    elif cfg.kind == FPFormatKind.HIF8:
        from spire.arithmetic.floating_point.hif8 import build_hif8_mul_logic
        module = build_hif8_mul_logic("HiFP8Mul_Logic_Ref")
    else:
        raise ValueError(f"Unsupported format kind: {cfg.kind}")
    return module


def run_configuration(
    cfg: FPConfig,
    num_vectors: int,
    sigmas: list[float],
    all_sigmas: bool = True,
    plot_histograms: bool = False,
    do_flowy_optimize: bool = True
):
    reset_shared_cache()

    # Build module
    if cfg.kind == FPFormatKind.IEEE:
        a_fmt = IEEEFormat(EW=int(cfg.EW), FW=int(cfg.FW), subnormals=bool(cfg.subnormals))
    elif cfg.kind == FPFormatKind.HIF8:
        a_fmt = HiF8Format()
    else:
        raise ValueError(f"Unsupported format kind: {cfg.kind}")
    module = get_module(cfg)

    # Smoke test vectors
    smoke_tb_sigma = None
    if cfg.dist in (FPDist.NORMAL_LINEAR, FPDist.NORMAL_LOG):
        smoke_tb_sigma = (sigmas[0] if sigmas else 1.0)
    vecs = FPMultiplierTestVectors(
        a_fmt=a_fmt,
        num_vectors=min(16, num_vectors),
        dist=cfg.dist,
        tb_sigma=smoke_tb_sigma,
    ).generate()

    decoder = functools.partial(fp_decode, EW=cfg.EW, FW=cfg.FW) if cfg.kind == FPFormatKind.IEEE else hif8_to_float

    raise_on_fail = False

    run_vectors(module, vecs, decoder=decoder, raise_on_fail=raise_on_fail)  # smoke test

    if do_flowy_optimize:
        module = flowy_optimize(module)

    # -- swact --
    m_aig = refactor_module_to_aig(module)

    # AIG network test sim
    print("Sim (AIG) …")
    run_vectors(m_aig, vecs, decoder=decoder, raise_on_fail=raise_on_fail)  # smoke test

    exprs = m_aig.all_exprs()
    all_ands = [e for e in exprs if isinstance(e, Op2) and e.op == "&"]

    def run_and_count(vecs_run) -> int:
        states_list = run_vectors(m_aig, vecs_run, exprs=all_ands, decoder=decoder, raise_on_fail=raise_on_fail)
        return get_switch_count(states_list)

    switches = []
    _sigmas = sigmas
    if not all_sigmas and sigmas:
        idx = _choose_sigma_index(sigmas, cfg.get_n_bits())
        _sigmas = [sigmas[idx]]

    vecs_all = {}    

    for sigma in _sigmas if _sigmas else [None]:
        vecs = FPMultiplierTestVectors(
            a_fmt=a_fmt,
            num_vectors=num_vectors,
            dist=cfg.dist,
            tb_sigma=(float(sigma) if (sigma is not None and cfg.dist in (FPDist.NORMAL_LINEAR, FPDist.NORMAL_LOG)) else None),
        ).generate()
        switches.append(run_and_count(vecs))
        print(f"Average AND switches (sigma={sigma}): {switches[-1]}")

    if plot_histograms:
        plot_input_output_histograms(cfg, vecs, sigma, decoder)

    gr = m_aig.module_analyze()
    tc = get_yosys_transistor_count(m_aig)
    ym = get_yosys_metrics(m_aig)
    aig_stats = get_aig_stats(m_aig)

    run_id = uuid.uuid4().hex
    t_now = time.time()

    rows = []
    for sigma_val, sw_val in zip(_sigmas if _sigmas else [None], switches):
        row = FPMultiplierRow(
            run_id=run_id,
            timestamp=t_now,
            module_name=module.name,
            fmt_kind=cfg.kind.value,
            EW=(int(cfg.EW) if cfg.EW is not None else None),
            FW=(int(cfg.FW) if cfg.FW is not None else None),
            subnormals=bool(cfg.subnormals),
            a_w=cfg.get_n_bits(),
            b_w=cfg.get_n_bits(),
            y_w=cfg.get_n_bits(),
            num_vectors=int(num_vectors),
            dist=cfg.dist.value,
            sigma=(float(sigma_val) if sigma_val is not None else float("nan")),
            switches=int(sw_val),
            n_sigmas=len(_sigmas) if _sigmas else 0,
            multiple_sigmas=bool(_sigmas and len(_sigmas) > 1),
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
        )
        rows.append(row)
    return rows

def run_stage_multiplier_ext_demo(config_items: list[FPConfig]) -> None:  # pragma: no cover - demonstration only

    num_vectors = 10000
    # When using NORMAL_* distributions we sweep sigma in exponent domain (log) or linear.
    # For UNIFORM_* distributions, sigmas are ignored but we keep the loop for uniformity.
    n_steps_sigma = 8
    sigma_min = 0.5/10
    sigma_max = 12

    sigmas = np.logspace(np.log10(sigma_min), np.log10(sigma_max), n_steps_sigma).tolist()


    parallel = False
    max_workers = 8

    sys.setrecursionlimit(4000)

    # datecode
    run_id = time.strftime("%Y%m%d_%H%M%S")
    out_file = f"data/fp_multiplier_runs_{run_id}.parquet"
    collector = ParquetCollector(out_file, row_cls=FPMultiplierRow)

    results = []
    errors = []

    if not parallel:
        for cfg in config_items:
            #try:
            rows = run_configuration(cfg, num_vectors, sigmas, all_sigmas=cfg.all_sigma)
            collector.extend(rows)
            results.extend(rows)
            #except Exception as e:
                # errors.append(f"Error in config {cfg.name}: {e}")
                # print(f"Error in config {cfg.name}: {e}")
    else:
        _worker = run_configuration
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(_worker, cfg, num_vectors, sigmas, cfg.all_sigma) for cfg in config_items]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Running FP demos", unit="cfg"):
                try:
                    rec = fut.result()
                    if collector is not None and rec is not None:
                        collector.extend(rec)
                        results.extend(rec)
                except Exception as e:
                    errors.append(f"Error in demo: {str(e)}")
                    print(f"Error in demo: {str(e)}")

    if errors:
        print(f"\nEncountered {len(errors)} errors:")
        for i, error in enumerate(errors, 1):
            print(f"{i}. {error}")
        print(f"\n(Encountered {len(errors)} errors)")

    print(f"Total configs: {len(config_items)}")
    print(f"Total rows collected: {collector.n_rows()}")
    collector.save(append=True)
    print(f"Saved to '{out_file}'")


# ------------------------------ helpers ------------------------------ #

def plot_input_output_histograms(cfg: FPConfig, vecs: list[tuple], sigma: float | None, decoder: callable | None) -> None:
    """Plot two histograms (inputs combined, outputs) for the given vectors and save to file.

    - Left subplot: inputs (a and b combined)
    - Right subplot: outputs (y)
    File is saved into fp_value_distribution_outputs/ with config name and sigma in filename.
    """

    # Choose decoder for bit patterns if not provided
    if decoder is None:
        if cfg.kind == FPFormatKind.IEEE:
            assert cfg.EW is not None and cfg.FW is not None
            decode_val = functools.partial(fp_decode, EW=int(cfg.EW), FW=int(cfg.FW))
        else:
            from spire.arithmetic.floating_point.hif8 import hif8_to_float
            decode_val = hif8_to_float
    else:
        decode_val = decoder

    inputs_vals: list[float] = []  # combined a and b
    outputs_vals: list[float] = []  # y
    a_vals: list[float] = []
    b_vals: list[float] = []
    a_raw_vals: list[float] = []
    b_raw_vals: list[float] = []
    y_raw_vals: list[float] = []

    for _, ins, outs in vecs:
        # Optional raw values captured at generation-time
        av_val = ins.get("_a_val", None)
        bv_val = ins.get("_b_val", None)
        yv_val = outs.get("_y_val", None)

        av = decode_val(ins.get("a", 0))
        bv = decode_val(ins.get("b", 0))
        yv = decode_val(outs.get("y", 0))

        if math.isfinite(av):
            inputs_vals.append(float(av))
            a_vals.append(float(av))
        if isinstance(av_val, (int, float)) and math.isfinite(av_val):
            a_raw_vals.append(float(av_val))

        if math.isfinite(bv):
            inputs_vals.append(float(bv))
            b_vals.append(float(bv))
        if isinstance(bv_val, (int, float)) and math.isfinite(bv_val):
            b_raw_vals.append(float(bv_val))

        if math.isfinite(yv):
            outputs_vals.append(float(yv))
        if isinstance(yv_val, (int, float)) and math.isfinite(yv_val):
            y_raw_vals.append(float(yv_val))

    out_dir = "fp_value_distribution_outputs"
    os.makedirs(out_dir, exist_ok=True)
    sigma_tag = ("NA" if sigma is None else f"{float(sigma):.4g}")
    fmt_tag = (f"E{cfg.EW}_F{cfg.FW}" if cfg.kind == FPFormatKind.IEEE else "HiF8")
    base = f"fp_vec_hist_{cfg.name}_{fmt_tag}_sigma_{sigma_tag}"
    # sanitize filename
    def _safe(name: str) -> str:
        return "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "_" for ch in name)
    out_path = os.path.join(out_dir, _safe(base + ".png"))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    # Inputs histogram (a and b combined)
    if inputs_vals:
        axes[0].hist(inputs_vals, bins=100, color="C0", alpha=0.85)
    axes[0].set_title("Inputs (a,b)")
    axes[0].set_xlabel("value")
    axes[0].set_ylabel("count")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    #try:
    #    axes[0].set_xscale("symlog", linthresh=1e-6)
    #except Exception:
    #    pass

    # Outputs histogram (y)
    if outputs_vals:
        axes[1].hist(outputs_vals, bins=100, color="C1", alpha=0.85)
    axes[1].set_title("Outputs (y)")
    axes[1].set_xlabel("value")
    axes[1].set_ylabel("count")
    axes[1].grid(True, linestyle="--", alpha=0.4)
    # try:
    #     axes[1].set_xscale("symlog", linthresh=1e-6)
    # except Exception:
    #     pass

    ctx = f"{cfg.name} | {fmt_tag} | dist={cfg.dist.value} | sigma={sigma_tag}"
    fig.suptitle(ctx)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[plot] wrote {out_path}")

    # Overlay a vs b distribution
    if a_vals or b_vals:
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        if a_vals:
            ax2.hist(a_vals, bins=100, alpha=0.6, label="a", color="C0")
        if b_vals:
            ax2.hist(b_vals, bins=100, alpha=0.6, label="b", color="C2")
        ax2.set_title("Inputs distribution (a vs b)")
        ax2.set_xlabel("value")
        ax2.set_ylabel("count")
        ax2.grid(True, linestyle="--", alpha=0.4)
        # try:
        #     ax2.set_xscale("symlog", linthresh=1e-6)
        # except Exception:
        #     pass
        ax2.legend(fontsize=7, frameon=False)
        out2 = os.path.join(out_dir, _safe(base + "_inputs_ab.png"))
        fig2.suptitle(ctx)
        fig2.tight_layout()
        fig2.savefig(out2, dpi=140)
        plt.close(fig2)
        print(f"[plot] wrote {out2}")

    # Outputs only (decoded)
    if outputs_vals:
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        ax3.hist(outputs_vals, bins=100, alpha=0.85, color="C1")
        ax3.set_title("Outputs distribution (y)")
        ax3.set_xlabel("value")
        ax3.set_ylabel("count")
        ax3.grid(True, linestyle="--", alpha=0.4)
        # try:
        #     ax3.set_xscale("symlog", linthresh=1e-6)
        # except Exception:
        #     pass
        out3 = os.path.join(out_dir, _safe(base + "_outputs_y.png"))
        fig3.suptitle(ctx)
        fig3.tight_layout()
        fig3.savefig(out3, dpi=140)
        plt.close(fig3)
        print(f"[plot] wrote {out3}")

    # Raw-value plots if provided in vecs
    if a_raw_vals:
        fig4, ax4 = plt.subplots(figsize=(6, 4))
        ax4.hist(a_raw_vals, bins=100, alpha=0.85, color="C0")
        ax4.set_title("Raw inputs distribution (a_val)")
        ax4.set_xlabel("value")
        ax4.set_ylabel("count")
        ax4.grid(True, linestyle="--", alpha=0.4)
        # try:
        #     ax4.set_xscale("symlog", linthresh=1e-6)
        # except Exception:
        #     pass
        out4 = os.path.join(out_dir, _safe(base + "_a_val.png"))
        fig4.suptitle(ctx)
        fig4.tight_layout()
        fig4.savefig(out4, dpi=140)
        plt.close(fig4)
        print(f"[plot] wrote {out4}")

    if b_raw_vals:
        fig5, ax5 = plt.subplots(figsize=(6, 4))
        ax5.hist(b_raw_vals, bins=100, alpha=0.85, color="C2")
        ax5.set_title("Raw inputs distribution (b_val)")
        ax5.set_xlabel("value")
        ax5.set_ylabel("count")
        ax5.grid(True, linestyle="--", alpha=0.4)
        # try:
        #     ax5.set_xscale("symlog", linthresh=1e-6)
        # except Exception:
        #     pass
        out5 = os.path.join(out_dir, _safe(base + "_b_val.png"))
        fig5.suptitle(ctx)
        fig5.tight_layout()
        fig5.savefig(out5, dpi=140)
        plt.close(fig5)
        print(f"[plot] wrote {out5}")

    if y_raw_vals:
        fig6, ax6 = plt.subplots(figsize=(6, 4))
        ax6.hist(y_raw_vals, bins=100, alpha=0.85, color="C1")
        ax6.set_title("Raw outputs distribution (y_val)")
        ax6.set_xlabel("value")
        ax6.set_ylabel("count")
        ax6.grid(True, linestyle="--", alpha=0.4)
        # try:
        #     ax6.set_xscale("symlog", linthresh=1e-6)
        # except Exception:
        #     pass
        out6 = os.path.join(out_dir, _safe(base + "_y_val.png"))
        fig6.suptitle(ctx)
        fig6.tight_layout()
        fig6.savefig(out6, dpi=140)
        plt.close(fig6)
        print(f"[plot] wrote {out6}")

    # Comparison plots: raw vs decoded for a and y
    if a_raw_vals and a_vals:
        fig7, axes7 = plt.subplots(1, 2, figsize=(10, 4))
        axes7[0].hist(a_raw_vals, bins=100, alpha=0.85, color="C0")
        axes7[0].set_title("a_val (raw)")
        axes7[0].set_xlabel("value")
        axes7[0].set_ylabel("count")
        axes7[0].grid(True, linestyle="--", alpha=0.4)
        # try:
        #     axes7[0].set_xscale("symlog", linthresh=1e-6)
        # except Exception:
        #     pass
        axes7[1].hist(a_vals, bins=100, alpha=0.85, color="C4")
        axes7[1].set_title("a (encoded→decoded)")
        axes7[1].set_xlabel("value")
        axes7[1].set_ylabel("count")
        axes7[1].grid(True, linestyle="--", alpha=0.4)
        # try:
        #     axes7[1].set_xscale("symlog", linthresh=1e-6)
        # except Exception:
        #     pass
        out7 = os.path.join(out_dir, _safe(base + "_a_raw_vs_decoded.png"))
        fig7.suptitle(ctx + " — a: raw vs encoded→decoded")
        fig7.tight_layout()
        fig7.savefig(out7, dpi=140)
        plt.close(fig7)
        print(f"[plot] wrote {out7}")

    if y_raw_vals and outputs_vals:
        fig8, axes8 = plt.subplots(1, 2, figsize=(10, 4))
        axes8[0].hist(y_raw_vals, bins=100, alpha=0.85, color="C1")
        axes8[0].set_title("y_val (raw)")
        axes8[0].set_xlabel("value")
        axes8[0].set_ylabel("count")
        axes8[0].grid(True, linestyle="--", alpha=0.4)
        # try:
        #     axes8[0].set_xscale("symlog", linthresh=1e-6)
        # except Exception:
        #     pass
        axes8[1].hist(outputs_vals, bins=100, alpha=0.85, color="C3")
        axes8[1].set_title("y (encoded→decoded)")
        axes8[1].set_xlabel("value")
        axes8[1].set_ylabel("count")
        axes8[1].grid(True, linestyle="--", alpha=0.4)
        # try:
        #     axes8[1].set_xscale("symlog", linthresh=1e-6)
        # except Exception:
        #     pass
        out8 = os.path.join(out_dir, _safe(base + "_y_raw_vs_decoded.png"))
        fig8.suptitle(ctx + " — y: raw vs encoded→decoded")
        fig8.tight_layout()
        fig8.savefig(out8, dpi=140)
        plt.close(fig8)
        print(f"[plot] wrote {out8}")


if __name__ == "__main__":  # small example

    dist = FPDist.NORMAL_LINEAR

    # Example: IEEE(5,10) and HiF8 with simple uniform distributions
    cfgs = [
        FPConfig(name="fp8_e4m3", kind=FPFormatKind.IEEE, EW=4, FW=3, subnormals=True, dist=dist, all_sigma=True),
        FPConfig(name="fp8_e5m2", kind=FPFormatKind.IEEE, EW=5, FW=2, subnormals=True, dist=dist, all_sigma=True),
        FPConfig(name="hif8", kind=FPFormatKind.HIF8, dist=dist, all_sigma=True),
    ]
    run_stage_multiplier_ext_demo(cfgs)