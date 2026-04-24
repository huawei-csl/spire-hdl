#!/usr/bin/env python3
"""
Matplotlib plotter for multiplier exploration results (single or multiple Parquet files).

NEW:
- 'scaling' plot: metric vs n_bits with one line per category (via color_by),
  per-(category, n_bits) aggregation, optional log–log axes, and power-law fit
  y ≈ a * n^p (p shown in legend).

Existing:
- Scatter: any metric vs any metric (one point per configuration; includes switches_at_target)
- Line: metric vs sigma (lines grouped by run_id)
- Filters (categoricals + bit-width ranges)
- color_by + legend on/off
- Example plots

Usage:
  python mpl_multiplier_plots.py --file data1.parquet --file data2.parquet \
      --out plots_mpl --legend on
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt


# ----------------------------- config ----------------------------- #

STATIC_METRICS = [
    "estimated_num_transistors",
    "transistor_count",
    "num_cells",
    "num_wires",
    "max_depth",
    "depth_y",
    "total_expr_nodes",
    "num_aig_gates",
    "aig_depth",
    "switches"
]

HOVER_BASE = [
    "design_label",
    "multiplier_opt",
    "ppg_opt",
    "ppa_opt",
    "fsa_opt",
    "a_enc",
    "b_enc",
    "y_enc",
    "n_bits",
    "a_w",
    "b_w",
    "y_w",
]


def design_label_from_row(row: pd.Series) -> str:
    def g(k: str, default: str = "NA"):
        return row[k] if (k in row.index and pd.notna(row[k])) else default

    return f"{g('multiplier_opt')} | " f"PPG={g('ppg_opt')} | PPA={g('ppa_opt')} | FSA={g('fsa_opt')} | " f"enc={g('a_enc')}/{g('b_enc')}→{g('y_enc')}"


def target_sigma_for(n_bits: int) -> float:
    return (3.0 / 16.0) * (2 ** int(n_bits))


def choose_area_column(df: pd.DataFrame) -> str:
    if "estimated_num_transistors" in df.columns:
        return "estimated_num_transistors"
    if "transistor_count" in df.columns:
        return "transistor_count"
    if "num_cells" in df.columns:
        return "num_cells"
    raise ValueError("No area-like column found (need one of: estimated_num_transistors, transistor_count, num_cells).")


# --------------------------- data helpers -------------------------- #
def augment_df_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add convenient derived columns (strings) for labeling/coloring."""
    df = df.copy()

    # enc: "a_enc/b_enc→y_enc"
    if all(c in df.columns for c in ("a_enc", "b_enc", "y_enc")):
        df["enc"] = df["a_enc"].astype("string") + "/" + df["b_enc"].astype("string") + "→" + df["y_enc"].astype("string")

    # io_widths: "a_w/b_w→y_w" (uses pandas Int64 so NaNs are allowed)
    if all(c in df.columns for c in ("a_w", "b_w", "y_w")):
        df["io_widths"] = (
            df["a_w"].astype("Int64").astype("string") + "/" + df["b_w"].astype("Int64").astype("string") + "→" + df["y_w"].astype("Int64").astype("string")
        )

    # stages combined: "PPG=... | PPA=... | FSA=..."
    if all(c in df.columns for c in ("ppg_opt", "ppa_opt", "fsa_opt")):
        df["stages"] = "PPG=" + df["ppg_opt"].astype("string") + " | PPA=" + df["ppa_opt"].astype("string") + " | FSA=" + df["fsa_opt"].astype("string")

    return df


def load_df(file_path: str) -> pd.DataFrame:
    df = pd.read_parquet(file_path, engine="pyarrow").copy()

    # Derive n_bits if missing (best effort)
    if "n_bits" not in df.columns:
        if "a_w" in df.columns and pd.api.types.is_numeric_dtype(df["a_w"]):
            try:
                df["n_bits"] = df["a_w"].astype("Int64")
            except Exception:
                pass
        if "n_bits" not in df.columns and "y_w" in df.columns and pd.api.types.is_numeric_dtype(df["y_w"]):
            try:
                df["n_bits"] = (df["y_w"] // 2).astype("Int64")
            except Exception:
                pass
        if "n_bits" not in df.columns:
            raise ValueError("Could not derive 'n_bits'; please include it or provide a_w / y_w columns.")

    # Label
    df["design_label"] = df.apply(design_label_from_row, axis=1).astype("string")

    # Types
    for c in ["multiplier_opt", "ppg_opt", "ppa_opt", "fsa_opt", "a_enc", "b_enc", "y_enc"]:
        if c in df.columns:
            df[c] = df[c].astype("string")

    df = augment_df_columns(df)

    for c in ["enc", "io_widths", "stages"]:
        if c in df.columns:
            df[c] = df[c].astype("string")

    return df


def load_many_dfs(file_paths: List[str]) -> pd.DataFrame:
    """Load one or more Parquet files and concatenate them."""
    frames = [load_df(path) for path in file_paths or []]
    if not frames:
        return pd.DataFrame()
    if len(frames) == 1:
        return frames[0]
    return pd.concat(frames, ignore_index=True)


def compute_switches_at_target(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per configuration with `switches_at_target`."""
    key_cols = [
        "n_bits",
        "a_w",
        "b_w",
        "y_w",
        "multiplier_opt",
        "ppg_opt",
        "ppa_opt",
        "fsa_opt",
        "a_enc",
        "b_enc",
        "y_enc",
        "design_label",
    ]
    for k in key_cols:
        if k not in df.columns:
            df[k] = np.nan

    parts = []
    for _, g in df.groupby(key_cols, dropna=False):
        nbits_val = int(g["n_bits"].iloc[0]) if pd.notna(g["n_bits"].iloc[0]) else 0
        tgt = target_sigma_for(nbits_val)
        gg = g.copy()
        gg["sigma_dist"] = (gg["sigma"].astype(float) - tgt).abs()
        idx = gg["sigma_dist"].idxmin()
        best = gg.loc[[idx], key_cols + ["switches"]].rename(columns={"switches": "switches_at_target"})
        parts.append(best)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=key_cols + ["switches_at_target"])


def build_design_level_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per configuration (static metrics + switches_at_target)."""
    key_cols = [
        "n_bits",
        "a_w",
        "b_w",
        "y_w",
        "multiplier_opt",
        "ppg_opt",
        "ppa_opt",
        "fsa_opt",
        "a_enc",
        "b_enc",
        "y_enc",
        "design_label",
    ]
    static_cols = [c for c in STATIC_METRICS if c in df.columns]
    base = df[key_cols + static_cols].drop_duplicates(subset=key_cols).reset_index(drop=True)
    sat = compute_switches_at_target(df)
    out = pd.merge(base, sat, on=key_cols, how="left")
    return out


# ------------------------------ filters ---------------------------- #


# @dataclass
# class Filters:
#     multiplier_opt: Optional[Sequence[str]] = None
#     ppg_opt: Optional[Sequence[str]] = None
#     ppa_opt: Optional[Sequence[str]] = None
#     fsa_opt: Optional[Sequence[str]] = None
#     a_enc: Optional[Sequence[str]] = None
#     b_enc: Optional[Sequence[str]] = None
#     y_enc: Optional[Sequence[str]] = None
#     design_label: Optional[Sequence[str]] = None
#     n_bits: Optional[Tuple[int, int]] = None
#     a_w: Optional[Tuple[int, int]] = None
#     b_w: Optional[Tuple[int, int]] = None
#     y_w: Optional[Tuple[int, int]] = None


# def apply_filters(df: pd.DataFrame, f: Filters) -> pd.DataFrame:
#     out = df.copy()

#     def filt(col, values):
#         nonlocal out
#         if values is not None and len(values) > 0 and col in out.columns:
#             out = out[out[col].astype("string").isin(values)]

#     filt("multiplier_opt", f.multiplier_opt)
#     filt("ppg_opt", f.ppg_opt)
#     filt("ppa_opt", f.ppa_opt)
#     filt("fsa_opt", f.fsa_opt)
#     filt("a_enc", f.a_enc)
#     filt("b_enc", f.b_enc)
#     filt("y_enc", f.y_enc)
#     filt("design_label", f.design_label)

#     def rng(col, r):
#         nonlocal out
#         if r is not None and col in out.columns:
#             out = out[(out[col] >= r[0]) & (out[col] <= r[1])]

#     rng("n_bits", f.n_bits)
#     rng("a_w", f.a_w)
#     rng("b_w", f.b_w)
#     rng("y_w", f.y_w)

#     return out

# from dataclasses import dataclass, field
# from typing import Dict, Optional, Sequence, Tuple
# import pandas as pd
# import numpy as np

# # Exclude one family everywhere
# f = Filters(not_multiplier_opt=["STAR_MULTIPLIER"])

# # Keep only unsigned encodings, but drop a specific FSA option
# f = Filters(
#     a_enc=["unsigned"], b_enc=["unsigned"], y_enc=["unsigned"],
#     not_fsa_opt=["RIPPLE_ADDER_STRONG"]
# )

# # Keep 4–32b, exclude the odd widths (numeric excludes)
# f = Filters(n_bits=(4, 32), n_bits_not=[5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31])

# # Generic include/exclude on arbitrary columns
# f = Filters(
#     include={"enc": ["unsigned/unsigned→unsigned", "sme/sme→unsigned"]},
#     exclude={"multiplier_opt": ["MULTIPLIER_FROM_OPTIMIZED_4BIT_BLOCKS_STRONG"]}
# )

@dataclass
class Filters:
    # -------- includes (same semantics as before) --------
    multiplier_opt: Optional[Sequence[str]] = None
    ppg_opt: Optional[Sequence[str]] = None
    ppa_opt: Optional[Sequence[str]] = None
    fsa_opt: Optional[Sequence[str]] = None
    a_enc: Optional[Sequence[str]] = None
    b_enc: Optional[Sequence[str]] = None
    y_enc: Optional[Sequence[str]] = None
    design_label: Optional[Sequence[str]] = None

    # numeric ranges (inclusive)
    n_bits: Optional[Tuple[int, int]] = None
    a_w: Optional[Tuple[int, int]] = None
    b_w: Optional[Tuple[int, int]] = None
    y_w: Optional[Tuple[int, int]] = None

    # -------- NEW: excludes (typed) --------
    not_multiplier_opt: Optional[Sequence[str]] = None
    not_ppg_opt: Optional[Sequence[str]] = None
    not_ppa_opt: Optional[Sequence[str]] = None
    not_fsa_opt: Optional[Sequence[str]] = None
    not_a_enc: Optional[Sequence[str]] = None
    not_b_enc: Optional[Sequence[str]] = None
    not_y_enc: Optional[Sequence[str]] = None
    not_design_label: Optional[Sequence[str]] = None

    # numeric value excludes (exact values)
    n_bits_not: Optional[Sequence[int]] = None
    a_w_not: Optional[Sequence[int]] = None
    b_w_not: Optional[Sequence[int]] = None
    y_w_not: Optional[Sequence[int]] = None

    # -------- NEW: generic include/exclude by column name --------
    # Example: include={"enc": ["unsigned/unsigned→unsigned"]},
    #          exclude={"multiplier_opt": ["STAR_MULTIPLIER"]}
    include: Dict[str, Sequence[str]] = field(default_factory=dict)
    exclude: Dict[str, Sequence[str]] = field(default_factory=dict)


def apply_filters(df: pd.DataFrame, f: Filters) -> pd.DataFrame:
    out = df.copy()

    # -------- helpers --------
    def include_cat(col: str, values: Optional[Sequence[str]]):
        nonlocal out
        if values and col in out.columns:
            out = out[out[col].astype("string").isin(values)]

    def exclude_cat(col: str, values: Optional[Sequence[str]]):
        nonlocal out
        if values and col in out.columns:
            out = out[~out[col].astype("string").isin(values)]

    def include_range(col: str, rng: Optional[Tuple[int, int]]):
        nonlocal out
        if rng and col in out.columns:
            out = out[(out[col] >= rng[0]) & (out[col] <= rng[1])]

    def exclude_values(col: str, vals: Optional[Sequence[int]]):
        nonlocal out
        if vals and col in out.columns:
            out = out[~out[col].isin(vals)]

    # -------- includes (categorical) --------
    include_cat("multiplier_opt", f.multiplier_opt)
    include_cat("ppg_opt",        f.ppg_opt)
    include_cat("ppa_opt",        f.ppa_opt)
    include_cat("fsa_opt",        f.fsa_opt)
    include_cat("a_enc",          f.a_enc)
    include_cat("b_enc",          f.b_enc)
    include_cat("y_enc",          f.y_enc)
    include_cat("design_label",   f.design_label)

    # generic includes
    for col, vals in (f.include or {}).items():
        include_cat(col, vals)

    # -------- includes (numeric ranges) --------
    include_range("n_bits", f.n_bits)
    include_range("a_w",    f.a_w)
    include_range("b_w",    f.b_w)
    include_range("y_w",    f.y_w)

    # -------- excludes (categorical) --------
    exclude_cat("multiplier_opt", f.not_multiplier_opt)
    exclude_cat("ppg_opt",        f.not_ppg_opt)
    exclude_cat("ppa_opt",        f.not_ppa_opt)
    exclude_cat("fsa_opt",        f.not_fsa_opt)
    exclude_cat("a_enc",          f.not_a_enc)
    exclude_cat("b_enc",          f.not_b_enc)
    exclude_cat("y_enc",          f.not_y_enc)
    exclude_cat("design_label",   f.not_design_label)

    # generic excludes
    for col, vals in (f.exclude or {}).items():
        exclude_cat(col, vals)

    # -------- excludes (numeric exact values) --------
    exclude_values("n_bits", f.n_bits_not)
    exclude_values("a_w",    f.a_w_not)
    exclude_values("b_w",    f.b_w_not)
    exclude_values("y_w",    f.y_w_not)

    return out


# ------------------------------ plotting --------------------------- #


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _slug(*parts: str) -> str:
    s = "_".join(p for p in parts if p).replace("/", "-").replace("→", "to").replace(" ", "_")
    return "".join(ch for ch in s if ch.isalnum() or ch in ("_", "-", "."))


def _category_colors(values: Sequence) -> Dict:
    """Map unique category values to matplotlib default cycle colors."""
    uniq = list(dict.fromkeys([str(v) for v in values]))
    palette = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not palette:
        palette = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]
    colors = {u: palette[i % len(palette)] for i, u in enumerate(uniq)}
    return colors


# ----- new: helpers for scaling plots ----- #


def _agg_from_name(name: str) -> Union[str, Callable[[pd.Series], float]]:
    name = (name or "median").lower()
    if name == "min":
        return "min"
    if name == "max":
        return "max"
    if name == "mean":
        return "mean"
    if name == "median":
        return "median"
    if name == "p10":
        return lambda x: float(np.quantile(x, 0.10))
    if name == "p90":
        return lambda x: float(np.quantile(x, 0.90))
    raise ValueError(f"Unknown aggregator '{name}'")


def _fit_power_law(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """
    Fit y ≈ a * x^p  (x>0, y>0) via linear regression in log-space:
       log(y) = log(a) + p * log(x)
    Returns: (p, a, r2)
    """
    mask = (x > 0) & (y > 0)
    x = x[mask]
    y = y[mask]
    if len(x) < 2:
        return (np.nan, np.nan, np.nan)
    lx = np.log(x)
    ly = np.log(y)
    p, b = np.polyfit(lx, ly, deg=1)  # slope, intercept
    a = float(np.exp(b))
    # R^2 in log space
    yhat = b + p * lx
    ss_res = float(np.sum((ly - yhat) ** 2))
    ss_tot = float(np.sum((ly - np.mean(ly)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return (float(p), a, r2)


def scatter_plot(
    design_df: pd.DataFrame,
    x_metric: str,
    y_metric: str,
    color_by: Optional[str],
    legend: bool,
    title: str,
    outfile: str,
) -> None:
    if design_df.empty or x_metric not in design_df.columns or y_metric not in design_df.columns:
        print(f"[warn] scatter skipped: missing metric(s) or empty data ({x_metric}, {y_metric})")
        return

    _ensure_dir(os.path.dirname(outfile) or ".")
    fig, ax = plt.subplots(figsize=(8, 6))

    if color_by and color_by in design_df.columns:
        vals = design_df[color_by].astype("string")
        cmap = _category_colors(vals)
        for val, g in design_df.groupby(vals):
            ax.scatter(g[x_metric], g[y_metric], label=str(val), s=28, alpha=0.85, color=cmap[str(val)])
    else:
        ax.scatter(design_df[x_metric], design_df[y_metric], s=28, alpha=0.85)

    ax.set_xlabel(x_metric)
    ax.set_ylabel(y_metric)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)
    if legend and color_by and color_by in design_df.columns:
        ax.legend(fontsize=6, loc="best", frameon=False, markerscale=0.9, handlelength=1.0)
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)


def line_plot_sigma(
    full_df: pd.DataFrame,
    y_metric: str,
    color_by: Optional[str],
    legend: bool,
    title: str,
    outfile: str,
) -> None:
    """Lines connect ONLY within the same run_id."""
    if full_df.empty or "sigma" not in full_df.columns or y_metric not in full_df.columns:
        print(f"[warn] line skipped: missing 'sigma' or '{y_metric}' or empty data")
        return

    _ensure_dir(os.path.dirname(outfile) or ".")
    fig, ax = plt.subplots(figsize=(8, 6))

    # Guarantee run_id exists
    if "run_id" not in full_df.columns:
        full_df = full_df.copy()
        full_df["run_id"] = (
            full_df["multiplier_opt"].astype(str)
            + "|"
            + full_df["ppg_opt"].astype(str)
            + "|"
            + full_df["ppa_opt"].astype(str)
            + "|"
            + full_df["fsa_opt"].astype(str)
            + "|"
            + full_df["a_enc"].astype(str)
            + "/"
            + full_df["b_enc"].astype(str)
            + "→"
            + full_df["y_enc"].astype(str)
        )

    # Prepare colors by category (one color per value of color_by)
    if color_by and color_by in full_df.columns:
        cat_vals_per_run = full_df.groupby("run_id")[color_by].first().astype("string")
        cmap = _category_colors(cat_vals_per_run.values)
        color_for_run = {rid: cmap[str(cat_vals_per_run.loc[rid])] for rid in cat_vals_per_run.index}
    else:
        color_for_run = {}

    # Plot each run as one line (sorted by sigma; unique per (run_id, sigma))
    for rid, g in full_df.groupby("run_id"):
        gg = g.dropna(subset=["sigma", y_metric]).sort_values("sigma").drop_duplicates(subset=["run_id", "sigma"], keep="first")
        if gg.empty:
            continue
        color = color_for_run.get(rid, None)
        label = None
        if legend and color_by and color_by in full_df.columns:
            label = str(cat_vals_per_run.loc[rid])
        ax.plot(gg["sigma"], gg[y_metric], marker="o", markersize=3.5, linewidth=1.2, alpha=0.9, color=color, label=label)

    ax.set_xlabel("sigma")
    ax.set_ylabel(y_metric)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)

    if legend and color_by and color_by in full_df.columns:
        # Deduplicate identical legend entries
        handles, labels = ax.get_legend_handles_labels()
        seen = set()
        uniq = [(h, l) for h, l in zip(handles, labels) if not (l in seen or seen.add(l))]
        if uniq:
            ax.legend(*zip(*uniq), fontsize=6, loc="best", frameon=False, markerscale=0.9, handlelength=1.0)

    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)


def scaling_plot(
    design_df: pd.DataFrame,
    y_metric: str,
    color_by: Optional[str],
    legend: bool,
    title: str,
    outfile: str,
    agg: str = "min",
    loglog: bool = True,
    fit_power: bool = True,
    min_points: int = 2,
) -> None:
    """
    Scaling law plot: metric vs n_bits; one line per category (color_by).
    - Aggregates duplicates per (category, n_bits) using 'agg'.
    - If loglog=True: both axes in log scale (x base 2).
    - If fit_power=True: fit y ≈ a*n^p per line and annotate p in legend.
    """
    if design_df.empty or "n_bits" not in design_df.columns or y_metric not in design_df.columns:
        print(f"[warn] scaling skipped: empty data or missing 'n_bits'/'{y_metric}'")
        return

    _ensure_dir(os.path.dirname(outfile) or ".")
    fig, ax = plt.subplots(figsize=(8, 6))

    # Prepare grouping
    if color_by and color_by in design_df.columns:
        cat_series = design_df[color_by].astype("string")
        cmap = _category_colors(cat_series.values)
        groups = [(val, g.copy()) for val, g in design_df.groupby(cat_series)]
    else:
        cmap = {"All": None}
        design_df = design_df.copy()
        design_df["__all__"] = "All"
        groups = [("All", design_df)]

    agg_fn = _agg_from_name(agg)

    legend_entries = []
    for cat_value, g in groups:
        gg = g.dropna(subset=[y_metric, "n_bits"]).copy()
        if gg.empty:
            continue
        # per-bit aggregation
        agg_df = gg.groupby("n_bits", as_index=False)[y_metric].agg(value=agg_fn).sort_values("n_bits")
        x = agg_df["n_bits"].astype(float).to_numpy()
        y = agg_df["value"].astype(float).to_numpy()

        if len(x) == 0:
            continue

        color = cmap.get(str(cat_value))
        # Plot aggregated points/lines
        ax.plot(x, y, marker="o", markersize=4, linewidth=1.4, alpha=0.95, label=str(cat_value), color=color)

        # Optional power-law fit
        if fit_power and len(x) >= max(2, min_points):
            p, a, r2 = _fit_power_law(x, y)
            if not np.isnan(p):
                # Smooth fit across observed x-range
                xs = np.linspace(x.min(), x.max(), 200)
                ys = a * (xs**p)
                ax.plot(xs, ys, linestyle="--", linewidth=1.0, alpha=0.9, color=color)
                legend_entries.append((str(cat_value), p, r2))
            else:
                legend_entries.append((str(cat_value), np.nan, np.nan))
        else:
            legend_entries.append((str(cat_value), np.nan, np.nan))

    ax.set_xlabel("n_bits")
    ax.set_ylabel(y_metric)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)

    if loglog:
        # base-2 x-axis is nice for bit-width
        try:
            ax.set_xscale("log", base=2)
        except TypeError:
            # older mpl: 'basex'
            ax.set_xscale("log")
        ax.set_yscale("log")

    if legend:
        # Build legend labels with p if available
        handles, labels = ax.get_legend_handles_labels()
        label_to_p = {name: (p, r2) for (name, p, r2) in legend_entries}
        new_labels = []
        for lab in labels:
            p, r2 = label_to_p.get(lab, (np.nan, np.nan))
            if not np.isnan(p):
                if not np.isnan(r2):
                    new_labels.append(f"{lab}  (p={p:.2f}, R²={r2:.3f})")
                else:
                    new_labels.append(f"{lab}  (p={p:.2f})")
            else:
                new_labels.append(lab)
        # Deduplicate while preserving order
        seen = set()
        uniq = [(h, l) for h, l in zip(handles, new_labels) if not (l in seen or seen.add(l))]
        if uniq:
            ax.legend(*zip(*uniq), fontsize=7, loc="best", frameon=False, markerscale=0.9, handlelength=1.2)

    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)

def relative_scaling_plot(
    design_df: pd.DataFrame,
    y_metric: str,
    color_by: Optional[str],
    baseline_category: str,
    legend: bool,
    title: str,
    outfile: str,
    agg: str = "min",
    loglog: bool = False,
    fit_power: bool = False,
    min_points: int = 2,
    show_baseline_line: bool = True,
) -> None:
    """
    Relative scaling: for each category in `color_by`, plot
        (agg(y_metric) at n_bits) / (agg(y_metric) of baseline_category at n_bits).
    One line per category (excluding the baseline itself).
    """
    if design_df.empty or "n_bits" not in design_df.columns or y_metric not in design_df.columns:
        print(f"[warn] rel_scaling skipped: empty data or missing 'n_bits'/'{y_metric}'")
        return
    if not color_by or color_by not in design_df.columns:
        print("[warn] rel_scaling skipped: 'color_by' must be provided and exist in the dataframe")
        return
    if baseline_category is None:
        print("[warn] rel_scaling skipped: 'baseline_category' must be provided")
        return

    _ensure_dir(os.path.dirname(outfile) or ".")
    fig, ax = plt.subplots(figsize=(8, 6))

    cat_series = design_df[color_by].astype("string")
    base_key = str(baseline_category)
    cmap = _category_colors(cat_series.values)
    agg_fn = _agg_from_name(agg)

    # ---- baseline aggregate per n_bits ----
    base_df = (
        design_df[cat_series == base_key]
        .dropna(subset=[y_metric, "n_bits"])
        .groupby("n_bits", as_index=False)[y_metric]
        .agg(agg_fn)
        .rename(columns={y_metric: "base_value"})
        .sort_values("n_bits")
    )
    if base_df.empty:
        print(f"[warn] rel_scaling: baseline '{base_key}' has no data")
        return

    # optional guide at y=1 over baseline x-range
    x_base = base_df["n_bits"].astype(float).to_numpy()
    if show_baseline_line and len(x_base) > 0:
        ax.plot(x_base, np.ones_like(x_base), linestyle="--", linewidth=1.0,
                color="black", alpha=0.8, label=f"baseline = {base_key}")

    legend_entries: List[Tuple[str, float, float]] = []  # (cat, p_rel, r2)

    # ---- plot each category relative to baseline ----
    for cat_val, g in design_df.groupby(cat_series):
        cat_val = str(cat_val)
        if cat_val == base_key:
            continue

        gg = g.dropna(subset=[y_metric, "n_bits"])
        if gg.empty:
            continue

        agg_df = (
            gg.groupby("n_bits", as_index=False)[y_metric]
              .agg(agg_fn)
              .rename(columns={y_metric: "value"})
        )
        merged = pd.merge(agg_df, base_df, on="n_bits", how="inner").sort_values("n_bits")
        if merged.empty:
            continue

        x = merged["n_bits"].astype(float).to_numpy()
        ratio = (merged["value"] / merged["base_value"]).astype(float).to_numpy()
        color = cmap.get(cat_val)

        ax.plot(x, ratio, marker="o", markersize=4, linewidth=1.4, alpha=0.95,
                color=color, label=cat_val)

        # Optional: fit power law to ratio -> estimates Δp = p_cat - p_baseline
        if fit_power and len(x) >= max(2, min_points):
            p_rel, a_rel, r2 = _fit_power_law(x, ratio)
            if not np.isnan(p_rel):
                xs = np.linspace(x.min(), x.max(), 200)
                ys = a_rel * (xs ** p_rel)
                ax.plot(xs, ys, linestyle="--", linewidth=1.0, alpha=0.9, color=color)
                legend_entries.append((cat_val, p_rel, r2))
            else:
                legend_entries.append((cat_val, np.nan, np.nan))
        else:
            legend_entries.append((cat_val, np.nan, np.nan))

    ax.set_xlabel("n_bits")
    ax.set_ylabel(f"{y_metric} / baseline({base_key})")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)

    if loglog:
        try:
            ax.set_xscale("log", base=2)
        except TypeError:
            ax.set_xscale("log")
        ax.set_yscale("log")

    if legend:
        handles, labels = ax.get_legend_handles_labels()
        label_to_p = {name: (p, r2) for (name, p, r2) in legend_entries}
        new_labels = []
        for lab in labels:
            p, r2 = label_to_p.get(lab, (np.nan, np.nan))
            if not np.isnan(p):
                new_labels.append(f"{lab}  (Δp={p:.2f}, R²={r2:.3f})")
            else:
                new_labels.append(lab)
        seen = set()
        uniq = [(h, l) for h, l in zip(handles, new_labels) if not (l in seen or seen.add(l))]
        if uniq:
            ax.legend(*zip(*uniq), fontsize=7, loc="best", frameon=False, markerscale=0.9, handlelength=1.2)

    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)

# ------------------------- example orchestration -------------------- #


@dataclass
class PlotConfig:
    kind: str  # 'scatter' or 'line' or 'scaling'
    title: str
    filename: str
    x_metric: Optional[str] = None  # scatter only
    y_metric: Optional[str] = None  # scatter + line + scaling (scaling uses only y_metric)
    color_by: Optional[str] = None
    legend: bool = True
    filters: Filters = field(default_factory=Filters)
    # scaling-specific options
    agg: str = "min"
    loglog: bool = True
    fit_power: bool = True
    min_points: int = 2
    # --- NEW: for relative scaling ---
    baseline_category: Optional[str] = None   # value from the 'color_by' column
    show_baseline_line: bool = True           # draw y=1 guide (over baseline x-range)


def run_plot(cfg: PlotConfig, full_df: pd.DataFrame, design_df: pd.DataFrame, out_dir: str) -> None:
    out_file = os.path.join(out_dir, _slug(cfg.filename))
    if cfg.kind == "scatter":
        ddf = apply_filters(design_df, cfg.filters)
        if cfg.x_metric is None or cfg.y_metric is None:
            print(f"[warn] scatter '{cfg.title}' skipped: x_metric/y_metric not set")
            return
        scatter_plot(ddf, cfg.x_metric, cfg.y_metric, cfg.color_by, cfg.legend, cfg.title, out_file)
    elif cfg.kind == "line":
        fdf = apply_filters(full_df, cfg.filters)
        if cfg.y_metric is None:
            print(f"[warn] line '{cfg.title}' skipped: y_metric not set")
            return
        line_plot_sigma(fdf, cfg.y_metric, cfg.color_by, cfg.legend, cfg.title, out_file)
    elif cfg.kind == "scaling":
        ddf = apply_filters(design_df, cfg.filters)
        if cfg.y_metric is None:
            print(f"[warn] scaling '{cfg.title}' skipped: y_metric not set")
            return
        scaling_plot(
            ddf, cfg.y_metric, cfg.color_by, cfg.legend, cfg.title, out_file, agg=cfg.agg, loglog=cfg.loglog, fit_power=cfg.fit_power, min_points=cfg.min_points
        )
    elif cfg.kind == "rel_scaling":
        ddf = apply_filters(design_df, cfg.filters)
        if cfg.y_metric is None or cfg.color_by is None or cfg.baseline_category is None:
            print(f"[warn] rel_scaling '{cfg.title}' skipped: need y_metric, color_by, baseline_category")
        else:
            relative_scaling_plot(
                ddf,
                y_metric=cfg.y_metric,
                color_by=cfg.color_by,
                baseline_category=cfg.baseline_category,
                legend=cfg.legend,
                title=cfg.title,
                outfile=os.path.join(out_dir, _slug(cfg.filename)),
                agg=cfg.agg,
                loglog=cfg.loglog,
                fit_power=cfg.fit_power,
                min_points=cfg.min_points,
                show_baseline_line=cfg.show_baseline_line,
            )
    
    else:
        print(f"[warn] unknown plot kind: {cfg.kind}")


# ------------------------------- main ------------------------------ #


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--file",
        dest="files",
        action="append",
        required=True,
        help="Path to a Parquet file (repeat to load and concatenate multiple files).",
    )
    ap.add_argument("--out", default="plots_mpl", help="Output directory for PNG files.")
    ap.add_argument("--legend", choices=["on", "off"], default="on", help="Show legends globally.")
    args = ap.parse_args()

    df = load_many_dfs(args.files)
    design_df = build_design_level_table(df)

    # Resolve a sensible area metric for examples
    try:
        area_col = choose_area_column(design_df)
    except Exception:
        area_col = None

    # -------------------- example plots -------------------- #
    PLOTS: List[PlotConfig] = []

    # 1) Scatter: switches_at_target vs area, colored by n_bits
    if area_col:
        PLOTS.append(
            PlotConfig(
                kind="scatter",
                title=f"switches_at_target vs {area_col} (one point per configuration; categorized by n_bits)",
                filename=f"scatter_switches_at_target_vs_{area_col}.png",
                x_metric=area_col,
                y_metric="switches_at_target",
                color_by="n_bits",
                legend=(args.legend == "on"),
                filters=Filters(),
            )
        )

    # 2) Scatter: switches_at_target vs max_depth (fallback to total_expr_nodes), colored by multiplier_opt
    depth_y = "max_depth" if "max_depth" in design_df.columns else ("total_expr_nodes" if "total_expr_nodes" in design_df.columns else None)
    if depth_y:
        PLOTS.append(
            PlotConfig(
                kind="scatter",
                title=f"switches_at_target vs {depth_y} (categorized by multiplier_opt)",
                filename=f"scatter_switches_at_target_vs_{depth_y}.png",
                x_metric=depth_y,
                y_metric="switches_at_target",
                color_by="multiplier_opt",
                legend=(args.legend == "on"),
                filters=Filters(),
            )
        )

    # 3) Line: switches vs sigma, colored by n_bits (lines connect per run_id)
    if "switches" in df.columns:
        PLOTS.append(
            PlotConfig(
                kind="line",
                title="switches vs sigma (lines per run_id; categorized by n_bits)",
                filename="line_switches_vs_sigma.png",
                y_metric="switches",
                color_by="n_bits",
                legend=(args.legend == "on"),
                filters=Filters(),
            )
        )

    # 4) Line: switches vs sigma, colored by multiplier_opt
    if "switches" in df.columns:
        PLOTS.append(
            PlotConfig(
                kind="line",
                title="switches vs sigma (lines per run_id; categorized by multiplier_opt)",
                filename="line_switches_vs_sigma_by_multiplier_opt.png",
                y_metric="switches",
                color_by="multiplier_opt",
                legend=(args.legend == "on"),
                filters=Filters(),
            )
        )

    # 5) NEW: Scaling — area vs n_bits, min per (category, n_bits), log–log with power-law fits
    if area_col and "n_bits" in design_df.columns:
        PLOTS.append(
            PlotConfig(
                kind="scaling",
                title=f"Scaling of {area_col} vs n_bits (min per category; log–log; categorized by multiplier_opt)",
                filename=f"scaling_{area_col}_vs_n_bits_by_multiplier_opt.png",
                y_metric=area_col,
                color_by="multiplier_opt",  # change to 'enc', 'stages', etc. if you prefer
                legend=(args.legend == "on"),
                agg="min",  # min, median, mean, p10, p90, max
                loglog=True,
                fit_power=True,
                min_points=2,
                filters=Filters(),  # e.g., Filters(a_enc=["unsigned"], b_enc=["unsigned"])
            )
        )

    filter_sel = Filters(not_multiplier_opt=["STAR_MULTIPLIER"], a_enc=["unsigned"], b_enc=["unsigned"])
    stage_unsigned_filters = Filters(multiplier_opt=["STAGE_BASED_MULTIPLIER"], a_enc=["unsigned"], b_enc=["unsigned"])

    # 6) NEW: Scaling — depth vs n_bits, min per (category, n_bits), linear axes (depth often ~ O(log n) or O(n))
    if "max_depth" in design_df.columns and "n_bits" in design_df.columns:
        PLOTS.append(
            PlotConfig(
                kind="scaling",
                title="Scaling of max_depth vs n_bits (min per category; categorized by multiplier_opt)",
                filename="scaling_max_depth_vs_n_bits_by_multiplier_opt.png",
                y_metric="max_depth",
                color_by="multiplier_opt",
                legend=(args.legend == "on"),
                agg="min",
                loglog=False,  # keep linear to see ~O(n) vs ~O(log n) trends
                fit_power=True,  # still fit p on linear data via log-fit (interprets as power law)
                min_points=2,
                filters=filter_sel,  # e.g., exclude outliers
            )
        )

    if "num_aig_gates" in design_df.columns and "n_bits" in design_df.columns:
        PLOTS.append(
            PlotConfig(
                kind="scaling",
                title="Scaling of num_aig_gates vs n_bits (min per category; categorized by multiplier_opt)",
                filename="scaling_num_aig_gates_vs_n_bits_by_multiplier_opt.png",
                y_metric="num_aig_gates",
                color_by="multiplier_opt",
                legend=(args.legend == "on"),
                agg="min",
                loglog=True,  # keep linear to see ~O(n) vs ~O(log n) trends
                fit_power=True,  # still fit p on linear data via log-fit (interprets as power law)
                min_points=2,
                filters=filter_sel,  # e.g., exclude outliers
            )
        )

        PLOTS.append(
            PlotConfig(
                kind="scaling",
                title="Scaling of num_aig_gates vs n_bits (min per category; categorized by multiplier_opt)",
                filename="scaling_num_aig_gates_vs_n_bits_by_multiplier_opt_linear.png",
                y_metric="num_aig_gates",
                color_by="multiplier_opt",
                legend=(args.legend == "on"),
                agg="min",
                loglog=False,  # keep linear to see ~O(n) vs ~O(log n) trends
                fit_power=True,  # still fit p on linear data via log-fit (interprets as power law)
                min_points=2,
                filters=filter_sel,  # e.g., exclude outliers
            )
        )

        PLOTS.append(
            PlotConfig(
                kind="rel_scaling",
                title="Relative Scaling of num_aig_gates vs n_bits (min per category; categorized by multiplier_opt)",
                filename="rel_scaling_num_aig_gates_vs_n_bits_by_multiplier_opt.png",
                y_metric="num_aig_gates",
                color_by="multiplier_opt",
                legend=(args.legend == "on"),
                agg="min",
                loglog=False,  # keep linear to see ~O(n) vs ~O(log n) trends
                fit_power=False,  # still fit p on linear data via log-fit (interprets as power law)
                min_points=2,
                filters=filter_sel,  # e.g., exclude outliers
                baseline_category="STAGE_BASED_MULTIPLIER",
            )
        )

    if "aig_depth" in design_df.columns and "n_bits" in design_df.columns:
        PLOTS.append(
            PlotConfig(
                kind="scaling",
                title="Scaling of aig_depth vs n_bits (min per category; categorized by multiplier_opt)",
                filename="scaling_aig_depth_vs_n_bits_by_multiplier_opt.png",
                y_metric="aig_depth",
                color_by="multiplier_opt",
                legend=(args.legend == "on"),
                agg="min",
                loglog=True,  # keep linear to see ~O(n) vs ~O(log n) trends
                fit_power=True,  # still fit p on linear data via log-fit (interprets as power law)
                min_points=2,
                filters=filter_sel,  # e.g., exclude outliers
            )
        )

        PLOTS.append(
            PlotConfig(
                kind="scaling",
                title="Scaling of aig_depth vs n_bits (min per category; categorized by multiplier_opt)",
                filename="scaling_aig_depth_vs_n_bits_by_multiplier_opt_linear.png",
                y_metric="aig_depth",
                color_by="multiplier_opt",
                legend=(args.legend == "on"),
                agg="min",
                loglog=False,  # keep linear to see ~O(n) vs ~O(log n) trends
                fit_power=True,  # still fit p on linear data via log-fit (interprets as power law)
                min_points=2,
                filters=filter_sel,  # e.g., exclude outliers
            )
        )

        PLOTS.append(
            PlotConfig(
                kind="rel_scaling",
                title="Relative Scaling of aig_depth vs n_bits (min per category; categorized by multiplier_opt)",
                filename="rel_scaling_aig_depth_vs_n_bits_by_multiplier_opt.png",
                y_metric="aig_depth",
                color_by="multiplier_opt",
                legend=(args.legend == "on"),
                agg="min",
                loglog=False,  # keep linear to see ~O(n) vs ~O(log n) trends
                fit_power=False,  # still fit p on linear data via log-fit (interprets as power law)
                min_points=2,
                filters=filter_sel,  # e.g., exclude outliers
                baseline_category="STAGE_BASED_MULTIPLIER",
            )
        )

    if "switches" in design_df.columns and "n_bits" in design_df.columns:
        PLOTS.append(
            PlotConfig(
                kind="scaling",
                title="Scaling of switches vs n_bits (min per category; categorized by multiplier_opt)",
                filename="scaling_switches_vs_n_bits_by_multiplier_opt.png",
                y_metric="switches",
                color_by="multiplier_opt",
                legend=(args.legend == "on"),
                agg="min",
                loglog=True,  # keep linear to see ~O(n) vs ~O(log n) trends
                fit_power=True,  # still fit p on linear data via log-fit (interprets as power law)
                min_points=2,
                filters=filter_sel,  # e.g., exclude outliers
            )
        )

        PLOTS.append(
            PlotConfig(
                kind="scaling",
                title="Scaling of switches vs n_bits (min per category; categorized by multiplier_opt)",
                filename="scaling_switches_vs_n_bits_by_multiplier_opt_linear.png",
                y_metric="switches",
                color_by="multiplier_opt",
                legend=(args.legend == "on"),
                agg="min",
                loglog=False,  # keep linear to see ~O(n) vs ~O(log n) trends
                fit_power=True,  # still fit p on linear data via log-fit (interprets as power law)
                min_points=2,
                filters=filter_sel,  # e.g., exclude outliers
            )
        )

        PLOTS.append(
            PlotConfig(
                kind="rel_scaling",
                title="Relative Scaling of switches vs n_bits (min per category; categorized by multiplier_opt)",
                filename="rel_scaling_switches_vs_n_bits_by_multiplier_opt.png",
                y_metric="switches",
                color_by="multiplier_opt",
                legend=(args.legend == "on"),
                agg="min",
                loglog=False,  # keep linear to see ~O(n) vs ~O(log n) trends
                fit_power=False,  # still fit p on linear data via log-fit (interprets as power law)
                min_points=2,
                filters=filter_sel,  # e.g., exclude outliers
                baseline_category="STAGE_BASED_MULTIPLIER",
            )
        )

    if "num_aig_gates" in design_df.columns and "n_bits" in design_df.columns:

        PLOTS.append(
            PlotConfig(
                kind="scaling",
                title="Scaling of num_aig_gates vs n_bits (min per category; categorized by ppg_opt)",
                filename="scaling_num_aig_gates_vs_n_bits_by_ppg_opt.png",
                y_metric="num_aig_gates",
                color_by="ppg_opt",
                legend=(args.legend == "on"),
                agg="min",
                loglog=True,  # keep linear to see ~O(n) vs ~O(log n) trends
                fit_power=True,  # still fit p on linear data via log-fit (interprets as power law)
                min_points=2,
                filters=stage_unsigned_filters,  # e.g., exclude outliers
                baseline_category="AND",
            )
        )

        PLOTS.append(
            PlotConfig(
                kind="rel_scaling",
                title="Relative Scaling of num_aig_gates vs n_bits (min per category; categorized by ppg_opt)",
                filename="rel_scaling_num_aig_gates_vs_n_bits_by_ppg_opt.png",
                y_metric="num_aig_gates",
                color_by="ppg_opt",
                legend=(args.legend == "on"),
                agg="min",
                loglog=False,  # keep linear to see ~O(n) vs ~O(log n) trends
                fit_power=False,  # still fit p on linear data via log-fit (interprets as power law)
                min_points=2,
                filters=stage_unsigned_filters,  # e.g., exclude outliers
                baseline_category="AND",
            )
        )

    if "aig_depth" in design_df.columns and "n_bits" in design_df.columns:

        PLOTS.append(
            PlotConfig(
                kind="scaling",
                title="Scaling of aig_depth vs n_bits (min per category; categorized by ppg_opt)",
                filename="scaling_aig_depth_vs_n_bits_by_ppg_opt.png",
                y_metric="aig_depth",
                color_by="ppg_opt",
                legend=(args.legend == "on"),
                agg="min",
                loglog=True,  # keep linear to see ~O(n) vs ~O(log n) trends
                fit_power=True,  # still fit p on linear data via log-fit (interprets as power law)
                min_points=2,
                filters=stage_unsigned_filters,  # e.g., exclude outliers
                baseline_category="AND",
            )
        )

        PLOTS.append(
            PlotConfig(
                kind="rel_scaling",
                title="Relative Scaling of aig_depth vs n_bits (min per category; categorized by ppg_opt)",
                filename="rel_scaling_aig_depth_vs_n_bits_by_ppg_opt.png",
                y_metric="aig_depth",
                color_by="ppg_opt",
                legend=(args.legend == "on"),
                agg="min",
                loglog=False,  # keep linear to see ~O(n) vs ~O(log n) trends
                fit_power=False,  # still fit p on linear data via log-fit (interprets as power law)
                min_points=2,
                filters=stage_unsigned_filters,  # e.g., exclude outliers
                baseline_category="AND",
            )
        )

    if "switches" in design_df.columns and "n_bits" in design_df.columns:

        PLOTS.append(
            PlotConfig(
                kind="scaling",
                title="Scaling of switches vs n_bits (min per category; categorized by ppg_opt)",
                filename="scaling_switches_vs_n_bits_by_ppg_opt.png",
                y_metric="switches",
                color_by="ppg_opt",
                legend=(args.legend == "on"),
                agg="min",
                loglog=True,  # keep linear to see ~O(n) vs ~O(log n) trends
                fit_power=True,  # still fit p on linear data via log-fit (interprets as power law)
                min_points=2,
                filters=stage_unsigned_filters,  # e.g., exclude outliers
                baseline_category="AND",
            )
        )

        PLOTS.append(
            PlotConfig(
                kind="rel_scaling",
                title="Relative Scaling of switches vs n_bits (min per category; categorized by ppg_opt)",
                filename="rel_scaling_switches_vs_n_bits_by_ppg_opt.png",
                y_metric="switches",
                color_by="ppg_opt",
                legend=(args.legend == "on"),
                agg="min",
                loglog=False,  # keep linear to see ~O(n) vs ~O(log n) trends
                fit_power=False,  # still fit p on linear data via log-fit (interprets as power law)
                min_points=2,
                filters=stage_unsigned_filters,  # e.g., exclude outliers
                baseline_category="AND",
            )
        )

    if "num_aig_gates" in design_df.columns and "n_bits" in design_df.columns:

        PLOTS.append(
            PlotConfig(
                kind="scaling",
                title="Scaling of num_aig_gates vs n_bits (min per category; categorized by ppa_opt)",
                filename="scaling_num_aig_gates_vs_n_bits_by_ppa_opt.png",
                y_metric="num_aig_gates",
                color_by="ppa_opt",
                legend=(args.legend == "on"),
                agg="min",
                loglog=True,  # keep linear to see ~O(n) vs ~O(log n) trends
                fit_power=True,  # still fit p on linear data via log-fit (interprets as power law)
                min_points=2,
                filters=stage_unsigned_filters,  # e.g., exclude outliers
                baseline_category="WALLACE_TREE",
            )
        )

        PLOTS.append(
            PlotConfig(
                kind="rel_scaling",
                title="Relative Scaling of num_aig_gates vs n_bits (min per category; categorized by ppa_opt)",
                filename="rel_scaling_num_aig_gates_vs_n_bits_by_ppa_opt.png",
                y_metric="num_aig_gates",
                color_by="ppa_opt",
                legend=(args.legend == "on"),
                agg="min",
                loglog=False,  # keep linear to see ~O(n) vs ~O(log n) trends
                fit_power=False,  # still fit p on linear data via log-fit (interprets as power law)
                min_points=2,
                filters=stage_unsigned_filters,  # e.g., exclude outliers
                baseline_category="WALLACE_TREE",
            )
        )

    if "aig_depth" in design_df.columns and "n_bits" in design_df.columns:

        PLOTS.append(
            PlotConfig(
                kind="scaling",
                title="Scaling of aig_depth vs n_bits (min per category; categorized by ppa_opt)",
                filename="scaling_aig_depth_vs_n_bits_by_ppa_opt.png",
                y_metric="aig_depth",
                color_by="ppa_opt",
                legend=(args.legend == "on"),
                agg="min",
                loglog=True,  # keep linear to see ~O(n) vs ~O(log n) trends
                fit_power=True,  # still fit p on linear data via log-fit (interprets as power law)
                min_points=2,
                filters=stage_unsigned_filters,  # e.g., exclude outliers
                baseline_category="WALLACE_TREE",
            )
        )

        PLOTS.append(
            PlotConfig(
                kind="rel_scaling",
                title="Relative Scaling of aig_depth vs n_bits (min per category; categorized by ppa_opt)",
                filename="rel_scaling_aig_depth_vs_n_bits_by_ppa_opt.png",
                y_metric="aig_depth",
                color_by="ppa_opt",
                legend=(args.legend == "on"),
                agg="min",
                loglog=False,  # keep linear to see ~O(n) vs ~O(log n) trends
                fit_power=False,  # still fit p on linear data via log-fit (interprets as power law)
                min_points=2,
                filters=stage_unsigned_filters,  # e.g., exclude outliers
                baseline_category="WALLACE_TREE",
            )
        )

    if "switches" in design_df.columns and "n_bits" in design_df.columns:

        PLOTS.append(
            PlotConfig(
                kind="scaling",
                title="Scaling of switches vs n_bits (min per category; categorized by ppa_opt)",
                filename="scaling_switches_vs_n_bits_by_ppa_opt.png",
                y_metric="switches",
                color_by="ppa_opt",
                legend=(args.legend == "on"),
                agg="min",
                loglog=True,  # keep linear to see ~O(n) vs ~O(log n) trends
                fit_power=True,  # still fit p on linear data via log-fit (interprets as power law)
                min_points=2,
                filters=stage_unsigned_filters,  # e.g., exclude outliers
                baseline_category="WALLACE_TREE",
            )
        )

        PLOTS.append(
            PlotConfig(
                kind="rel_scaling",
                title="Relative Scaling of switches vs n_bits (min per category; categorized by ppa_opt)",
                filename="rel_scaling_switches_vs_n_bits_by_ppa_opt.png",
                y_metric="switches",
                color_by="ppa_opt",
                legend=(args.legend == "on"),
                agg="min",
                loglog=False,  # keep linear to see ~O(n) vs ~O(log n) trends
                fit_power=False,  # still fit p on linear data via log-fit (interprets as power law)
                min_points=2,
                filters=stage_unsigned_filters,  # e.g., exclude outliers
                baseline_category="WALLACE_TREE",
            )
        )

    # Run all examples
    for cfg in PLOTS:
        run_plot(cfg, df, design_df, args.out)

    print(f"[done] Wrote plots to: {os.path.abspath(args.out)}")
    print(f"[info] Generated {len(PLOTS)} plots.")
    print("Tip: add more PlotConfig entries to the PLOTS list or generate them programmatically.")
    print("For scaling plots, tune 'color_by', 'agg', 'loglog', and 'fit_power' as needed.")


if __name__ == "__main__":
    main()
