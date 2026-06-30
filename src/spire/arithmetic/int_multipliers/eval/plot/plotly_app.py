#!/usr/bin/env python3
"""
Dash + Plotly app for interactive visualization of multiplier exploration results
stored in one or more Parquet files (pandas/pyarrow only).

Features
--------
- Two plot types (tabs):
  1) Scatter: any metric vs any metric (includes "switches_at_target");
     one point per multiplier configuration (deduped across sigma).
  2) Line: metric vs sigma (choose any metric; "switches" gives switches-per-sigma lines).

- Filters (dropdowns/range sliders):
  * multiplier_opt, ppg_opt, ppa_opt, fsa_opt
  * a_enc, b_enc, y_enc
  * combined "design label" = "{multiplier_opt} | PPG=... | PPA=... | FSA=... | enc=a/b→y"
  * n_bits, a_w, b_w, y_w ranges
  * color-by (for both plot types)

Usage
-----
python dash_multiplier_app.py --file data/multiplier_runs.parquet --host 0.0.0.0 --port 8050
(repeat --file to load and concatenate multiple Parquet files)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np

import dash
from dash import Dash, dcc, html, Output, Input, State, callback, no_update
import plotly.express as px
import plotly.graph_objects as go


# ----------------------------- config ----------------------------- #

STATIC_METRICS = [
    #"estimated_num_transistors",
    "transistor_count",
    "num_cells",
    "num_wires",
    "max_depth",
    "depth_y",
    "total_expr_nodes",
    "num_aig_gates",
    "aig_depth",
]

# These are always useful to show in hover
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

extra_cols = ["enc", "io_widths", "stages", "optimization_effort"]

# Full label format requested by you
def design_label_from_row(row: pd.Series) -> str:
    def g(k: str, default: str = "NA"):
        return row[k] if (k in row.index and pd.notna(row[k])) else default

    return f"{g('multiplier_opt')} | " f"PPG={g('ppg_opt')} | PPA={g('ppa_opt')} | FSA={g('fsa_opt')} | " f"enc={g('a_enc')}/{g('b_enc')}→{g('y_enc')}"


# --------------------------- data helpers -------------------------- #
def augment_df_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add convenient derived columns (strings) for labeling/coloring."""
    df = df.copy()

    # enc: "a_enc/b_enc→y_enc"
    if all(c in df.columns for c in ("a_enc", "b_enc", "y_enc")):
        df["enc"] = (
            df["a_enc"].astype("string") + "/" +
            df["b_enc"].astype("string") + "→" +
            df["y_enc"].astype("string")
        )

    # io_widths: "a_w/b_w→y_w" (uses pandas Int64 so NaNs are allowed)
    if all(c in df.columns for c in ("a_w", "b_w", "y_w")):
        df["io_widths"] = (
            df["a_w"].astype("Int64").astype("string") + "/" +
            df["b_w"].astype("Int64").astype("string") + "→" +
            df["y_w"].astype("Int64").astype("string")
        )

    # stages combined: "PPG=... | PPA=... | FSA=..."
    if all(c in df.columns for c in ("ppg_opt", "ppa_opt", "fsa_opt")):
        df["stages"] = (
            "PPG=" + df["ppg_opt"].astype("string") +
            " | PPA=" + df["ppa_opt"].astype("string") +
            " | FSA=" + df["fsa_opt"].astype("string")
        )

    return df


def load_df(file_path: str) -> pd.DataFrame:
    """Read Parquet and perform light normalization."""
    df = pd.read_parquet(file_path, engine="pyarrow")
    # Derive n_bits if missing
    if "n_bits" not in df.columns:
        if "a_w" in df.columns:
            df["n_bits"] = df["a_w"].astype("int64")
        elif "y_w" in df.columns:
            df["n_bits"] = (df["y_w"].astype("int64") // 2).clip(lower=1)
        else:
            df["n_bits"] = 0

    # Ensure expected columns exist even if missing in file
    for col in ["a_w", "b_w", "y_w"]:
        if col not in df.columns:
            df[col] = pd.Series([np.nan] * len(df), dtype="float64")

    # Build design label
    df = df.copy()
    df["design_label"] = df.apply(design_label_from_row, axis=1)

    # Mark known categoricals as string
    for c in ["multiplier_opt", "ppg_opt", "ppa_opt", "fsa_opt", "a_enc", "b_enc", "y_enc", "design_label"]:
        if c in df.columns:
            df[c] = df[c].astype("string")

    # <<< NEW: add derived columns >>>
    df = augment_df_columns(df)
    for c in extra_cols:
        if c in df.columns:
            df[c] = df[c].astype("string")

    return df


def load_many_ds(file_paths: List[str]) -> pd.DataFrame:
    """Load one or more Parquet files and concatenate them."""
    frames = [load_df(path) for path in file_paths or []]
    if not frames:
        return pd.DataFrame()
    if len(frames) == 1:
        return frames[0]
    return pd.concat(frames, ignore_index=True)


def target_sigma_for(n_bits: int) -> float:
    """σ_target = 3/16 * 2**n_bits."""
    return (3.0 / 16.0) * (2 ** int(n_bits))


def compute_switches_at_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each configuration (n_bits + encodings + stages + widths),
    pick the row at sigma closest to the target and return switches_at_target.
    """
    if df.empty:
        return df

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

    # Ensure all keys exist
    for k in key_cols:
        if k not in df.columns:
            df[k] = np.nan

    # Compute per-group nearest sigma
    parts = []
    for (nbits, *rest), g in df.groupby(key_cols, dropna=False):
        nbits_val = int(nbits) if pd.notna(nbits) else 0
        tgt = target_sigma_for(nbits_val)
        # nearest
        g = g.copy()
        g["sigma_dist"] = (g["sigma"].astype(float) - tgt).abs()
        idx = g["sigma_dist"].idxmin()
        best = g.loc[[idx], key_cols + ["switches"]].rename(columns={"switches": "switches_at_target"})
        parts.append(best)
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=key_cols + ["switches_at_target"])
    return out


def build_design_level_table(df: pd.DataFrame) -> pd.DataFrame:
    key_cols = [
        "n_bits","a_w","b_w","y_w",
        "multiplier_opt","ppg_opt","ppa_opt","fsa_opt",
        "a_enc","b_enc","y_enc","design_label",
    ]
    static_cols = [c for c in STATIC_METRICS if c in df.columns]
    # keep derived label columns if present
    extra_cols_f = [c for c in extra_cols if c in df.columns]

    to_keep = list(dict.fromkeys(key_cols + static_cols + extra_cols_f))
    base = df[to_keep].drop_duplicates(subset=key_cols).reset_index(drop=True)

    sat = compute_switches_at_target(df)
    design_df = pd.merge(base, sat, on=key_cols, how="left")

    # numeric cleanup (unchanged)
    for col in STATIC_METRICS + ["switches_at_target"]:
        if col in design_df.columns:
            design_df[col] = pd.to_numeric(design_df[col], errors="coerce")

    return design_df


def list_metrics_for_scatter(design_df: pd.DataFrame) -> List[str]:
    """Metrics available in the 'one point per config' scatter."""
    cols = []
    for c in STATIC_METRICS + ["switches_at_target"]:
        if c in design_df.columns:
            cols.append(c)
    # De-dup while preserving order
    return list(dict.fromkeys(cols))


def list_metrics_for_line(df: pd.DataFrame) -> List[str]:
    """Metrics that vary across sigma (line plot)."""
    cols = []
    if "switches" in df.columns:
        cols.append("switches")
    # add others if present (some teams like to inspect 'max_depth' stability vs sigma – usually constant)
    for c in STATIC_METRICS:
        if c in df.columns:
            cols.append(c)
    return list(dict.fromkeys(cols))


def apply_filters(
    df: pd.DataFrame,
    multiplier_opt: List[str] | None,
    ppg_opt: List[str] | None,
    ppa_opt: List[str] | None,
    fsa_opt: List[str] | None,
    a_enc: List[str] | None,
    b_enc: List[str] | None,
    y_enc: List[str] | None,
    design_labels: List[str] | None,
    n_bits_range: Tuple[float, float] | None,
    a_w_range: Tuple[float, float] | None,
    b_w_range: Tuple[float, float] | None,
    y_w_range: Tuple[float, float] | None,
) -> pd.DataFrame:
    out = df.copy()

    def filt(col, values):
        nonlocal out
        if values:
            out = out[out[col].astype("string").isin(values)]

    filt("multiplier_opt", multiplier_opt)
    filt("ppg_opt", ppg_opt)
    filt("ppa_opt", ppa_opt)
    filt("fsa_opt", fsa_opt)
    filt("a_enc", a_enc)
    filt("b_enc", b_enc)
    filt("y_enc", y_enc)
    filt("design_label", design_labels)

    def rng(col, r):
        nonlocal out
        if (r is not None) and (col in out.columns):
            out = out[(out[col] >= r[0]) & (out[col] <= r[1])]

    rng("n_bits", n_bits_range)
    rng("a_w", a_w_range)
    rng("b_w", b_w_range)
    rng("y_w", y_w_range)

    return out


# --------------------------- app + layout --------------------------- #


def make_app(df: pd.DataFrame) -> Dash:
    app = Dash(__name__, title="Multiplier Explorer", suppress_callback_exceptions=True)

    # Prepare design-level table (for scatter) once
    design_df_all = build_design_level_table(df)

    # Filter choices
    def opts(col: str, frame: pd.DataFrame) -> List[Dict[str, str]]:
        if col not in frame.columns:
            return []
        vals = frame[col].dropna().astype("string").unique()
        vals = sorted(vals, key=lambda x: (str(x).lower(), str(x)))
        return [{"label": v, "value": v} for v in vals]

    # Ranges
    def minmax(col: str, frame: pd.DataFrame, default=(0, 0)) -> Tuple[float, float]:
        if col not in frame.columns or frame[col].dropna().empty:
            return default
        return float(frame[col].min()), float(frame[col].max())

    nb_min, nb_max = minmax("n_bits", df, (0, 0))
    aw_min, aw_max = minmax("a_w", df, (0, 0))
    bw_min, bw_max = minmax("b_w", df, (0, 0))
    yw_min, yw_max = minmax("y_w", df, (0, 0))

    scatter_metrics = list_metrics_for_scatter(design_df_all)
    line_metrics = list_metrics_for_line(df)

    color_options = [
        "design_label",
        "n_bits",
        "multiplier_opt",
        "ppg_opt",
        "ppa_opt",
        "fsa_opt",
        "a_enc",
        "b_enc",
        "y_enc",
    ]
    color_options += extra_cols
    color_opts = [{"label": c, "value": c} for c in color_options]

    app.layout = html.Div(
        style={
            "fontFamily": "Inter, system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Noto Sans, Helvetica, Arial, sans-serif",
            "padding": "12px",
        },
        children=[
            html.H2("Multiplier Explorer"),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Color by"),
                            dcc.Dropdown(id="color-by", options=color_opts, value="n_bits", clearable=False),
                        ],
                        style={"minWidth": "220px", "marginRight": "12px"},
                    ),
                ],
                style={"display": "flex", "flexWrap": "wrap", "alignItems": "center"},
            ),
            html.Hr(),
            # Filters row 1
            html.Div(
                [
                    html.Div(
                        [html.Label("multiplier_opt"), dcc.Dropdown(id="f-mult", options=opts("multiplier_opt", df), multi=True)],
                        style={"flex": 1, "minWidth": "220px", "marginRight": "8px"},
                    ),
                    html.Div(
                        [html.Label("PPG"), dcc.Dropdown(id="f-ppg", options=opts("ppg_opt", df), multi=True)],
                        style={"flex": 1, "minWidth": "220px", "marginRight": "8px"},
                    ),
                    html.Div(
                        [html.Label("PPA"), dcc.Dropdown(id="f-ppa", options=opts("ppa_opt", df), multi=True)],
                        style={"flex": 1, "minWidth": "220px", "marginRight": "8px"},
                    ),
                    html.Div([html.Label("FSA"), dcc.Dropdown(id="f-fsa", options=opts("fsa_opt", df), multi=True)], style={"flex": 1, "minWidth": "220px"}),
                ],
                style={"display": "flex", "flexWrap": "wrap", "marginBottom": "8px"},
            ),
            # Filters row 2
            html.Div(
                [
                    html.Div(
                        [html.Label("a_enc"), dcc.Dropdown(id="f-aenc", options=opts("a_enc", df), multi=True)],
                        style={"flex": 1, "minWidth": "220px", "marginRight": "8px"},
                    ),
                    html.Div(
                        [html.Label("b_enc"), dcc.Dropdown(id="f-benc", options=opts("b_enc", df), multi=True)],
                        style={"flex": 1, "minWidth": "220px", "marginRight": "8px"},
                    ),
                    html.Div([html.Label("y_enc"), dcc.Dropdown(id="f-yenc", options=opts("y_enc", df), multi=True)], style={"flex": 1, "minWidth": "220px"}),
                ],
                style={"display": "flex", "flexWrap": "wrap", "marginBottom": "8px"},
            ),
            # Filter row 3: design label
            html.Div(
                [
                    html.Div(
                        [html.Label("Design label"), dcc.Dropdown(id="f-label", options=opts("design_label", df), multi=True)],
                        style={"flex": 1, "minWidth": "320px"},
                    ),
                ],
                style={"display": "flex", "flexWrap": "wrap", "marginBottom": "12px"},
            ),
            # Bit width ranges
            html.Div(
                [
                    html.Div(
                        [
                            html.Label(f"n_bits [{int(nb_min)}–{int(nb_max)}]"),
                            dcc.RangeSlider(
                                id="r-nbits",
                                min=int(nb_min or 0),
                                max=int(nb_max or 0),
                                step=1,
                                value=[int(nb_min or 0), int(nb_max or 0)],
                                tooltip={"always_visible": False},
                            ),
                        ],
                        style={"flex": 1, "minWidth": "280px", "marginRight": "12px"},
                    ),
                    html.Div(
                        [
                            html.Label(f"a_w [{int(aw_min)}–{int(aw_max)}]"),
                            dcc.RangeSlider(
                                id="r-aw",
                                min=int(aw_min or 0),
                                max=int(aw_max or 0),
                                step=1,
                                value=[int(aw_min or 0), int(aw_max or 0)],
                                tooltip={"always_visible": False},
                            ),
                        ],
                        style={"flex": 1, "minWidth": "280px", "marginRight": "12px"},
                    ),
                    html.Div(
                        [
                            html.Label(f"b_w [{int(bw_min)}–{int(bw_max)}]"),
                            dcc.RangeSlider(
                                id="r-bw",
                                min=int(bw_min or 0),
                                max=int(bw_max or 0),
                                step=1,
                                value=[int(bw_min or 0), int(bw_max or 0)],
                                tooltip={"always_visible": False},
                            ),
                        ],
                        style={"flex": 1, "minWidth": "280px", "marginRight": "12px"},
                    ),
                    html.Div(
                        [
                            html.Label(f"y_w [{int(yw_min)}–{int(yw_max)}]"),
                            dcc.RangeSlider(
                                id="r-yw",
                                min=int(yw_min or 0),
                                max=int(yw_max or 0),
                                step=1,
                                value=[int(yw_min or 0), int(yw_max or 0)],
                                tooltip={"always_visible": False},
                            ),
                        ],
                        style={"flex": 1, "minWidth": "280px"},
                    ),
                ],
                style={"display": "flex", "flexWrap": "wrap", "marginBottom": "16px"},
            ),
            dcc.Tabs(
                id="tab",
                value="scatter",
                children=[
                    dcc.Tab(
                        label="Scatter: metric vs metric (one point per configuration)",
                        value="scatter",
                        children=[
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Label("X metric"),
                                            dcc.Dropdown(
                                                id="x-metric",
                                                options=[{"label": c, "value": c} for c in scatter_metrics],
                                                value=(scatter_metrics[0] if scatter_metrics else None),
                                                clearable=False,
                                            ),
                                        ],
                                        style={"minWidth": "240px", "marginRight": "8px"},
                                    ),
                                    html.Div(
                                        [
                                            html.Label("Y metric"),
                                            dcc.Dropdown(
                                                id="y-metric",
                                                options=[{"label": c, "value": c} for c in scatter_metrics],
                                                value=(scatter_metrics[1] if len(scatter_metrics) > 1 else (scatter_metrics[0] if scatter_metrics else None)),
                                                clearable=False,
                                            ),
                                        ],
                                        style={"minWidth": "240px", "marginRight": "8px"},
                                    ),
                                ],
                                style={"display": "flex", "flexWrap": "wrap", "marginBottom": "8px"},
                            ),
                            dcc.Graph(id="scatter-graph", style={"height": "72vh"}),
                        ],
                    ),
                    dcc.Tab(
                        label="Line: metric vs sigma (lines per configuration)",
                        value="line",
                        children=[
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Label("Y metric"),
                                            dcc.Dropdown(
                                                id="line-metric",
                                                options=[{"label": c, "value": c} for c in line_metrics],
                                                value=("switches" if "switches" in line_metrics else (line_metrics[0] if line_metrics else None)),
                                                clearable=False,
                                            ),
                                        ],
                                        style={"minWidth": "240px", "marginRight": "8px"},
                                    ),
                                ],
                                style={"display": "flex", "flexWrap": "wrap", "marginBottom": "8px"},
                            ),
                            dcc.Graph(id="line-graph", style={"height": "72vh"}),
                        ],
                    ),
                ],
            ),
            dcc.Store(id="design-table", data=design_df_all.to_json(orient="records")),
            dcc.Store(id="full-table", data=df.to_json(orient="records")),
        ],
    )

    # ----------------------- callbacks: scatter ----------------------- #

    @callback(
        Output("scatter-graph", "figure"),
        [
            Input("design-table", "data"),
            Input("color-by", "value"),
            Input("x-metric", "value"),
            Input("y-metric", "value"),
            Input("f-mult", "value"),
            Input("f-ppg", "value"),
            Input("f-ppa", "value"),
            Input("f-fsa", "value"),
            Input("f-aenc", "value"),
            Input("f-benc", "value"),
            Input("f-yenc", "value"),
            Input("f-label", "value"),
            Input("r-nbits", "value"),
            Input("r-aw", "value"),
            Input("r-bw", "value"),
            Input("r-yw", "value"),
        ],
    )
    def update_scatter(design_table_json, color_by, x_metric, y_metric, mult, ppg, ppa, fsa, aenc, benc, yenc, labels, nbits_r, aw_r, bw_r, yw_r):
        if not design_table_json or not x_metric or not y_metric:
            return go.Figure()
        design_df = pd.DataFrame.from_records(pd.read_json(design_table_json, orient="records"))
        # Apply same filter logic on design_df
        design_df = apply_filters(design_df, mult, ppg, ppa, fsa, aenc, benc, yenc, labels, nbits_r, aw_r, bw_r, yw_r)

        if design_df.empty or x_metric not in design_df.columns or y_metric not in design_df.columns:
            return go.Figure()

        hover_cols = [c for c in HOVER_BASE + [x_metric, y_metric, "switches_at_target"] if c in design_df.columns]
        fig = px.scatter(
            design_df,
            x=x_metric,
            y=y_metric,
            color=color_by if color_by in design_df.columns else None,
            hover_data=hover_cols,
            template="plotly_white",
        )
        fig.update_traces(marker=dict(size=10, opacity=0.85), selector=dict(mode="markers"))
        fig.update_layout(
            legend=dict(font=dict(size=10)),
            margin=dict(l=40, r=10, t=50, b=40),
            title=f"{y_metric} vs {x_metric} (one point per configuration)",
        )
        return fig

    # ------------------------- callbacks: line ------------------------ #

    # @callback(
    #     Output("line-graph", "figure"),
    #     [
    #         Input("full-table", "data"),
    #         Input("color-by", "value"),
    #         Input("line-metric", "value"),
    #         Input("f-mult", "value"),
    #         Input("f-ppg", "value"),
    #         Input("f-ppa", "value"),
    #         Input("f-fsa", "value"),
    #         Input("f-aenc", "value"),
    #         Input("f-benc", "value"),
    #         Input("f-yenc", "value"),
    #         Input("f-label", "value"),
    #         Input("r-nbits", "value"),
    #         Input("r-aw", "value"),
    #         Input("r-bw", "value"),
    #         Input("r-yw", "value"),
    #     ],
    # )
    # def update_line(full_table_json, color_by, y_metric, mult, ppg, ppa, fsa, aenc, benc, yenc, labels, nbits_r, aw_r, bw_r, yw_r):
    #     if not full_table_json or not y_metric:
    #         return go.Figure()

    #     full_df = pd.DataFrame.from_records(pd.read_json(full_table_json, orient="records"))
    #     full_df = apply_filters(full_df, mult, ppg, ppa, fsa, aenc, benc, yenc, labels, nbits_r, aw_r, bw_r, yw_r)

    #     if "sigma" not in full_df.columns or y_metric not in full_df.columns or full_df.empty:
    #         return go.Figure()

    #     # Build label to color lines consistently
    #     if "design_label" not in full_df.columns:
    #         full_df["design_label"] = full_df.apply(design_label_from_row, axis=1)

    #     hover_cols = [c for c in HOVER_BASE + ["sigma", y_metric] if c in full_df.columns]
    #     fig = px.line(
    #         full_df.sort_values(["design_label", "sigma"]),
    #         x="sigma",
    #         y=y_metric,
    #         color=color_by if color_by in full_df.columns else "design_label",
    #         line_group="design_label",
    #         hover_data=hover_cols,
    #         template="plotly_white",
    #     )
    #     fig.update_traces(mode="lines+markers", marker=dict(size=6, opacity=0.8))
    #     fig.update_layout(
    #         legend=dict(font=dict(size=10)),
    #         margin=dict(l=40, r=10, t=50, b=40),
    #         title=f"{y_metric} vs sigma (lines per configuration)",
    #     )
    #     return fig

    @callback(
        Output("line-graph", "figure"),
        [
            Input("full-table", "data"),
            Input("color-by", "value"),
            Input("line-metric", "value"),
            Input("f-mult", "value"), Input("f-ppg", "value"), Input("f-ppa", "value"), Input("f-fsa", "value"),
            Input("f-aenc", "value"), Input("f-benc", "value"), Input("f-yenc", "value"), Input("f-label", "value"),
            Input("r-nbits", "value"), Input("r-aw", "value"), Input("r-bw", "value"), Input("r-yw", "value"),
        ],
    )
    def update_line(full_table_json, color_by, y_metric,
                    mult, ppg, ppa, fsa, aenc, benc, yenc, labels,
                    nbits_r, aw_r, bw_r, yw_r):
        if not full_table_json or not y_metric:
            return go.Figure()

        full_df = pd.DataFrame.from_records(pd.read_json(full_table_json, orient="records"))

        # Apply filters
        full_df = apply_filters(full_df, mult, ppg, ppa, fsa, aenc, benc, yenc, labels,
                                nbits_r, aw_r, bw_r, yw_r)

        # Guard rails
        if full_df.empty or "sigma" not in full_df.columns or y_metric not in full_df.columns:
            return go.Figure()
        if "run_id" not in full_df.columns:
            # Fallback: synthesize a line id if run_id is missing (shouldn't happen with your data)
            full_df["run_id"] = (
                full_df["multiplier_opt"].astype(str) + "|" +
                full_df["ppg_opt"].astype(str) + "|" +
                full_df["ppa_opt"].astype(str) + "|" +
                full_df["fsa_opt"].astype(str) + "|" +
                full_df["a_enc"].astype(str) + "/" +
                full_df["b_enc"].astype(str) + "→" +
                full_df["y_enc"].astype(str)
            )

        # Make sure we have design labels (for coloring/hover if chosen)
        if "design_label" not in full_df.columns:
            full_df["design_label"] = full_df.apply(design_label_from_row, axis=1)

        # Keep one point per (run_id, sigma); order by (run_id, sigma) so lines are drawn correctly
        lines_df = (full_df
                    .dropna(subset=["run_id", "sigma", y_metric])
                    .sort_values(["run_id", "sigma"])
                    .drop_duplicates(subset=["run_id", "sigma"], keep="first"))

        if lines_df.empty:
            return go.Figure()

        hover_cols = [c for c in HOVER_BASE + ["run_id", "sigma", y_metric] if c in lines_df.columns]

        fig = px.line(
            lines_df,
            x="sigma",
            y=y_metric,
            color=color_by if color_by in lines_df.columns else "design_label",
            line_group="run_id",            # <-- connects only within the same run
            hover_data=hover_cols,
            template="plotly_white",
        )
        fig.update_traces(mode="lines+markers", marker=dict(size=6, opacity=0.8))
        fig.update_layout(
            legend=dict(font=dict(size=10)),
            margin=dict(l=40, r=10, t=50, b=40),
            title=f"{y_metric} vs sigma (each line = one run_id)",
        )
        return fig

    return app


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
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8050)
    ap.add_argument("--debug", action="store_true", help="Run app in debug mode (auto-reload on code changes).")
    args = ap.parse_args()

    df = load_many_ds(args.files)
    app = make_app(df)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
