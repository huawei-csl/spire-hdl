from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any, List
import json
import os
import time
import uuid

import pandas as pd


def _enum_name(x) -> str:
    try:
        return x.name
    except AttributeError:
        return str(x)


def _flatten_op_nodes(op_nodes: Dict[str, int]) -> str:
    return json.dumps(op_nodes or {}, sort_keys=True)


@dataclass
class MultiplierRow:
    # IDs / meta
    run_id: str
    timestamp: float
    module_name: str

    # Config
    n_bits: int
    multiplier_opt: str
    ppg_opt: str
    ppa_opt: str
    fsa_opt: str
    a_enc: str
    b_enc: str
    y_enc: str
    a_w: int
    b_w: int
    y_w: int
    num_vectors: int
    
    optimization_effort: int

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

# --- add these helpers ---
from dataclasses import fields, is_dataclass
from typing import get_origin

def _base_type(t):
    # unwrap Optional[T] / Union[T, None]
    origin = get_origin(t)
    if origin is None:
        return t
    args = getattr(t, "__args__", ())
    return args[0] if args else t

def _apply_dtypes_from_dataclass(df, dc_cls):
    """Cast columns using dc_cls annotations (int->int64, float->float64, str->string). Soft-fail if casting fails."""
    if not is_dataclass(dc_cls):
        return df
    for f in fields(dc_cls):
        col = f.name
        if col not in df.columns:
            continue
        bt = _base_type(f.type)
        try:
            if bt is int:
                df[col] = pd.to_numeric(df[col], errors="raise").astype("int64")
            elif bt is float:
                df[col] = pd.to_numeric(df[col], errors="raise").astype("float64")
            elif bt is str:
                df[col] = df[col].astype("string")
            # else: leave as-is (e.g., JSON strings, enums already stringified)
        except Exception:
            # If any bad value sneaks in, just keep pandas’ inferred dtype.
            pass
    return df

class ParquetCollector:
    def __init__(self, out_file: str, engine: str = "pyarrow", row_cls=MultiplierRow):
        self.out_file = out_file
        self.engine = engine
        self.row_cls = row_cls
        self._rows = []
        os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)

    def add(self, row):
        self._rows.append(asdict(row) if is_dataclass(row) else dict(row))
    
    def extend(self, rows: List[Any]):
        for row in rows:
            self.add(row)

    def _to_df(self) -> pd.DataFrame:
        if not self._rows:
            return pd.DataFrame()
        df = pd.DataFrame(self._rows)
        # Either keep this line for a light, annotation-driven cast...
        df = _apply_dtypes_from_dataclass(df, self.row_cls)
        # ...or remove it entirely and rely on pandas inference.
        return df

    def save(self, append: bool = True) -> None:
        df = self._to_df()
        if df.empty:
            raise Warning("No rows to save in ParquetCollector.save()")
            return
        if append and os.path.exists(self.out_file):
            existing = pd.read_parquet(self.out_file, engine=self.engine)
            pd.concat([existing, df], ignore_index=True).to_parquet(self.out_file, engine=self.engine, index=False)
        else:
            df.to_parquet(self.out_file, engine=self.engine, index=False)
        self._rows.clear()
        
    def n_rows(self) -> int:
        return len(self._rows)

