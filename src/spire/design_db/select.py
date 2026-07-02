"""Selection over a slot's admitted designs — pure, deterministic functions on ``index.json``.

Selection never generates anything: it reads the metric vectors the insert gate stamped and picks a
``design_id``. The measurement system (``metric``) resolves deterministically when not given:
technology PPA (once ``db score`` ran) → ``transistors`` → ``aig``. Objectives can be a plain metric
name (``"area" | "delay" | "adp" | "edap"``, argmin) or one of the combinators below.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from spire.design_db.store import DesignDB, DesignDBError

OBJECTIVES = ("area", "delay", "adp", "edap")

ObjectiveSpec = Union[str, "constrained", "weighted", "lexicographic"]


@dataclass(frozen=True)
class constrained:
    """Minimize one objective subject to upper bounds on others, e.g.
    ``constrained(minimize="area", subject_to={"delay": 500})``."""
    minimize: str
    subject_to: Mapping[str, float]


@dataclass(frozen=True)
class weighted:
    """Minimize a weighted sum, e.g. ``weighted({"area": 1.0, "delay": 0.1})``."""
    weights: Mapping[str, float]


@dataclass(frozen=True)
class lexicographic:
    """Minimize objectives in priority order, e.g. ``lexicographic(("area", "delay"))``."""
    objectives: Sequence[str]


@dataclass
class SelectionResult:
    design_id: str
    entry: Dict[str, Any]
    objective: str          # repr of the objective spec
    metric: str             # resolved measurement system


_ALIASES = {"aig": "aig", "intrinsic": "aig", "transistors": "transistors",
            "transistors_heavy": "transistors"}


def _systems(index: Dict[str, Any]) -> Tuple[List[str], bool]:
    """(technology systems present, any transistor stamp present)."""
    techs, has_t = set(), False
    for entry in index.values():
        metrics = entry.get("metrics", {})
        for k, v in metrics.items():
            if isinstance(v, dict) and k != "intrinsic":
                techs.add(k)
        if metrics.get("transistors_heavy") is not None:
            has_t = True
    return sorted(techs), has_t


def resolve_metric(index: Dict[str, Any], metric: Optional[str]) -> str:
    """Deterministic system resolution: explicit (validated) or technology → transistors → aig."""
    techs, has_t = _systems(index)
    if metric is not None:
        if metric in _ALIASES:
            return _ALIASES[metric]
        if metric in techs:
            return metric
        raise DesignDBError(f"metric {metric!r} not available for this slot — "
                            f"available: {techs + (['transistors'] if has_t else []) + ['aig']}")
    if techs:
        return techs[0]
    return "transistors" if has_t else "aig"


def metric_value(metrics: Dict[str, Any], objective: str, system: str) -> Optional[float]:
    """The value of one objective under one measurement system, or None if not measurable."""
    intrinsic = metrics.get("intrinsic") or {}
    if system == "aig":
        nodes, depth = intrinsic.get("aig_nodes"), intrinsic.get("aig_depth")
        return {"area": nodes, "delay": depth,
                "adp": nodes * depth if nodes is not None and depth is not None else None,
                "edap": None}.get(objective)
    if system == "transistors":
        t, depth = metrics.get("transistors_heavy"), intrinsic.get("aig_depth")
        return {"area": t, "delay": depth,
                "adp": t * depth if t is not None and depth is not None else None,
                "edap": None}.get(objective)
    tech = metrics.get(system) or {}
    return tech.get(objective)


def _score(metrics: Dict[str, Any], objective: ObjectiveSpec, system: str) -> Optional[Tuple]:
    """A sortable score tuple (lower is better), or None if the candidate is ineligible."""
    if isinstance(objective, str):
        v = metric_value(metrics, objective, system)
        return None if v is None else (v,)
    if isinstance(objective, lexicographic):
        vals = tuple(metric_value(metrics, o, system) for o in objective.objectives)
        return None if any(v is None for v in vals) else vals
    if isinstance(objective, weighted):
        total = 0.0
        for o, w in objective.weights.items():
            v = metric_value(metrics, o, system)
            if v is None:
                return None
            total += w * v
        return (total,)
    if isinstance(objective, constrained):
        for o, bound in objective.subject_to.items():
            v = metric_value(metrics, o, system)
            if v is None or v > bound:
                return None
        v = metric_value(metrics, objective.minimize, system)
        return None if v is None else (v,)
    raise DesignDBError(f"unsupported objective spec: {objective!r}")


def _obj_repr(objective: ObjectiveSpec) -> str:
    return objective if isinstance(objective, str) else repr(objective)


def select_design(spec_key: str, *, objective: ObjectiveSpec = "area",
                  metric: Optional[str] = None, pin: Optional[str] = None,
                  sources: Optional[Sequence[str]] = None,
                  db: Optional[Any] = None, record: bool = False) -> Optional[SelectionResult]:
    """Pick one admitted design of a slot. Returns None when no eligible design exists.

    ``pin`` bypasses metrics and demands an exact ``design_id`` — a missing pin is a broken
    reproducibility lock and raises. With ``record=True`` the resolved
    ``(selected_id, objective, metric)`` is written into the manifest.
    """
    d = DesignDB.open(db)
    index = d.read_json(d.slot_dir(spec_key) / "index.json", {})

    if pin is not None:
        if pin not in index:
            raise DesignDBError(f"pinned design {pin!r} not found in slot {spec_key[:12]}… "
                                f"(broken reproducibility lock)")
        result = SelectionResult(pin, index[pin], f"pin:{pin}", metric or "pinned")
    else:
        if sources:
            index = {k: v for k, v in index.items()
                     if any(v.get("source", "").startswith(s) for s in sources)}
        if not index:
            return None
        system = resolve_metric(index, metric)
        eligible = []
        for design_id, entry in index.items():
            score = _score(entry.get("metrics", {}), objective, system)
            if score is not None:
                eligible.append((score, design_id, entry))
        if not eligible:
            return None
        eligible.sort(key=lambda t: (t[0], t[1]))       # value, then id — fully deterministic
        _, design_id, entry = eligible[0]
        result = SelectionResult(design_id, entry, _obj_repr(objective), system)

    if record:
        d.update_manifest_selection(spec_key, {"selected_id": result.design_id,
                                               "objective": result.objective,
                                               "metric": result.metric})
    return result


def pareto_front(spec_key: str, objectives: Sequence[str] = ("area", "delay"), *,
                 metric: Optional[str] = None, db: Optional[Any] = None) -> List[Dict[str, Any]]:
    """The non-dominated set (minimizing all ``objectives``), sorted by the first objective."""
    d = DesignDB.open(db)
    index = d.read_json(d.slot_dir(spec_key) / "index.json", {})
    if not index:
        return []
    system = resolve_metric(index, metric)
    pts = []
    for design_id, entry in index.items():
        vals = [metric_value(entry.get("metrics", {}), o, system) for o in objectives]
        if all(v is not None for v in vals):
            pts.append((vals, design_id, entry))
    front = []
    for vals, design_id, entry in pts:
        dominated = any(
            all(ov <= v for ov, v in zip(ovals, vals)) and any(ov < v for ov, v in zip(ovals, vals))
            for ovals, oid, _ in pts if oid != design_id
        )
        if not dominated:
            front.append({"design_id": design_id, "source": entry.get("source"), "metric": system,
                          **dict(zip(objectives, vals))})
    front.sort(key=lambda r: (r[objectives[0]], r["design_id"]))
    return front
