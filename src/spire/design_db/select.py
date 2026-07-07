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


# Built-in systems the insert gate stamps, in default-resolution preference (after technologies).
BUILTIN_SYSTEMS = ("transistors", "aig")


def _is_system_block(v: Any) -> bool:
    """A measurement system is a self-describing block: raw ``metrics`` + an ``objectives`` map."""
    return isinstance(v, dict) and isinstance(v.get("metrics"), dict)


def _system_names(index: Dict[str, Any]) -> set:
    """Every measurement system present across the slot's designs."""
    names: set = set()
    for entry in index.values():
        for k, v in (entry.get("metrics") or {}).items():
            if _is_system_block(v):
                names.add(k)
    return names


def resolve_metric(index: Dict[str, Any], metric: Optional[str]) -> str:
    """Deterministic system resolution: explicit (validated) or technology → transistors → aig."""
    names = _system_names(index)
    if metric is not None:
        if metric in names:
            return metric
        raise DesignDBError(f"metric {metric!r} not available for this slot — "
                            f"available: {sorted(names)}")
    techs = sorted(n for n in names if n not in BUILTIN_SYSTEMS)
    if techs:
        return techs[0]
    for n in BUILTIN_SYSTEMS:
        if n in names:
            return n
    raise DesignDBError("slot has no measurement systems (no admitted designs?)")


def _lookup(metrics: Dict[str, Any], system: str, path: str) -> Optional[float]:
    """Resolve an objectives path: ``field`` (this system's metrics) or ``other.field`` (a sibling
    system's metrics — how the transistor system borrows the AIG depth for its delay axis)."""
    sys_name, field = path.split(".", 1) if "." in path else (system, path)
    block = metrics.get(sys_name)
    if not isinstance(block, dict):
        return None
    return (block.get("metrics") or {}).get(field)


def metric_value(metrics: Dict[str, Any], objective: str, system: str) -> Optional[float]:
    """The value of one objective under one measurement system, or None if not measurable.

    Fully data-driven: each system block carries an ``objectives`` map (objective → a ``field`` in
    its own metrics, or a ``sibling.field`` path). ``adp`` is derived (area·delay) when a system
    does not map it explicitly; ``edap`` and any other objective are only available if mapped.
    """
    block = metrics.get(system)
    if not isinstance(block, dict):
        return None
    mapping = block.get("objectives") or {}
    if objective in mapping:
        return _lookup(metrics, system, mapping[objective])
    if objective == "adp":                                   # derived by default
        a = metric_value(metrics, "area", system)
        delay = metric_value(metrics, "delay", system)
        return a * delay if a is not None and delay is not None else None
    return None


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
    index = d.read_index(spec_key)

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
    index = d.read_index(spec_key)
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
