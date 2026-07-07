"""``annotate`` — attach a measurement block to a stored design.

Spire's insert gate stamps only what it can compute with yosys (the ``aig`` structural system +
the ``transistors`` estimate). Richer per-technology PPA (e.g. an ASAP7 place-and-route flow) is
produced *outside* spire — by RTLScout's cost pipeline, or by hand — and handed back through this
one write path so spire keeps ownership of the on-disk schema (``metrics.json`` is authoritative;
the ``index.json`` cache re-derives from it).

A ``tech`` names a new *measurement system* on the design, written in the same self-describing
form the gate uses — ``{"metrics": {...}, "objectives": {area→…, delay→…}}`` — so that after
annotating ``asap7`` the design is selectable with ``select_design(..., metric="asap7")`` /
``@from_design_db(metric="asap7")``. The ``objectives`` map is the identity over the standard axes
present (``area``/``delay``/``adp``/``edap``); ``adp`` is derived by selection when unmapped.
Reserved system names (the gate-stamped ones) are refused; re-annotating an existing system needs
``force`` (a measurement is a commitment, like a freeze); and the map must agree with any sibling
design's map for the same system, so one slot never mixes interpretations.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from spire.design_db.store import DesignDB, DesignDBError

# Systems written by the insert gate (and legacy spellings) — never overwritten via annotate.
RESERVED_SYSTEMS = {"intrinsic", "aig", "transistors", "transistors_heavy"}
OBJECTIVE_AXES = ("area", "delay", "adp", "edap")


def _resolve_design(d: DesignDB, spec_key: str, design_ref: str) -> str:
    """A design reference: an exact design_id or a unique prefix of one, within the slot."""
    index = d.derive_index(spec_key)
    if not index:
        raise DesignDBError(f"slot {spec_key[:12]}… has no admitted designs to annotate")
    if design_ref in index:
        return design_ref
    hits = [did for did in index if did.startswith(design_ref)]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        raise DesignDBError(f"ambiguous design ref {design_ref!r}: {len(hits)} matches")
    raise DesignDBError(f"unknown design {design_ref!r} in slot {spec_key[:12]}… — "
                        f"have: {sorted(index)}")


def _assert_consistent_with_siblings(d: DesignDB, spec_key: str, tech: str,
                                     objectives: Dict[str, str], exclude_id: str) -> None:
    """No other design in the slot may map the same objective of this system to a different field —
    keeps every design comparable under one interpretation (the guarantee a shared registry would
    give for free)."""
    index = d.derive_index(spec_key)
    for did, entry in index.items():
        if did == exclude_id:
            continue
        block = (entry.get("metrics") or {}).get(tech)
        sib = block.get("objectives") if isinstance(block, dict) else None
        if not isinstance(sib, dict):
            continue
        for obj, path in objectives.items():
            if obj in sib and sib[obj] != path:
                raise DesignDBError(
                    f"objectives map for {tech!r} disagrees with design {did} on {obj!r} "
                    f"({sib[obj]!r} vs {path!r}) — all designs of a slot must share one "
                    f"interpretation")


def annotate(spec_key: str, design_ref: str, *, tech: str, values: Dict[str, float],
             raw: Optional[Any] = None, force: bool = False,
             db: Optional[Any] = None) -> Dict[str, Any]:
    """Attach a self-describing measurement block ``metrics[tech]`` to one stored design.

    ``values`` are numeric readings; the block's ``objectives`` map is the identity over the
    standard axes (``area``/``delay``/``adp``/``edap``) present in ``values`` — other numeric keys
    are stored but not selectable. Written to the design's ``metrics.json`` (authoritative);
    the slot's ``index.json`` cache refreshes from it. Returns the updated metrics dict.
    """
    if tech in RESERVED_SYSTEMS:
        raise DesignDBError(f"{tech!r} is a reserved (gate-stamped) measurement system — "
                            f"annotate a technology name such as 'asap7', not {tech!r}")
    if not values:
        raise DesignDBError("no metric values given (e.g. area=… delay=…)")
    for k, v in values.items():
        if not isinstance(k, str) or not k.isidentifier():
            raise DesignDBError(f"metric name {k!r} must be an identifier")
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise DesignDBError(f"metric {k!r} must be numeric, got {v!r}")

    d = DesignDB.open(db, create=False)
    design_id = _resolve_design(d, spec_key, design_ref)
    slot = d.slot_dir(spec_key)
    mfile = slot / "designs" / design_id / "metrics.json"
    metrics = d.read_json(mfile, None)
    if metrics is None:
        raise DesignDBError(f"design {design_id} has no metrics.json")
    if tech in metrics and not force:
        raise DesignDBError(f"design {design_id} already has a {tech!r} measurement — pass "
                            f"force=True (--force) to overwrite it")

    objectives = {axis: axis for axis in OBJECTIVE_AXES if axis in values}
    _assert_consistent_with_siblings(d, spec_key, tech, objectives, design_id)
    block: Dict[str, Any] = {"metrics": {k: v for k, v in values.items()}, "objectives": objectives}
    if raw is not None:
        block["raw"] = raw
    metrics[tech] = block
    d.write_json(mfile, metrics)                   # metrics.json is the source of truth …
    d.read_index(spec_key)                         # … the index.json cache refreshes from it
    return {"spec_key": spec_key, "design_id": design_id, "tech": tech, "metrics": metrics}
