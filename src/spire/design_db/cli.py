"""``spire db`` — the human/agent window into the design DB.

Subcommands (S2): ``init | ls | show <name-or-key> [--pareto] | insert <design.v> --slot <key>``.
``show`` prints JSON so agents and humans share the same interface. ``verify`` arrives with the
sim tiers (S3).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

from spire.design_db.store import DB_ENV, DesignDB, DesignDBError, resolve_db_root
from spire.design_db.verify import VerificationError


def _open(db_arg: Optional[str], *, create: bool = False) -> DesignDB:
    """Read-only commands must never create a DB as a side effect (create=False default)."""
    return DesignDB.open(db_arg, create=create)


def _resolve_slot(d: DesignDB, token: str) -> str:
    """A slot reference: a manifest name, a full spec_key, or a unique key prefix (≥ 8 chars)."""
    manifest = d.read_json(d.manifest_path, {"slots": {}})
    entry = manifest.get("slots", {}).get(token)
    if entry:
        return entry["spec_key"]
    if (d.v1 / token).is_dir():
        return token
    if len(token) >= 8 and d.v1.is_dir():
        hits = [p.name for p in d.v1.iterdir() if p.is_dir() and p.name.startswith(token)]
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            raise DesignDBError(f"ambiguous slot prefix {token!r}: {len(hits)} matches")
    raise DesignDBError(f"unknown slot {token!r} (not a manifest name, key, or unique key prefix)")


def _cmd_init(args: argparse.Namespace) -> int:
    root = resolve_db_root(args.db, create=True)
    print(root)
    return 0


def _cmd_ls(args: argparse.Namespace) -> int:
    d = _open(args.db)
    manifest = d.read_json(d.manifest_path, {"slots": {}})
    if args.json:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0
    rows = manifest.get("slots", {})
    if not rows:
        if d.root.exists():
            print(f"(empty design DB at {d.root})")
        else:
            print(f"(no design DB found — `spire db init` would create {d.root})")
        return 0
    for name, e in sorted(rows.items()):
        sel = e.get("selected_id", "-")
        print(f"{name:32s} {e.get('class', '?'):13s} designs={e.get('n_designs', 0):<3d} "
              f"key={e['spec_key'][:12]}…  selected={sel}")
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    d = _open(args.db)
    key = _resolve_slot(d, args.slot)
    slot = d.slot_dir(key)
    out = {
        "spec_key": key,
        "spec": d.read_json(slot / "spec.json"),
        "verification": d.read_json(slot / "verification.json"),
        "designs": d.read_json(slot / "index.json", {}),
    }
    if args.pareto:
        from spire.design_db.select import pareto_front
        out["pareto"] = pareto_front(key, db=args.db)
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


def _cmd_insert(args: argparse.Namespace) -> int:
    import contextlib
    import io

    from spire.design_db.insert import insert_design
    d = _open(args.db, create=True)
    key = _resolve_slot(d, args.slot)
    try:
        with contextlib.redirect_stdout(io.StringIO()):     # keep stdout = our JSON only
            res = insert_design(key, Path(args.design), source=args.source, db=args.db,
                                budget_s=args.budget)
    except VerificationError as exc:
        print(f"REJECTED ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"design_id": res.design_id, "deduped": res.deduped,
                      "metrics": res.metrics}, indent=2, sort_keys=True))
    return 0


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(prog="spire", description="Spire command-line tools")
    top = parser.add_subparsers(dest="ns", required=True)
    dbp = top.add_parser("db", help=f"design DB (root: --db / ${DB_ENV} / nearest design_db/)")
    sub = dbp.add_subparsers(dest="cmd", required=True)

    def _common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--db", default=None, help="DB root (default: resolve/auto-create)")

    p = sub.add_parser("init", help="create (or print) the DB root")
    _common(p); p.set_defaults(func=_cmd_init)

    p = sub.add_parser("ls", help="list registered slots")
    _common(p); p.add_argument("--json", action="store_true"); p.set_defaults(func=_cmd_ls)

    p = sub.add_parser("show", help="dump one slot as JSON")
    _common(p); p.add_argument("slot", help="manifest name, spec_key, or unique key prefix")
    p.add_argument("--pareto", action="store_true", help="include the area/delay Pareto front")
    p.set_defaults(func=_cmd_show)

    p = sub.add_parser("insert", help="insert a Verilog design through the verification gate")
    _common(p); p.add_argument("design", help="path to the candidate .v/.sv file")
    p.add_argument("--slot", required=True, help="manifest name, spec_key, or unique key prefix")
    p.add_argument("--source", default="cli", help="provenance source tag (default: cli)")
    p.add_argument("--budget", type=float, default=None, help="CEC budget in seconds")
    p.set_defaults(func=_cmd_insert)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except DesignDBError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
