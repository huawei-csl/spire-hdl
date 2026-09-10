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

from spire.design_db.store import DB_ENV, DesignDB, DesignDBError, fmt_ts_local, resolve_db_root
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
    rows = manifest.get("slots", {})
    indexes = {e["spec_key"]: d.derive_index(e["spec_key"])         # derived, never stale
               for e in rows.values() if e.get("spec_key")}
    counts = {k: len(idx) for k, idx in indexes.items()}
    last = {k: max((e["created"] for e in idx.values()              # newest admit (epoch; legacy
                    if isinstance(e.get("created"), (int, float))), default=None)   # stamps skipped)
            for k, idx in indexes.items()}
    if args.json:
        out = {**manifest, "slots": {n: {**e, "n_designs": counts.get(e.get("spec_key"), 0),
                                         "last_created": last.get(e.get("spec_key"))}
                                     for n, e in rows.items()}}
        print(json.dumps(out, indent=2, sort_keys=True))
        return 0
    if not rows:
        if d.root.exists():
            print(f"(empty design DB at {d.root})")
        else:
            print(f"(no design DB found — `spire db init` would create {d.root})")
        return 0
    for name, e in sorted(rows.items()):
        print(f"{name:32s} {e.get('class', '?'):13s} designs={counts.get(e.get('spec_key'), 0):<3d} "
              f"last={fmt_ts_local(last.get(e.get('spec_key'))) or '-':19s} key={e['spec_key'][:12]}…")
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    d = _open(args.db)
    key = _resolve_slot(d, args.slot)
    slot = d.slot_dir(key)
    designs = d.read_index(key)                     # derived from designs/ (cache refreshed)
    for entry in designs.values():                  # local-time renderings: display-only, never stored
        for f in ("created", "last_rediscovered"):
            if isinstance(entry.get(f), (int, float)):
                entry[f + "_local"] = fmt_ts_local(entry[f])
    out = {
        "spec_key": key,
        "spec": d.read_json(slot / "spec.json"),
        "verification": d.read_json(slot / "verification.json"),
        "designs": designs,
    }
    if args.pareto:
        from spire.design_db.select import pareto_front
        out["pareto"] = pareto_front(key, db=args.db)
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


def _gated_insert(run) -> int:
    """Run an insert-like callable with clean stdout; print the result JSON or REJECTED."""
    import contextlib
    import io
    try:
        with contextlib.redirect_stdout(io.StringIO()):     # keep stdout = our JSON only
            res = run()
    except VerificationError as exc:
        print(f"REJECTED ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"design_id": res.design_id, "deduped": res.deduped,
                      "metrics": res.metrics}, indent=2, sort_keys=True))
    return 0


def _cmd_insert(args: argparse.Namespace) -> int:
    from spire.design_db.insert import insert_design
    d = _open(args.db, create=True)
    key = _resolve_slot(d, args.slot)
    return _gated_insert(lambda: insert_design(key, Path(args.design), source=args.source, db=args.db,
                                               budget_s=args.budget, proof=_opt_path(args.proof)))


def _opt_path(value: Optional[str]) -> Optional[Path]:
    return Path(value) if value else None


def _cmd_add_spec(args: argparse.Namespace) -> int:
    from spire.design_db.verify import VerificationError
    from spire.design_db.verify_lean import DEFAULT_LAKE_BUDGET_S, add_spec
    d = _open(args.db, create=False)
    key = _resolve_slot(d, args.slot)
    try:
        record = add_spec(key, args.name, Path(args.spec), Path(args.proof), author=args.author, db=args.db,
                          dry_run=args.check, keep_dir=_opt_path(args.workspace),
                          lake_budget_s=args.budget if args.budget is not None else DEFAULT_LAKE_BUDGET_S)
    except VerificationError as exc:
        print(f"REJECTED ({type(exc).__name__}): {str(exc)[:1500]}", file=sys.stderr)
        return 2
    print(json.dumps({k: v for k, v in record.items() if k != "log"}, indent=2, sort_keys=True))
    return 0


def _cmd_seed(args: argparse.Namespace) -> int:
    from spire.design_db.insert import seed_original
    d = _open(args.db, create=True)
    key = _resolve_slot(d, args.slot)
    return _gated_insert(lambda: seed_original(key, db=args.db, budget_s=args.budget))


def _parse_metric_pairs(pairs: list) -> dict:
    values = {}
    for p in pairs:
        if "=" not in p:
            raise DesignDBError(f"metric must be KEY=VALUE, got {p!r}")
        k, v = p.split("=", 1)
        try:
            values[k.strip()] = float(v)
        except ValueError:
            raise DesignDBError(f"metric {k.strip()!r} must be numeric, got {v!r}")
    return values


def _cmd_annotate(args: argparse.Namespace) -> int:
    from spire.design_db.annotate import annotate
    d = _open(args.db, create=False)
    key = _resolve_slot(d, args.slot)
    values = _parse_metric_pairs(args.values)
    raw = json.loads(Path(args.raw).read_text()) if args.raw else None
    result = annotate(key, args.design, tech=args.tech, values=values, raw=raw,
                      force=args.force, db=args.db)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    """Advisory verification: run the slot's set oracle against a candidate, no admit, no write."""
    from spire.design_db.insert import check_design
    from spire.design_db.verify import CECInapplicable, SlotUnverified, VerificationError
    d = _open(args.db, create=False)
    key = _resolve_slot(d, args.slot)
    workdir = args.workspace
    try:
        result = check_design(key, Path(args.design), db=args.db, budget_s=args.budget,
                              proof=_opt_path(args.proof), workdir=workdir)
    except (SlotUnverified, CECInapplicable) as exc:
        raise DesignDBError(str(exc))                    # a setup problem, not a candidate verdict
    except VerificationError as exc:
        fail = {"verdict": "FAIL", "type": type(exc).__name__, "reason": str(exc).splitlines()[0][:300]}
        if workdir:
            fail["workspace"] = workdir
            fail["next"] = f"write/repair the D<hash>_Proof.lean in {workdir}, `lake build` there, then insert --proof"
        print(json.dumps(fail, indent=2, sort_keys=True))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _cmd_set_verification(args: argparse.Namespace) -> int:
    """Configure (and, for sim tiers, freeze) a slot's verification oracle — the method every
    later insert/verify for this slot is judged by. Fail-and-choose, never an auto-fallback."""
    from spire.design_db.verify import DEFAULT_CEC_BUDGET_S, VERIFICATION_SCHEMA
    if args.check and args.stimulus is None and not args.lean:
        raise DesignDBError("--check requires --stimulus <file> or --lean (dry run, nothing frozen)")
    if args.workspace is not None and not args.lean:
        raise DesignDBError("--workspace only applies to --lean (keeps the Lean freeze project)")
    d = _open(args.db, create=True)
    key = _resolve_slot(d, args.slot)
    slot = d.slot_dir(key)
    spec = d.read_json(slot / "spec.json", None)
    if spec is None:
        raise DesignDBError(f"unknown slot {args.slot!r}")
    chosen = [m for m, on in (("cec", args.cec), ("auto", args.auto),
                              ("stimulus", args.stimulus is not None), ("lean", args.lean)) if on]
    if len(chosen) > 1:
        raise DesignDBError("choose exactly one of --cec | --auto | --stimulus | --lean")
    if not chosen:
        if spec.get("class") == "combinational":
            chosen = ["cec"]                                # the default-picker (class check)
        else:
            raise DesignDBError("sequential slot — choose the verification explicitly: "
                                "--auto (Tier-1 sim harness) | --stimulus <file> (authored); "
                                "CEC is inapplicable")
    mode = chosen[0]
    if args.check and mode == "stimulus":
        from spire.design_db.verify_sim import check_stimulus
        result = check_stimulus(key, stimulus_file=args.stimulus, n_vectors=args.vectors,
                                seed=args.seed, db=args.db)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.author is not None and mode not in ("stimulus", "lean"):
        raise DesignDBError("--author only applies to --stimulus and --lean freezes")
    existing = d.read_json(slot / "verification.json", None)
    if mode == "lean":
        if existing is not None and int(existing.get("tier", 0)) >= 1 and not args.check:
            raise DesignDBError("slot already has a frozen verification (immutable); --check still dry-runs")
        from spire.design_db.verify_lean import DEFAULT_LAKE_BUDGET_S, freeze_lean_verification
        verification = freeze_lean_verification(
            key, author=args.author, db=args.db, dry_run=args.check, keep_dir=_opt_path(args.workspace),
            lake_budget_s=args.budget if args.budget is not None else DEFAULT_LAKE_BUDGET_S)
        print(json.dumps(verification, indent=2, sort_keys=True))
        return 0
    if mode == "cec":
        if spec.get("class") == "sequential":
            raise DesignDBError("CEC is inapplicable to sequential slots (no register mapping) — "
                                "options: --auto | --stimulus <file>")
        if existing is not None and int(existing.get("tier", 0)) >= 1:
            raise DesignDBError("slot has a frozen verification (immutable) — CEC cannot replace it")
        verification = {"schema": VERIFICATION_SCHEMA, "tier": 0, "method": "cec",
                        "budget_s": args.budget if args.budget is not None
                        else DEFAULT_CEC_BUDGET_S}
        d.write_json(slot / "verification.json", verification)
    else:
        from spire.design_db.verify_sim import freeze_sim_verification
        verification = freeze_sim_verification(
            key, stimulus_file=args.stimulus, n_vectors=args.vectors, seed=args.seed,
            sim_budget_s=args.budget if args.budget is not None else 300.0,
            stimulus_author=args.author, db=args.db)
    print(json.dumps(verification, indent=2, sort_keys=True))
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

    p = sub.add_parser("insert", help="insert a design through the verification gate — spire "
                                      "(.py with build(); source stored) or Verilog (.v/.sv)")
    _common(p); p.add_argument("design", help="candidate: .py (spire, elaborated here) or .v/.sv")
    p.add_argument("--slot", required=True, help="manifest name, spec_key, or unique key prefix")
    p.add_argument("--source", default="cli", help="provenance source tag (default: cli)")
    p.add_argument("--budget", type=float, default=None, help="verification budget in seconds")
    p.add_argument("--proof", default=None, metavar="FILE",
                   help="Lean-gated slots: the candidate's D<hash>_Proof.lean")
    p.set_defaults(func=_cmd_insert)

    p = sub.add_parser("seed", help="insert the slot's own golden as the baseline candidate "
                                    "(source=original) — a selection floor + report baseline")
    _common(p); p.add_argument("--slot", required=True,
                               help="manifest name, spec_key, or unique key prefix")
    p.add_argument("--budget", type=float, default=None, help="CEC budget in seconds")
    p.set_defaults(func=_cmd_seed)

    p = sub.add_parser("annotate", help="attach a per-technology metric block to a stored design "
                                        "(makes metric=<tech> selectable)")
    _common(p); p.add_argument("--slot", required=True,
                               help="manifest name, spec_key, or unique key prefix")
    p.add_argument("--design", required=True, help="design_id or a unique prefix of one")
    p.add_argument("--tech", required=True,
                   help="measurement-system name, e.g. asap7 (selectable via metric=<tech>)")
    p.add_argument("values", nargs="+", metavar="KEY=VALUE",
                   help="numeric metric readings, e.g. area=123.4 delay=456.7 adp=56789")
    p.add_argument("--raw", default=None, metavar="FILE",
                   help="optional JSON file: the full tool stats blob (stored under .raw)")
    p.add_argument("--force", action="store_true",
                   help="overwrite an existing block for this tech")
    p.set_defaults(func=_cmd_annotate)

    p = sub.add_parser("set-verification",
                       help="choose (and, for sim tiers, freeze) a slot's verification oracle — "
                            "the method every later insert/verify is judged by (explicit; "
                            "no auto-fallback; combinational defaults to CEC at registration)")
    _common(p); p.add_argument("--slot", required=True,
                               help="manifest name, spec_key, or unique key prefix")
    p.add_argument("--cec", action="store_true", help="Tier-0 CEC (combinational only)")
    p.add_argument("--auto", action="store_true", help="freeze the Tier-1 auto sim harness")
    p.add_argument("--stimulus", default=None, metavar="FILE",
                   help="freeze a Tier-2 sim verification from an authored stimulus generator")
    p.add_argument("--budget", type=float, default=None,
                   help="verification budget in seconds (cec: yosys/abc, default 120; sim: verilator, default 300; "
                        "lean: lake, default 600)")
    p.add_argument("--lean", action="store_true",
                   help="freeze the Lean oracle: Spec = equivalence to the golden (model captured at registration); "
                        "add nicer, proven-equivalent specs later with `add-spec`")
    p.add_argument("--vectors", type=int, default=256, help="number of stimulus vectors (sim)")
    p.add_argument("--seed", type=int, default=0, help="stimulus RNG seed (sim, auto)")
    p.add_argument("--author", default=None,
                   help="recorded author for --stimulus / --lean-spec freezes (unset if omitted; "
                        "agent layers pass e.g. agent:rtl-dv-prep, agent:lean-spec-author)")
    p.add_argument("--check", action="store_true",
                   help="dry run, nothing frozen: --stimulus loads the generator and produces vectors; "
                        "--lean-spec runs both freeze obligations")
    p.add_argument("--workspace", default=None, metavar="DIR",
                   help="--lean: keep the Lean freeze project in DIR")
    p.set_defaults(func=_cmd_set_verification)

    p = sub.add_parser("verify", help="advisory: run the slot's set oracle against a candidate "
                                      "design (no admit, no write) — the check `insert` gates on")
    _common(p); p.add_argument("design", help="candidate: .py (spire, elaborated here) or .v/.sv")
    p.add_argument("--slot", required=True, help="manifest name, spec_key, or unique key prefix")
    p.add_argument("--budget", type=float, default=None, help="verification budget in seconds")
    p.add_argument("--proof", default=None, metavar="FILE",
                   help="Lean-gated slots: D<hash>_Proof.lean (omitted with --workspace: write the workspace only)")
    p.add_argument("--workspace", default=None, metavar="DIR",
                   help="Lean-gated slots: keep the Lean project in DIR (pass or fail) — the agent workspace "
                        "(frozen files, spec layers, admitted designs, D<hash>_Circuit.lean)")
    p.set_defaults(func=_cmd_verify)

    p = sub.add_parser("add-spec", help="Lean-gated slots: admit a spec layer NAME — NAME.lean defines NAME.Correct, "
                                        "PROOF proves NAME.equiv : ∀ e, NAME.Correct e ↔ Spec.Correct e (append-only)")
    _common(p); p.add_argument("name", help="layer name (capitalised identifier, e.g. Mmac)")
    p.add_argument("spec", help="NAME.lean")
    p.add_argument("proof", help="NAMEProof.lean")
    p.add_argument("--slot", required=True, help="manifest name, spec_key, or unique key prefix")
    p.add_argument("--author", default=None, help="recorded author, e.g. agent:lean-spec-author")
    p.add_argument("--budget", type=float, default=None, help="lake build budget in seconds (default 600)")
    p.add_argument("--check", action="store_true", help="dry run: build and audit, admit nothing")
    p.add_argument("--workspace", default=None, metavar="DIR", help="keep the Lean project in DIR")
    p.set_defaults(func=_cmd_add_spec)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except DesignDBError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
