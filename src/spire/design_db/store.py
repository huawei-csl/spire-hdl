"""Design-DB store: layout, atomic writes, DB-root resolution, manifest, slot registration.

Layout (schema v1)::

    <db root>/v1/<spec_key>/
        spec.json           # {name, ports, class, golden_sha, created, registered_from:[...]}
        golden.v            # the golden reference candidates are verified against
        verification.json   # frozen verification (combinational default: Tier-0 CEC); absent = unverified
        designs/<id>/       # verification-gated implementations (design.v, metrics.json, provenance.json)
        index.json          # roll-up {design_id -> {struct_hash, metrics, source, created}}
    <db root>/v1/manifest.json   # reverse index {registered name -> {spec_key, class, n_designs}}

DB-root resolution (zero-config): explicit ``db=`` → ``$SPIREHDL_DB_PATH`` → nearest ``design_db/``
upward from cwd → auto-create ``./design_db`` (one-line note on first creation).
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

DB_ENV = "SPIREHDL_DB_PATH"
DB_DIRNAME = "design_db"
VERSION_DIR = "v1"
SPEC_SCHEMA = 1

_autocreate_noted = False


class DesignDBError(Exception):
    """Base class for design-DB store/gate errors."""


def resolve_db_root(db: Optional[str | Path] = None, *, create: bool = True) -> Path:
    """Resolve the DB root: explicit → env → nearest existing ``design_db/`` upward → auto-create."""
    global _autocreate_noted
    if db is not None:
        root = Path(db)
    elif os.environ.get(DB_ENV):
        root = Path(os.environ[DB_ENV])
    else:
        cur = Path.cwd().resolve()
        for p in (cur, *cur.parents):
            cand = p / DB_DIRNAME
            if cand.is_dir():
                return cand
        root = cur / DB_DIRNAME
        if create and not root.exists():
            (root / VERSION_DIR).mkdir(parents=True, exist_ok=True)
            if not _autocreate_noted:
                print(f"[spire.design_db] created new design DB at {root}")
                _autocreate_noted = True
        return root
    if create:
        (root / VERSION_DIR).mkdir(parents=True, exist_ok=True)
    return root


class DesignDB:
    """Thin handle on a DB root with atomic JSON/text IO and manifest helpers."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.v1 = self.root / VERSION_DIR

    @classmethod
    def open(cls, db: Optional[str | Path] = None, *, create: bool = True) -> "DesignDB":
        return cls(resolve_db_root(db, create=create))

    # -- paths

    @property
    def manifest_path(self) -> Path:
        return self.v1 / "manifest.json"

    def slot_dir(self, spec_key: str) -> Path:
        return self.v1 / spec_key

    # -- atomic IO (tmp + os.replace, the optimize.py _write_cache_entry pattern)

    def atomic_write_text(self, path: Path, text: str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + f".tmp{os.getpid()}")
        tmp.write_text(text)
        os.replace(tmp, path)

    def write_json(self, path: Path, obj: Any) -> None:
        self.atomic_write_text(path, json.dumps(obj, indent=2, sort_keys=True) + "\n")

    def read_json(self, path: Path, default: Any = None) -> Any:
        try:
            return json.loads(Path(path).read_text())
        except FileNotFoundError:
            return default

    # -- manifest

    def update_manifest(self, name: str, entry: Dict[str, Any]) -> None:
        manifest = self.read_json(self.manifest_path, {"schema": 1, "slots": {}})
        slot = manifest["slots"].get(name, {})
        slot.update(entry)
        manifest["slots"][name] = slot
        self.write_json(self.manifest_path, manifest)

    def refresh_manifest_counts(self, spec_key: str, n_designs: int) -> None:
        manifest = self.read_json(self.manifest_path, None)
        if not manifest:
            return
        changed = False
        for entry in manifest.get("slots", {}).values():
            if entry.get("spec_key") == spec_key:
                entry["n_designs"] = n_designs
                changed = True
        if changed:
            self.write_json(self.manifest_path, manifest)


def register_slot(module_or_component: Any, db: Optional[str | Path] = None, *,
                  name: Optional[str] = None) -> str:
    """Register a slot for a design: write ``spec.json`` + ``golden.v`` (+ the default Tier-0 CEC
    ``verification.json`` for combinational slots) and update the manifest. Idempotent — returns the
    content-addressed ``spec_key``. Sequential slots register fine but stay unverified (inserts are
    refused until a sim-tier verification is frozen, S3)."""
    from spire.design_db import keys as _keys
    from spire.design_db.verify import default_verification, detect_class

    module = _keys.normalize(module_or_component)
    verilog, ports, key = _keys.golden_and_key(module)
    circuit_class = detect_class(module)

    d = DesignDB.open(db)
    slot = d.slot_dir(key)
    (slot / "designs").mkdir(parents=True, exist_ok=True)
    reg_name = name or getattr(module, "name", "design")
    now = time.strftime("%Y-%m-%dT%H:%M:%S")

    spec = d.read_json(slot / "spec.json", None)
    if spec is None:
        spec = {"schema": SPEC_SCHEMA, "name": reg_name, "ports": ports, "class": circuit_class,
                "golden_sha": hashlib.sha256(verilog.encode("utf-8")).hexdigest(),
                "created": now, "registered_from": []}
    if reg_name not in [r["name"] for r in spec["registered_from"]]:
        spec["registered_from"].append({"name": reg_name, "at": now})
    d.write_json(slot / "spec.json", spec)

    if not (slot / "golden.v").exists():
        d.atomic_write_text(slot / "golden.v", verilog)
    if not (slot / "verification.json").exists():
        verification = default_verification(circuit_class)
        if verification is not None:
            d.write_json(slot / "verification.json", verification)

    index = d.read_json(slot / "index.json", {})
    d.update_manifest(reg_name, {"spec_key": key, "class": circuit_class, "n_designs": len(index)})
    return key
