"""Design-DB store: layout, atomic writes, DB-root resolution, manifest, slot registration.

Layout (schema v1)::

    <db root>/v1/<spec_key>/
        spec.json           # {name, ports, class, golden_sha, created, registered_from:[...]}
        golden.v            # the golden reference candidates are verified against
        verification.json   # frozen verification (combinational default: Tier-0 CEC); absent = unverified
        designs/<id>/       # verification-gated implementations (design.v, metrics.json, provenance.json)
        index.json          # roll-up {design_id -> {struct_hash, metrics, source, created}}
    <db root>/v1/manifest.json   # reverse index {registered name -> {spec_key, class, selection}}

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

    # -- the design index: derived from the design dirs (the source of truth)

    def derive_index(self, spec_key: str) -> Dict[str, Any]:
        """The slot's design index, derived from ``designs/<id>/`` — each admitted dir carries
        its own ``provenance.json`` + ``metrics.json`` (+ ``design.aag``), and the atomic dir
        rename at admit is the only write that matters. Concurrent admissions can therefore
        never lose an entry; ``index.json`` is merely a materialized cache of this."""
        designs = self.slot_dir(spec_key) / "designs"
        index: Dict[str, Any] = {}
        if not designs.is_dir():
            return index
        for ddir in sorted(p for p in designs.iterdir()
                           if p.is_dir() and not p.name.endswith(".tmp")):
            prov = self.read_json(ddir / "provenance.json", {})
            struct_hash = prov.get("struct_hash")
            if not struct_hash:                     # pre-derivation dirs: hash the stored AAG
                aag = ddir / "design.aag"
                if aag.exists():
                    text = aag.read_text()
                    text = text[:-1] if text.endswith("\n") else text
                    struct_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            index[ddir.name] = {"struct_hash": struct_hash,
                                "source": prov.get("source", ddir.name.rsplit(":", 1)[0]),
                                "created": prov.get("created"),
                                "metrics": self.read_json(ddir / "metrics.json", {})}
        return index

    def read_index(self, spec_key: str, *, materialize: bool = True) -> Dict[str, Any]:
        """Derived index + best-effort refresh of the ``index.json`` cache (kept on disk for
        direct inspection — never authoritative, self-healing on every read)."""
        index = self.derive_index(spec_key)
        if materialize and self.slot_dir(spec_key).is_dir():   # never create a slot on a read
            try:
                cache = self.slot_dir(spec_key) / "index.json"
                if self.read_json(cache, None) != index:
                    self.write_json(cache, index)
            except OSError:
                pass                                 # cache refresh must never break a read
        return index

    # -- manifest (small primary record: name bindings + selection provenance; counts derived)

    def _locked_manifest_write(self, mutate) -> None:
        """Read-modify-write the manifest under an fcntl lock — manifest writes are rare
        (registration, selection recording) but must not lose entries under concurrency."""
        import fcntl
        self.v1.mkdir(parents=True, exist_ok=True)
        lock_path = self.v1 / ".manifest.lock"
        with open(lock_path, "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                manifest = self.read_json(self.manifest_path, {"schema": 1, "slots": {}})
                if mutate(manifest):
                    self.write_json(self.manifest_path, manifest)
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def update_manifest(self, name: str, entry: Dict[str, Any]) -> None:
        def _mutate(manifest: Dict[str, Any]) -> bool:
            slot = manifest["slots"].get(name, {})
            slot.update(entry)
            manifest["slots"][name] = slot
            return True
        self._locked_manifest_write(_mutate)

    def update_manifest_selection(self, spec_key: str, fields: Dict[str, Any]) -> None:
        """Record the resolved selection on every manifest entry of this slot."""
        def _mutate(manifest: Dict[str, Any]) -> bool:
            changed = False
            for entry in manifest.get("slots", {}).values():
                if entry.get("spec_key") == spec_key:
                    entry.update(fields)
                    changed = True
            return changed
        self._locked_manifest_write(_mutate)


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
    reg_name = name or getattr(module, "name", "design")
    existing = d.read_json(d.manifest_path, {"slots": {}}).get("slots", {}).get(reg_name)
    if existing and existing.get("spec_key") != key:
        raise DesignDBError(
            f"slot name {reg_name!r} is already bound to slot "
            f"{(existing.get('spec_key') or '?')[:12]}… but this subcircuit hashes to "
            f"{key[:12]}… — slot names are permanent bindings; register the changed subcircuit "
            f"under a new name (rename the function or pass name=)")
    slot = d.slot_dir(key)
    (slot / "designs").mkdir(parents=True, exist_ok=True)
    now = time.strftime("%Y-%m-%dT%H:%M:%S")

    clk = getattr(module, "clk", None)
    rst = getattr(module, "rst", None)
    spec = d.read_json(slot / "spec.json", None)
    if spec is None:
        spec = {"schema": SPEC_SCHEMA, "name": reg_name, "ports": ports, "class": circuit_class,
                "clock": {"clk": clk.name if clk is not None else None,
                          "rst": rst.name if rst is not None else None},
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

    d.update_manifest(reg_name, {"spec_key": key, "class": circuit_class})
    # (design counts are derived from designs/ at read time — see derive_index; nothing to stamp)
    return key
