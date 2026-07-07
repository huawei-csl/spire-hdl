"""The derived index: designs/ dirs are the source of truth; index.json is a self-healing cache.

Admission = one atomic dir rename, so concurrent inserts can never lose an entry — the property
that unlocks parallel slot dispatch. The cache stays materialized for direct inspection.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from spire.design_db import DesignDB, insert_design, register_slot, select_design, seed_original
from spire.design_db.cli import main as cli_main
from spire.design_db.store import DB_ENV, VERSION_DIR

from test_s1_core import Adder, EQUIV_V


@pytest.fixture
def db(tmp_path, monkeypatch):
    root = tmp_path / "design_db"
    monkeypatch.setenv(DB_ENV, str(root))
    return root


# Three structurally distinct correct adder forms (a+b == (a^b)+2(a&b) == (a|b)+(a&b)).
VARIANTS = {
    "plain": "assign s = {1'b0, a} + {1'b0, b};",
    "carry_save": "assign s = {1'b0, a ^ b} + {(a & b), 1'b0};",
    "or_and": "assign s = {1'b0, (a | b)} + {1'b0, (a & b)};",
}


def _adder_v(body):
    return ("module cand(input [7:0] a, input [7:0] b, output [8:0] s);\n"
            f"  {body}\nendmodule\n")


def test_index_is_derived_and_self_healing(db):
    key = register_slot(Adder(), name="adder8")
    seed_original(key)
    insert_design(key, _adder_v(VARIANTS["carry_save"]), source="test")
    d = DesignDB.open(db)
    cache = d.slot_dir(key) / "index.json"
    assert cache.exists()                                 # materialized at admit
    golden_view = json.loads(cache.read_text())
    assert len(golden_view) == 2

    cache.unlink()                                        # lost cache → reads still whole
    assert d.read_index(key) == golden_view
    assert cache.exists(), "read_index re-materializes the cache"

    cache.write_text("{}")                                # stale/corrupt cache → self-heals
    sel = select_design(key, objective="area")
    assert sel is not None
    assert json.loads(cache.read_text()) == golden_view

    # dedup still works with no cache at all (derivation feeds it)
    cache.unlink()
    res = insert_design(key, _adder_v(VARIANTS["carry_save"]), source="test")
    assert res.deduped


def test_struct_hash_stamped_and_legacy_fallback(db):
    key = register_slot(Adder(), name="adder8")
    res = insert_design(key, _adder_v(VARIANTS["plain"]), source="test")
    d = DesignDB.open(db)
    ddir = d.slot_dir(key) / "designs" / res.design_id
    prov = json.loads((ddir / "provenance.json").read_text())
    assert len(prov["struct_hash"]) == 64                 # stamped at admit

    # legacy dir (pre-derivation): no struct_hash in provenance → derived from design.aag
    # (must reproduce the stamped hash exactly), and dedup still recognizes the design
    stamped = prov["struct_hash"]
    del prov["struct_hash"]
    (ddir / "provenance.json").write_text(json.dumps(prov))
    derived = d.derive_index(key)
    assert derived[res.design_id]["struct_hash"] == stamped
    again = insert_design(key, _adder_v(VARIANTS["plain"]), source="test")
    assert again.deduped and again.design_id == res.design_id


def test_parallel_inserts_lose_nothing(db, tmp_path):
    """Concurrent admissions into ONE slot from separate processes — every design lands."""
    key = register_slot(Adder(), name="adder8")
    script = tmp_path / "ins.py"
    script.write_text(
        "import sys\n"
        "from spire.design_db import insert_design\n"
        "insert_design(sys.argv[1], sys.argv[2], source='p' + sys.argv[3])\n")
    files = []
    for i, body in enumerate(VARIANTS.values()):
        f = tmp_path / f"cand{i}.v"
        f.write_text(_adder_v(body))
        files.append(f)
    env = {**os.environ, DB_ENV: str(db)}
    procs = [subprocess.Popen([sys.executable, str(script), key, str(f), str(i)],
                              env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
             for i, f in enumerate(files)]
    for p in procs:
        _out, err = p.communicate(timeout=300)
        assert p.returncode == 0, err.decode()[-500:]

    d = DesignDB.open(db)
    index = d.read_index(key)
    assert len(index) == 3, sorted(index)                 # nothing lost to the race
    assert len({e["struct_hash"] for e in index.values()}) == 3


def test_counts_derived_in_ls_and_manifest_free_of_counts(db, capsys):
    key = register_slot(Adder(), name="adder8")
    seed_original(key)
    manifest = json.loads((db / VERSION_DIR / "manifest.json").read_text())
    assert "n_designs" not in manifest["slots"]["adder8"]     # counts are not stored anymore

    (DesignDB.open(db).slot_dir(key) / "index.json").unlink() # even without the cache…
    assert cli_main(["db", "ls"]) == 0
    out = capsys.readouterr().out
    assert "designs=1" in out and "adder8" in out             # …ls derives the right count