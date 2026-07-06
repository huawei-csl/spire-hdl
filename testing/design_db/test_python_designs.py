"""Python (spire) designs in the DB: the ``.py`` insert path, source storage, and the seed copy.

Spire is the authoring language; Verilog is the intermediate representation. A ``.py`` insert is
elaborated by the gate itself (``build()`` → Verilog), so the stored ``design.py`` is correct by
construction; ``design.v`` stays canonical for all downstream processing.
"""
import json

import pytest

from spire import UInt
from spire.component import Netlist
from spire.design_db import (DesignDBError, VerificationFailed, check_design, from_design_db,
                             insert_design, register_slot, seed_original)
from spire.design_db.cli import main as cli_main
from spire.design_db.store import DB_ENV, VERSION_DIR

from test_s1_core import Adder


@pytest.fixture
def db(tmp_path, monkeypatch):
    root = tmp_path / "design_db"
    monkeypatch.setenv(DB_ENV, str(root))
    return root


# A structurally distinct correct adder, authored in spire (carry-save first stage).
CS_ADDER_PY = """\
from spire import UInt
from spire.component import Netlist


def build():
    m = Netlist("cs_adder", with_clock=False, with_reset=False)
    a = m.input(UInt(8), "a")
    b = m.input(UInt(8), "b")
    s = m.output(UInt(9), "s")
    s <<= (a ^ b) + ((a & b) << 1)
    return m
"""

WRONG_PY = CS_ADDER_PY.replace("(a ^ b) + ((a & b) << 1)", "(a ^ b) + ((a & b) << 1) + 1")

MULTIFILE_ENTRY = """\
from spire import UInt
from spire.component import Netlist
from cs_helper import carry_save_sum


def build():
    m = Netlist("cs_adder_mf", with_clock=False, with_reset=False)
    a = m.input(UInt(8), "a")
    b = m.input(UInt(8), "b")
    s = m.output(UInt(9), "s")
    s <<= carry_save_sum(a, b)
    return m
"""

MULTIFILE_HELPER = """\
def carry_save_sum(a, b):
    return (a ^ b) + ((a & b) << 1)
"""


def _design_dir(db, key, design_id):
    return db / VERSION_DIR / key / "designs" / design_id


def test_py_insert_stores_source_and_gates(db, tmp_path):
    key = register_slot(Adder(), name="adder8")
    cand = tmp_path / "cs_adder.py"
    cand.write_text(CS_ADDER_PY)

    res = insert_design(key, cand, source="agent:rtl-subcircuit")
    assert not res.deduped
    ddir = _design_dir(db, key, res.design_id)
    assert (ddir / "design.v").exists() and (ddir / "design.aag").exists()   # IR stays canonical
    assert (ddir / "design.py").read_text() == CS_ADDER_PY                    # source stored
    prov = json.loads((ddir / "provenance.json").read_text())
    assert prov["python_source"] == {"kind": "elaborated", "entry": "design.py",
                                     "local_modules": []}

    # a wrong .py is rejected by the same gate that rejects wrong Verilog
    bad = tmp_path / "bad.py"
    bad.write_text(WRONG_PY)
    with pytest.raises(VerificationFailed):
        insert_design(key, bad, source="agent:rtl-subcircuit")

    # contract errors are clean
    nb = tmp_path / "nobuild.py"
    nb.write_text("x = 1\n")
    with pytest.raises(DesignDBError, match="must define build"):
        insert_design(key, nb, source="test")
    crash = tmp_path / "crash.py"
    crash.write_text("raise RuntimeError('boom')\n")
    with pytest.raises(DesignDBError, match="failed to execute"):
        insert_design(key, crash, source="test")


def test_py_insert_multifile_closure(db, tmp_path):
    key = register_slot(Adder(), name="adder8")
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "cs_entry.py").write_text(MULTIFILE_ENTRY)
    (proj / "cs_helper.py").write_text(MULTIFILE_HELPER)

    res = insert_design(key, proj / "cs_entry.py", source="human")
    ddir = _design_dir(db, key, res.design_id)
    assert (ddir / "design.py").read_text() == MULTIFILE_ENTRY
    assert (ddir / "source" / "cs_helper.py").read_text() == MULTIFILE_HELPER
    prov = json.loads((ddir / "provenance.json").read_text())
    assert prov["python_source"]["local_modules"] == ["cs_helper.py"]


def test_py_verify_advisory_and_cli(db, tmp_path, capsys):
    key = register_slot(Adder(), name="adder8")
    cand = tmp_path / "cs_adder.py"
    cand.write_text(CS_ADDER_PY)

    assert check_design(key, cand)["verdict"] == "PASS"      # advisory, .py accepted
    assert cli_main(["db", "verify", str(cand), "--slot", "adder8"]) == 0
    assert json.loads(capsys.readouterr().out)["verdict"] == "PASS"
    assert cli_main(["db", "insert", str(cand), "--slot", "adder8",
                     "--source", "agent:rtl-subcircuit"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["deduped"] is False
    assert (_design_dir(db, key, out["design_id"]) / "design.py").exists()


def test_seed_copies_starting_point(db):
    # decorator-registered slot: starting_point.py is captured → seed carries it
    @from_design_db(objective="area", name="seeded_add")
    def add8(a, b):
        return a + b

    m = Netlist("top", with_clock=False, with_reset=False)
    a, b = m.input(UInt(8), "a"), m.input(UInt(8), "b")
    y = m.output(UInt(9), "y")
    y <<= add8(a, b)
    m.to_verilog()                                            # registers the slot (miss → original)

    manifest = json.loads((db / VERSION_DIR / "manifest.json").read_text())
    key = manifest["slots"]["seeded_add"]["spec_key"]
    res = seed_original(key)
    ddir = _design_dir(db, key, res.design_id)
    sp = (db / VERSION_DIR / key / "starting_point.py").read_text()
    assert (ddir / "design.py").read_text() == sp
    prov = json.loads((ddir / "provenance.json").read_text())
    assert prov["python_source"]["kind"] == "copied"

    # plain register_slot slot (no starting point): seed works, stores no python
    key2 = register_slot(Adder(), name="adder_plain")
    res2 = seed_original(key2)
    assert not (_design_dir(db, key2, res2.design_id) / "design.py").exists()
