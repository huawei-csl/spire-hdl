"""S1 tests: registration, the verification seam (Tier-0 CEC), and the insert gate.

Offline but tool-real: yosys / yosys-abc must be on PATH (they are in the dev container).
Each test gets a fresh DB via the ``db`` fixture (SPIREHDL_DB_PATH → tmp_path).
"""
import json
from pathlib import Path

import pytest

from spire import Component, IORecord, Input, Output, UInt
from spire.component import Netlist
from spire.design_db import (CECTimeout, DesignDBError, SlotUnverified, VerificationFailed,
                             insert_design, register_slot, resolve_db_root)
from spire.design_db.store import DB_ENV, VERSION_DIR


class Adder(Component):
    def __init__(self):
        self.io = IORecord(a=Input(UInt(8)), b=Input(UInt(8)), s=Output(UInt(9)))
        self.elaborate()

    def elaborate(self):
        self.io.s <<= self.io.a + self.io.b


EQUIV_V = """\
module cand_add(input [7:0] a, input [7:0] b, output [8:0] s);
  assign s = {1'b0, a} + {1'b0, b};
endmodule
"""

WRONG_V = """\
module cand_bad(input [7:0] a, input [7:0] b, output [8:0] s);
  assign s = {1'b0, a} + {1'b0, b} + 9'd1;
endmodule
"""


@pytest.fixture
def db(tmp_path, monkeypatch):
    root = tmp_path / "design_db"
    monkeypatch.setenv(DB_ENV, str(root))
    return root


def _slot_dir(db_root: Path, key: str) -> Path:
    return db_root / VERSION_DIR / key


def test_register_creates_slot_with_default_cec(db):
    key = register_slot(Adder(), name="adder8")
    slot = _slot_dir(db, key)
    assert (slot / "golden.v").exists()
    spec = json.loads((slot / "spec.json").read_text())
    assert spec["class"] == "combinational"
    assert [p["name"] for p in spec["ports"]] == ["a", "b", "s"]
    verification = json.loads((slot / "verification.json").read_text())
    assert verification["method"] == "cec" and verification["tier"] == 0
    manifest = json.loads((db / VERSION_DIR / "manifest.json").read_text())
    assert manifest["slots"]["adder8"]["spec_key"] == key


def test_insert_equivalent_design(db):
    key = register_slot(Adder(), name="adder8")
    res = insert_design(key, EQUIV_V, source="test")
    assert not res.deduped
    assert res.metrics["intrinsic"]["aig_nodes"] > 0
    assert isinstance(res.metrics["transistors_heavy"], int)
    ddir = _slot_dir(db, key) / "designs" / res.design_id
    assert (ddir / "design.v").exists() and (ddir / "provenance.json").exists()
    prov = json.loads((ddir / "provenance.json").read_text())
    assert prov["verification"]["verdict"] == "PASS"
    index = json.loads((_slot_dir(db, key) / "index.json").read_text())
    assert res.design_id in index


def test_reject_non_equivalent_and_stay_atomic(db):
    key = register_slot(Adder(), name="adder8")
    with pytest.raises(VerificationFailed):
        insert_design(key, WRONG_V, source="test")
    designs = _slot_dir(db, key) / "designs"
    assert list(designs.iterdir()) == []            # nothing admitted, no .tmp leftovers
    assert not (_slot_dir(db, key) / "index.json").exists()


class AdderCarrySave(Component):
    """Equivalent adder written differently: a + b == (a ^ b) + 2*(a & b)."""

    def __init__(self):
        self.io = IORecord(a=Input(UInt(8)), b=Input(UInt(8)), s=Output(UInt(9)))
        self.elaborate()

    def elaborate(self):
        self.io.s <<= (self.io.a ^ self.io.b) + ((self.io.a & self.io.b) << 1)


def test_insert_spire_component_directly(db):
    key = register_slot(Adder(), name="adder8")
    res = insert_design(key, AdderCarrySave(), source="spire")
    assert not res.deduped
    ddir = _slot_dir(db, key) / "designs" / res.design_id
    assert "module" in (ddir / "design.v").read_text()
    assert res.metrics["intrinsic"]["aig_nodes"] > 0


def test_dedup_by_structural_hash(db):
    key = register_slot(Adder(), name="adder8")
    first = insert_design(key, EQUIV_V, source="test")
    second = insert_design(key, EQUIV_V, source="test")
    assert second.deduped and second.design_id == first.design_id
    index = json.loads((_slot_dir(db, key) / "index.json").read_text())
    assert len(index) == 1


def test_sequential_slot_registers_but_refuses_insert(db):
    m = Netlist("seqm", with_clock=True, with_reset=True)
    din = m.input(UInt(4), "din")
    q = m.reg(UInt(4), "q", init=0)
    q <<= din
    out = m.output(UInt(4), "dout")
    out <<= q
    key = register_slot(m)
    slot = _slot_dir(db, key)
    assert json.loads((slot / "spec.json").read_text())["class"] == "sequential"
    assert not (slot / "verification.json").exists()  # unverified until a sim tier is frozen (S3)
    with pytest.raises(SlotUnverified):
        insert_design(key, EQUIV_V, source="test")


def test_cec_timeout_fails_cleanly_with_options(db):
    key = register_slot(Adder(), name="adder8")
    with pytest.raises(CECTimeout) as excinfo:
        insert_design(key, EQUIV_V, source="test", budget_s=1e-4)
    assert "Options:" in str(excinfo.value)
    assert list((_slot_dir(db, key) / "designs").iterdir()) == []


def test_registration_idempotent_with_backrefs(db):
    k1 = register_slot(Adder(), name="alpha")
    k2 = register_slot(Adder(), name="beta")
    assert k1 == k2
    spec = json.loads((_slot_dir(db, k1) / "spec.json").read_text())
    assert [r["name"] for r in spec["registered_from"]] == ["alpha", "beta"]
    manifest = json.loads((db / VERSION_DIR / "manifest.json").read_text())
    assert manifest["slots"]["alpha"]["spec_key"] == k1
    assert manifest["slots"]["beta"]["spec_key"] == k1


def test_unknown_slot_is_an_error(db):
    with pytest.raises(DesignDBError):
        insert_design("0" * 64, EQUIV_V, source="test")


def test_zero_config_autocreate_in_cwd(tmp_path, monkeypatch):
    monkeypatch.delenv(DB_ENV, raising=False)
    monkeypatch.chdir(tmp_path)
    key = register_slot(Adder(), name="adder8")
    assert (tmp_path / "design_db" / VERSION_DIR / key / "golden.v").exists()


def test_env_override_wins_over_cwd(tmp_path, monkeypatch):
    monkeypatch.setenv(DB_ENV, str(tmp_path / "elsewhere"))
    monkeypatch.chdir(tmp_path)
    root = resolve_db_root()
    assert root == tmp_path / "elsewhere"
