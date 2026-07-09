"""Temporary selection overrides: $SPIREHDL_DB_PINS and selection_overrides(...).

Both doors force ``pick_design`` (and therefore ``@from_design_db`` splices) to specific
design_ids without touching source. Picking is a pure query — nothing is written.
"""
import json
import os
import subprocess
import sys

import pytest

from spire import UInt
from spire.aiger import AigerExporter
from spire.component import Netlist
from spire.design_db import (DesignDBError, PINS_ENV, from_design_db, insert_design,
                             register_slot, seed_original, pick_design, selection_overrides)
from spire.design_db.store import DB_ENV, DesignDB

from test_s1_core import Adder, EQUIV_V


@pytest.fixture
def db(tmp_path, monkeypatch):
    root = tmp_path / "design_db"
    monkeypatch.setenv(DB_ENV, str(root))
    return root


def _two_design_slot():
    """A named slot with two admitted designs; returns (key, argmin_id, other_id)."""
    key = register_slot(Adder(), name="adder8")
    a = seed_original(key).design_id
    b = insert_design(key, EQUIV_V, source="verilog").design_id
    argmin = pick_design(key, objective="area").design_id
    other = b if argmin == a else a
    return key, argmin, other


def test_env_pins_force_selection(db, monkeypatch):
    key, argmin, other = _two_design_slot()
    monkeypatch.setenv(PINS_ENV, json.dumps({key: other}))
    assert pick_design(key, objective="area").design_id == other    # override beat the argmin
    monkeypatch.delenv(PINS_ENV)
    assert pick_design(key, objective="area").design_id == argmin   # back to normal


def test_context_manager_scoped_and_nested(db):
    key, argmin, other = _two_design_slot()
    with selection_overrides({key: other}):
        assert pick_design(key).design_id == other
        with selection_overrides({key: argmin}):        # innermost wins
            assert pick_design(key).design_id == argmin
        assert pick_design(key).design_id == other
    assert pick_design(key).design_id == argmin       # scope ended


def test_manifest_names_as_override_keys(db):
    key, _, other = _two_design_slot()
    with selection_overrides({"adder8": other}):
        assert pick_design(key).design_id == other


def test_explicit_pin_wins_over_override(db):
    key, argmin, other = _two_design_slot()
    with selection_overrides({key: other}):
        assert pick_design(key, pin=argmin).design_id == argmin


def test_unknown_override_id_raises(db):
    key, _, _ = _two_design_slot()
    with selection_overrides({key: "nope:0000000000"}):
        with pytest.raises(DesignDBError, match="pinned design"):
            pick_design(key)


def test_invalid_env_json_raises(db, monkeypatch):
    key, _, _ = _two_design_slot()
    monkeypatch.setenv(PINS_ENV, "not json")
    with pytest.raises(DesignDBError, match="not valid JSON"):
        pick_design(key)


# equivalent to a truncating 8-bit add, ports a,b -> y (a traced slot's single output is `y`)
CAND_Y = """\
module cand(input [7:0] a, input [7:0] b, output [7:0] y);
  wire [8:0] cs = {1'b0, a ^ b} + {(a & b), 1'b0};
  assign y = cs[7:0];
endmodule
"""


def test_override_reaches_the_decorator_splice(db):
    @from_design_db(objective="area", name="adder8_dec")
    def add8(a, b):
        return (a + b)[0:8]

    def build(tag):
        m = Netlist(f"T_{tag}", with_clock=False, with_reset=False)
        a, b = m.input(UInt(8), "a"), m.input(UInt(8), "b")
        y = m.output(UInt(8), "y")
        y <<= add8(a, b)
        return m

    build("probe")                        # first compile registers the decorated slot (miss)
    man = DesignDB.open(db).read_json(DesignDB.open(db).manifest_path)
    dec_key = man["slots"]["adder8_dec"]["spec_key"]
    seed_original(dec_key)
    insert_design(dec_key, CAND_Y, source="verilog")
    a_id, b_id = sorted(DesignDB.open(db).read_index(dec_key))

    with selection_overrides({dec_key: a_id}):
        aag_a = AigerExporter(build("a")).get_aag()
    with selection_overrides({dec_key: b_id}):
        aag_b = AigerExporter(build("b")).get_aag()
    assert aag_a != aag_b                               # the override reached the splice


def test_env_pins_cross_process(db, monkeypatch):
    """The actual composition-sweep use case: pins set in the environment of a child process."""
    key, argmin, other = _two_design_slot()
    code = ("from spire.design_db import pick_design; "
            f"print(pick_design({key!r}, objective='area').design_id)")
    env = {**os.environ, PINS_ENV: json.dumps({key: other})}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         timeout=120, env=env)
    assert out.returncode == 0, out.stderr[-300:]
    assert out.stdout.strip().splitlines()[-1] == other


def test_selection_log_records_each_splice(db, tmp_path, monkeypatch):
    """$SPIREHDL_DB_SELECTION_LOG: the compile's caller gets one JSON line per splice —
    artifact-side observability; the DB and manifest stay untouched."""
    from spire.design_db import SELECTION_LOG_ENV

    @from_design_db(objective="area", name="add8_logged")
    def add8(a, b):
        return (a + b)[0:8]

    def build(tag):
        m = Netlist(f"L_{tag}", with_clock=False, with_reset=False)
        a, b = m.input(UInt(8), "a"), m.input(UInt(8), "b")
        y = m.output(UInt(8), "y")
        y <<= add8(a, b)
        return m

    build("probe")                                   # registers the slot (miss -> no log entry)
    man = DesignDB.open(db).read_json(DesignDB.open(db).manifest_path)
    key = man["slots"]["add8_logged"]["spec_key"]
    seed_original(key)

    log = tmp_path / "selections.jsonl"
    monkeypatch.setenv(SELECTION_LOG_ENV, str(log))
    build("logged")                                  # splice -> one entry
    (entry,) = [json.loads(l) for l in log.read_text().splitlines()]
    assert entry["spec_key"] == key and entry["name"] == "add8_logged"
    assert entry["design_id"].startswith("original:")
    assert "selected_id" not in (db / "v1" / "manifest.json").read_text()
