"""S2 tests: selection engine, ``@from_design_db``, starting-point capture, and the CLI."""
import json
from pathlib import Path

import pytest

from spire import UInt
from spire.aiger import AigerExporter
from spire.component import Netlist
from spire.design_db import (DesignDBError, cec_check, constrained, from_design_db, insert_design,
                             lexicographic, metric_value, pareto_front, register_slot,
                             select_design, weighted)
from spire.design_db.cli import main as cli_main
from spire.design_db.store import DB_ENV, DesignDB, VERSION_DIR

from test_s1_core import Adder, AdderCarrySave, EQUIV_V


@pytest.fixture
def db(tmp_path, monkeypatch):
    root = tmp_path / "design_db"
    monkeypatch.setenv(DB_ENV, str(root))
    return root


def _filled_slot(db):
    """A slot with two structurally distinct, verified designs; returns (key, index)."""
    key = register_slot(Adder(), name="adder8")
    insert_design(key, EQUIV_V, source="verilog")
    insert_design(key, AdderCarrySave(), source="spire")
    index = json.loads((db / VERSION_DIR / key / "index.json").read_text())
    assert len(index) == 2
    return key, index


# --- selection engine ---------------------------------------------------------------------------


def test_argmin_area_transistors_default(db):
    key, index = _filled_slot(db)
    expected = min(index.items(), key=lambda kv: (kv[1]["metrics"]["transistors_heavy"], kv[0]))[0]
    sel = select_design(key, objective="area")
    assert sel.design_id == expected and sel.metric == "transistors"


def test_objective_delay_under_aig_metric(db):
    key, index = _filled_slot(db)
    expected = min(index.items(),
                   key=lambda kv: (kv[1]["metrics"]["intrinsic"]["aig_depth"], kv[0]))[0]
    sel = select_design(key, objective="delay", metric="aig")
    assert sel.design_id == expected and sel.metric == "aig"


def test_constrained_weighted_lexicographic(db):
    key, index = _filled_slot(db)
    depths = {k: v["metrics"]["intrinsic"]["aig_depth"] for k, v in index.items()}
    areas = {k: v["metrics"]["transistors_heavy"] for k, v in index.items()}
    min_area_id = min(areas, key=lambda k: (areas[k], k))
    loose = max(depths.values())
    sel = select_design(key, objective=constrained(minimize="area", subject_to={"delay": loose}))
    assert sel.design_id == min_area_id
    sel_w = select_design(key, objective=weighted({"area": 1.0, "delay": 0.0}))
    assert sel_w.design_id == min_area_id
    sel_l = select_design(key, objective=lexicographic(("area", "delay")))
    assert sel_l.design_id == min_area_id


def test_constrained_infeasible_returns_none(db):
    key, _ = _filled_slot(db)
    assert select_design(key, objective=constrained(minimize="area",
                                                    subject_to={"delay": -1})) is None


def test_pin_and_broken_pin(db):
    key, index = _filled_slot(db)
    some_id = sorted(index)[0]
    assert select_design(key, pin=some_id).design_id == some_id
    with pytest.raises(DesignDBError):
        select_design(key, pin="nope:0000000000")
    with pytest.raises(DesignDBError):
        select_design(key, objective="area", metric="asap7")   # no technology scored yet


def test_pareto_front_nondominated(db):
    key, index = _filled_slot(db)
    front = pareto_front(key)
    assert front, "front must not be empty"
    for a in front:
        for b in front:
            if a["design_id"] == b["design_id"]:
                continue
            assert not (b["area"] <= a["area"] and b["delay"] <= a["delay"]
                        and (b["area"] < a["area"] or b["delay"] < a["delay"]))


def test_selection_recorded_in_manifest(db):
    key, _ = _filled_slot(db)
    sel = select_design(key, objective="area", record=True)
    manifest = json.loads((db / VERSION_DIR / "manifest.json").read_text())
    entry = manifest["slots"]["adder8"]
    assert entry["selected_id"] == sel.design_id and entry["metric"] == "transistors"


# --- the decorator -------------------------------------------------------------------------------


def _plain_add(a, b):
    return a + b


@from_design_db                      # bare form
def add8_bare(a, b):
    return a + b


@from_design_db(objective="area")    # parameterized form — same golden, same slot as add8_bare
def add8_area(a, b):
    return a + b


def _build_top(fn):
    m = Netlist("Top", with_clock=False, with_reset=False)
    a = m.input(UInt(8), "a")
    b = m.input(UInt(8), "b")
    y = m.output(UInt(9), "y")
    y <<= fn(a, b)
    return m


def _slot_key_of(db, qualname_suffix):
    manifest = json.loads((db / VERSION_DIR / "manifest.json").read_text())
    hits = [e["spec_key"] for n, e in manifest["slots"].items() if n.endswith(qualname_suffix)]
    assert len(set(hits)) == 1, f"expected one slot for {qualname_suffix}, got {hits}"
    return hits[0]


def test_decorator_miss_uses_original_logic(db):
    top = _build_top(add8_bare)
    plain = _build_top(_plain_add)
    assert AigerExporter(top).get_aag() == AigerExporter(plain).get_aag()
    key = _slot_key_of(db, "add8_bare")
    slot = db / VERSION_DIR / key
    assert (slot / "golden.v").exists() and (slot / "verification.json").exists()
    spec = json.loads((slot / "spec.json").read_text())
    assert spec["source_ref"]["qualname"].endswith("add8_bare")
    assert spec["starting_point"]["fidelity"] == "self-contained"
    sp = (slot / "starting_point.py").read_text()
    assert "class StartingPoint(Component)" in sp and "def add8_bare(a, b):" in sp


CARRY_SAVE_Y_V = """\
module cand_cs(input [7:0] a, input [7:0] b, output [8:0] y);
  assign y = {1'b0, a ^ b} + {a & b, 1'b0};
endmodule
"""


def test_decorator_splices_selected_design(db, tmp_path):
    _build_top(add8_area)                                   # registers the slot (miss)
    key = _slot_key_of(db, "add8_area")
    insert_design(key, CARRY_SAVE_Y_V, source="verilog")    # fill (matches the slot ports a,b -> y)
    top = _build_top(add8_area)                             # fresh build → splice
    plain = _build_top(_plain_add)
    assert AigerExporter(top).get_aag() != AigerExporter(plain).get_aag(), "should differ structurally"
    tv, pv = tmp_path / "top.v", tmp_path / "plain.v"
    tv.write_text(top.to_verilog())
    pv.write_text(plain.to_verilog())
    cec_check(tv, pv, tmp_path / "cec")                     # …but must stay equivalent
    manifest = json.loads((db / VERSION_DIR / "manifest.json").read_text())
    assert any(e.get("selected_id") for e in manifest["slots"].values()
               if e["spec_key"] == key)


def test_decorator_fill_hook_fires_once(db):
    calls = []

    def filler(spec_key, db_root, objective, metric):
        calls.append(spec_key)
        insert_design(spec_key, ADD4_EQUIV, source="fill")

    @from_design_db(objective="area", fill=filler)
    def add4(a, b):
        return a + b

    m = Netlist("T4", with_clock=False, with_reset=False)
    a, b = m.input(UInt(4), "a"), m.input(UInt(4), "b")
    y = m.output(UInt(5), "y")
    y <<= add4(a, b)
    assert len(calls) == 1
    m2 = Netlist("T4b", with_clock=False, with_reset=False)
    a2, b2 = m2.input(UInt(4), "a"), m2.input(UInt(4), "b")
    y2 = m2.output(UInt(5), "y")
    y2 <<= add4(a2, b2)
    assert len(calls) == 1, "fill must not fire again once the slot is populated"


ADD4_EQUIV = """\
module cand4(input [3:0] a, input [3:0] b, output [4:0] y);
  assign y = {1'b0, a} + {1'b0, b};
endmodule
"""


def test_decorator_broken_pin_raises(db):
    @from_design_db(pin="ghost:1234567890")
    def add8_pinned(a, b):
        return a + b

    m = Netlist("TP", with_clock=False, with_reset=False)
    a, b = m.input(UInt(8), "a"), m.input(UInt(8), "b")
    y = m.output(UInt(9), "y")
    with pytest.raises(DesignDBError):
        y <<= add8_pinned(a, b)


MISMATCHED_PORTS_V = """\
module cand_add(input [7:0] a, input [7:0] b, output [8:0] sum);
  assign sum = {1'b0, a} + {1'b0, b};
endmodule
"""


def test_insert_rejects_port_mismatch(db):
    key = register_slot(Adder(), name="adder8")
    from spire.design_db import VerificationFailed
    with pytest.raises(VerificationFailed, match="port mismatch"):
        insert_design(key, MISMATCHED_PORTS_V, source="test")


# --- CLI ------------------------------------------------------------------------------------------


def test_cli_read_commands_do_not_create_db(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv(DB_ENV, raising=False)
    monkeypatch.chdir(tmp_path)
    assert cli_main(["db", "ls"]) == 0
    assert not (tmp_path / "design_db").exists(), "read-only ls must not create a DB"
    assert cli_main(["db", "show", "nothing"]) == 1
    assert not (tmp_path / "design_db").exists()


def test_cli_roundtrip(db, tmp_path, capsys):
    assert cli_main(["db", "init"]) == 0
    key = register_slot(Adder(), name="adder8")
    design = tmp_path / "cand.v"
    design.write_text(EQUIV_V)
    capsys.readouterr()                                      # drain init/registration output
    assert cli_main(["db", "insert", str(design), "--slot", "adder8", "--source", "cli"]) == 0
    inserted = json.loads(capsys.readouterr().out)
    assert inserted["deduped"] is False and inserted["metrics"]["intrinsic"]["aig_nodes"] > 0
    assert cli_main(["db", "ls"]) == 0
    out = capsys.readouterr().out
    assert "adder8" in out and "designs=1" in out
    assert cli_main(["db", "show", "adder8", "--pareto"]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["spec_key"] == key and len(shown["designs"]) == 1 and shown["pareto"]
    assert cli_main(["db", "show", key[:12]]) == 0           # unique key prefix works
    bad = tmp_path / "bad.v"
    bad.write_text(EQUIV_V.replace("+ {1'b0, b}", "+ {1'b0, b} + 9'd1"))
    assert cli_main(["db", "insert", str(bad), "--slot", "adder8"]) == 2
    assert cli_main(["db", "show", "unknown-slot"]) == 1
