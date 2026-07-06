"""S3 tests: the Tier-1/2 sim verification harness + the sim-tier gate path.

Tool-real: verilator (and yosys for the gate's AAG/dedup path) must be on PATH.
"""
import json
import os
import stat

import pytest

from spire import UInt
from spire.component import Netlist
from spire.design_db import (DesignDBError, VerificationFailed, freeze_sim_verification,
                             insert_design, register_slot)
from spire.design_db.cli import main as cli_main
from spire.design_db.store import DB_ENV, VERSION_DIR

from test_s1_core import Adder, EQUIV_V, WRONG_V


@pytest.fixture
def db(tmp_path, monkeypatch):
    root = tmp_path / "design_db"
    monkeypatch.setenv(DB_ENV, str(root))
    return root


def _seq_module():
    m = Netlist("seqm", with_clock=True, with_reset=True)
    din = m.input(UInt(4), "din")
    q = m.reg(UInt(4), "q", init=0)
    q <<= din
    dout = m.output(UInt(4), "dout")
    dout <<= q
    return m


CORRECT_SEQ_V = """\
module cand_seq(input clk, input rst, input [3:0] din, output [3:0] dout);
  reg [3:0] q;
  always @(posedge clk or posedge rst)
    if (rst) q <= 4'd0;
    else q <= din;
  assign dout = q;
endmodule
"""

WRONG_SEQ_V = CORRECT_SEQ_V.replace("assign dout = q;", "assign dout = q ^ 4'd1;")


def _slot(db, key):
    return db / VERSION_DIR / key


def test_comb_switch_to_sim_and_gate(db):
    """Combinational slot: explicit CEC→sim switch; the gate then checks via the frozen trace."""
    key = register_slot(Adder(), name="adder8")
    ver = freeze_sim_verification(key, n_vectors=64, seed=1)
    assert ver["tier"] == 1 and ver["method"] == "sim" and ver["stimulus_author"] == "auto"
    slot = _slot(db, key)
    assert (slot / "tb.sv").exists() and (slot / "vectors.dat").exists()
    res = insert_design(key, EQUIV_V, source="test")
    prov = json.loads((slot / "designs" / res.design_id / "provenance.json").read_text())
    assert prov["verification"]["method"] == "sim"
    with pytest.raises(VerificationFailed, match="vector mismatches"):
        insert_design(key, WRONG_V, source="test")


def test_sequential_freeze_and_gate(db):
    key = register_slot(_seq_module())
    spec = json.loads((_slot(db, key) / "spec.json").read_text())
    assert spec["class"] == "sequential" and spec["clock"]["clk"] == "clk"
    ver = freeze_sim_verification(key, n_vectors=48, seed=7)
    assert ver["tier"] == 1 and ver["sequential"] is True
    res = insert_design(key, CORRECT_SEQ_V, source="test")
    assert res.metrics["aig"]["metrics"]["aig_latches"] > 0
    with pytest.raises(VerificationFailed, match="vector mismatches"):
        insert_design(key, WRONG_SEQ_V, source="test")


def test_frozen_verification_is_immutable(db):
    key = register_slot(Adder(), name="adder8")
    freeze_sim_verification(key, n_vectors=32)
    slot = _slot(db, key)
    for name in ("tb.sv", "vectors.dat"):
        mode = (slot / name).stat().st_mode
        assert not (mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)), f"{name} is writable"
    with pytest.raises(DesignDBError, match="immutable"):
        freeze_sim_verification(key, n_vectors=32)


def test_human_stimulus_is_tier_2(db, tmp_path):
    key = register_slot(_seq_module())
    stim = tmp_path / "stim.py"
    stim.write_text(
        "def generate(ports, n_vectors, seed):\n"
        "    for i in range(n_vectors):\n"
        "        yield {p['name']: i * 3 + 1 for p in ports}\n")
    ver = freeze_sim_verification(key, stimulus_file=stim, n_vectors=40)
    assert ver["tier"] == 2 and ver["stimulus_author"] == "human" and ver["n_vectors"] == 40
    insert_design(key, CORRECT_SEQ_V, source="test")


def test_stimulus_author_attribution(db, tmp_path, capsys):
    """--author / stimulus_author= records who wrote the generator (still Tier 2); it is
    rejected outside authored freezes — auto stimulus is always recorded as "auto"."""
    stim = tmp_path / "stim.py"
    stim.write_text(
        "def generate(ports, n_vectors, seed):\n"
        "    for i in range(n_vectors):\n"
        "        yield {p['name']: i * 3 + 1 for p in ports}\n")

    key = register_slot(_seq_module())
    with pytest.raises(DesignDBError, match="stimulus_author only applies"):
        freeze_sim_verification(key, n_vectors=32, stimulus_author="agent:rtl-dv-prep")
    ver = freeze_sim_verification(key, stimulus_file=stim, n_vectors=32,
                                  stimulus_author="agent:rtl-dv-prep")
    assert ver["tier"] == 2 and ver["stimulus_author"] == "agent:rtl-dv-prep"
    assert ver["seed"] is None                          # authored freeze: seed not recorded

    # CLI: --author refused outside --stimulus (guard fires before any state change) …
    assert cli_main(["db", "set-verification", "--slot", key[:12], "--auto", "--author", "agent:x"]) == 1
    assert "--author only applies" in capsys.readouterr().err
    # … and flows through on --stimulus
    key2 = register_slot(Adder(), name="adder_auth")
    assert cli_main(["db", "set-verification", "--slot", key2[:12], "--stimulus", str(stim),
                     "--vectors", "16", "--author", "agent:rtl-dv-prep"]) == 0
    assert json.loads(capsys.readouterr().out)["stimulus_author"] == "agent:rtl-dv-prep"


def test_stimulus_check_dry_run(db, tmp_path, capsys):
    """--check / check_stimulus(): the cheap front half of a --stimulus freeze, no side effects
    — so the one-shot freeze is preserved for after the stimulus is worth committing."""
    from spire.design_db import check_stimulus

    key = register_slot(_seq_module())
    stim = tmp_path / "stim.py"
    stim.write_text(
        "def generate(ports, n_vectors, seed):\n"
        "    for i in range(n_vectors):\n"
        "        yield {p['name']: i for p in ports}\n")
    res = check_stimulus(key, stimulus_file=stim, n_vectors=24)
    assert res["check"] == "ok" and res["n_vectors"] == 24 and res["data_inputs"] == ["din"]
    slot = _slot(db, key)
    for name in ("verification.json", "vectors.dat", "tb.sv"):
        assert not (slot / name).exists(), f"--check must write nothing ({name})"

    bad = tmp_path / "bad.py"
    bad.write_text("def generate(ports, n_vectors, seed):\n    raise RuntimeError('boom')\n")
    with pytest.raises(DesignDBError, match="stimulus check failed: RuntimeError"):
        check_stimulus(key, stimulus_file=bad)

    # CLI: --check requires --stimulus; a passing check does not consume the one-shot freeze
    assert cli_main(["db", "set-verification", "--slot", key[:12], "--check"]) == 1
    assert "requires --stimulus" in capsys.readouterr().err
    assert cli_main(["db", "set-verification", "--slot", key[:12], "--stimulus", str(stim),
                     "--check"]) == 0
    assert json.loads(capsys.readouterr().out)["check"] == "ok"
    ver = freeze_sim_verification(key, stimulus_file=stim, n_vectors=24)
    assert ver["tier"] == 2                                    # still freezable after checks


def test_cli_verify_fail_and_choose(db, capsys):
    seq_key = register_slot(_seq_module())
    # sequential slot, no explicit method → clean failure listing the options
    assert cli_main(["db", "set-verification", "--slot", seq_key[:12]]) == 1
    assert "--auto" in capsys.readouterr().err
    # explicit --cec on sequential → guardrail
    assert cli_main(["db", "set-verification", "--slot", seq_key[:12], "--cec"]) == 1
    assert "inapplicable" in capsys.readouterr().err
    # explicit --auto → freezes tier 1
    assert cli_main(["db", "set-verification", "--slot", seq_key[:12], "--auto", "--vectors", "32"]) == 0
    assert json.loads(capsys.readouterr().out)["tier"] == 1
    # combinational slot: bare verify defaults to CEC (the default-picker)
    register_slot(Adder(), name="adder8")
    assert cli_main(["db", "set-verification", "--slot", "adder8"]) == 0
    assert json.loads(capsys.readouterr().out)["method"] == "cec"
    # sim-frozen slots are immutable, even against --cec
    assert cli_main(["db", "set-verification", "--slot", seq_key[:12], "--auto"]) == 1
    assert "immutable" in capsys.readouterr().err


def test_cli_verify_advisory(db, tmp_path, capsys):
    """`db verify <design>` runs the *set* oracle against a candidate — no admit, no write — the
    same check `db insert` gates on."""
    from spire.design_db.store import DesignDB
    key = register_slot(Adder(), name="adder8")          # combinational → CEC by default
    good = tmp_path / "good.v"; good.write_text(EQUIV_V)
    bad = tmp_path / "bad.v"; bad.write_text(WRONG_V)

    assert cli_main(["db", "verify", str(good), "--slot", "adder8"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["verdict"] == "PASS" and out["method"] == "cec"
    assert cli_main(["db", "verify", str(bad), "--slot", "adder8"]) == 2
    assert json.loads(capsys.readouterr().out)["verdict"] == "FAIL"
    assert DesignDB.open(db).read_json(_slot(db, key) / "index.json", {}) == {}   # nothing admitted

    # a slot with no oracle set → a setup error, not a candidate verdict
    seq_key = register_slot(_seq_module())
    assert cli_main(["db", "verify", str(good), "--slot", seq_key[:12]]) == 1
    assert "set-verification" in capsys.readouterr().err
