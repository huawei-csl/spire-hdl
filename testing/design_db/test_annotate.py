"""Tests for `annotate` (API + `spire db annotate` CLI): per-technology metric enrichment and
the selection path it unlocks."""
import json

import pytest

from spire.design_db import annotate, insert_design, register_slot, pick_design
from spire.design_db.cli import main as cli_main
from spire.design_db.store import DB_ENV, DesignDBError, VERSION_DIR

from test_s1_core import Adder   # noqa: F401

# A correct adder that is *structurally distinct* from the golden (carry-save), so it admits as a
# second design rather than deduping against the seeded original.
DISTINCT_ADD = """\
module cand_cs(input [7:0] a, input [7:0] b, output [8:0] s);
  assign s = {1'b0, (a ^ b)} + {(a & b), 1'b0};
endmodule
"""


@pytest.fixture
def db(tmp_path, monkeypatch):
    root = tmp_path / "design_db"
    monkeypatch.setenv(DB_ENV, str(root))
    return root


def _seeded_slot(db):
    """A slot with two distinct admitted designs (the golden as original + a carry-save candidate)."""
    key = register_slot(Adder(), name="adder8")
    from spire.design_db import seed_original
    seed_original(key)
    cand = insert_design(key, DISTINCT_ADD, source="agent")
    assert not cand.deduped, "candidate must be structurally distinct from the golden"
    return key, cand.design_id


ASAP7_BLOCK = {"metrics": {"area": 118.3, "delay": 94.6},
               "objectives": {"area": "area", "delay": "delay"}}


def test_annotate_adds_block_and_mirrors(db):
    key, design_id = _seeded_slot(db)
    slot = db / VERSION_DIR / key
    before = json.loads((slot / "designs" / design_id / "metrics.json").read_text())
    assert "asap7" not in before and "transistors" in before          # gate stamped aig+transistors

    res = annotate(key, design_id, tech="asap7", values={"area": 118.3, "delay": 94.6})
    assert res["metrics"]["asap7"] == ASAP7_BLOCK                      # self-describing block

    after = json.loads((slot / "designs" / design_id / "metrics.json").read_text())
    assert after["asap7"] == ASAP7_BLOCK
    assert after["transistors"] == before["transistors"]              # gate stamp untouched
    index = json.loads((slot / "index.json").read_text())
    assert index[design_id]["metrics"]["asap7"] == ASAP7_BLOCK        # mirror


def test_stamped_shape_and_borrow(db):
    """The gate stamps self-describing blocks; the transistors system borrows aig.aig_depth for
    delay (explicit path, not a duplicated value), and metric_value reads it generically."""
    from spire.design_db.select import metric_value
    key, design_id = _seeded_slot(db)
    m = json.loads((db / VERSION_DIR / key / "designs" / design_id / "metrics.json").read_text())
    assert set(m["aig"]["metrics"]) == {"aig_nodes", "aig_depth", "aig_latches"}
    assert m["aig"]["objectives"] == {"area": "aig_nodes", "delay": "aig_depth"}
    assert m["transistors"]["objectives"] == {"area": "transistors_heavy", "delay": "aig.aig_depth"}
    # generic reader: transistors area = own field; transistors delay = borrowed aig depth
    assert metric_value(m, "area", "transistors") == m["transistors"]["metrics"]["transistors_heavy"]
    assert metric_value(m, "delay", "transistors") == m["aig"]["metrics"]["aig_depth"]
    # adp is derived (area·delay) with no stored adp field
    assert metric_value(m, "adp", "aig") == m["aig"]["metrics"]["aig_nodes"] * m["aig"]["metrics"]["aig_depth"]
    assert metric_value(m, "edap", "aig") is None                 # unmapped, not derivable


def test_annotate_sibling_consistency(db):
    """One slot never mixes interpretations: if a sibling design already draws an objective from a
    different field, annotating another design for that system is refused."""
    from spire.design_db.store import DesignDB
    key, cand_id = _seeded_slot(db)
    d = DesignDB.open(db)
    idx_path = d.slot_dir(key) / "index.json"
    original_id = next(i for i in d.read_json(idx_path, {}) if i.startswith("original:"))

    annotate(key, cand_id, tech="asap7", values={"area": 100.0, "delay": 88.0})
    annotate(key, original_id, tech="asap7", values={"area": 120.0, "delay": 90.0})  # same map: fine

    # plant a conflicting interpretation (area drawn from a different field) in the design's own
    # metrics.json (the source of truth — index.json is only a derived cache), then a normal
    # annotate of the sibling must refuse rather than silently mix interpretations
    mfile = d.slot_dir(key) / "designs" / cand_id / "metrics.json"
    metrics = d.read_json(mfile)
    metrics["asap7"]["objectives"]["area"] = "footprint"
    d.write_json(mfile, metrics)
    with pytest.raises(DesignDBError, match="disagrees with design"):
        annotate(key, original_id, tech="asap7", values={"area": 111.0, "delay": 91.0}, force=True)


def test_annotate_guards(db):
    key, design_id = _seeded_slot(db)
    with pytest.raises(DesignDBError, match="reserved"):
        annotate(key, design_id, tech="transistors", values={"area": 1.0})
    with pytest.raises(DesignDBError, match="numeric"):
        annotate(key, design_id, tech="asap7", values={"area": "big"})
    annotate(key, design_id, tech="asap7", values={"area": 100.0})
    with pytest.raises(DesignDBError, match="already has a 'asap7'"):
        annotate(key, design_id, tech="asap7", values={"area": 999.0})
    annotate(key, design_id, tech="asap7", values={"area": 90.0}, force=True)    # overwrite ok
    with pytest.raises(DesignDBError, match="unknown design"):
        annotate(key, "nope:1234", tech="asap7", values={"area": 1.0})


def test_annotate_unlocks_metric_selection(db):
    """After annotating both designs, metric='asap7' selects on the annotated area — which can
    disagree with the transistor default."""
    key, cand_id = _seeded_slot(db)
    from spire.design_db.store import DesignDB
    d = DesignDB.open(db)
    original_id = next(i for i in d.read_json(d.slot_dir(key) / "index.json", {})
                       if i.startswith("original:"))

    # transistors: the two adders tie (same structure) → original wins the tie-break floor.
    # asap7: make the candidate strictly better, so metric='asap7' must pick it.
    annotate(key, original_id, tech="asap7", values={"area": 120.0, "delay": 90.0})
    annotate(key, cand_id, tech="asap7", values={"area": 100.0, "delay": 88.0})
    sel = pick_design(key, objective="area", metric="asap7", db=db)
    assert sel.design_id == cand_id and sel.metric == "asap7"

    with pytest.raises(DesignDBError, match="not available"):
        pick_design(key, objective="area", metric="nosuchtech", db=db)


def test_cli_annotate(db, capsys):
    key, design_id = _seeded_slot(db)
    rc = cli_main(["db", "annotate", "--slot", "adder8", "--design", design_id[:16],
                   "--tech", "asap7", "area=118.3", "delay=94.6", "adp=11190.2"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["design_id"] == design_id and out["tech"] == "asap7"
    assert out["metrics"]["asap7"] == {
        "metrics": {"area": 118.3, "delay": 94.6, "adp": 11190.2},
        "objectives": {"area": "area", "delay": "delay", "adp": "adp"}}
    # a non-numeric pair is a clean CLI error (exit 1, not a traceback)
    rc = cli_main(["db", "annotate", "--slot", "adder8", "--design", design_id[:16],
                   "--tech", "asap7", "area=oops"])
    assert rc == 1 and "must be numeric" in capsys.readouterr().err
