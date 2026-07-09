"""The gate on a plain-pip environment (no yosys / yosys-abc binaries — e.g. CI): AAG
conversion runs through pyosys in a child interpreter, Tier-0 CEC through yosys' own
``equiv_make``/``equiv_simple`` flow. A bad candidate must be a clean rejection — never a dead
host process (pyosys ``log_error`` exits, which is why the child interpreter exists)."""
import shutil

import pytest

pytest.importorskip("pyosys", reason="the no-binary path needs the pyosys wheel")

from spire import UInt
from spire.component import Netlist
from spire.design_db import (VerificationFailed, insert_design, register_slot, seed_original,
                             pick_design)
from spire.design_db.store import DB_ENV


@pytest.fixture
def db(tmp_path, monkeypatch):
    root = tmp_path / "design_db"
    monkeypatch.setenv(DB_ENV, str(root))
    return root


@pytest.fixture
def no_binaries(monkeypatch):
    """Hide the yosys / yosys-abc binaries from the modules that probe for them."""
    import spire.design_db._yosys as yos
    import spire.design_db.verify as ver
    real = shutil.which

    def fake(name, *a, **kw):
        return None if name in ("yosys", "yosys-abc") else real(name, *a, **kw)

    monkeypatch.setattr(yos.shutil, "which", fake)
    monkeypatch.setattr(ver.shutil, "which", fake)


def _adder_slot():
    m = Netlist("adder", with_clock=False, with_reset=False)
    a = m.input(UInt(8), "a")
    b = m.input(UInt(8), "b")
    y = m.output(UInt(8), "y")
    y <<= a + b
    return register_slot(m)


# Equivalent carry-save form — with the ports deliberately declared in a DIFFERENT order than
# the golden: the equiv fallback must pair ports by NAME (like abc's named-BLIF cec), not index.
CARRY_SAVE_SWAPPED_PORTS = """\
module cand(input [7:0] b, input [7:0] a, output [7:0] y);
  wire [8:0] cs = {1'b0, a ^ b} + {(a & b), 1'b0};
  assign y = cs[7:0];
endmodule
"""

WRONG = """\
module cand(input [7:0] a, input [7:0] b, output [7:0] y);
  assign y = a - b;
endmodule
"""


def test_gate_without_binaries(db, no_binaries):
    key = _adder_slot()
    seeded = seed_original(key)                    # golden: pyosys-child AAG + equiv CEC vs itself
    assert seeded.design_id.startswith("original:")
    res = insert_design(key, CARRY_SAVE_SWAPPED_PORTS, source="verilog")
    assert not res.deduped                         # structurally new, name-matched ports, admitted
    with pytest.raises(VerificationFailed):        # clean rejection — the host process survives
        insert_design(key, WRONG, source="verilog")
    again = insert_design(key, CARRY_SAVE_SWAPPED_PORTS, source="verilog")
    assert again.deduped
    assert pick_design(key, objective="area") is not None
