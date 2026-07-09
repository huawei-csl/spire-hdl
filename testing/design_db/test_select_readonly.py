"""Readers never create: select_design / pareto_front on a missing DB touch nothing."""
from spire.design_db import pareto_front, select_design
from spire.design_db.store import DB_ENV


def test_select_readers_never_create_a_db(tmp_path, monkeypatch):
    monkeypatch.delenv(DB_ENV, raising=False)
    monkeypatch.chdir(tmp_path)
    assert select_design("0" * 64, objective="area") is None
    assert pareto_front("0" * 64) == []
    assert not (tmp_path / "design_db").exists()      # a query never materializes a DB
