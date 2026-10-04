"""Signal names inferred from Python variables must not collide with SystemVerilog keywords."""
import shutil
import subprocess

from spire import Component, IORecord, Input, Output
from spire.expr import UInt, reset_shared_cache
from spire.signal_name_inference import VERILOG_KEYWORDS


def test_sv_keywords_are_reserved():
    for kw in ("inside", "class", "string", "type", "with", "bind", "import", "package", "this", "ref"):
        assert kw in VERILOG_KEYWORDS


class _Kw(Component):
    def __init__(self):
        self.io = IORecord(a=Input(UInt(4)), y=Output(UInt(4)))
        self.elaborate()

    def elaborate(self):
        inside = self.io.a < 9          # Python variable named like an SV keyword
        string = self.io.a ^ 3
        self.io.y <<= (string + inside)[0:4]


def test_keyword_named_variables_emit_legal_sv(tmp_path):
    reset_shared_cache()
    text = _Kw().to_verilog(name="kw")
    assert " inside;" not in text and " string;" not in text, text
    if shutil.which("iverilog"):
        src = tmp_path / "kw.v"
        src.write_text(text)
        r = subprocess.run(["iverilog", "-g2012", "-o", str(tmp_path / "a.out"), str(src)], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr + text
