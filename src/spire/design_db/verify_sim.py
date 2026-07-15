"""Tier-1/2 sim verification: stimulus → golden-simulated vectors → a frozen tb — and the gate check.

**Semantics: cycle-accurate trace equivalence under the frozen stimulus.** Expected outputs are
always produced by simulating the golden (Verilator); a candidate must reproduce the golden's
output trace exactly (for sequential slots cycle for cycle — a re-pipelined design with different
latency is rejected, by design for v1). Combinational slots may freeze a sim verification too —
the explicit caller choice after a CEC timeout.

Stimulus is **auto** (per-input corners + seeded random; exhaustive for tiny combinational input
spaces) or a **human-authored generator** (the non-agentic Tier-2 path): a Python file defining
``generate(ports, n_vectors, seed) -> iterable of {input_name: int}``.

The frozen artifacts (``tb.sv`` + ``vectors.dat``) follow the rtlscout testbench contract
(``module tb`` … ``TB_SUMMARY total=N errors=M`` … ``PASS``), so external fillers can reuse them
directly. The DUT module name is bound at compile time via the ``DUT`` macro (``-DDUT=<top>``).
"""
from __future__ import annotations

import itertools
import random
import re
import runpy
import shutil
import stat
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from spire.design_db.store import DesignDB, DesignDBError
from spire.design_db.verify import VerificationError, VerificationFailed

DEFAULT_N_VECTORS = 256
DEFAULT_SIM_BUDGET_S = 300.0
EXHAUSTIVE_BITS = 12
_VL_FLAGS = ["--sv", "-Wno-fatal", "-Wno-lint"]


class SimTimeout(VerificationError):
    """Verilator build/run blew past its budget. Raise ``--sim-budget`` or trim the stimulus."""


# --- ports & stimulus -----------------------------------------------------------------------


def _ports_split(spec: Dict[str, Any]) -> Tuple[List[dict], List[dict], Optional[str], Optional[str]]:
    clock = spec.get("clock") or {}
    clk, rst = clock.get("clk"), clock.get("rst")
    ins = [p for p in spec["ports"] if p["dir"] == "input" and p["name"] not in (clk, rst)]
    outs = [p for p in spec["ports"] if p["dir"] == "output"]
    return ins, outs, clk, rst


def _mask(v: int, w: int) -> int:
    return v & ((1 << w) - 1)


def generate_auto_stimulus(ins: List[dict], n_vectors: int, seed: int,
                           sequential: bool) -> List[Dict[str, int]]:
    """Corners + seeded random; exhaustive when a combinational input space is tiny."""
    total_bits = sum(p["width"] for p in ins)
    if not sequential and total_bits <= EXHAUSTIVE_BITS:
        ranges = [range(1 << p["width"]) for p in ins]
        return [{p["name"]: v for p, v in zip(ins, combo)}
                for combo in itertools.product(*ranges)]
    corners: Dict[str, List[int]] = {}
    for p in ins:
        w = p["width"]
        vals = [0, _mask(-1, w), 1, _mask(-2, w),
                _mask(0xAAAAAAAAAAAAAAAA, w), _mask(0x5555555555555555, w)]
        corners[p["name"]] = list(dict.fromkeys(vals))       # dedup, keep order
    n_corner = max(len(v) for v in corners.values())
    vectors = [{name: vals[i % len(vals)] for name, vals in corners.items()}
               for i in range(n_corner)]
    rng = random.Random(seed)
    while len(vectors) < max(n_vectors, n_corner):
        vectors.append({p["name"]: rng.randrange(1 << p["width"]) for p in ins})
    return vectors[:max(n_vectors, n_corner)]


def load_stimulus_file(path: Path, ins: List[dict], n_vectors: int,
                       seed: int) -> List[Dict[str, int]]:
    ns = runpy.run_path(str(path))
    gen = ns.get("generate")
    if not callable(gen):
        raise DesignDBError(f"stimulus file {path} must define generate(ports, n_vectors, seed)")
    vectors = [{p["name"]: _mask(int(vec[p["name"]]), p["width"]) for p in ins}
               for vec in gen([dict(p) for p in ins], n_vectors, seed)]
    if not vectors:
        raise DesignDBError("stimulus generator produced no vectors")
    return vectors


def check_stimulus(spec_key: str, *, stimulus_file: str | Path,
                   n_vectors: int = DEFAULT_N_VECTORS, seed: int = 0,
                   db: Optional[Any] = None) -> Dict[str, Any]:
    """Dry-run an authored stimulus generator against a slot's interface — the cheap front half
    of a ``--stimulus`` freeze with **no side effects**: nothing is simulated, written, or
    frozen. The iteration aid for stimulus authors, since the freeze itself is one-shot."""
    d = DesignDB.open(db, create=False)
    slot = d.slot_dir(spec_key)
    spec = d.read_json(slot / "spec.json", None)
    if spec is None:
        raise DesignDBError(f"unknown slot {spec_key[:12]}… — register it first")
    ins, outs, _clk, _rst = _ports_split(spec)
    if not ins or not outs:
        raise DesignDBError("slot has no data inputs/outputs to stimulate")
    try:
        vectors = load_stimulus_file(Path(stimulus_file), ins, n_vectors, seed)
    except DesignDBError:
        raise
    except Exception as exc:            # generator bugs surface as a clean check verdict
        raise DesignDBError(f"stimulus check failed: {type(exc).__name__}: "
                            f"{str(exc).splitlines()[0][:300]}") from exc
    return {"check": "ok", "n_vectors": len(vectors),
            "data_inputs": [p["name"] for p in ins],
            "note": "generator loads and produces masked vectors (clk/rst are driven by the "
                    "testbench, not the generator); freeze with: spire db verify --slot <key> "
                    "--stimulus <file> [--author …]"}


# --- testbench generation -------------------------------------------------------------------


def _decl(name: str, w: int) -> str:
    return f"  logic {name};" if w == 1 else f"  logic [{w - 1}:0] {name};"


def _tb_text(mode: str, ins: List[dict], outs: List[dict], clk: Optional[str],
             rst: Optional[str]) -> str:
    """One template, two modes: ``gen`` (drive golden, record outputs) / ``check`` (compare)."""
    seq = clk is not None
    L: List[str] = ["module tb;", "  int total_checks;", "  int total_errors;", ""]
    for p in ins:
        L.append(_decl(p["name"], p["width"]))
    for p in outs:
        L.append(_decl(p["name"], p["width"]))
    if mode == "check":
        for p in outs:
            L.append(_decl("expected_" + p["name"], p["width"]))
    if seq:
        L.append("  logic clk_i, rst_i;")
        L.append("  always #5 clk_i = ~clk_i;")
    L.append("")
    ports = [f".{p['name']}({p['name']})" for p in ins + outs]
    if seq:
        ports.append(f".{clk}(clk_i)")
        if rst:
            ports.append(f".{rst}(rst_i)")
    L.append(f"  `DUT dut ({', '.join(ports)});")
    L += ["", "  integer fd, rc, line_num;", "  string line_buf;"]
    if mode == "gen":
        L.append("  integer ofd;")
    L += ["", "  initial begin", "    total_checks = 0;", "    total_errors = 0;"]
    if seq:
        L += ["    clk_i = 0;"] + ([f"    rst_i = 1;"] if rst else [])
        L += [f"    {p['name']} = 0;" for p in ins]
        L += ["    repeat (2) @(posedge clk_i);", "    #1;"]
        if rst:
            L.append("    rst_i = 0;")
    src = "inputs.dat" if mode == "gen" else "vectors.dat"
    L += [f'    fd = $fopen("{src}", "r");',
          "    if (fd == 0) begin",
          f'      $display("ERROR: cannot open {src}");',
          "      $fatal(1);",
          "    end"]
    if mode == "gen":
        L += ['    ofd = $fopen("vectors.dat", "w");']
    read_names = [p["name"] for p in ins] + \
        (["expected_" + p["name"] for p in outs] if mode == "check" else [])
    fmt = " ".join(["%d"] * len(read_names))
    L += ["    line_num = 0;",
          "    while (!$feof(fd)) begin",
          "      line_num = line_num + 1;",
          "      void'($fgets(line_buf, fd));",
          "      if (line_buf.len() == 0) continue;",
          '      if (line_buf.substr(0, 0) == "#") continue;',
          f'      rc = $sscanf(line_buf, "{fmt}", {", ".join(read_names)});',
          f"      if (rc != {len(read_names)}) continue;"]
    if seq:
        L += ["      @(negedge clk_i);", "      @(posedge clk_i);", "      #1;"]
    else:
        L += ["      #1;"]
    L.append("      total_checks = total_checks + 1;")
    if mode == "gen":
        wfmt = " ".join(["%0d"] * (len(ins) + len(outs)))
        wargs = ", ".join([p["name"] for p in ins] + [p["name"] for p in outs])
        L.append(f'      $fdisplay(ofd, "{wfmt}", {wargs});')
    else:
        for p in outs:
            n = p["name"]
            L += [f"      if ({n} !== expected_{n}) begin",
                  f'        $display("TB_ERROR line=%0d expected_{n}=%0d actual_{n}=%0d", '
                  f"line_num, expected_{n}, {n});",
                  "        total_errors = total_errors + 1;",
                  "      end"]
    L += ["    end", "    $fclose(fd);"]
    if mode == "gen":
        L.append("    $fclose(ofd);")
    L += ['    $display("TB_SUMMARY total=%0d errors=%0d", total_checks, total_errors);',
          '    if (total_errors != 0) $fatal(1, "FAIL");',
          '    $display("PASS");',
          "    $finish;",
          "  end",
          "endmodule"]
    return "\n".join(L) + "\n"


# --- verilator ------------------------------------------------------------------------------


def _top_module_name(verilog_text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", verilog_text, flags=re.S)
    for line in text.splitlines():
        m = re.match(r"\s*module\s+([A-Za-z_]\w*)", line.split("//")[0])
        if m:
            return m.group(1)
    raise VerificationError("could not find a module declaration in the design")


def _sub(args: List[str], cwd: Path, budget_s: float, what: str) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(args, cwd=str(cwd), capture_output=True, text=True,
                              timeout=budget_s)
    except subprocess.TimeoutExpired:
        raise SimTimeout(f"{what} timed out after {budget_s:g} s — raise the sim budget "
                         f"(--sim-budget / verification.json sim_budget_s)") from None


def _run_verilator(sources: List[Path], workdir: Path, dut_top: str, budget_s: float) -> str:
    if shutil.which("verilator") is None:
        raise VerificationError("verilator not found on PATH — sim verification unavailable")
    args = (["verilator", "--binary", "--top-module", "tb", "-o", "simv"] + _VL_FLAGS
            + [f"-DDUT={dut_top}"] + [str(s.resolve()) for s in sources])
    build = _sub(args, workdir, budget_s, "verilator build")
    if build.returncode != 0:
        raise VerificationError("verilator build failed:\n" + (build.stdout + build.stderr)[-800:])
    run = _sub([str(workdir / "obj_dir" / "simv")], workdir, budget_s, "simulation")
    out = run.stdout + run.stderr
    # A tb $fatal exits non-zero but still prints TB_SUMMARY; anything else non-zero is a crash.
    if run.returncode != 0 and "TB_SUMMARY" not in out:
        raise VerificationError(f"simulation crashed (rc={run.returncode}):\n"
                                + (out[-800:] or "(no output)"))
    return out


def _parse_summary(out: str) -> Tuple[int, int, bool]:
    m = re.search(r"TB_SUMMARY total=(\d+) errors=(\d+)", out)
    if not m:
        raise VerificationError("simulation produced no TB_SUMMARY:\n" + out[-800:])
    total, errors = int(m.group(1)), int(m.group(2))
    return total, errors, errors == 0 and "PASS" in out


# --- freeze + gate check --------------------------------------------------------------------


def freeze_sim_verification(spec_key: str, *, stimulus_file: Optional[str | Path] = None,
                            n_vectors: int = DEFAULT_N_VECTORS, seed: int = 0,
                            sim_budget_s: float = DEFAULT_SIM_BUDGET_S,
                            stimulus_author: Optional[str] = None,
                            db: Optional[Any] = None) -> Dict[str, Any]:
    """Build + freeze a sim verification for a slot: golden-simulated ``vectors.dat`` + ``tb.sv``.

    Tier 1 with auto stimulus, Tier 2 with an authored generator file. ``stimulus_author``
    records who authored the generator (``None`` when not given — no assumed identity; agent
    layers pass e.g. ``"agent:rtl-dv-prep"``) — it only applies to a ``stimulus_file`` freeze;
    Tier-1 freezes always record ``"auto"``. Immutable once
    frozen (a re-freeze would silently change the oracle designs were admitted against).
    """
    d = DesignDB.open(db)
    slot = d.slot_dir(spec_key)
    spec = d.read_json(slot / "spec.json", None)
    if spec is None:
        raise DesignDBError(f"unknown slot {spec_key[:12]}… — register it first")
    existing = d.read_json(slot / "verification.json", None)
    if existing is not None and int(existing.get("tier", 0)) >= 1:
        raise DesignDBError("this slot's sim verification is already frozen (immutable — a "
                            "re-freeze would change the oracle admitted designs were checked "
                            "against)")
    ins, outs, clk, rst = _ports_split(spec)
    if not ins or not outs:
        raise DesignDBError("slot has no data inputs/outputs to stimulate")
    if spec.get("class") == "sequential" and clk is None:
        raise DesignDBError("sequential slot without recorded clock info — re-register the slot")
    sequential = clk is not None

    authored = stimulus_file is not None
    if stimulus_author is not None and not authored:
        raise DesignDBError("stimulus_author only applies to an authored (stimulus-file) freeze — "
                            "auto stimulus is always recorded as \"auto\"")
    author = stimulus_author if authored else "auto"
    if authored:
        vectors = load_stimulus_file(Path(stimulus_file), ins, n_vectors, seed)
    else:
        vectors = generate_auto_stimulus(ins, n_vectors, seed, sequential)

    golden_text = (slot / "golden.v").read_text()
    dut_top = _top_module_name(golden_text)
    with tempfile.TemporaryDirectory(prefix="spire_ddb_sim_") as td:
        w = Path(td)
        (w / "golden.v").write_text(golden_text)
        (w / "inputs.dat").write_text(
            "# " + " ".join(p["name"] for p in ins) + "\n"
            + "\n".join(" ".join(str(v[p["name"]]) for p in ins) for v in vectors) + "\n")
        (w / "tb_gen.sv").write_text(_tb_text("gen", ins, outs, clk, rst))
        out = _run_verilator([w / "tb_gen.sv", w / "golden.v"], w, dut_top, sim_budget_s)
        total, _errors, passed = _parse_summary(out)
        if not passed or total != len(vectors):
            raise VerificationError(f"golden simulation failed while recording the trace "
                                    f"({total}/{len(vectors)} vectors):\n" + out[-800:])
        d.atomic_write_text(slot / "vectors.dat", (w / "vectors.dat").read_text())
    d.atomic_write_text(slot / "tb.sv", _tb_text("check", ins, outs, clk, rst))
    for f in (slot / "tb.sv", slot / "vectors.dat"):                # frozen = read-only
        f.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)

    verification = {"schema": 1, "tier": 2 if authored else 1, "method": "sim",
                    "stimulus_author": author, "n_vectors": len(vectors),
                    "seed": None if authored else seed, "sim_budget_s": sim_budget_s,
                    "sequential": sequential,
                    "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    d.write_json(slot / "verification.json", verification)
    return verification


def run_frozen_tb(spec_key: str, candidate_v: Path, workdir: Path, *,
                  db: Optional[Any] = None, budget_s: Optional[float] = None) -> None:
    """The sim-tier gate check: run the slot's frozen tb against a candidate; raise on non-PASS."""
    d = DesignDB.open(db)
    slot = d.slot_dir(spec_key)
    tb, vectors = slot / "tb.sv", slot / "vectors.dat"
    if not tb.exists() or not vectors.exists():
        raise VerificationError("slot has no frozen sim verification artifacts (tb.sv/vectors.dat)")
    budget = float(budget_s) if budget_s is not None else \
        float((d.read_json(slot / "verification.json", {}) or {}).get("sim_budget_s",
                                                                      DEFAULT_SIM_BUDGET_S))
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(vectors, workdir / "vectors.dat")
    dut_top = _top_module_name(Path(candidate_v).read_text())
    out = _run_verilator([tb, Path(candidate_v)], workdir, dut_top, budget)
    total, errors, passed = _parse_summary(out)
    if not passed:
        detail = "\n".join(line for line in out.splitlines() if "TB_ERROR" in line)[:600]
        raise VerificationFailed(f"candidate fails the frozen trace: {errors}/{total} vector "
                                 f"mismatches\n{detail}")
