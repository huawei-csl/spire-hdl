# roundtrip_suite.py
from __future__ import annotations
import os, tempfile, random
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

# --- your libs (adjust paths as needed) ---
from aigverse import equivalence_checking, simulate, write_aiger
from spire.aig.aig_aigerverse import _get_aag_sym, file_to_lines, read_aag_into_aig, conv_aag_into_aig
from spire.helpers import run_vectors
from spire.expr import UInt, Bool, reset_shared_cache
from spire.aiger import AigerExporter, AigerImporter
from spire.component import Module
from spire.simulator import Simulator
from spire.component import IOCollector
from spire.aig.aig_yosys import verilog_to_aag_via_yosys
from spire.arithmetic.floating_point.fp_encoding import fp_decode
from spire.arithmetic.floating_point.fp_mul_testvectors import build_fp_vectors  # generic EW,FW vectors/decoder
from spire.arithmetic.floating_point.spire_hdl_float_mult_sn import build_fp_mul_sn

# Pyosys

# -----------------------------
# Utilities
# -----------------------------


def write_temp_verilog(m: Module, top_name: str | None = None) -> str:
    """Write module Verilog to a temporary file and return the path."""
    if top_name and m.name != top_name:
        # rename for convenience
        v = m.to_verilog().replace(f"module {m.name} ", f"module {top_name} ")
    else:
        v = m.to_verilog()
    fd, path = tempfile.mkstemp(suffix=".v")
    os.close(fd)
    with open(path, "w") as f:
        f.write(v)
    return path


def spire_to_aig_via_exporter(m: Module):
    """SpireHDL → AAGER lines (ASCII) and AIG object (via read_aag_into_aig)."""
    aag_lines = AigerExporter(m).get_aag()
    # If you want an AIG object as well:
    fd, tmp = tempfile.mkstemp(suffix=".aag")
    os.close(fd)
    with open(tmp, "w") as f:
        f.write("\n".join(aag_lines) + "\n")
    aig = read_aag_into_aig(tmp)
    return aag_lines, aig


def roundtrip_aiger_back_to_spire(aag_lines: List[str], *, name="Imported") -> Module:
    """Import AAG (with symbols kept) back into SpireHDL."""
    # Keep symbol table (last lines); leave as-is if already present
    aag_sym = _get_aag_sym(aag_lines)
    aag_for_import = aag_lines[:-2] + aag_sym if aag_sym else aag_lines
    return AigerImporter(aag_for_import).get_spire_module(name)


# -----------------------------
# Simple modules + vectors
# -----------------------------


def build_logic3() -> Tuple[Module, Dict[str, UInt]]:
    """y = x0 & (x1 | x2)"""
    m = Module("Logic3", with_clock=False, with_reset=False)
    x0 = m.input(Bool(), "x0")
    x1 = m.input(Bool(), "x1")
    x2 = m.input(Bool(), "x2")
    y = m.output(Bool(), "y")
    y <<= x0 & (x1 | x2)
    # vectors: exhaustive 3-bit
    vecs = []
    for v in range(8):
        ins = {"x0": (v >> 0) & 1, "x1": (v >> 1) & 1, "x2": (v >> 2) & 1}
        exp = {"y": ins["x0"] & (ins["x1"] | ins["x2"])}
        vecs.append((f"v={v:03b}", ins, exp))
    spec = {"x0": UInt(1), "x1": UInt(1), "x2": UInt(1), "y": UInt(1)}
    return m, spec, vecs


def build_adder(W: int = 8) -> Tuple[Module, Dict[str, UInt], List]:
    m = Module(f"Add{W}", with_clock=False, with_reset=False)
    a = m.input(UInt(W), "a")
    b = m.input(UInt(W), "b")
    y = m.output(UInt(W + 1), "y")
    y <<= a + b
    vecs = []
    for _ in range(64):
        va = random.getrandbits(W)
        vb = random.getrandbits(W)
        vecs.append((f"{va}+{vb}", {"a": va, "b": vb}, {"y": (va + vb) & ((1 << (W + 1)) - 1)}))
    spec = {"a": UInt(W), "b": UInt(W), "y": UInt(W + 1)}
    return m, spec, vecs


def build_fp_mul_case(EW: int, FW: int, *, subnormals: bool = False):
    bits_tot = 1 + EW + FW
    m = build_fp_mul_sn(f"FPMul_E{EW}_F{FW}_{'SN' if subnormals else 'NZ'}", EW=EW, FW=FW, subnormals=subnormals)
    # use your generic vector builder
    vecs_basic = []
    for name, a, b, exp in build_fp_vectors(EW, FW):
        vecs_basic.append((name, {"a": a, "b": b}, {"y": exp}))
    spec = {"a": UInt(bits_tot), "b": UInt(bits_tot), "y": UInt(bits_tot)}
    return m, spec, vecs_basic, (lambda bits: fp_decode(bits, EW, FW))


# primary pi permutation handler
def build_io_order(aag_lines):
    """Return (pi_order, po_order) lists using symbol lines (iN/oN name)."""
    syms_i: Dict[int, str] = {}
    syms_o: Dict[int, str] = {}
    for ln in aag_lines:
        if not ln or ln[0] in "ca":  # stop at 'c' or skip blanks
            if ln.startswith("c"):
                break
            continue
        if ln[0] == "i":
            # i<number> <name>
            try:
                n_str, name = ln[1:].lstrip().split(" ", 1)
            except ValueError:
                n_str, name = ln[1:].lstrip(), ""
            syms_i[int(n_str)] = name
        elif ln[0] == "o":
            # o<number> <name>
            try:
                n_str, name = ln[1:].lstrip().split(" ", 1)
            except ValueError:
                n_str, name = ln[1:].lstrip(), ""
            syms_o[int(n_str)] = name
    pi_order = [syms_i[i] for i in sorted(syms_i)]
    po_order = [syms_o[i] for i in sorted(syms_o)]
    return pi_order, po_order


def permute_aag_lines_by_pi_order(
    aag_lines: List[str],
    target_order: List[str],
    target_po_order: List[str] | None = None,
) -> List[str]:
    """
    Reorder AAG input lines (and input symbols) to match target_order by name.
    Returns a new list of AAG lines.
    """
    if not aag_lines or not target_order:
        return aag_lines
    header = aag_lines[0].strip().split()
    if len(header) < 6 or header[0] != "aag":
        return aag_lines
    try:
        I = int(header[2])
        L = int(header[3])
        O = int(header[4])
        A = int(header[5])
    except ValueError:
        return aag_lines

    pos = 1
    if len(aag_lines) < pos + I + L + O + A:
        return aag_lines
    input_lines = aag_lines[pos : pos + I]
    pos += I
    latch_lines = aag_lines[pos : pos + L]
    pos += L
    output_lines = aag_lines[pos : pos + O]
    pos += O
    and_lines = aag_lines[pos : pos + A]
    pos += A
    sym_lines = aag_lines[pos:]

    sym_i: Dict[int, str] = {}
    sym_o: Dict[int, str] = {}
    preserved_sym_lines: List[str] = []
    comment_lines: List[str] = []
    in_comment = False
    for line in sym_lines:
        if in_comment:
            comment_lines.append(line)
            continue
        if not line:
            preserved_sym_lines.append(line)
            continue
        if line.startswith("c"):
            in_comment = True
            comment_lines.append(line)
            continue
        tag = line[0]
        rest = line[1:].lstrip()
        n_str, name = rest.split(" ", 1)
        if tag == "i":
            sym_i[int(n_str)] = name
        elif tag == "o":
            sym_o[int(n_str)] = name
        else:
            preserved_sym_lines.append(line)

    def _permute_io(
        *,
        tag: str,
        count: int,
        lines: List[str],
        sym_map: Dict[int, str],
        target: List[str] | None,
        default_prefix: str,
    ) -> tuple[List[str], List[str]] | None:
        if target is None:
            if not sym_map:
                return lines, []
            sym_out = []
            for idx in range(count):
                name = sym_map.get(idx, f"{default_prefix}{idx}")
                sym_out.append(f"{tag}{idx} {name}" if name else f"{tag}{idx}")
            return lines, sym_out
        if not sym_map or len(target) != count:
            return None
        name_to_idx: Dict[str, int] = {}
        for idx in range(count):
            nm = sym_map.get(idx, f"{default_prefix}{idx}")
            if nm in name_to_idx:
                return None
            name_to_idx[nm] = idx
        perm_indices = [name_to_idx[nm] for nm in target]
        lines_perm = [lines[i] for i in perm_indices]
        sym_out = [f"{tag}{idx} {name}" if name else f"{tag}{idx}" for idx, name in enumerate(target)]
        return lines_perm, sym_out

    perm_in = _permute_io(
        tag="i",
        count=I,
        lines=input_lines,
        sym_map=sym_i,
        target=target_order,
        default_prefix="pi",
    )
    if perm_in is None:
        return aag_lines
    input_lines_perm, new_input_sym_lines = perm_in

    perm_out = _permute_io(
        tag="o",
        count=O,
        lines=output_lines,
        sym_map=sym_o,
        target=target_po_order,
        default_prefix="po",
    )
    if perm_out is None:
        return aag_lines
    output_lines_perm, new_output_sym_lines = perm_out

    new_sym_lines = new_input_sym_lines + new_output_sym_lines + preserved_sym_lines
    if comment_lines:
        new_sym_lines += comment_lines

    return [aag_lines[0]] + input_lines_perm + latch_lines + output_lines_perm + and_lines + new_sym_lines


# -----------------------------
# Round-trip test for ONE module case
# -----------------------------


def run_test_one_module(m: Module, spec: Dict[str, UInt], vectors, *, decoder=None, equivalence_check=True) -> None:
    print(f"\n=== {m.name} ===")

    # 1) Original sim
    print("Sim (original) …")
    run_vectors(m, vectors, decoder=decoder)

    # 2) SpireHDL → AIGER (exporter) → AIG
    aag_lines, aig_exp = spire_to_aig_via_exporter(m)

    # optional agi to SpireHDL module
    # m2 = roundtrip_aiger_back_to_spire(aag_lines, name=m.name+"_exp")

    # 3) Verilog → Pyosys → AIGER → AIG
    vpath = write_temp_verilog(m, top_name=m.name)

    # for testing copy vpath to local dir
    os.system(f"cp {vpath} ./{m.name}.v")
    aag_path, map_path = verilog_to_aag_via_yosys(vpath, top=m.name, embed_symbols=True, no_startoffset=True)
    aig_pyo = read_aag_into_aig(aag_path)

    aag_pyo_lines = file_to_lines(aag_path)

    # after you produced aag_path with yosys:
    aag_back_lines = file_to_lines(aag_path)
    m_back_raw = AigerImporter(aag_back_lines).get_spire_module("BackRaw")
    # exporter AIG from the raw imported module (no IOCollector yet)
    aag_from_raw, aig_from_raw = spire_to_aig_via_exporter(m_back_raw)
    # yosys AIG
    aig_yosys = read_aag_into_aig(aag_path)

    assert equivalence_checking(aig_from_raw, aig_yosys), "Importer produced a non-equivalent SpireHDL netlist BEFORE regrouping"

    # Normalize PI order (by name) before equivalence
    order_exp, po_order_exp = build_io_order(aag_lines)  # e.g., ["a[0]",...,"b[0]",...]
    order_pyo, po_order_pyo = build_io_order(aag_pyo_lines)  # e.g., ["b[0]",...,"a[0]",...]
    if order_exp != order_pyo or po_order_exp != po_order_pyo:
        aag_pyo_lines = permute_aag_lines_by_pi_order(
            aag_pyo_lines,
            order_exp,
            target_po_order=po_order_exp if po_order_exp else None,
        )
        aig_pyo = conv_aag_into_aig(aag_pyo_lines)

    # 4) Equivalence check
    if equivalence_check:
        print("AIG equivalence (exporter vs pyosys) …")
        assert equivalence_checking(aig_exp, aig_pyo), "AIGs not equivalent!"

    # 5) AAG (with symbols) → SpireHDL
    aag_back = file_to_lines(aag_path)
    m_back = AigerImporter(aag_back).get_spire_module(m.name + "_back")
    # m_back = roundtrip_aiger_back_to_spire(aag_back, name=m.name + "_back")

    # 6) Regroup I/Os to match original port widths
    IOCollector().group(m_back, spec)

    # 7) Re-sim round-tripped module
    print("Sim (round-tripped) …")
    run_vectors(m_back, vectors, decoder=decoder)


def gen_m_case(i: int) -> Tuple[Module, Dict[str, UInt], List, Callable | None]:
    reset_shared_cache()
    if i == 0:
        return build_logic3() + (None,)
    elif i == 1:
        return build_adder(8) + (None,)
    elif i == 2:
        return build_fp_mul_case(5, 3, subnormals=False)
    elif i == 3:
        return build_fp_mul_case(5, 10, subnormals=False)


# -----------------------------
# Main: build a list and run
# -----------------------------


def test_run():
    random.seed(123)

    # Run everything
    n_cases = 4
    for i in range(n_cases):

        m, spec, vecs, dec = gen_m_case(i)

        print(f"Running case {i + 1}/{n_cases}")
        run_test_one_module(
            m, spec, vecs, decoder=dec, equivalence_check=True if i != 3 else False
        )  # skip equivalence for last case (FP mul E5F10) due to complexity

    print("\nAll module cases passed round-trip + sim checks.")
    print(f"Tested {n_cases} module(s).")


if __name__ == "__main__":
    test_run()
