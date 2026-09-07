"""Shared helpers for the differential conformance harness (docs/README_semantics.md §5).

Backends compared: the Python simulator (the Domain-A specification), the IEEE-1364 reference evaluation of the
emitted Verilog text (ieee_eval.VlogModule), and the AIGER export reloaded through AigerImporter. All comparisons
are on masked bit patterns at each output's declared width.
"""
from __future__ import annotations

import itertools
import random
from collections import OrderedDict

from spire.aiger import AigerExporter, AigerImporter
from spire.component import IOCollector
from spire.expr import Bool, SInt, UInt, cat, mux, reset_shared_cache
from spire.ir import Netlist
from spire.simulator import Simulator

from .ieee_eval import VlogModule


def mask(w: int) -> int:
    return (1 << w) - 1


def corners(w: int):
    """Boundary-loaded value set for a w-bit input (both sign interpretations covered pattern-wise)."""
    return sorted({0, 1, 2, mask(w) >> 1, (mask(w) >> 1) + 1, mask(w) - 1, mask(w)})


def sweep(*widths):
    """Cartesian product of per-input corner values."""
    return list(itertools.product(*[corners(w) for w in widths]))


def exhaustive(*widths):
    return list(itertools.product(*[range(1 << w) for w in widths]))


def fresh_netlist(name: str) -> Netlist:
    """Combinational netlist with a clean shared-wire cache (deterministic emission per test)."""
    reset_shared_cache()
    return Netlist(name, with_clock=False, with_reset=False)


def output_like(m: Netlist, expr, name: str):
    """Declare an output whose type equals the expression's own type and drive it (no assignment resize)."""
    t = SInt(expr.typ.width) if expr.typ.signed else UInt(expr.typ.width)
    y = m.output(t, name)
    y <<= expr
    return y


def diff_sim_vs_verilog(m: Netlist, input_names, vectors, out_prefix: str = "y"):
    """Return ([(out, vec, sim_pattern, verilog_pattern), ...], emitted_text). Empty list = conformant."""
    text = m.to_verilog()
    vm = VlogModule(text)
    sim = Simulator(m)
    spec = m.get_spec()
    outs = [n for n in spec if n.startswith(out_prefix)]
    assert outs, "netlist has no outputs to compare"
    bad = []
    for vec in vectors:
        for n, v in zip(input_names, vec):
            sim.set(n, v)
            vm.set(n, v)
        sim.eval()
        for out in outs:
            w = spec[out].width
            got_sim = sim.get(out) & mask(w)
            got_vlog = vm.get(out) & mask(w)
            if got_sim != got_vlog:
                bad.append((out, tuple(vec), got_sim, got_vlog))
    return bad, text


def aiger_roundtrip(m: Netlist) -> Netlist:
    """Export to AAG, re-import, and re-group ports to the original widths (name-preserving)."""
    aag = AigerExporter(m).get_aag()
    try:
        from spire.aig.aig_aigerverse import _get_aag_sym

        aag = aag[:-2] + _get_aag_sym(aag)
    except Exception:
        pass
    m2 = AigerImporter(aag).get_spire_netlist()
    group_spec = OrderedDict(
        (p.name, UInt(1 if p.typ.is_bool else p.typ.width)) for p in m._ports if p.name not in ("clk", "rst")
    )
    IOCollector().group(m2, group_spec)
    return m2


def diff_sim_vs_aiger(m: Netlist, input_names, vectors, out_prefix: str = "y"):
    """Return [(out, vec, sim_pattern, aiger_pattern), ...] comparing m against its AIGER round-trip."""
    m2 = aiger_roundtrip(m)
    sim = Simulator(m)
    sim2 = Simulator(m2)
    spec = m.get_spec()
    outs = [n for n in spec if n.startswith(out_prefix)]
    bad = []
    for vec in vectors:
        for n, v in zip(input_names, vec):
            sim.set(n, v)
            sim2.set(n, v)
        sim.eval()
        sim2.eval()
        for out in outs:
            w = spec[out].width
            a = sim.get(out) & mask(w)
            b = sim2.get(out) & mask(w)
            if a != b:
                bad.append((out, tuple(vec), a, b))
    return bad


def format_mismatches(bad, input_names, limit: int = 4) -> str:
    lines = [f"{len(bad)} mismatching vectors; first {min(len(bad), limit)}:"]
    for out, vec, a, b in bad[:limit]:
        ins = ", ".join(f"{n}={v}" for n, v in zip(input_names, vec))
        lines.append(f"  {out}: sim={a}  other={b}  ({ins})")
    return "\n".join(lines)


# ------------------------- seeded random circuit generator -------------------------

_UNARY = ("not", "neg")
_BINARY = ("add", "sub", "mul", "and", "or", "xor", "shl_const", "shl_var", "shr", "lt", "eq", "mux", "cat")


def gen_random_module(seed: int, n_inputs: int = 3, n_ops: int = 6, max_w: int = 6, signed_ok: bool = True):
    """Deterministic random combinational netlist. Returns (netlist, input_names)."""
    rng = random.Random(seed)
    m = fresh_netlist(f"fuzz_{seed}")
    names = [f"i{k}" for k in range(n_inputs)]
    pool = []
    for name in names:
        w = rng.randint(1, max_w)
        s = signed_ok and rng.random() < 0.5
        pool.append(m.input(SInt(w) if s else UInt(w), name))
    exprs = []
    used_ids = set()
    for _ in range(n_ops):
        op = rng.choice(_BINARY + _UNARY)
        # Chain bias: prefer the freshest expression as the LEFT operand. A single-use compound left operand is
        # emitted inline (no shared-wire width boundary), which is exactly the nesting the harness must exercise;
        # any additional reference (another operand use, or an output drive) shares the node into a named wire and
        # would mask it.
        a = exprs[-1] if exprs and rng.random() < 0.7 else rng.choice(pool + exprs)
        b = rng.choice(pool + exprs)
        used_ids.add(id(a))
        used_ids.add(id(b))
        try:
            if op == "add":
                e = a + b
            elif op == "sub":
                e = a - b
            elif op == "mul":
                e = a * b
            elif op == "and":
                e = a & b
            elif op == "or":
                e = a | b
            elif op == "xor":
                e = a ^ b
            elif op == "shl_const":
                e = a << rng.randint(0, 2)
            elif op == "shl_var":
                e = a << b
            elif op == "shr":
                e = a >> (b if rng.random() < 0.5 else rng.randint(0, max(a.typ.width - 1, 0)))
            elif op == "lt":
                e = a < b
            elif op == "eq":
                e = a == b
            elif op == "mux":
                e = mux(a < b, a, b)
            elif op == "cat":
                e = cat(a, b)
            elif op == "not":
                e = ~a
            else:  # neg
                e = -a
            if e.typ.width > 24:
                continue
        except Exception:
            continue
        exprs.append(e)
    if not exprs:
        exprs.append(pool[0] + pool[-1])
    # Only sink expressions (never consumed as an operand) become outputs: divergence in chain interiors is observed
    # through their consumers, while an output drive on an interior node would add a second reference and share it
    # into a named wire, erasing the inline nesting this generator exists to produce.
    sinks = [e for e in exprs if id(e) not in used_ids] or [exprs[-1]]
    for k, e in enumerate(sinks):
        output_like(m, e, f"y{k}")
    return m, names


def random_vectors(rng: random.Random, m: Netlist, input_names, count: int):
    spec = m.get_spec()
    widths = [spec[n].width for n in input_names]
    return [tuple(rng.randrange(1 << w) for w in widths) for _ in range(count)]
