"""Bit-level emit: replace an FSM's mux-tree next-state / output logic with
minimised bit-level Boolean expressions over the state-register bits.

Why this exists
---------------
``optimized_encoding`` (and the user's ``switch_``/``case_`` body) leave the
FSM as a *structural mux tree* driving the state register, with outputs written
as ``state == CODE`` comparisons. Downstream Yosys recognises that shape as an
FSM and **re-encodes** it with its own state assignment — discarding whatever
encoding the search picked (see ``INVESTIGATION_fsm_encoding_api.md``). The
realised PPA is then encoding-invariant and the search is wasted.

This pass rewrites the FSM so it no longer *looks* like an FSM to Yosys:

* recover the abstract transition + output table (reusing
  :func:`extract_transition_table`, which enumerates
  ``(state_value × input) -> (next_state, outputs)`` with the dependency-free
  ``eval_with`` evaluator);
* for each next-state bit and each output bit, build a truth table over
  ``(state_bits, input_bits)``; **unused state codes become don't-cares**;
* minimise each with Quine–McCluskey (``sympy.SOPform``);
* replace ``state_reg._driver`` and each FSM output's ``_driver`` with the
  freshly-built bit-level ``Expr`` DAG (``&`` / ``|`` / ``~`` / ``Signal[i]`` /
  ``Concat`` — all core operators, no core changes).

The two user-visible knobs are:

* **bit-level emit** (this rewrite) — recovers the realised-PPA win that the
  encoding choice should buy, because the emitted logic is plain Boolean over
  the state bits rather than an FSM Yosys will re-encode.
* **don't-cares** (``dont_cares``) — feed unused state codes as don't-cares to
  the minimiser. This is objective-dependent: it shrinks gate/area count but
  can lengthen the critical path, so it helps a cells/area objective and can
  hurt an ADP/delay objective. Default ``False``.

``sympy`` is imported lazily so the rest of the FSM package keeps working
without it; only callers that opt into bit-level emit need it installed.
"""
from __future__ import annotations

from itertools import product
from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple

from spire.optimize.fsm._table import extract_transition_table
from spire.optimize.fsm._walker import find_state_consts

if TYPE_CHECKING:
    from spire.expr import Expr, Signal
    from spire.component import Module
    from spire.state import State


def _require_sympy():
    try:
        import sympy  # noqa: F401
        from sympy import SOPform, symbols  # noqa: F401
        return sympy
    except ImportError as e:  # pragma: no cover - exercised only without sympy
        raise ImportError(
            "bit-level emit / the 'adp_proxy' objective require sympy for "
            "Boolean minimisation (`pip install sympy`). It is imported lazily "
            "so the rest of the FSM API works without it."
        ) from e


# ---------------------------------------------------------------------------
# Discovery: state register + FSM outputs
# ---------------------------------------------------------------------------
def find_state_register(module: "Module", state_cls: "type[State]") -> "Signal":
    """Return the single ``reg`` whose value holds ``state_cls`` states.

    A register qualifies if its ``_init`` is a Const of this state class, or its
    driver references state Consts of this class. Raises if zero or >1 match.
    """
    module.collect_signals()
    cands: List["Signal"] = []
    for s in module._signals:
        if s.kind != "reg":
            continue
        init = getattr(s, "_init", None)
        if init is not None and getattr(init, "_state_class", None) is state_cls:
            cands.append(s)
            continue
        drv = getattr(s, "_driver", None)
        if drv is not None and find_state_consts([drv], state_cls):
            cands.append(s)
    # dedupe by identity
    uniq = {id(c): c for c in cands}
    cands = list(uniq.values())
    if not cands:
        raise ValueError(
            f"find_state_register: no register found for state class "
            f"{state_cls.__name__!r}")
    if len(cands) > 1:
        names = ", ".join(c.name for c in cands)
        raise ValueError(
            f"find_state_register: multiple candidate registers ({names}) for "
            f"{state_cls.__name__!r}; bit-level emit supports a single state register")
    return cands[0]


def find_fsm_outputs(
    module: "Module", state_reg: "Signal", state_cls: "type[State]",
) -> List["Signal"]:
    """Module outputs whose driver depends on ``state_reg`` (and only on the
    state register + module inputs). Outputs that depend on *other* registers
    are skipped — we only rewrite logic we can fully recover from the
    (state × input) enumeration.
    """
    from spire.optimize.fsm._walker import find_input_signals
    module.collect_signals()
    inputs_ok = {id(s) for s in module._signals if s.kind == "input"}
    inputs_ok.add(id(state_reg))
    outs: List["Signal"] = []
    for s in module._signals:
        if s.kind != "output" or s._driver is None:
            continue
        # support = non-state signal leaves of the driver, plus whether it uses the state reg
        leaves = find_input_signals([s._driver], state_cls)  # excludes state Consts
        # driver must reference the state register (else it's not state logic)
        uses_state = any(id(l) == id(state_reg) for l in leaves) or bool(
            find_state_consts([s._driver], state_cls))
        if not uses_state:
            continue
        # every other leaf must be a module input (or the state reg itself)
        if all(id(l) in inputs_ok for l in leaves):
            outs.append(s)
    return outs


# ---------------------------------------------------------------------------
# Truth-table construction + minimisation
# ---------------------------------------------------------------------------
def _bit_layout(state_width: int, input_signals: Sequence["Signal"]):
    """Return (n_vars, list_of_(kind, idx, bit)) describing the minterm bit order.

    Bit 0..state_width-1 = state bits (LSB first). Then each input signal's bits
    (LSB first), in ``input_signals`` order. ``kind`` is "state" or an int index
    into ``input_signals``.
    """
    layout = [("state", b) for b in range(state_width)]
    for i, s in enumerate(input_signals):
        for b in range(s.typ.width):
            layout.append((i, b))
    return len(layout), layout


def _minterm_bits(state_value: int, input_combo: Tuple[int, ...],
                  layout) -> List[int]:
    """Bit-list for one (state_value, input_combo) cell, in ``layout`` order."""
    bits: List[int] = []
    for entry in layout:
        kind, b = entry
        if kind == "state":
            bits.append((state_value >> b) & 1)
        else:
            bits.append((input_combo[kind] >> b) & 1)
    return bits


def _minimize_to_sympy(onset_bitlists, dcset_bitlists, syms):
    """Quine–McCluskey via sympy.SOPform. Returns a sympy boolean expr.

    Constant handling: empty onset -> false. If every assignment is in
    onset ∪ dcset, SOPform returns true.
    """
    sympy = _require_sympy()
    from sympy import SOPform
    if not onset_bitlists:
        return sympy.false
    return SOPform(syms, onset_bitlists, dcset_bitlists)


# ---------------------------------------------------------------------------
# sympy expr -> spire Expr
# ---------------------------------------------------------------------------
def _sympy_to_spire(node, sym_to_expr) -> "Expr":
    """Convert a sympy boolean expr (And/Or/Not/Symbol/true/false) to a spire
    1-bit ``Expr`` using ``sym_to_expr`` (maps a sympy Symbol -> spire Expr)."""
    import sympy
    from sympy.logic.boolalg import And, Or, Not
    from spire.expr import Const, UInt

    if node is sympy.true:
        return Const(1, UInt(1))
    if node is sympy.false:
        return Const(0, UInt(1))
    if node.is_Symbol:
        return sym_to_expr(node)
    if isinstance(node, Not):
        return ~_sympy_to_spire(node.args[0], sym_to_expr)
    if isinstance(node, And):
        acc = None
        for a in node.args:
            e = _sympy_to_spire(a, sym_to_expr)
            acc = e if acc is None else (acc & e)
        return acc
    if isinstance(node, Or):
        acc = None
        for a in node.args:
            e = _sympy_to_spire(a, sym_to_expr)
            acc = e if acc is None else (acc | e)
        return acc
    raise ValueError(f"unrenderable sympy node: {node!r} ({type(node)})")


# ---------------------------------------------------------------------------
# Core: build minimised per-bit expressions
# ---------------------------------------------------------------------------
class MinimizedFSM:
    """Result of minimisation: per-output sympy exprs over an abstract bit
    layout, ready to materialise against either the live state register
    (for in-place rewrite) or a fresh state input (for a combinational cone)."""

    def __init__(self, state_reg, input_signals, output_signals,
                 ns_exprs, out_exprs, layout, syms):
        self.state_reg = state_reg
        self.input_signals = input_signals
        self.output_signals = output_signals
        self.ns_exprs = ns_exprs          # list[sympy expr], one per state bit
        self.out_exprs = out_exprs        # list[list[sympy expr]], per output, per bit
        self.layout = layout
        self.syms = syms

    def _make_sym_to_expr(self, state_bit_exprs, input_bit_exprs):
        mapping = {}
        for sym, entry in zip(self.syms, self.layout):
            kind, b = entry
            if kind == "state":
                mapping[sym] = state_bit_exprs[b]
            else:
                mapping[sym] = input_bit_exprs[kind][b]
        return lambda s: mapping[s]


def minimize_fsm_logic(
    module: "Module",
    state_cls: "type[State]",
    *,
    state_reg: Optional["Signal"] = None,
    outputs: Optional[Sequence["Signal"]] = None,
    dont_cares: bool = False,
) -> MinimizedFSM:
    """Recover the FSM tables and minimise each next-state bit + output bit.

    Does **not** mutate the module — returns a :class:`MinimizedFSM` the caller
    materialises (rewrite in place, or build a combinational cone).
    """
    _require_sympy()
    from sympy import symbols

    if state_reg is None:
        state_reg = find_state_register(module, state_cls)
    if outputs is None:
        outputs = find_fsm_outputs(module, state_reg, state_cls)
    outputs = list(outputs)

    table = extract_transition_table(state_reg, state_cls, outputs=outputs)
    w = state_reg.typ.width
    used_codes = set(state_cls._values.values())
    n_vars, layout = _bit_layout(w, table.input_signals)
    syms = list(symbols(f"v0:{n_vars}")) if n_vars else []

    # don't-care minterms: every (unused_code × every input combo).
    dc_bitlists: List[List[int]] = []
    if dont_cares:
        all_codes = set(range(1 << w))
        for code in sorted(all_codes - used_codes):
            for ic in table.all_input_combos():
                dc_bitlists.append(_minterm_bits(code, ic, layout))

    # next-state bits
    ns_exprs = []
    for b in range(w):
        onset = []
        for sv in used_codes:
            for ic in table.all_input_combos():
                nsv = table.transitions[sv][ic]
                if (nsv >> b) & 1:
                    onset.append(_minterm_bits(sv, ic, layout))
        ns_exprs.append(_minimize_to_sympy(onset, dc_bitlists, syms))

    # output bits
    out_exprs = []
    for oi, o in enumerate(outputs):
        ow = o.typ.width
        per_bit = []
        for b in range(ow):
            onset = []
            for sv in used_codes:
                for ic in table.all_input_combos():
                    ov = table.outputs[sv][ic][oi]
                    if (ov >> b) & 1:
                        onset.append(_minterm_bits(sv, ic, layout))
            per_bit.append(_minimize_to_sympy(onset, dc_bitlists, syms))
        out_exprs.append(per_bit)

    return MinimizedFSM(state_reg, list(table.input_signals), outputs,
                        ns_exprs, out_exprs, layout, syms)


# ---------------------------------------------------------------------------
# Materialise: rewrite the live module in place
# ---------------------------------------------------------------------------
def _concat_bits(bit_exprs: List["Expr"]) -> "Expr":
    """Concat 1-bit exprs (LSB first) into a wider value."""
    from spire.expr import Concat
    if len(bit_exprs) == 1:
        return bit_exprs[0]
    return Concat(bit_exprs)  # parts[0] = LSB (matches simulator + to_verilog)


def minimize_and_rewrite(
    module: "Module",
    state_cls: "type[State]",
    *,
    state_reg: Optional["Signal"] = None,
    outputs: Optional[Sequence["Signal"]] = None,
    dont_cares: bool = False,
) -> MinimizedFSM:
    """Minimise + replace the live ``state_reg._driver`` and each FSM output's
    ``_driver`` with bit-level Boolean expressions over the state register bits.
    """
    from spire.expr import flat_emit

    mf = minimize_fsm_logic(module, state_cls, state_reg=state_reg,
                            outputs=outputs, dont_cares=dont_cares)
    sreg = mf.state_reg

    # Build the bit-level DAG with opportunistic CSE OFF: SpireHDL otherwise
    # wraps every subexpression into a named wire, and that wire-split structure
    # maps ~5-7% worse on PPA than flat inline logic (abc is structurally
    # sensitive — see flat_emit / INVESTIGATION_fsm_encoding_api.md). The whole
    # point of bit-level emit is PPA, so we always build flat here. Forced
    # slice shares (needed for valid Verilog) are unaffected.
    with flat_emit():
        # state bit leaves come from the live register; inputs from live signals
        state_bit_exprs = [sreg[b] for b in range(sreg.typ.width)]
        input_bit_exprs = [[s[b] for b in range(s.typ.width)] for s in mf.input_signals]
        sym_to_expr = mf._make_sym_to_expr(state_bit_exprs, input_bit_exprs)

        # next-state
        ns_bits = [_sympy_to_spire(e, sym_to_expr) for e in mf.ns_exprs]
        sreg._driver = _concat_bits(ns_bits)

        # outputs
        for o, per_bit in zip(mf.output_signals, mf.out_exprs):
            bit_exprs = [_sympy_to_spire(e, sym_to_expr) for e in per_bit]
            o._driver = _concat_bits(bit_exprs)

    return mf


# ---------------------------------------------------------------------------
# Combinational cone (for the PDK-free adp_proxy cost objective)
# ---------------------------------------------------------------------------
def build_comb_cone(
    module: "Module",
    state_cls: "type[State]",
    *,
    dont_cares: bool = False,
) -> "Module":
    """Build a fresh **combinational** Module: the FSM's next-state + output
    logic with the state register turned into an input port. This is exactly
    the logic the encoding affects, and being combinational it can be measured
    by aigverse (the sequential design cannot — the AIG reader rejects latches).
    """
    from spire.expr import UInt, flat_emit
    from spire.component import Module

    mf = minimize_fsm_logic(module, state_cls, dont_cares=dont_cares)
    w = mf.state_reg.typ.width

    cone = Module(f"{module.name}__cone", with_clock=False, with_reset=False)
    state_in = cone.input(UInt(w), "state")
    in_ports = [cone.input(s.typ, s.name) for s in mf.input_signals]

    # Flat build (see minimize_and_rewrite) so the cone measured by the cost
    # oracle matches the flat form the committed design will use.
    with flat_emit():
        state_bit_exprs = [state_in[b] for b in range(w)]
        input_bit_exprs = [[p[b] for b in range(p.typ.width)] for p in in_ports]
        sym_to_expr = mf._make_sym_to_expr(state_bit_exprs, input_bit_exprs)

        ns_bits = [_sympy_to_spire(e, sym_to_expr) for e in mf.ns_exprs]
        ns_out = cone.output(UInt(w), "ns")
        ns_out <<= _concat_bits(ns_bits)

        for o, per_bit in zip(mf.output_signals, mf.out_exprs):
            bit_exprs = [_sympy_to_spire(e, sym_to_expr) for e in per_bit]
            op = cone.output(o.typ, o.name)
            op <<= _concat_bits(bit_exprs)

    return cone
