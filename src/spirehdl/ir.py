"""SpireHDL internal IR.

`Netlist` (formerly `Module`) is the flat, lowered netlist that every backend consumes —
the post-elaboration port+signal list. It is one layer below `Component` (the user-facing
authoring abstraction in spirehdl_module): the dependency is strictly one-way,
`component -> ir -> spirehdl`. Power users may import `Netlist` from here; the docs do not
lead with it. `Module` / `IOCollector` remain available as deprecated aliases.
"""
from __future__ import annotations

import hashlib
import random
import time
from typing import Any, Dict, List, Optional

from spirehdl import VERILOG_BANNER
from spirehdl.spirehdl import (Bool, Const, Expr, ExprLike, HDLType, Signal, UInt, cat,
                               fit_width, get_shared_wires, reset_shared_cache)
from spirehdl.spirehdl_memory import _MemoryArray
from spirehdl.spirehdl_analyzer import _Analyzer, GraphReport
from spirehdl.spirehdl_visitor import ExprVisitor, expr_children
from spirehdl.composite.record_dynamic import iter_values  # used by _SignalCollector


class Netlist:
    def __init__(self, name: str, with_clock: bool = True, with_reset: bool = True):
        self.name = name
        self.with_clock = with_clock
        self.with_reset = with_reset
        self._signals: List[Signal] = []
        self._ports: List[Signal] = []
        # default clock/reset inputs
        if with_clock:
            self.clk = self.input(Bool(), "clk")
        else:
            self.clk = None
        if with_reset:
            self.rst = self.input(Bool(), "rst")
        else:
            self.rst = None
        self.component : Optional["Component"] = None

    # Signal constructors
    def input(self, typ: HDLType, name: str) -> Signal:
        s = Signal(typ=typ, kind="input", name=name) #, self)
        self._signals.append(s)
        self._ports.append(s)
        return s

    def add_input(self, signal: Signal) -> None:
        if signal.kind != "input":
            # change to input
            signal.kind = "input"
        if id(signal) in [id(s) for s in self._signals]:
            raise ValueError("Signal already exists in module.")
        self._signals.append(signal)
        self._ports.append(signal)

    def output(self, typ: HDLType, name: str) -> Signal:
        s = Signal(typ=typ, kind="output", name=name) #, self)
        self._signals.append(s)
        self._ports.append(s)
        return s

    def add_output(self, signal: Signal) -> None:
        if signal.kind != "output":
            # change to output
            signal.kind = "output"
        if id(signal) in [id(s) for s in self._signals]:
            raise ValueError("Signal already exists in module.")
        self._signals.append(signal)
        self._ports.append(signal)

    def wire(self, typ: HDLType, name: str) -> Signal:
        s = Signal(typ=typ, kind="wire", name=name) #, self)
        self._signals.append(s)
        return s

    def reg(self, typ: HDLType, name: str, init: Optional[ExprLike] = None) -> Signal:
        s = Signal(typ=typ, kind="reg", name=name) #, self)
        if init is not None:
            s.set_init(init)
        self._signals.append(s)
        return s

    # Introspection helpers
    def _ports_of(self, kind: str) -> List[Signal]:
        return [s for s in self._ports if s.kind == kind]

    def _is_port(self, s: "Signal") -> bool:
        # Use identity, not equality, so we don't trigger Expr.__eq__.
        return any(s is p for p in self._ports)

    def _internals_of(self, kind: str) -> List[Signal]:
        # Avoid `s not in self._ports` (it calls __eq__). Use identity instead.
        return [s for s in self._signals if s.kind == kind and not self._is_port(s)]

    def get_spec(self) -> Dict[str, UInt]:
        spec = {}
        for p in self._ports:
            spec[p.name] = p.typ
        return spec

    def collect_signals(self) -> None:
        """Rebuild ``_signals`` to hold every Signal reachable from any port or user-added internal (anything
        currently in ``_signals`` that wasn't produced by the CSE share-cache). Orphan auto-shared wires drop out
        on each call — that's how simplify/CSE rewrites prune dead `assign` lines.
        """
        port_ids = {id(p) for p in self._ports}
        user_internals = [s for s in self._signals
                          if id(s) not in port_ids
                          and not getattr(s, "_auto_generated", False)]
        seeds = self._ports_of("output") + user_internals
        self._signals = list(self._ports)
        _SignalCollector(self).run(seeds)

        # print(f"Collected {len(self._signals)} signals.")

    def to_component(self):
        """Deprecated: use ``Component.from_netlist(netlist)``.

        Kept as a thin shim for back-compat. The real implementation lives on ``Component`` so
        the IR has no upward dependency on the user-facing layer (the import below is function-local,
        so there is no module-load-time cycle: ``component -> ir -> spirehdl`` stays one-way).
        """
        from spirehdl.spirehdl_module import Component
        return Component.from_netlist(self)

    # Verilog generation
    def to_verilog_lines(self, collect_signals=True, simplify=False, cse=True,
                          balance_mux_trees=False, balance_mux_min_n=16) -> list[str]:

        if collect_signals:
            self.collect_signals()
            # Post-construction peephole simplification (opt_expr / opt_muxtree analogue): constant folding, boolean
            # identities, trivial-mux collapse, and mux-tree guard substitution. Runs before CSE so that any newly-
            # exposed shared sub-expressions (e.g. mux(c, x, x) → x exposes x as a sharing target) get collapsed
            # afterward. Re-collect signals so auto-shared wires orphaned by guard substitution drop out of _signals
            # (and don't emit as dead `assign sig_N = …` lines).
            if simplify:
                from spirehdl.spirehdl_simplify import apply_simplify
                if apply_simplify(self):
                    self.collect_signals()
            # Optional mux-tree balance pass: detect linear cascades of the form
            #   mux(sel == Const(0), v_0, mux(sel == Const(1), v_1, ... mux(sel == Const(N-1), v_{N-1}, default)))
            # with full power-of-2 coverage and N >= balance_mux_min_n, and rewrite as a balanced binary mux tree
            # using BITS of sel. Yosys+abc often fails to tree-balance the linear cascade back, leaving a deep AOI/OAI
            # chain on the critical path (see `benchmarks/dr_rtl_spirehdl/router/_debug/DEBUGGING.md` for the analysis).
            if balance_mux_trees:
                from spirehdl.spirehdl_simplify import apply_mux_tree_balance
                if apply_mux_tree_balance(self, min_n=balance_mux_min_n):
                    self.collect_signals()
            # Post-construction structural CSE (Common Subexpression Elimination): collapse any
            # duplicate subtrees. Then re-collect so the freshly-created shared wires land in self._signals for emission.
            if cse:
                from spirehdl.spirehdl_cse import apply_structural_cse
                if apply_structural_cse(self):
                    self.collect_signals()

        # Basic checks. Signals tagged `_no_emit_drive` (custom-Verilog replacement) are exempt — the custom block
        # provides their value. Memory stores (`_MemoryArray`) are sim-only and always no-emit, so they and their
        # rdata registers fall through these checks without special-casing.
        for s in self._signals:
            if s.kind in ("wire", "output") and s._driver is None and not s._no_emit_drive:
                if s.kind == "output":
                    raise ValueError(f"Output '{s.name}' has no driver.")
            if s.kind == "reg" and s._driver is None and not s._no_emit_drive:
                raise ValueError(f"Register '{s.name}' has no next-state assignment.")

        # Collect custom-Verilog blocks from any Component that owns part of this module's design graph (top-level
        # Component via `module.component`, plus embedded Components reached through `_owning_component` back-edges
        # on IO wires). Called here so port-wire names are already uniquified.
        custom_blocks: list[str] = []
        seen_owners: set = set()
        def _maybe_collect(comp):
            if comp is None or id(comp) in seen_owners:
                return
            if hasattr(comp, "custom_verilog"):
                seen_owners.add(id(comp))
                custom_blocks.append(comp.custom_verilog())
        _maybe_collect(self.component)
        for s in self._signals:
            _maybe_collect(getattr(s, "_owning_component", None))

        lines: List[str] = [VERILOG_BANNER, ""]
        # Ports list
        port_names = [p.name for p in self._ports]
        ports_csv = ", ".join(port_names)
        lines.append(f"module {self.name} ({ports_csv});")

        # Declarations
        lines.append("// Ports")
        for p in self._ports:
            dir_ = "input" if p.kind == "input" else "output"
            sign = "signed " if p.typ.signed else ""
            rng = p.typ.range_str()
            lines.append(f"  {dir_} {sign}{rng} {p.name};")

        wires = self._internals_of("wire")
        if not collect_signals:
            wires += [s for s in get_shared_wires() if not any(s is w for w in wires)]

        regs_all = self._internals_of("reg")
        regs = regs_all

        lines.append('// Wires')
        for w in wires:
            if w._no_emit_decl:
                continue
            sign = "signed " if w.typ.signed else ""
            rng = w.typ.range_str()
            lines.append(f"  wire {sign}{rng} {w.name};")
        lines.append('// Registers')
        for r in regs_all:
            if r._no_emit_decl:
                continue
            sign = "signed " if r.typ.signed else ""
            rng = r.typ.range_str()
            lines.append(f"  reg {sign}{rng} {r.name};")

        # Combinational assigns for wires/outputs. `_MemoryArray` port wires are no-emit (the wrapping primitive's
        # custom_verilog drives the storage), so they're skipped here via the `_no_emit_drive` guard.
        lines.append("// Combinational assignments")
        for s in [*wires, *self._ports_of("output")]:
            if s._no_emit_drive:
                continue
            if s._driver is not None:
                rhs = fit_width(s._driver, s.typ).to_verilog()
                lines.append(f"  assign {s.name} = {rhs};")

        # Sequential logic for normal regs.
        lines.append("// Sequential logic")
        emit_regs = [r for r in regs if not r._no_emit_drive]
        if emit_regs:
            if not self.with_clock:
                raise ValueError("Registers present but module has no clock input.")
            sens = f"posedge {self.clk.name}"
            if self.with_reset:
                sens += f" or posedge {self.rst.name}"
            lines.append(f"  always @({sens}) begin")
            if self.with_reset:
                lines.append(f"    if ({self.rst.name}) begin")
                for r in emit_regs:
                    init = r._init.to_verilog() if r._init is not None else f"{r.typ.width}'d0"
                    lines.append(f"      {r.name} <= {init};")
                lines.append("    end else begin")
                for r in emit_regs:
                    lines.append(f"      {r.name} <= {fit_width(r._driver, r.typ).to_verilog()};")
                lines.append("    end")
            else:
                for r in emit_regs:
                    lines.append(f"    {r.name} <= {fit_width(r._driver, r.typ).to_verilog()};")
            lines.append("  end")

        # Custom Verilog blocks (one per Component that provides `custom_verilog`).
        if custom_blocks:
            lines.append("// Custom Verilog")
            for block in custom_blocks:
                lines.extend(block.splitlines())

        lines.append("endmodule")
        return lines

    def to_verilog(self, simplify=False, cse=True,
                    balance_mux_trees=False, balance_mux_min_n=16) -> str:
        lines = self.to_verilog_lines(
            simplify=simplify, cse=cse,
            balance_mux_trees=balance_mux_trees,
            balance_mux_min_n=balance_mux_min_n,
        ) + [""]  # final newline
        return "\n".join(lines)

    def to_verilog_file(self, filepath: str, simplify=False, cse=True,
                         balance_mux_trees=False, balance_mux_min_n=16) -> None:
        verilog_str = self.to_verilog(
            simplify=simplify, cse=cse,
            balance_mux_trees=balance_mux_trees,
            balance_mux_min_n=balance_mux_min_n,
        )
        with open(filepath, "w") as f:
            f.write(verilog_str)

    def analyze(self: "Netlist",
                *,
                include_wiring: bool = False,
                        include_consts: bool = False,
                        include_reg_cones: bool = True) -> GraphReport:
        """
        Analyze combinational cones of this module.
          - include_wiring=False → don't count Concat/Slice/Resize in node counts (still traversed)
          - include_consts=False → don't count Const in node counts
          - include_reg_cones=True → also traverse register driver cones (depth to sequential inputs)
        Depth model:
          - Op1/Op2/Ternary each add 1 level
          - Concat/Slice/Resize add 0 (transparent wiring)
          - Signals: inputs/regs are sources (depth=0); wires/outputs inline their driver
          - Const: depth=0
        """
        return _Analyzer(include_wiring, include_consts, include_reg_cones).run(self)

    def all_exprs(self) -> List[Expr]:
        """Depth-first traversal of every expression in the module."""
        seen = set()
        exprs = []

        def add_expr(e: Expr):
            if id(e) not in seen:
                seen.add(id(e))
                exprs.append(e)

        def visit(e: Expr):

            if id(e) in seen:
                return

            add_expr(e)
            # Recurse through children
            if hasattr(e, "a"):
                visit(e.a)
            if hasattr(e, "b"):
                visit(e.b)
            if hasattr(e, "sel"):
                visit(e.sel)
            if hasattr(e, "parts"):
                for p in e.parts:
                    visit(p)
            if hasattr(e, "_driver"):
                if e._driver is not None:
                    visit(e._driver)

        for s in self._signals:
            # outputs
            if s.kind == "output":
                add_expr(s)
            if s._driver is not None:
                visit(s._driver)

        return exprs


class _SignalCollector(ExprVisitor[None]):
    """Walk a design and rebuild `Module._signals`.

    Side-effects only: each visited Signal is uniquified and appended to `_signals` (once). Traversal goes through
    `expr_children`, which knows about Memory port-wires and the port-wire back-edge to its parent.
    """

    def __init__(self, module: "Module") -> None:
        super().__init__()
        self.m = module
        self.port_ids = {id(p) for p in module._ports}
        self.in_list = set(self.port_ids)
        self.name_to_sig: Dict[str, "Signal"] = {p.name: p for p in module._ports}

    def run(self, seeds: List["Signal"]) -> None:
        for s in seeds:
            self.visit(s)

    def visit_signal(self, s: Signal) -> None:
        sid = id(s)
        if sid not in self.port_ids:
            if s.kind in ("input", "output"):
                raise RuntimeError(
                    f"Internal signal '{s.name}' has port kind '{s.kind}'. "
                    "Use wire/reg for internals. For internal components use make_internal()")
            self._uniquify(s)
            if sid not in self.in_list:
                self.m._signals.append(s)
                self.in_list.add(sid)
        # Walk children explicitly. Unlike `expr_children` (which treats regs as comb-depth-zero leaves for
        # analyzer purposes), the collector must traverse reg drivers too — otherwise externally-created
        # Registers/Wires chained to module outputs would be missed.
        if isinstance(s, _MemoryArray):
            for p in s._iter_ports():
                self.visit(p)
        else:
            parent = getattr(s, "_memory_parent", None)
            if parent is not None:
                self.visit(parent)
            if s._driver is not None:
                self.visit(s._driver)
        # Blackbox support: visiting any IO wire of a blackbox triggers seeding from the peer IO wires too.
        # Without this, the parent's input-side wiring wouldn't be reached (blackbox outputs have no driver chain
        # back to inputs).
        owner = getattr(s, "_owning_component", None)
        if owner is not None and getattr(owner, "_is_blackbox", False):
            for peer in iter_values(owner.io):
                if peer is not s:
                    self.visit(peer)

    # Default-recurse for non-Signal nodes via expr_children.
    def _walk(self, e: Expr) -> None:
        for ch in expr_children(e):
            self.visit(ch)

    visit_const = staticmethod(lambda e: None)
    visit_array_index = staticmethod(lambda e: None)
    visit_op1     = _walk
    visit_op2     = _walk
    visit_ternary = _walk
    visit_concat  = _walk
    visit_slice   = _walk
    visit_resize  = _walk

    def _uniquify(self, sig: Signal) -> None:
        # Memory port wires: name is `{mem.name}__suffix` derived from the current parent name. If the parent is
        # renamed later, ``_propagate_mem_rename`` (below) updates port wire names too — so we don't need to force
        # the parent to be visited first here (that would recurse infinitely because the parent's children include
        # this very port).
        parent = getattr(sig, "_memory_parent", None)
        if parent is not None:
            sig.name = f"{parent.name}__{sig._port_suffix}"
        base = sig.name
        existing = self.name_to_sig.get(base)
        if existing is sig:
            return
        if existing is None:
            self.name_to_sig[base] = sig
            return
        # Collision: suffix until free.
        idx = 1
        while True:
            candidate = f"{base}_{idx}"
            if candidate not in self.name_to_sig:
                sig.name = candidate
                self.name_to_sig[candidate] = sig
                # If we just renamed a Memory, propagate to its port wires that have already been uniquified
                # (we'll see new ports later via traversal, but already-cached ones won't be revisited).
                if isinstance(sig, _MemoryArray):
                    for port in sig._iter_ports():
                        old = port.name
                        new = f"{sig.name}__{port._port_suffix}"
                        port.name = new
                        if old in self.name_to_sig and self.name_to_sig[old] is port:
                            del self.name_to_sig[old]
                        self.name_to_sig[new] = port
                return
            idx += 1


def get_rand_hash() -> str:
    random_string = str(random.random()) + str(time.time())
    hash_object = hashlib.sha256(random_string.encode())
    name = str(hash_object.hexdigest())
    return name


class _PortGrouper:
    """
    Group scattered 1-bit ports like a[0], a[1], ..., a[N-1] into a wide UInt port 'a' of width N.
    Mutates the module in-place:
      - The old bit-ports are converted to internal 'wire's.
      - New composited ports are created and connected.
    API:
        IOCollector().group(m, {"a": UInt(16), "b": UInt(16), "y": UInt(16)})
    """

    def group(self, m: Module, spec: Dict[str, Any]) -> Dict[str, Any]:
        """
        spec: { base_name -> SpireHDL type (e.g., UInt(16)) }
        Returns a mapping { base_name -> composited Signal } for convenience.
        """
        out: Dict[str, Any] = {}
        for base, typ in spec.items():
            width = typ.width
            if width == 1:
                continue  # nothing to do for 1-bit ports
            # Gather all ports that look like base[i]
            bits = self._find_bit_ports(m, base, width)

            if not bits:
                raise ValueError(f"No ports found for base '{base}'")

            k = bits[0].kind
            if any(b.kind != k for b in bits):
                raise ValueError(f"Mixed directions for '{base}[i]': {[b.kind for b in bits]}")

            if k == "input":
                agg = self._create_agg_input_and_wire_bits(m, base, typ, bits)
            elif k == "output":
                agg = self._create_agg_output_from_bits(m, base, typ, bits)
            else:
                raise ValueError(f"Ports for '{base}[i]' are not inputs/outputs (found kind='{k}')")

            out[base] = agg
        return out

    # ---------------- internals ----------------

    def _find_bit_ports(self, m: Module, base: str, width: int):
        """Return ports [bit0, bit1, ..., bit{width-1}] by exact bracketed name."""
        # Build precise name map: "a[0]" -> Signal
        name_to_sig = {p.name: p for p in m._ports}
        bits = []
        for i in range(width):
            # Try underscore notation first (abc style: prod_0_),
            # then bracket notation (yosys/cleaned style: prod[0])
            nm = f"{base}_{i}_"
            s = name_to_sig.get(nm)
            if s is None: # try bracket notation, might be able to remove this fallback
                nm = f"{base}[{i}]"
                s = name_to_sig.get(nm)
            if s is None:
                raise ValueError(f"Missing bit-port '{base}_{i}_' or '{base}[{i}]'")
            # sanity: 1-bit?
            if s.typ.width != 1:
                raise ValueError(f"Expected 1-bit for '{nm}', got {s.typ}")
            bits.append(s)
        return bits

    def _demote_port_to_wire(self, m: Module, s):
        """Turn an input/output port into an internal wire (keeps drivers/uses intact)."""
        # if s in m._ports:
        #     m._ports.remove(s)
        # s.kind = "wire"  # from input/output → wire
        m._ports[:] = [p for p in m._ports if p is not s]
        s.kind = "wire"

    def _create_agg_input_and_wire_bits(self, m: Module, base: str, typ: Any, bits: List[Any]):
        """Create 'input <typ> base' and drive each former port-bit (now wire) from base[i]."""
        # Name clash?
        if any(p.name == base for p in m._ports):
            raise ValueError(f"Port '{base}' already exists.")
        agg = m.input(typ, base)
        # Demote and connect LSB..MSB
        for i, b in enumerate(bits):
            self._demote_port_to_wire(m, b)
            b <<= agg[i]  # drive internal bit-wire from wide input
        return agg

    def _create_agg_output_from_bits(self, m: Module, base: str, typ: Any, bits: List[Any]):
        """Create 'output <typ> base' as concat of LSB..MSB of the (now internal) bit signals."""
        if any(p.name == base for p in m._ports):
            raise ValueError(f"Port '{base}' already exists.")
        # Demote old bit-ports first (they already have drivers from the existing logic)
        for b in bits:
            self._demote_port_to_wire(m, b)

        agg = m.output(typ, base)
        # Build y = cat(LSB ... MSB)
        parts_lsb_to_msb = [bits[i] for i in range(typ.width)]
        agg <<= cat(*parts_lsb_to_msb)
        return agg


# `_to_composite` / `iter_values` now live in spirehdl.composite.record_dynamic
# (the layer that owns CompositeRecordDynamic); `iter_values` is imported at the top of this module.


# ---------------------------------------------------------------------------
# Deprecated aliases (kept for one release; migrate to the new names).
# ---------------------------------------------------------------------------
Module = Netlist            # `Module` was renamed to `Netlist`
IOCollector = _PortGrouper  # `IOCollector` was renamed to `_PortGrouper`
Netlist.module_analyze = Netlist.analyze  # `module_analyze` was renamed to `analyze`
