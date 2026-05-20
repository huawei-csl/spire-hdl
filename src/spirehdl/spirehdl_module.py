import abc
from dataclasses import dataclass, make_dataclass
import hashlib
import random
import time


from spirehdl import VERILOG_BANNER
from spirehdl.spirehdl import Bool, Const, Expr, ExprLike, HDLType, MemRead, Memory, Signal, UInt, cat, fit_width, get_shared_wires, reset_shared_cache


from typing import Any, Dict, Iterable, List, Optional
from dataclasses import is_dataclass, fields

from spirehdl.aig.aig_yosys import verilog_to_aag_lines_via_yosys

try:  # Python 3.10 compatibility
    from typing import Self  # type: ignore
except ImportError:
    from typing_extensions import Self  # type: ignore

from spirehdl.spirehdl_analyzer import _Analyzer, GraphReport


class Component(abc.ABC):

    io: dataclass | Dict

    # define attribute name
    @property
    def name(self) -> str:
        return self.__class__.__name__

    # @abc.abstractmethod
    def elaborate(self) -> None:  # pragma: no cover - structural hook
        # raise NotImplementedError
        pass

    # convenience helpers -------------------------------------------------------

    def to_module(self, name: Optional[str] = None, with_clock: bool = False, with_reset: bool = False) -> 'Module':
        module = Module(
            name or f"comp_{get_rand_hash()}",
            with_clock=with_clock,
            with_reset=with_reset,
        )

        for sig in iter_values(self.io):
            sig: Signal

            # if is clock/reset assign to module clk/rst
            if sig.name == "clk":
                if module.clk is None:
                    module.clk = sig
                else:
                    # module already has a clock signal
                    pass
                continue
            if sig.name == "rst":
                if module.rst is None:
                    module.rst = sig
                else:
                    # module already has a reset signal
                    pass
                continue

            if sig.kind == "input":
                module.add_input(sig)
            elif sig.kind == "output":
                module.add_output(sig)
            else:
                raise ValueError(f"Signal {sig.name} has unsupported kind '{sig.kind}'")
        module.component = self # can be used for debugging
        #reset_shared_cache() # no longer needed as we collect signals
        module._collect_signals_from_outputs([s for s in iter_values(self.io) if s.kind == "output"])
        return module 

    def from_module(self, module: 'Module', make_internal=False, group=False) -> Self:
        if group:
            IOCollector().group(module, self.get_spec())

        # Map the AIG module's ports to this component's IO fields.
        # Use _ports (which have canonical grouped names after group())
        # rather than _signals (which may contain renamed duplicates).
        for sig in module._ports:
            if sig.kind in ('input', 'output'):
                setattr(self.io, sig.name, sig)
            else:
                raise ValueError(f"Signal {sig.name} has unsupported kind '{sig.kind}'")
        self.elaborate()  # re-elaborate to rebuild internal structure
        if make_internal:
            self.make_internal()

    def from_verilog(self, verilog_str: str, top=None, group=True) -> Self:
        from spirehdl.aig.aig_yosys import aig_file_to_aag_lines_via_yosys

        aag_lines = verilog_to_aag_lines_via_yosys(verilog_str, top=top, embed_symbols=True, no_startoffset=True)
        self.from_aag_lines(aag_lines, group=group)

    def from_aig_file(self, aig_path: str, map_file: str|None = None, group=True, make_internal=False) -> Self:
        from spirehdl.aig.aig_yosys import aig_file_to_aag_lines_via_yosys

        aag_lines = aig_file_to_aag_lines_via_yosys(aig_path, map_file=map_file)
        self.from_aag_lines(aag_lines, group=group, make_internal=make_internal)
        return self

    def from_aag_lines(self, aag_lines: List[str], group=True, make_internal=True) -> Self:
        from spirehdl.spirehdl_aiger import AigerImporter

        m = AigerImporter(aag_lines).get_spirehdl_module()
        self.from_module(m, make_internal=make_internal, group=group)

    def make_internal(self) -> Self:
        # go through all signals in io and change to 'wire'
        ios_dict = self.io if isinstance(self.io, dict) else self.io.__dict__
        for sig in ios_dict.values():
            if sig.kind in ('input', 'output'):
                sig.kind = 'wire'
            else:
                raise ValueError(f"Signal {sig.name} has unsupported kind '{sig.kind}'")
        return self

    def get_spec(self) -> Dict[str, UInt]:
        return gen_spec(self)


class Module:
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
        s = Signal(name, typ, "input") #, self)
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
        s = Signal(name, typ, "output") #, self)
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
        s = Signal(name, typ, "wire") #, self)
        self._signals.append(s)
        return s

    def reg(self, typ: HDLType, name: str, init: Optional[ExprLike] = None) -> Signal:
        s = Signal(name, typ, "reg") #, self)
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
        """
        Walk the design starting from outputs
        """
        self._collect_signals_from_outputs(self._ports_of("output"))

    # fast version
    def _collect_signals_from_outputs(self, outputs: List["Signal"]) -> None:
        """
        Walk the design starting from outputs, pulling every reachable Signal into _signals.
        Internal signals must be wire/reg; encountering an input/output that is not a port raises.
        Name collisions are avoided by suffixing internal signal names.
        """

        # Optional: comment these out on huge graphs (printing itself can be slow)
        # print("Collecting signals...")

        # Start with ports, keep their names stable
        self._signals = list(self._ports)
        port_ids = {id(p) for p in self._ports}

        # O(1) membership for "already appended to _signals"
        signals_in_list = set(port_ids)

        # Name tracking (ports first)
        name_to_sig: Dict[str, "Signal"] = {p.name: p for p in self._ports}

        visited_signal_ids: set[int] = set()
        visited_expr_ids: set[int] = set()

        # Cache: expr type -> function that yields child expressions
        child_extractor_cache: Dict[type, callable] = {}

        def uniquify_internal(sig: "Signal") -> None:
            # Ports keep their names
            if id(sig) in port_ids:
                return

            base = sig.name
            existing = name_to_sig.get(base)

            # If mapping already points to this exact signal, nothing to do
            if existing is sig:
                return

            # If name is free, claim it
            if existing is None:
                name_to_sig[base] = sig
                return

            # Otherwise, suffix until free
            idx = 1
            while True:
                candidate = f"{base}_{idx}"
                if candidate not in name_to_sig:
                    sig.name = candidate
                    name_to_sig[candidate] = sig
                    return
                idx += 1

        def get_children(e: "Expr"):
            """Yield child expressions of e, with per-type caching of which fields to traverse."""
            t = type(e)
            fn = child_extractor_cache.get(t)
            if fn is None:
                if t is MemRead:
                    # MemRead has non-standard fields (mem, addr); the generic attr-probe
                    # below would miss both.
                    def fn(x):
                        yield x.mem
                        yield x.addr
                else:
                    # Probe once for this type using the current instance.
                    # This assumes structure is consistent across instances of the same class (usually true).
                    names = []
                    for n in ("a", "b", "sel", "_driver"):
                        if hasattr(e, n):
                            names.append(n)
                    has_parts = hasattr(e, "parts")

                    def fn(x, names=tuple(names), has_parts=has_parts):
                        for n in names:
                            yield getattr(x, n)
                        if has_parts:
                            for p in x.parts:
                                yield p

                child_extractor_cache[t] = fn
            return fn(e)

        # Iterative DFS stack over Expr|Signal|None
        stack = list(outputs)

        # Localize lookups for speed in tight loops
        v_sig = visited_signal_ids
        v_expr = visited_expr_ids
        v_sig_add = v_sig.add
        v_expr_add = v_expr.add
        s_in = signals_in_list
        s_in_add = s_in.add
        append_sig = self._signals.append

        while stack:
            node = stack.pop()
            if node is None:
                continue

            # Signals are Expr in your system, so check Signal first
            if isinstance(node, Signal):
                sid = id(node)
                if sid in v_sig:
                    continue
                v_sig_add(sid)

                if sid not in port_ids:
                    if node.kind in ("input", "output"):
                        raise RuntimeError(
                            f"Internal signal '{node.name}' has port kind '{node.kind}'. "
                            "Use wire/reg for internals. For internal components use make_internal()"
                        )
                    uniquify_internal(node)

                    if sid not in s_in:
                        append_sig(node)
                        s_in_add(sid)

                drv = node._driver
                if drv is not None:
                    stack.append(drv)

                # Memory-managed registers (registered-read outputs) have no _driver — their
                # next-state is the memory's MemRead. Push the parent memory so it gets walked.
                mem_owner = getattr(node, "_memory_managed_by", None)
                if mem_owner is not None:
                    stack.append(mem_owner)

                # Memory: its port exprs (write/reset/registered-read) aren't reachable via _driver,
                # so push them here so any expressions feeding them get collected.
                if node.kind == "mem":
                    for e in (node._write_addr, node._write_data, node._write_enable,
                              node._reset_enable, node._reset_value,
                              node._reg_read_addr, node._reg_read_enable):
                        if e is not None:
                            stack.append(e)
                    if node._reg_read_output is not None:
                        stack.append(node._reg_read_output)
                continue

            # Otherwise it's an Expr (non-Signal)
            eid = id(node)
            if eid in v_expr:
                continue
            v_expr_add(eid)

            # Push children
            for ch in get_children(node):
                if ch is not None:
                    stack.append(ch)

        # print(f"Collected {len(self._signals)} signals.")

    def to_component(self) -> Component:
        """
        Create a lightweight Component wrapper that exposes this module's ports as
        the component IO. This mirrors Component.to_module() in spirit by sharing
        the same Signal objects for ports (no copying), so drivers and expressions
        remain intact.

        Notes:
        - Field names in the IO dataclass must be valid Python identifiers. If a
          port name contains characters like '[', ']', etc. (e.g., bit-ports
          "a[0]"), the field name is sanitized (e.g., "a_0"). The underlying
          Signal keeps its original .name, so codegen and analysis are unaffected.
        """
        # Build IO dataclass fields for inputs/outputs
        port_signals = [p for p in self._ports if p.kind in ("input", "output")]

        def sanitize(n: str) -> str:
            s = ''.join(c if (c.isalnum() or c == '_') else '_' for c in n)
            if not s or s[0].isdigit():
                s = f"p_{s}"
            return s

        # Ensure unique field names after sanitization
        used: Dict[str, int] = {}
        fields: List[tuple[str, type]] = []
        values: Dict[str, Signal] = {}
        for sig in port_signals:
            base = sanitize(sig.name)
            idx = used.get(base, 0)
            used[base] = idx + 1
            field_name = base if idx == 0 else f"{base}_{idx}"
            fields.append((field_name, Signal))
            values[field_name] = sig

        IO = make_dataclass("IO", fields)

        # Minimal concrete Component instance with populated IO
        # class _ModuleWrappedComponent(Component):
        #    pass

        comp = Component() #_ModuleWrappedComponent()
        comp.io = IO(**values)
        return comp

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

        # Basic checks
        for s in self._signals:
            if s.kind in ("wire", "output") and s._driver is None:
                if s.kind == "output":
                    raise ValueError(f"Output '{s.name}' has no driver.")
            if s.kind == "reg" and s._driver is None:
                # Memory-managed registers (registered-read outputs) get their next-state
                # written by the memory's always block, not via _driver.
                if getattr(s, "_memory_managed_by", None) is None:
                    raise ValueError(f"Register '{s.name}' has no next-state assignment.")

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

        # Internals
        # wires = self._internals_of("wire") + get_shared_wires()
        # instead of the above merge to avoid duplication if called multiple times
        wires = self._internals_of("wire")
        if not collect_signals:
            wires += [s for s in get_shared_wires() if not any(s is w for w in wires)]

        regs_all = self._internals_of("reg")
        # Memory-managed registers (registered-read outputs) get their next-state in the
        # memory's own always block; exclude from the normal reg always block.
        regs = [r for r in regs_all if getattr(r, "_memory_managed_by", None) is None]
        mems = self._internals_of("mem")
        lines.append('// Wires')
        for w in wires:
            sign = "signed " if w.typ.signed else ""
            rng = w.typ.range_str()
            lines.append(f"  wire {sign}{rng} {w.name};")
        lines.append('// Registers')
        for r in regs_all:
            sign = "signed " if r.typ.signed else ""
            rng = r.typ.range_str()
            lines.append(f"  reg {sign}{rng} {r.name};")
        if mems:
            lines.append('// Memories')
            for m in mems:
                sign = "signed " if m.typ.signed else ""
                rng = m.typ.range_str()
                lines.append(f"  reg {sign}{rng} {m.name}[0:{m.depth-1}];")
        # Combinational assigns for wires/outputs
        lines.append("// Combinational assignments")
        for s in [*wires, *self._ports_of("output")]:
            if s._driver is not None:
                rhs = fit_width(s._driver, s.typ).to_verilog()
                lines.append(f"  assign {s.name} = {rhs};")

        # Memory initial values (ROM-style array literal)
        for m in mems:
            if m.init is not None:
                lines.append("initial begin")
                for i, v in enumerate(m.init):
                    lit = Const(v, m.typ).to_verilog()
                    lines.append(f"  {m.name}[{i}] = {lit};")
                lines.append("end")

        # Sequential logic
        lines.append("// Sequential logic")
        if regs:
            if not self.with_clock:
                raise ValueError("Registers present but module has no clock input.")
            sens = f"posedge {self.clk.name}"
            if self.with_reset:
                sens += f" or posedge {self.rst.name}"
            lines.append(f"  always @({sens}) begin")
            if self.with_reset:
                lines.append(f"    if ({self.rst.name}) begin")
                for r in regs:
                    init = r._init.to_verilog() if r._init is not None else f"{r.typ.width}'d0"
                    lines.append(f"      {r.name} <= {init};")
                lines.append("    end else begin")
                for r in regs:
                    lines.append(f"      {r.name} <= {fit_width(r._driver, r.typ).to_verilog()};")
                lines.append("    end")
            else:
                for r in regs:
                    lines.append(f"    {r.name} <= {fit_width(r._driver, r.typ).to_verilog()};")
            lines.append("  end")

        # Memory sequential logic — one always block per memory. Clock-only sensitivity
        # (no async rst) since memory reset is via the memory's own .reset(enable) port,
        # not the module's rst — matches the verilog idiom yosys's memory pass recognises.
        for m in mems:
            has_write = m._write_addr is not None
            has_reset = m._reset_enable is not None
            has_rread = m._reg_read_addr is not None
            if not (has_write or has_reset or has_rread):
                continue  # ROM only (init-only) — no sequential block
            if not self.with_clock:
                raise ValueError(f"Memory '{m.name}' has sequential ports but module has no clock.")
            lines.append(f"  always @(posedge {self.clk.name}) begin")
            if has_reset and has_write:
                re_v = m._reset_enable.to_verilog()
                rv_v = m._reset_value.to_verilog()
                wa_v = m._write_addr.to_verilog()
                wd_v = fit_width(m._write_data, m.typ).to_verilog()
                lines.append(f"    if ({re_v}) begin")
                for i in range(m.depth):
                    lines.append(f"      {m.name}[{i}] <= {rv_v};")
                if m._write_enable is not None:
                    we_v = m._write_enable.to_verilog()
                    lines.append(f"    end else if ({we_v}) begin")
                else:
                    lines.append(f"    end else begin")
                lines.append(f"      {m.name}[{wa_v}] <= {wd_v};")
                lines.append(f"    end")
            elif has_reset:
                re_v = m._reset_enable.to_verilog()
                rv_v = m._reset_value.to_verilog()
                lines.append(f"    if ({re_v}) begin")
                for i in range(m.depth):
                    lines.append(f"      {m.name}[{i}] <= {rv_v};")
                lines.append(f"    end")
            elif has_write:
                wa_v = m._write_addr.to_verilog()
                wd_v = fit_width(m._write_data, m.typ).to_verilog()
                if m._write_enable is not None:
                    we_v = m._write_enable.to_verilog()
                    lines.append(f"    if ({we_v}) begin")
                    lines.append(f"      {m.name}[{wa_v}] <= {wd_v};")
                    lines.append(f"    end")
                else:
                    lines.append(f"    {m.name}[{wa_v}] <= {wd_v};")
            if has_rread:
                ra_v = m._reg_read_addr.to_verilog()
                rout = m._reg_read_output.name
                if m._reg_read_enable is not None:
                    re_v = m._reg_read_enable.to_verilog()
                    lines.append(f"    if ({re_v}) begin")
                    lines.append(f"      {rout} <= {m.name}[{ra_v}];")
                    lines.append(f"    end")
                else:
                    lines.append(f"    {rout} <= {m.name}[{ra_v}];")
            lines.append("  end")

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

    def module_analyze(self: "Module",
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

def gen_spec(class_instance: Component) -> Dict[str, UInt]:
    spec = {}
    for sig in class_instance.io.__dict__.values():
        spec[sig.name] = sig.typ
    return spec

def get_rand_hash() -> str:
    random_string = str(random.random()) + str(time.time())
    hash_object = hashlib.sha256(random_string.encode())
    name = str(hash_object.hexdigest())
    return name


class IOCollector:
    """
    Group scattered 1-bit ports like a[0], a[1], ..., a[N-1] into a wide UInt port 'a' of width N.
    Mutates the module in-place:
      - The old bit-ports are converted to internal 'wire's.
      - New aggregated ports are created and connected.
    API:
        IOCollector().group(m, {"a": UInt(16), "b": UInt(16), "y": UInt(16)})
    """

    def group(self, m: Module, spec: Dict[str, Any]) -> Dict[str, Any]:
        """
        spec: { base_name -> SpireHDL type (e.g., UInt(16)) }
        Returns a mapping { base_name -> aggregated Signal } for convenience.
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


def _to_aggregate(obj: Any) -> "AggregateRecordDynamic":
    """Convert any IO container into an AggregateRecordDynamic for uniform handling."""
    from spirehdl.aggregate.aggregate_record_dynamic import AggregateRecordDynamic
    if isinstance(obj, AggregateRecordDynamic):  # already an aggregate — no conversion needed
        return obj
    if is_dataclass(obj):  # @dataclass IO (most common Component IO pattern)
        pairs = [(f.name, getattr(obj, f.name)) for f in fields(obj)]
    elif hasattr(obj, "_fields"):  # namedtuple IO
        pairs = [(n, getattr(obj, n)) for n in obj._fields]
    elif isinstance(obj, dict):  # plain dict IO
        pairs = list(obj.items())
    else:  # plain object with __dict__
        pairs = list(vars(obj).items())
    DynIO = make_dataclass("DynIO", [(n, type(v)) for n, v in pairs], bases=(AggregateRecordDynamic,))
    return DynIO(**{n: v for n, v in pairs})


def iter_values(obj: Any) -> Iterable[Any]:
    """Extract leaf Signals from an IO container (dataclass, namedtuple, dict,
    or HDLAggregate).  Converts to AggregateRecordDynamic first, then flattens
    via to_list() — all aggregate fields (Array, Record, ...) and plain lists
    are recursively resolved into individual Signals."""
    return _to_aggregate(obj).to_list()
