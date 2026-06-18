import abc
from dataclasses import dataclass, make_dataclass
import hashlib
import random
import time


from spirehdl import VERILOG_BANNER
from spirehdl.spirehdl import Bool, Const, Expr, ExprLike, HDLType, Signal, UInt, cat, fit_width, get_shared_wires, reset_shared_cache
from spirehdl.spirehdl_memory import _MemoryArray


from typing import Any, Dict, Iterable, List, Optional
from dataclasses import is_dataclass, fields

from spirehdl.aig.aig_yosys import verilog_to_aag_lines_via_yosys

try:  # Python 3.10 compatibility
    from typing import Self  # type: ignore
except ImportError:
    from typing_extensions import Self  # type: ignore

from spirehdl.spirehdl_analyzer import _Analyzer, GraphReport
from spirehdl.spirehdl_visitor import ExprVisitor, expr_children



# The flat netlist IR now lives in spirehdl.ir (one-way layering: component -> ir -> spirehdl).
# Re-exported here so existing `from spirehdl.spirehdl_module import Module/IOCollector/...` keep working.
from spirehdl.ir import (Netlist, Module, _PortGrouper, IOCollector, _SignalCollector, get_rand_hash)
# IO normalization helpers live in the composite layer; re-exported here for back-compat
# (e.g. `from spirehdl.spirehdl_module import iter_values`).
from spirehdl.composite.record_dynamic import _to_composite, iter_values


class Component(abc.ABC):

    io: dataclass | Dict

    # define attribute name
    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abc.abstractmethod
    def elaborate(self) -> None:
        """Define the component's internal logic (drive outputs, instantiate sub-components).

        Abstract: every concrete Component must implement it. Components whose logic arrives via
        import rather than elaboration subclass :class:`ImportedComponent` (a no-op ``elaborate``).
        """
        ...

    def get_ios(self) -> "CompositeRecordDynamic":
        """Return this component's IO as an composite record — the single IO normalization point.

        Default: normalize ``self.io`` (dataclass / dict / namedtuple / ``IORecord``) via
        ``_to_composite``. Override for components whose IO is not a simple stored ``self.io``
        (e.g. generated or imported wrappers that build their ports dynamically).
        """
        return _to_composite(self.io)

    # convenience helpers -------------------------------------------------------

    def to_verilog(self, name: Optional[str] = None, *, with_clock: bool = False,
                   with_reset: bool = False, **emit_opts) -> str:
        """Lower to the netlist IR and emit Verilog. Users never touch the IR directly."""
        return self.to_netlist(name, with_clock=with_clock, with_reset=with_reset).to_verilog(**emit_opts)

    def to_verilog_file(self, filepath: str, name: Optional[str] = None, *, with_clock: bool = False,
                        with_reset: bool = False, **emit_opts) -> None:
        self.to_netlist(name, with_clock=with_clock, with_reset=with_reset).to_verilog_file(filepath, **emit_opts)

    def to_aag(self, name: Optional[str] = None, *, with_clock: bool = False,
               with_reset: bool = False) -> List[str]:
        """Lower to the netlist IR and export AIGER (AAG) lines."""
        from spirehdl.spirehdl_aiger import AigerExporter
        return AigerExporter(self.to_netlist(name, with_clock=with_clock, with_reset=with_reset)).get_aag()

    def analyze(self, name: Optional[str] = None, *, with_clock: bool = False,
                with_reset: bool = False, **opts) -> GraphReport:
        """Lower to the netlist IR and run combinational-cone analysis."""
        return self.to_netlist(name, with_clock=with_clock, with_reset=with_reset).analyze(**opts)

    def to_netlist(self, name: Optional[str] = None, with_clock: bool = False, with_reset: bool = False) -> 'Netlist':
        module = Netlist(
            name or f"comp_{get_rand_hash()}",
            with_clock=with_clock,
            with_reset=with_reset,
        )

        for sig in self.get_ios().to_list():
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
        # Top-level Component with custom Verilog: tag elaborate-created signals so the emitter skips them.
        # (Embedded Components do this in make_internal.)
        if hasattr(self, 'custom_verilog'):
            self._apply_custom_verilog_tags()
        #reset_shared_cache() # no longer needed as we collect signals
        module.collect_signals()
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
            self.inline()

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

    @classmethod
    def from_netlist(cls, net: "Netlist") -> "Component":
        """Wrap a netlist's ports as a Component IO (shares Signal objects, no copy).

        Absorbs the former ``Netlist.to_component()`` — this is the spec-free reinsertion path
        (e.g. an optimized AIG with no predefined Component subclass). Port names that are not valid
        Python identifiers (e.g. bit-ports ``a[0]``) are sanitized for the IO field name; the Signal
        keeps its original ``.name`` so codegen/analysis are unaffected. Returns an
        ``ImportedComponent`` (a concrete trivial subclass), so the ABC contract holds.
        """
        port_signals = [p for p in net._ports if p.kind in ("input", "output")]

        def sanitize(n: str) -> str:
            s = ''.join(c if (c.isalnum() or c == '_') else '_' for c in n)
            if not s or s[0].isdigit():
                s = f"p_{s}"
            return s

        used: Dict[str, int] = {}
        io_fields: List[tuple[str, type]] = []
        values: Dict[str, Signal] = {}
        for sig in port_signals:
            base = sanitize(sig.name)
            idx = used.get(base, 0)
            used[base] = idx + 1
            field_name = base if idx == 0 else f"{base}_{idx}"
            io_fields.append((field_name, Signal))
            values[field_name] = sig

        IO = make_dataclass("IO", io_fields)
        comp = ImportedComponent()
        comp.io = IO(**values)
        return comp

    def inline(self) -> Self:
        # Inline this Component's logic into the surrounding design: retag its IO signals as wires so the
        # parent's signal collector splices the child's graph in (it does NOT instantiate a sub-module).
        # If this Component supplies a custom Verilog body, tag its elaborate()-created internals so the parent's
        # emitter skips them. Do this BEFORE flipping kind="output" → "wire" so the tagger can find the outputs by
        # their original direction.
        if hasattr(self, 'custom_verilog'):
            self._apply_custom_verilog_tags()
        # go through all signals in io and change to 'wire'
        for sig in self.get_ios().to_list():
            if sig.kind in ('input', 'output'):
                sig.kind = 'wire'
                # Back-reference so the parent's emitter knows which Component owns this wire (and thus which
                # custom_verilog block to call).
                if hasattr(self, 'custom_verilog'):
                    sig._owning_component = self
            else:
                raise ValueError(f"Signal {sig.name} has unsupported kind '{sig.kind}'")
        return self

    def _apply_custom_verilog_tags(self) -> None:
        """Walk from IO outputs and tag elaborate()-created signals as no-emit.

        Internal signals (anything reachable from an IO output via driver chains that isn't itself an IO port) get
        both flags set — they don't appear in the emitted Verilog at all. IO outputs get only ``_no_emit_drive`` set:
        the port/wire declaration stays (parent code references it) but the elaborate-set ``assign`` is suppressed,
        leaving the custom Verilog block to provide the actual value.

        Sets ``self._is_blackbox = True`` iff *no* output had an elaborate-set driver — the signal the collector
        uses to decide whether peer-seeding is needed (only blackboxes need it).
        """
        from spirehdl.spirehdl_visitor import expr_children  # local to avoid cycles
        io_ids = {id(s) for s in self.get_ios().to_list()}
        stack: List = []
        for sig in self.get_ios().to_list():
            if sig.kind == "output":
                sig._no_emit_drive = True
                if sig._driver is not None:
                    stack.append(sig._driver)
        self._is_blackbox = not stack  # nothing to walk = no elaborate output drivers = blackbox
        visited: set = set()
        while stack:
            node = stack.pop()
            nid = id(node)
            if nid in visited:
                continue
            visited.add(nid)
            if isinstance(node, Signal):
                # Boundary check: signals owned by a *different* Component (i.e. a `make_internal`'d sub-Component)
                # are that Component's namespace — its tagging is already done at its own `make_internal` time. We
                # neither tag them here nor walk past them; the boundary stops us.
                owner = getattr(node, "_owning_component", None)
                if owner is not None and owner is not self:
                    continue
                if nid not in io_ids:
                    node._no_emit_decl = True
                    node._no_emit_drive = True
                if node._driver is not None:
                    stack.append(node._driver)
                # Mirror `collect_signals`' store traversal: the `_ArrayIndex` leaf stops the driver walk, so
                # follow store ↔ port-wire edges explicitly — else state reachable only through a store (e.g. a
                # FIFO's write pointer via `store.write_addr`) is collected but never tagged, leaking into the Verilog.
                if isinstance(node, _MemoryArray):
                    for p in node._iter_ports():
                        stack.append(p)
                parent = getattr(node, "_memory_parent", None)
                if parent is not None:
                    stack.append(parent)
            else:
                for ch in expr_children(node):
                    stack.append(ch)

    def get_spec(self) -> Dict[str, UInt]:
        return {s.name: s.typ for s in self.get_ios().to_list()}

    # Deprecated method aliases (renamed for clarity; kept for one release).
    to_module = to_netlist     # `to_module` was renamed to `to_netlist`
    make_internal = inline     # `make_internal` was renamed to `inline`


class ImportedComponent(Component):
    """A Component whose logic arrives via import (``from_netlist`` / ``from_aag_lines`` /
    ``from_verilog``) rather than ``elaborate()``. Satisfies the (now abstract) ``elaborate()``
    with a no-op — its logic is reinjected by ``from_module`` at import time, not built here."""

    def elaborate(self) -> None:
        pass


def gen_spec(class_instance: Component) -> Dict[str, UInt]:
    # Deprecated: superseded by Component.get_spec() / get_ios(). Routes through the same
    # normalization so dict / namedtuple / IORecord IO all work (the old io.__dict__ walk
    # silently broke for dict IO).
    return {sig.name: sig.typ for sig in iter_values(class_instance.io)}

