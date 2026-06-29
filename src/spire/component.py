import abc
from dataclasses import make_dataclass
import hashlib
import random
import time


from spire import VERILOG_BANNER
from spire.expr import Bool, Const, Expr, ExprLike, HDLType, Signal, UInt, cat, fit_width, get_shared_wires, reset_shared_cache
from spire.memory import _MemoryArray


from typing import Any, Dict, Iterable, List, Optional
from dataclasses import is_dataclass, fields

from spire.aig.aig_yosys import verilog_to_aag_lines_via_yosys

try:  # Python 3.10 compatibility
    from typing import Self  # type: ignore
except ImportError:
    from typing_extensions import Self  # type: ignore

from spire.analyzer import _Analyzer, GraphReport
from spire.visitor import ExprVisitor, expr_children


# The flat netlist IR now lives in spire.ir (one-way layering: component -> ir -> spire).
# Re-exported here so existing `from spire.component import IOCollector/...` keep working.
from spire.ir import (Netlist, _PortGrouper, IOCollector, _SignalCollector, get_rand_hash)
# IO normalization helpers live in the composite layer; re-exported here for back-compat
# (e.g. `from spire.component import iter_values`).
from spire.composite.record import CompositeRecord, _to_composite, iter_values


class _ComponentMeta(abc.ABCMeta):
    """Run ``_finalize()`` once, right after a Component is fully constructed.

    Hooking the metaclass ``__call__`` (instead of wrapping ``__init__`` in ``__init_subclass__``)
    fires the hook exactly once for the outermost ``Cls(...)``: a subclass ``__init__`` that calls
    ``super().__init__()`` does NOT re-trigger it, so no re-entrancy bookkeeping is needed and the
    user's ``__init__`` is left untouched (clean tracebacks; works with ``@dataclass`` IO).
    """

    def __call__(cls, *args, **kwargs):
        obj = super().__call__(*args, **kwargs)   # runs the full __init__ chain (ABCMeta enforces abstractness)
        obj._finalize()
        return obj


class Component(abc.ABC, metaclass=_ComponentMeta):

    io: "CompositeRecord | Any"

    def _finalize(self) -> None:
        """Tag each IO leaf with its owning Component once construction is complete.

        Ownership lets the netlist attribute IO leaves by identity (membership in ``_ports`` decides
        global-vs-internal) with no destructive ``kind`` mutation — this replaced the old ``inline()``.
        Subclasses extend it (see ``CustomVerilogComponent``).

        Contract: ``_ComponentMeta`` calls this the moment ``__init__`` returns, so every subclass must
        have ``self.io`` set — or ``get_ios()`` overridden — by then, else construction fails fast here.
        """
        for sig in self.get_ios().to_list():
            sig._owning_component = self

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

    def get_ios(self) -> "CompositeRecord":
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
        from spire.aiger import AigerExporter
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
        # reset_shared_cache() # no longer needed as we collect signals
        module.collect_signals()
        return module 

    def from_module(self, module: 'Netlist', group=False) -> Self:
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
        self._finalize()
        # No inlining step: if this imported component is later embedded in a parent, the parent's
        # emitter classifies its ports as internal wires by membership (see ir.py).

    def from_verilog(self, verilog_str: str, top=None, group=True) -> Self:
        from spire.aig.aig_yosys import aig_file_to_aag_lines_via_yosys

        aag_lines = verilog_to_aag_lines_via_yosys(verilog_str, top=top, embed_symbols=True, no_startoffset=True)
        self.from_aag_lines(aag_lines, group=group)

    def from_aig_file(self, aig_path: str, map_file: str|None = None, group=True) -> Self:
        from spire.aig.aig_yosys import aig_file_to_aag_lines_via_yosys

        aag_lines = aig_file_to_aag_lines_via_yosys(aig_path, map_file=map_file)
        self.from_aag_lines(aag_lines, group=group)
        return self

    def from_aag_lines(self, aag_lines: List[str], group=True) -> Self:
        from spire.aiger import AigerImporter

        m = AigerImporter(aag_lines).get_spire_module()
        self.from_module(m, group=group)

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

        IO = make_dataclass("IO", io_fields, bases=(CompositeRecord,))
        return ImportedComponent(IO(**values))

    def get_spec(self) -> Dict[str, UInt]:
        return {s.name: s.typ for s in self.get_ios().to_list()}

    # Deprecated method aliases (renamed for clarity; kept for one release).
    to_module = to_netlist     # `to_module` was renamed to `to_netlist`


class ImportedComponent(Component):
    """A Component whose logic arrives via import (``from_netlist`` / ``from_aag_lines`` /
    ``from_verilog``) rather than ``elaborate()``. Satisfies the (now abstract) ``elaborate()``
    with a no-op — its logic is reinjected by ``from_module`` at import time, not built here."""

    def __init__(self, io: "CompositeRecord | Any") -> None:
        self.io = io

    def elaborate(self) -> None:
        pass


class CustomVerilogComponent(Component):
    """Base for components whose emitted implementation is supplied by ``custom_verilog()``."""

    _is_blackbox: bool = False   # set in _apply_custom_verilog_tags: True iff no output has an elaborate driver

    def _finalize(self) -> None:
        super()._finalize()
        self._apply_custom_verilog_tags()

    @abc.abstractmethod
    def custom_verilog(self) -> str:
        ...

    def _apply_custom_verilog_tags(self) -> None:
        """Suppress the ``elaborate()`` logic of a custom-Verilog component so only its custom block emits.

        A custom-Verilog component keeps two implementations: the ``elaborate()`` graph (for simulation) and
        the ``custom_verilog()`` string (for emission). Walking the graph backward from the IO outputs, this:

        - tags internal signals (reachable from an output, not themselves IO) ``_no_emit_decl`` +
          ``_no_emit_drive`` — they drop out of the Verilog entirely;
        - tags IO outputs ``_no_emit_drive`` only — the declaration stays (parents reference it) but the
          elaborate ``assign`` is dropped, so the custom block provides the value;
        - for memory, follows store ↔ port-wire edges so state reachable only through a store is tagged too;
        - for sub-components, stops at any signal owned by a *different* Component (it self-tags) — neither
          tagging nor crossing it;
        - for blackboxes, sets ``self._is_blackbox`` when no output had an elaborate driver — the cue the
          collector uses to peer-seed the inputs.
        """
        io_ids = {id(s) for s in self.get_ios().to_list()}
        stack: List = []
        for sig in self.get_ios().to_list():
            if sig.kind == "output":
                sig._no_emit_drive = True
                if sig._driver is not None:
                    stack.append(sig._driver)
        self._is_blackbox = not stack
        visited: set = set()
        while stack:
            node = stack.pop()
            nid = id(node)
            if nid in visited:
                continue
            visited.add(nid)
            if not isinstance(node, Signal):
                stack.extend(expr_children(node))
                continue
            owner = getattr(node, "_owning_component", None)
            if owner is not None and owner is not self:
                continue   # sub-component boundary
            if nid not in io_ids:
                node._no_emit_decl = True
                node._no_emit_drive = True
            if node._driver is not None:
                stack.append(node._driver)
            if isinstance(node, _MemoryArray):
                stack.extend(node._iter_ports())
            parent = getattr(node, "_memory_parent", None)
            if parent is not None:
                stack.append(parent)


def gen_spec(class_instance: Component) -> Dict[str, UInt]:
    # Deprecated: superseded by Component.get_spec() / get_ios(). Routes through the same
    # normalization so dict / namedtuple / IORecord IO all work (the old io.__dict__ walk
    # silently broke for dict IO).
    return {sig.name: sig.typ for sig in iter_values(class_instance.io)}
