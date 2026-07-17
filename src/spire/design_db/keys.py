"""Spec-key canonicalization for the design DB.

A slot is content-addressed: ``spec_key = sha256(structural AAG ‖ port spec)``. The key is computed
over the *numeric* AAG section (header + inputs + latches + outputs + AND gates) exported by
``AigerExporter`` — internal wire names never appear there, so the key is independent of the build
context (the global ``_SharedCache`` name counter leaks context into ``to_verilog()`` output, which
is why the raw Verilog is *not* the key input; it is still what gets stored as ``golden.v``).
The interface (port names/widths/dirs) enters via the port-spec JSON.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Tuple

# Fixed top name used when lowering a Component: ``Component.to_netlist()``'s default name is
# randomized (``comp_<hash>``), which would leak into the Verilog and break content addressing.
CANONICAL_TOP = "design_db_top"

_PORT_KINDS = ("input", "output")


def normalize(module_or_component: Any, name: str = CANONICAL_TOP) -> Any:
    """Return a ``Netlist`` for a Component or Netlist input (Component → ``to_netlist(name)``)."""
    obj = module_or_component
    if hasattr(obj, "_ports"):                     # already a Netlist
        return obj
    if hasattr(obj, "to_netlist"):                 # Component — lower with a *fixed* top name
        return obj.to_netlist(name)
    raise TypeError(f"expected a spire Netlist or Component, got {type(obj).__name__}")


def port_spec(module: Any) -> List[Dict[str, Any]]:
    """Stable port description ``[{name, width, signed, dir}]`` in declaration order."""
    ports: List[Dict[str, Any]] = []
    for s in module._ports:
        if s.kind not in _PORT_KINDS:
            continue
        ports.append({
            "name": s.name,
            "width": s.typ.width,
            "signed": bool(getattr(s.typ, "signed", False)),
            "dir": s.kind,
        })
    return ports


def _structural_aag(aag_lines: List[str]) -> List[str]:
    """The numeric AAG section only: header + I + L + O + A lines (no symbols/comments)."""
    head = aag_lines[0].split()
    n_in, n_latch, n_out, n_and = (int(x) for x in head[2:6])
    return aag_lines[:1 + n_in + n_latch + n_out + n_and]


def spec_key(aag_lines: List[str], ports: List[Dict[str, Any]]) -> str:
    """sha256 over the structural AAG + the canonical JSON of the port spec."""
    h = hashlib.sha256()
    h.update("\n".join(_structural_aag(aag_lines)).encode("utf-8"))
    h.update(json.dumps(ports, sort_keys=True).encode("utf-8"))
    return h.hexdigest()


def golden_and_key(module: Any) -> Tuple[str, List[Dict[str, Any]], str]:
    """Emit the golden Verilog and compute ``(verilog, ports, spec_key)`` for a Netlist."""
    from spire.aiger import AigerExporter   # deferred: keeps `import spire.design_db` light
    verilog = module.to_verilog()
    ports = port_spec(module)
    return verilog, ports, spec_key(AigerExporter(module).get_aag(), ports)
