"""``@from_design_db`` — the pure selection decorator.

Trace the decorated function (exactly like ``@abc_optimized``) → register its slot (golden + spec +
default verification + starting-point capture) → select the best admitted implementation by
``objective``/``metric`` → splice it in place of the function. **Miss ⇒ the original logic** (a build
never fails because a slot isn't optimized yet). ``pin=`` demands an exact design (missing ⇒ error);
``fill=`` is the opt-in generate-on-miss hook (called once, then re-select).

The decorator never generates and never spends budget — it is a reader over the DB.
"""
from __future__ import annotations

import builtins
import functools
import inspect
import textwrap
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from spire.design_db import keys as _keys
from spire.design_db.select import ObjectiveSpec, select_design
from spire.design_db.store import DesignDB, DesignDBError, register_slot

_MISS_NOTED: set = set()

# Names importable from `spire` that a self-contained starting point may reference freely.
_ALLOWED_GLOBALS = {"UInt", "SInt", "Bool", "Signal", "Wire", "Register", "Input", "Output",
                    "IORecord", "Component"}


def from_design_db(_fn: Optional[Callable[..., Any]] = None, *,
                   objective: ObjectiveSpec = "area", metric: Optional[str] = None,
                   pin: Optional[str] = None, fill: Optional[Callable[..., Any]] = None,
                   db: Optional[Any] = None) -> Callable[..., Any]:
    """Select the decorated subcircuit's implementation from the design DB.

    Usage (bare or parameterized, like the other optimization decorators)::

        @from_design_db(objective="area")
        def mac(a, b, c):
            return a * b + c
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        fn_sig = inspect.signature(fn)

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            bound = fn_sig.bind(*args, **kwargs)
            logic_args: Dict[str, Tuple[int, bool]] = {}
            other_args: Dict[str, Any] = {}
            actual: Dict[str, Any] = {}
            for pname, param in fn_sig.parameters.items():
                if pname not in bound.arguments:
                    continue
                value = bound.arguments[pname]
                if param.kind == inspect.Parameter.VAR_POSITIONAL:
                    for i, v in enumerate(value):
                        vn = f"{pname}_{chr(ord('a') + i)}"
                        if hasattr(v, "typ"):
                            logic_args[vn] = (v.typ.width, v.typ.signed)
                            actual[vn] = v
                        else:
                            other_args[vn] = v
                elif hasattr(value, "typ"):                  # an Expr
                    logic_args[pname] = (value.typ.width, value.typ.signed)
                    actual[pname] = value
                else:
                    other_args[pname] = value
            if not logic_args:
                return fn(*args, **kwargs)

            # Heavy machinery only now (keeps `import spire.design_db` light).
            from spire.expr import HDLType
            from spire.optimize.optimize import _build_component, _instantiate_from_cache

            comp, output_names = _build_component(fn, logic_args, other_args)
            module = comp.to_netlist(_keys.CANONICAL_TOP)
            key = register_slot(module, db=db, name=fn.__qualname__)
            d = DesignDB.open(db)
            _capture_source(d, key, fn)

            sel = select_design(key, objective=objective, metric=metric, pin=pin, db=db, record=True)
            if sel is None and fill is not None:
                fill(key, db_root=d.root, objective=objective, metric=metric)
                sel = select_design(key, objective=objective, metric=metric, db=db, record=True)
            if sel is None:
                if key not in _MISS_NOTED:
                    print(f"[spire.design_db] slot {key[:12]}… has no admitted design — "
                          f"using the original logic for {fn.__qualname__}")
                    _MISS_NOTED.add(key)
                return fn(*args, **kwargs)                   # the original logic, inline

            aag_lines = _load_design_aag(d, key, sel.design_id)
            spec_ports = d.read_json(d.slot_dir(key) / "spec.json")["ports"]
            spec = {p["name"]: HDLType(p["width"], signed=p["signed"]) for p in spec_ports}
            return _instantiate_from_cache(aag_lines, spec, output_names, actual)

        return wrapper

    return decorator(_fn) if _fn is not None else decorator


# --- splice input -----------------------------------------------------------------------------


def _load_design_aag(d: DesignDB, spec_key: str, design_id: str) -> list:
    """AAG lines of an admitted design — precomputed at insert; converted on demand as fallback."""
    ddir = d.slot_dir(spec_key) / "designs" / design_id
    aag_path = ddir / "design.aag"
    if aag_path.exists():
        return aag_path.read_text().splitlines()
    from spire.aig.aig_yosys import verilog_to_aag_lines_via_yosys
    aag_lines = verilog_to_aag_lines_via_yosys(str(ddir / "design.v"))
    try:                                                     # cache for next time (best effort)
        d.atomic_write_text(aag_path, "\n".join(aag_lines) + "\n")
    except OSError:
        pass
    return aag_lines


# --- starting-point capture -------------------------------------------------------------------


def _capture_source(d: DesignDB, spec_key: str, fn: Callable[..., Any]) -> None:
    """Store the decorated function's *current* source as the slot's primary starting point:
    ``source_ref`` in spec.json (pointer to the defining file — full context on the shared
    filesystem) + ``starting_point.py`` (captured body; runnable wrapper when self-contained).
    Best-effort: capture problems must never break a build."""
    slot = d.slot_dir(spec_key)
    spec = d.read_json(slot / "spec.json", None)
    if spec is None or "source_ref" in spec:
        return
    try:
        src_file = inspect.getsourcefile(fn)
        src_lines, line = inspect.getsourcelines(fn)
    except (OSError, TypeError):
        return
    source_ref = {"file": src_file, "qualname": fn.__qualname__, "line": line}
    body_lines = textwrap.dedent("".join(src_lines)).splitlines()
    while body_lines and body_lines[0].lstrip().startswith("@"):
        body_lines.pop(0)                                    # strip decorator lines
    body = "\n".join(body_lines).rstrip() + "\n"

    unresolved = [n for n in fn.__code__.co_names
                  if not hasattr(builtins, n) and n not in _ALLOWED_GLOBALS]
    fidelity = "self-contained" if not unresolved else "fragment"

    spec["source_ref"] = source_ref
    spec["starting_point"] = {"fidelity": fidelity}
    d.write_json(slot / "spec.json", spec)
    if not (slot / "starting_point.py").exists():
        text = _starting_point_text(fn.__name__, body, spec["ports"], fidelity, spec_key, source_ref)
        d.atomic_write_text(slot / "starting_point.py", text)


def _starting_point_text(fn_name: str, body: str, ports: list, fidelity: str,
                         spec_key: str, source_ref: Dict[str, Any]) -> str:
    header = (f'"""Auto-generated starting point for design-DB slot {spec_key[:12]}…\n\n'
              f"fidelity: {fidelity}\n"
              f"captured from: {source_ref['file']}:{source_ref['line']} "
              f"({source_ref['qualname']})\n"
              '"""\n')
    if fidelity != "self-contained":
        return (header + "# Body fragment only — it references helpers/components not captured "
                "here;\n# read the source file above for the full context.\n\n" + body)
    inputs = [p for p in ports if p["dir"] == "input"]
    outputs = [p for p in ports if p["dir"] == "output"]

    def _typ(p):
        return f'{"SInt" if p["signed"] else "UInt"}({p["width"]})'

    io_items = ", ".join([f'{p["name"]}=Input({_typ(p)})' for p in inputs]
                         + [f'{p["name"]}=Output({_typ(p)})' for p in outputs])
    call = f'{fn_name}(' + ", ".join(f'self.io.{p["name"]}' for p in inputs) + ")"
    if len(outputs) == 1:
        assigns = f'        self.io.{outputs[0]["name"]} <<= {call}\n'
    else:
        assigns = ("        res = " + call + "\n"
                   + "".join(f'        self.io.{o["name"]} <<= res[{i}]\n'
                             for i, o in enumerate(outputs)))
    return (header
            + "from spire import Bool, Component, IORecord, Input, Output, Register, SInt, Signal, "
              "UInt, Wire\n\n\n"
            + body + "\n"
            + "class StartingPoint(Component):\n"
            + "    def __init__(self):\n"
            + f"        self.io = IORecord({io_items})\n"
            + "        self.elaborate()\n\n"
            + "    def elaborate(self):\n" + assigns + "\n"
            + 'if __name__ == "__main__":\n'
            + '    print(StartingPoint().to_verilog("starting_point"))\n')
