from __future__ import annotations

from dataclasses import fields, is_dataclass, make_dataclass
from typing import Any, Iterable, List, Tuple, Union

from spire.composite.base import HDLComposite
from spire.expr import Expr, Signal
from spire.hdl_traits import BitSerializable


class CompositeRecord(HDLComposite):
    """Record composite: field names become signal names, direction is explicit.

    Fields may be ``Signal`` ports (typically :class:`~spire.expr.Input` /
    :class:`~spire.expr.Output`) or nested ``HDLComposite`` values (``Array``, fixed/float
    types, other records); non-``BitSerializable`` instance attributes are ignored, not rejected.
    ``to_list()`` flattening, ``width``, ``to_bits`` and ``<<=`` / ``@=`` all come from
    :class:`HDLComposite`.

    Build it whichever way fits::

        # 1. inline / dynamic — the field set is decided at the call site
        io = CompositeRecord(addr=Input(UInt(8)), data=Output(SInt(16)))

        # 2. a subclass whose __init__ calls super().__init__(...) — reusable / parameterized
        class Packet(CompositeRecord):
            def __init__(self, n=4):
                super().__init__(addr=Wire(UInt(8)),
                                 lanes=Array([Wire(UInt(4)) for _ in range(n)]))

    Autocomplete / type hints:

        class Packet(CompositeRecord):
            addr:    Wire          # annotations -> autocomplete on packet.addr / packet.payload
            payload: Wire
            def __init__(self):
                super().__init__(addr=Wire(UInt(8)), payload=Wire(SInt(16)))
    """

    def __init__(self, **field_values: object) -> None:
        for field_name, val in field_values.items():
            # Direct Signal port: a leaf built without an explicit name (`_io_autoname`)
            # inherits the field key as its port name.
            if isinstance(val, Signal):
                if getattr(val, "_io_autoname", False):
                    val.name = field_name
                    val._io_autoname = False
            # Nested bundle/array: name its leaves hierarchically by the field path
            # (``up`` + ``valid`` -> ``up_valid``) so nested-interface ports don't collide.
            elif isinstance(val, HDLComposite):
                val._assign_port_names(field_name)
            setattr(self, field_name, val)

    def _raw_fields(self) -> list:
        """Field values without flattening (avoids recursion into to_list)."""
        if is_dataclass(self):
            return [getattr(self, f.name) for f in fields(self)]
        return list(vars(self).values())

    def to_list_first_level(self) -> List[BitSerializable]:
        return [v for v in self._raw_fields() if isinstance(v, BitSerializable)]

    def _named_children(self) -> List[Tuple[str, BitSerializable]]:
        """Field (name, value) pairs — used to build hierarchical port names."""
        if is_dataclass(self):
            items = [(f.name, getattr(self, f.name)) for f in fields(self)]
        else:
            items = list(vars(self).items())
        return [(k, v) for k, v in items if isinstance(v, BitSerializable)]


# IO normalization — lives here (the composite layer) so the dependency points one way:
# ir.py / component.py import these downward instead of importing the composite inside a
# function body. ``_to_composite`` references CompositeRecord directly (same module, no import).
def _to_composite(obj: Any) -> "CompositeRecord":
    """Convert any IO container into a CompositeRecord for uniform handling.

    Accepts an existing composite (returned as-is), a ``@dataclass``, a namedtuple, a plain dict,
    or a plain object with ``__dict__``. Single IO-normalization point behind ``Component.get_ios()``.
    """
    if isinstance(obj, CompositeRecord):  # already a composite — no conversion needed
        return obj
    if is_dataclass(obj):  # @dataclass IO
        pairs = [(f.name, getattr(obj, f.name)) for f in fields(obj)]
    elif hasattr(obj, "_fields"):  # namedtuple IO
        pairs = [(n, getattr(obj, n)) for n in obj._fields]
    elif isinstance(obj, dict):  # plain dict IO
        pairs = list(obj.items())
    else:  # plain object with __dict__
        pairs = list(vars(obj).items())
    DynIO = make_dataclass("DynIO", [(n, type(v)) for n, v in pairs], bases=(CompositeRecord,))
    return DynIO(**{n: v for n, v in pairs})


def iter_values(obj: Any) -> Iterable[Any]:
    """Flatten an IO container (dataclass, namedtuple, dict, or HDLComposite) into leaf Signals."""
    return _to_composite(obj).to_list()


# ===========================================================================
# DEPRECATED — class-template record (`TemplateRecord`) and its cloning helper.
# Kept only for the tests that exercise the cloning behaviour; not used anywhere
# else. Prefer `CompositeRecord` with an explicit `__init__` (plus field
# annotations for autocomplete). See the `TemplateRecord` docstring for why.
# ===========================================================================

def _clone_template(tmpl: BitSerializable, key: str) -> BitSerializable:
    """Clone a class-level field template so each instance gets its own leaves.

    A ``Signal`` is rebuilt fresh, preserving ``kind`` (wire/input/output/reg) and any reg
    ``init`` and taking ``key`` as its port name; a nested composite clones via its
    ``wire_like`` factory (``Array`` / fixed / float / nested record).
    """
    if isinstance(tmpl, Signal):
        clone = Signal(typ=tmpl.typ, kind=tmpl.kind, name=key)
        if tmpl.kind == "reg" and tmpl._init is not None:
            clone._init = tmpl._init
        return clone
    if isinstance(tmpl, HDLComposite):
        return type(tmpl).wire_like(tmpl)
    raise TypeError(f"Unsupported record field template for {key!r}: {type(tmpl)}")


class TemplateRecord(CompositeRecord):
    """**Deprecated.** Class-template record: declare fields as class attributes and they are
    cloned per instance (preserving ``kind`` and taking the field name as the port name)::

        class Bus(TemplateRecord):
            data  = Wire(UInt(8))
            valid = Output(UInt(1))

    .. deprecated::
        Bare class attributes read as state *shared across instances* in normal Python, but this
        class silently clones them per instance instead — which misleads anyone who hasn't read the
        implementation. Prefer :class:`CompositeRecord` with an explicit ``__init__`` (plus field
        annotations for autocomplete). Kept only for the tests that exercise the cloning behaviour;
        not used anywhere else.
    """

    # Class-level field templates captured from a subclass body by __init_subclass__.
    _record_field_templates: "dict[str, BitSerializable]" = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        templates = {
            name: value
            for name, value in cls.__dict__.items()
            if not name.startswith("_") and isinstance(value, BitSerializable)
        }
        if templates:
            cls._record_field_templates = templates

    def __init__(self, **field_values: object) -> None:
        templates = self._record_field_templates
        if templates:
            # Clone each declared field (an override kwarg wins), preserving declaration order; any
            # remaining kwargs are appended as extra fields (lenient). The base __init__ then does
            # the actual port-naming + setattr.
            merged = {
                key: field_values.pop(key) if key in field_values else _clone_template(tmpl, key)
                for key, tmpl in templates.items()
            }
            merged.update(field_values)
            field_values = merged
        super().__init__(**field_values)

    @classmethod
    def wire_like(cls, template: "TemplateRecord" = None) -> "TemplateRecord":
        """Clone factory used when a record is itself a field template: the shape is fixed by the
        class, so build a fresh default instance (``template`` is ignored)."""
        return cls()
