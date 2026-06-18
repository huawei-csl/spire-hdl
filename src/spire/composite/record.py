from typing import Any, Dict, List, Type, TypeVar, Union

from spire.composite.base import HDLComposite
from spire.expr import Expr, Signal, Wire

T_Record = TypeVar("T_Record", bound="CompositeRecord")


class CompositeRecord(HDLComposite):
    """
    Class-based record composite.

    Usage:

        class MyRecord():
            a: Signal = Wire(UInt(8))
            b: Signal = Wire(SInt(4))
            c: Array  = Array([Wire(UInt(8)) for _ in range(4)])

    On subclass definition (__init_subclass__), we scan the class
    attributes and treat any 'wire template' or HDLComposite as a field
    template. On instance creation, these templates are cloned so that
    each instance gets its own wires/composites (no sharing).
    """

    _record_field_templates: Dict[str, Union[Signal, HDLComposite]]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        templates: Dict[str, Union[Signal, HDLComposite]] = {}

        for name, value in cls.__dict__.items():
            # Skip private / dunder / methods / descriptors
            if name.startswith("_"):
                continue

            # Only instances are interesting (class-level templates)
            if isinstance(value, Signal):
                if value.kind != "wire":
                    raise TypeError(f"Record field '{name}' must be a wire Signal (kind='wire'), " f"got kind={value.kind!r}")
                templates[name] = value
            elif isinstance(value, HDLComposite):
                # For composites, we expect they are wire-backed; cloning
                # is done via type(value).wire_like(value).
                templates[name] = value

        cls._record_field_templates = templates

    def __init__(self, **overrides: Union[Signal, HDLComposite]) -> None:
        """
        Instantiate a record.

        - For each template field on the class, create a new instance:
            * Signal (wire)   → new Wire with same HDLType
            * HDLComposite    → type(template).wire_like(template)
        - If a field is passed via keyword argument, use that instead.

        Example:
            b = MyRecord()
            b2 = MyRecord(a=Wire(UInt(8)))  # override 'a'
        """
        # Clone all template fields
        for name, tmpl in self._record_field_templates.items():
            if name in overrides:
                # todo: check that override type matches template width
                val = overrides[name]
            else:
                if isinstance(tmpl, Signal):
                    if tmpl.kind != "wire":
                        raise TypeError(f"Record field '{name}' template must be a wire Signal, " f"got kind={tmpl.kind!r}")
                    # Clone: new Wire with same type
                    val = Wire(tmpl.typ)
                elif isinstance(tmpl, HDLComposite):
                    # Clone composite via its wire_like(template) API
                    val = type(tmpl).wire_like(tmpl)
                else:
                    raise TypeError(f"Unsupported template type for record field '{name}': " f"{type(tmpl)}")

            setattr(self, name, val)

        # Check for unknown overrides
        for k in overrides:
            if k not in self._record_field_templates:
                raise AttributeError(f"Unknown record field override '{k}' " f"for {self.__class__.__name__}")

    # ---------------- HDLComposite API ----------------

    def to_list_first_level(self) -> List[Expr | HDLComposite]:
        list_first_level: List[Expr | HDLComposite] = []
        for name in self._record_field_templates.keys():
            v = getattr(self, name)
            list_first_level.append(v)
        return list_first_level

    @classmethod
    def wire_like(
        cls: Type[T_Record],
        *args: Any,
        **kwargs: Any,
    ) -> T_Record:
        """
        Create a 'wire-filled' instance.

        For records, the shape is fully defined by the class, so we ignore
        any template argument and simply construct a fresh instance.
        """
        return cls()

    def __repr__(self) -> str:
        fields = ", ".join(self._record_field_templates.keys())
        return f"{self.__class__.__name__}(fields=[{fields}], width={self.width})"
