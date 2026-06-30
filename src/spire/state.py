"""FSM state enumeration for Spire.

Declare states as class variables using :func:`state` so the IDE can
see (and autocomplete) every state name::

    class MyFSM(State, encoding=Encoding.BINARY):
        IDLE = state()
        RUN  = state()
        DONE = state()

    reg = m.reg(MyFSM.typ, "state", init=MyFSM.IDLE)

Supported encodings: ``"binary"`` (default), ``"onehot"``, ``"gray"``.
"""

from __future__ import annotations

from enum import Enum

from spire.expr import Const, HDLType, UInt


class Encoding(Enum):
    BINARY = "binary"
    ONEHOT = "onehot"
    GRAY = "gray"


class _StatePlaceholder:
    """Sentinel returned by :func:`state` to mark class-level state entries."""


def state() -> Const:
    """Declare a state entry inside a :class:`State` subclass.

    The return type is annotated as ``Const`` so that the IDE treats the
    attribute as an expression usable in ``switch_``/``case_``/``==``.
    At class-creation time the placeholder is replaced with the real
    ``Const`` value.
    """
    return _StatePlaceholder()  # type: ignore[return-value]



class State:
    """Base class for FSM state enumerations.

    Subclass and use :func:`state` for each entry::

        class MyFSM(State, encoding=Encoding.ONEHOT):
            IDLE = state()
            RUN  = state()
            DONE = state()

        MyFSM.IDLE   # Const – IDE autocompletes this
        MyFSM.typ    # HDLType matching the encoding width
    """

    typ: HDLType
    encoding: Encoding
    names: list[str]
    _width: int
    _values: dict[str, int]

    def __init_subclass__(cls, encoding: Encoding = Encoding.BINARY, **kwargs):
        super().__init_subclass__(**kwargs)

        if not isinstance(encoding, Encoding):
            raise ValueError(f"Unknown encoding '{encoding}', expected one of {list(Encoding)}")

        # Collect state names in declaration order
        names: list[str] = []
        for attr in list(vars(cls)):
            if isinstance(getattr(cls, attr), _StatePlaceholder):
                names.append(attr)

        if not names:
            return  # allow intermediate base classes with no states

        n = len(names)

        if encoding == Encoding.BINARY:
            values = list(range(n))
            width = max(1, (n - 1).bit_length())
        elif encoding == Encoding.ONEHOT:
            values = [1 << i for i in range(n)]
            width = n
        elif encoding == Encoding.GRAY:
            values = [i ^ (i >> 1) for i in range(n)]
            width = max(1, (n - 1).bit_length())

        typ = UInt(width)

        cls.encoding = encoding
        cls.names = names
        cls._width = width
        cls._values = dict(zip(names, values))
        cls.typ = typ

        # Replace placeholders with real Const values. Tag each Const with
        # provenance back-pointers so downstream passes (Hopcroft FSM
        # minimisation, encoding search) can recognise state Consts
        # unambiguously by identity rather than by (value, width), which
        # would collide with literal zeros, register inits, mask constants,
        # etc. The sentinels are read by helpers in `spire.optimize.fsm.*`.
        for name, val in cls._values.items():
            c = Const(val, typ)
            c._state_class = cls
            c._state_name = name
            setattr(cls, name, c)

    def __len__(self) -> int:
        return len(self.names)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({', '.join(self.names)}, encoding={self.encoding!r}, width={self._width})"


# Re-export the FSM-optimisation surface (optimized_fsm, optimized_encoding, ...) from
# `spire.optimize.fsm`. The import is lazy to avoid a circular import, since that module imports this one.

def __getattr__(name):  # PEP 562 module-level __getattr__
    _exports = {
        "optimized_fsm",
        "optimized_encoding",
        "minimize_fsm",
        "search_encoding",
        "apply_encoding",
        "snapshot_encoding",
        "restore_encoding",
        "equivalence_classes",
    }
    if name in _exports:
        from spire.optimize import fsm as _fsm
        return getattr(_fsm, name)
    raise AttributeError(f"module 'spire.state' has no attribute {name!r}")
