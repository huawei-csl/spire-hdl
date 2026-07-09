from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Iterable, List, Optional, Union

from spire.expr import Expr, Signal
from spire.component import Netlist


class SimulatorBase(ABC):
    """Shared public API for simulator-like backends."""

    @abstractmethod
    def __init__(self, module: Netlist):
        self.m = module

    @abstractmethod
    def set(self, ref: Union[str, Signal], value: int):
        pass

    @abstractmethod
    def get(self, ref: Union[str, Signal], *, signed: Optional[bool] = None) -> int:
        pass

    @abstractmethod
    def eval(self):
        pass

    @abstractmethod
    def step(self, n: int = 1):
        pass

    @abstractmethod
    def reset(self, asserted: bool = True):
        pass

    @abstractmethod
    def deassert_reset(self):
        pass

    @abstractmethod
    def peek_outputs(self) -> Dict[str, int]:
        pass

    @abstractmethod
    def peek_inputs(self) -> Dict[str, int]:
        pass

    @abstractmethod
    def get_mem(self, ref) -> List[int]:
        pass

    @abstractmethod
    def watch(self, what, alias: Optional[str] = None):
        pass

    @abstractmethod
    def get_watch(self, name: str) -> int:
        pass

    @abstractmethod
    def clear_watches(self) -> None:
        pass

    @abstractmethod
    def list_signals(self) -> List[str]:
        pass

    @abstractmethod
    def peek(self, what):
        pass

    @abstractmethod
    def peek_next(self, reg_name):
        pass

    @abstractmethod
    def log_expression_states(self, expr_list: Iterable[Expr]):
        pass

    # Trace capture (VCD-ready history; see spire.various.vcd_writer.write_vcd). Backends also
    # expose `trace_enabled` (off by default) and `traced_expressions` as plain attributes or
    # properties; snapshots are recorded at every eval()/step() while enabled.
    @abstractmethod
    def record_expr_snapshot(self) -> None:
        pass

    @abstractmethod
    def get_traced_expr_names(self) -> Dict[int, str]:
        pass

    @abstractmethod
    def get_trace_by_names(self) -> Dict[str, List[int]]:
        pass


__all__ = ["SimulatorBase"]
