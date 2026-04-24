from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Iterable, List, Optional, Union

from spirehdl.spirehdl import Expr, Signal
from spirehdl.spirehdl_module import Module


class SimulatorBase(ABC):
    """Shared public API for simulator-like backends."""

    @abstractmethod
    def __init__(self, module: Module):
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


__all__ = ["SimulatorBase"]
