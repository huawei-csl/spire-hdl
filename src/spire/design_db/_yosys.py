"""Out-of-process yosys for the gate: the ``yosys`` binary when present, else pyosys in a child.

The gate is fed arbitrary candidate input, so yosys must run **out of process**: a pyosys
``log_error`` (e.g. ``write_aiger`` on an unsupported construct, or ``equiv_status -assert``
on a non-equivalence) hard-exits its host process. The binary path uses the installed
``yosys``; where only the ``pyosys`` wheel is available (e.g. a plain-pip CI), the same
command list runs through pyosys inside a **child interpreter** — identical crash isolation,
no PATH dependency. Either way, a yosys failure is a nonzero returncode, never a dead host.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import List

from spire.design_db.store import DesignDBError

#: the pyosys child: run each command on the (fresh) global design; log_error exits the child.
_PYOSYS_CHILD = (
    "import sys\n"
    "from pyosys import libyosys as ys\n"
    "for cmd in sys.argv[1:]:\n"
    "    ys.run_pass(cmd)\n"
)


def _have_pyosys() -> bool:
    try:
        import pyosys  # noqa: F401
        return True
    except ImportError:
        return False


def have_yosys() -> bool:
    """Is *some* yosys available (binary or pyosys wheel)?"""
    return shutil.which("yosys") is not None or _have_pyosys()


def run_yosys(cmds: List[str], cwd: Path, timeout_s: float) -> subprocess.CompletedProcess:
    """Run a yosys command list out of process; returns the completed process (rc + output).

    Raises ``DesignDBError`` if neither the binary nor pyosys is available, and lets
    ``subprocess.TimeoutExpired`` propagate (callers own the budget semantics).
    """
    if shutil.which("yosys") is not None:
        return subprocess.run(["yosys", "-q", "-p", "; ".join(cmds)], cwd=str(cwd),
                              capture_output=True, text=True, timeout=timeout_s)
    if _have_pyosys():
        return subprocess.run([sys.executable, "-c", _PYOSYS_CHILD, *cmds], cwd=str(cwd),
                              capture_output=True, text=True, timeout=timeout_s)
    raise DesignDBError("no yosys available (neither the `yosys` binary on PATH nor the "
                        "`pyosys` wheel) — insert/CEC unavailable")
