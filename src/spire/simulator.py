from spire.component import Component, Netlist
from spire.expr import *
from spire.expr import Signal, Expr, Const, Op1, Op2, Ternary, Concat, Slice, Resize, WIRE_LIKE_KINDS
from spire.memory import _MemoryArray, _ArrayIndex
from spire.simulator_base import SimulatorBase
from spire.visitor import ExprVisitor


# ---------------------------------------------------------------------------
# Expression evaluator (visitor) for bit-pattern simulation
# ---------------------------------------------------------------------------

class _SimExprEval(ExprVisitor[int]):
    """Evaluates an Expr tree to a Python int bit-pattern."""

    def __init__(self, sim: "Simulator") -> None:
        super().__init__()
        self._sim = sim
        self._visiting: set = set()

    def visit_const(self, e: Const) -> int:
        return _to_unsigned(e.value, e.typ.width)

    def visit_signal(self, e: Signal) -> int:
        return self._sim._eval_signal_bits(e)

    def visit_op1(self, e: Op1) -> int:
        a = self.visit(e.a)
        if e.op == "~":
            return _to_unsigned(~a, e.typ.width)
        raise NotImplementedError(f"Unary op '{e.op}' not implemented.")

    def visit_op2(self, e: Op2) -> int:
        op = e.op
        tw = e.typ.width

        if op in ("&", "|", "^"):
            av = self.visit(e.a)
            bv = self.visit(e.b)
            if op == "&":
                return _to_unsigned(av & bv, tw)
            elif op == "|":
                return _to_unsigned(av | bv, tw)
            else:
                return _to_unsigned(av ^ bv, tw)

        elif op == "nand":  # experimental feature
            av = self.visit(e.a)
            bv = self.visit(e.b)
            return _to_unsigned(~(av & bv), tw)

        elif op in ("+", "-"):
            aw = e.a.typ.width
            bw = e.b.typ.width
            av = _resize_bits(self.visit(e.a), aw, tw, e.a.typ.signed)
            bv = _resize_bits(self.visit(e.b), bw, tw, e.b.typ.signed)
            if op == "+":
                return _to_unsigned(av + bv, tw)
            else:
                return _to_unsigned(av - bv, tw)

        elif op == "*":
            aw = e.a.typ.width
            bw = e.b.typ.width
            a_raw = self.visit(e.a)
            b_raw = self.visit(e.b)
            a_int = _to_signed(a_raw, aw) if e.a.typ.signed else _to_unsigned(a_raw, aw)
            b_int = _to_signed(b_raw, bw) if e.b.typ.signed else _to_unsigned(b_raw, bw)
            prod = a_int * b_int
            return _to_unsigned(prod, tw)

        elif op in ("<<", ">>"):
            av = self.visit(e.a)
            bv = self.visit(e.b)
            shift = _to_unsigned(bv, max(e.b.typ.width, 32))
            if op == "<<":
                if shift >= tw:
                    return 0
                return _to_unsigned(av << shift, tw)
            else:
                src_w = e.a.typ.width
                av_src = _to_unsigned(av, src_w)
                return _to_unsigned(av_src >> shift, tw)

        elif op in ("==", "!=", "<", "<=", ">", ">="):
            # Exact integer compare (charter §2): decode each operand per its OWN signedness.
            # Operator-built IR arrives same-signed (op_cmp promotes); this also covers raw nodes.
            aw, bw = e.a.typ.width, e.b.typ.width
            a_raw, b_raw = self.visit(e.a), self.visit(e.b)
            ai = _to_signed(a_raw, aw) if e.a.typ.signed else _to_unsigned(a_raw, aw)
            bi = _to_signed(b_raw, bw) if e.b.typ.signed else _to_unsigned(b_raw, bw)
            val = {"==": ai == bi, "!=": ai != bi, "<": ai < bi,
                   "<=": ai <= bi, ">": ai > bi, ">=": ai >= bi}[op]
            return _to_unsigned(1 if val else 0, e.typ.width)

        else:
            raise NotImplementedError(f"Binary op '{op}' not implemented.")

    def visit_ternary(self, e: Ternary) -> int:
        sel = self.visit(e.sel)
        chosen = e.a if sel != 0 else e.b
        cbits = self.visit(chosen)
        return _resize_bits(cbits, chosen.typ.width, e.typ.width, chosen.typ.signed)

    def visit_concat(self, e: Concat) -> int:
        acc = 0
        shift = 0
        for p in e.parts:
            pv = self.visit(p)
            width = p.typ.width
            acc |= _to_unsigned(pv, width) << shift
            shift += width
        return _to_unsigned(acc, e.typ.width)

    def visit_slice(self, e: Slice) -> int:
        av = self.visit(e.a)
        shifted = av >> e.lsb
        return _to_unsigned(shifted, e.typ.width)

    def visit_resize(self, e: Resize) -> int:
        av = self.visit(e.a)
        return _resize_bits(av, e.a.typ.width, e.to_width, e.a.typ.signed)

    def visit_array_index(self, e: _ArrayIndex) -> int:
        addr_w = e.addr_wire.typ.width
        addr = _to_unsigned(self.visit(e.addr_wire), addr_w)
        arr = self._sim._mem_state.get(id(e.mem))
        if arr is None or addr >= len(arr):
            return 0
        return _to_unsigned(arr[addr], e.typ.width)


class Simulator(SimulatorBase):
    """
    Cycle-accurate simulator for a single Netlist.
    - step(): one rising edge of clk (if module has clock)
    - set(): set an input or (optionally) force a reg
    - get(): read any signal (inputs, wires, outputs, regs)
    - reset(): asynchronous reset (like posedge rst)
    - eval(): recompute combinational paths (lazy anyway)
    """

    def __init__(self, module: "Netlist | Component"):
        # Accept a Component directly and lower it to its netlist IR (a Netlist is used as-is).
        if isinstance(module, Component):
            module = module.to_netlist()
        self.m = module
        # A _MemoryArray and its port wires are constructed standalone. Collecting signals pulls them in via the
        # design graph (the walker traverses `read_data._memory_parent` back-edges and the store's port-children).
        # Registered reads are ordinary capture Registers in the primitive — no store-owned regs to special-case.
        self.m.collect_signals()
        self.inputs = [s for s in self.m._signals if self.m.is_global_io(s, "input")]
        self.outputs = [s for s in self.m._signals if self.m.is_global_io(s, "output")]
        self.regs = [s for s in self.m._signals if s.kind == "reg"]
        self.wires = [s for s in self.m._signals if s.kind in WIRE_LIKE_KINDS and not self.m.is_global_io(s)]
        self.mems = [s for s in self.m._signals if s.kind == "mem"]

        def check_or_duplicate_name(signals):
            seen = set()
            for s in signals:
                if s.name in seen:
                    raise ValueError(f"Duplicate signal name detected: '{s.name}'")
                seen.add(s.name)

        check_or_duplicate_name(self.m._signals)

        self._by_name = {s.name: s for s in self.m._signals}

        self._expr_eval = _SimExprEval(self)
        self._cache_sig: dict[int, int] = {}
        self._time_steps = 0

        # logging
        self.trace_enabled = False
        self.traced_expressions = self.m.all_exprs()
        self.trace_history = []

        # Use ids instead of Signal objects
        self._in: dict[int, int] = {_sid(i): 0 for i in self.inputs}
        self._reg: dict[int, int] = {}

        for r in self.regs:
            init_bits = 0
            if r._init is not None:
                init_bits = self._eval_expr_bits(r._init)
                init_bits = _resize_bits(init_bits, r._init.typ.width, r.typ.width, r._init.typ.signed)
            self._reg[_sid(r)] = _to_unsigned(init_bits, r.typ.width)

        # Memory contents: { id(mem) -> list[int] of length depth }. Initialisation comes from
        # Memory.init_sim_state (zeros, or `init=…` if given).
        self._mem_state: dict[int, list[int]] = {id(m): m.init_sim_state() for m in self.mems}

        if self.m.with_reset:
            self._in[_sid(self.m.rst)] = 0
        if self.m.with_clock:
            self._in[_sid(self.m.clk)] = 0

    # -----------------------------
    # Public API
    # -----------------------------

    def set(self, ref, value: int):
        s = self._resolve(ref)
        if s.kind == "input":
            self._in[_sid(s)] = _to_unsigned(value, s.typ.width)
        elif s.kind == "reg":
            self._reg[_sid(s)] = _to_unsigned(value, s.typ.width)
        else:
            raise ValueError("Only inputs and regs can be set directly.")
        self._invalidate()
        return self

    # for signals, converted to signed number in case signed=True, use peek for expressions
    def get(self, ref, *, signed: bool | None = None) -> int:
        s = self._resolve(ref)
        bits = self._eval_signal_bits(s)
        if signed is None:
            signed = s.typ.signed
        return _to_signed(bits, s.typ.width) if signed else bits

    def eval(self) -> "Simulator":
        """(Re)compute combinational network. (Lazy by default; this just clears caches.)"""
        self._invalidate()
        # Evaluate all outputs once to populate errors early.
        for y in self.outputs:
            _ = self._eval_signal_bits(y)
        self._capture_watches()
        self.record_expr_snapshot()
        return self

    def step(self, n: int = 1):
        for _ in range(n):
            if self.m.with_clock:
                self._in[_sid(self.m.clk)] = 0
            next_vals = self._compute_next_state()
            next_mem_vals = [(m, m.compute_next_state(self)) for m in self.mems]  # against pre-edge state
            for sid, v in next_vals.items():
                self._reg[sid] = v
            for m, nxt in next_mem_vals:
                m.commit_next_state(self, nxt)
            if self.m.with_clock:
                self._in[_sid(self.m.clk)] = 1
                self._in[_sid(self.m.clk)] = 0
            self._time_steps += 1
            self._invalidate()  # watches must see post-edge state, not stale pre-edge caches
            self._capture_watches()
            self.record_expr_snapshot()
        return self

    def reset(self, asserted: bool = True):
        if not self.m.with_reset:
            return self
        self._in[_sid(self.m.rst)] = 1 if asserted else 0
        if asserted:
            for r in self.regs:
                if r._init is not None:
                    v = self._eval_expr_bits(r._init)
                    v = _resize_bits(v, r._init.typ.width, r.typ.width, r._init.typ.signed)
                else:
                    v = 0
                self._reg[_sid(r)] = _to_unsigned(v, r.typ.width)
        self._invalidate()  # the rst line changed either way — combinational cones must recompute
        self._capture_watches()
        return self

    def deassert_reset(self):
        if self.m.with_reset:
            self._in[_sid(self.m.rst)] = 0
            self._invalidate()
            self._capture_watches()
        return self

    # Peek, for raw outputs and inputs
    def peek_outputs(self) -> dict[str, int]:
        return {y.name: self.peek(y) for y in self.outputs}

    def peek_inputs(self) -> dict[str, int]:
        return {x.name: self.peek(x) for x in self.inputs}

    def get_mem(self, ref) -> list[int]:
        """Return a copy of the current contents of a Memory.

        `ref` may be the Memory object itself or its name (string). Returns a Python list of length `mem.depth`
        with each entry as an unsigned bit-pattern of width `mem.typ.width`. The returned list is a copy —
        mutating it does not affect simulation state.
        """
        if isinstance(ref, _MemoryArray):
            mem = ref
        elif isinstance(ref, str):
            for m in self.mems:
                if m.name == ref:
                    mem = m
                    break
            else:
                raise KeyError(f"No memory named '{ref}' in module {self.m.name}.")
        else:
            raise TypeError(f"get_mem: expected Memory or str, got {type(ref)}")
        return list(self._mem_state[id(mem)])

    # -----------------------------
    # Internals
    # -----------------------------

    def _resolve(self, ref: Union[str, Signal]) -> Signal:
        if isinstance(ref, Signal):
            return ref
        if isinstance(ref, str):
            try:
                return self._by_name[ref]
            except KeyError:
                raise KeyError(f"No signal named '{ref}' in module {self.m.name}.")
        raise TypeError(f"Expected Signal or str, got {type(ref)}")

    def _invalidate(self):
        self._expr_eval.clear_cache()
        self._cache_sig.clear()

    def _compute_next_state(self) -> dict[int, int]:
        """Compute next-state values for all registers without committing.

        A memory's registered read is an ordinary capture ``Register`` in the primitive (it
        has a real ``_driver``), so there is no longer any externally-managed register to skip
        here — it evaluates like any other reg, sampling pre-edge memory before ``step`` runs.
        """
        res: dict[int, int] = {}
        rst_high = self.m.with_reset and self._in.get(_sid(self.m.rst), 0) != 0
        for r in self.regs:
            if rst_high:
                if r._init is not None:
                    init_bits = self._eval_expr_bits(r._init)
                    v = _resize_bits(init_bits, r._init.typ.width, r.typ.width, r._init.typ.signed)
                else:
                    v = 0
                res[_sid(r)] = _to_unsigned(v, r.typ.width)
            else:
                drv = r._driver
                if drv is None:
                    raise ValueError(f"Register '{r.name}' has no next-state assignment.")
                nxt_bits = self._eval_expr_bits(drv)
                res[_sid(r)] = _resize_bits(nxt_bits, drv.typ.width, r.typ.width, drv.typ.signed)
        return res

    # ------- Expression evaluation (to bit patterns) -------

    def _eval_signal_bits(self, s: Signal) -> int:
        sid = id(s)
        if sid in self._cache_sig:
            return self._cache_sig[sid]

        if self.m.is_global_io(s, "input"):
            bits = _to_unsigned(self._in.get(_sid(s), 0), s.typ.width)

        elif s.kind == "reg":
            bits = _to_unsigned(self._reg.get(_sid(s), 0), s.typ.width)

        elif s.kind in WIRE_LIKE_KINDS:
            if s._driver is None:
                # Blackbox-Component outputs have no Python sim model. The framework returns 0 as a stub so the
                # rest of the design can still simulate. Signature: `_no_emit_drive=True`.
                if s._no_emit_drive:
                    bits = 0
                    self._cache_sig[sid] = bits
                    return bits
                raise ValueError(f"Signal '{s.name}' ({s.kind}) has no driver.")
            visiting = self._expr_eval._visiting
            key = ("sig", sid)
            if key in visiting:
                raise RuntimeError(f"Combinational loop detected involving '{s.name}'.")
            visiting.add(key)
            try:
                drv_bits = self._expr_eval.visit(s._driver)
            finally:
                visiting.remove(key)
            bits = _resize_bits(drv_bits, s._driver.typ.width, s.typ.width, s._driver.typ.signed)
        else:
            if s.kind == "mem":
                raise TypeError(f"'{s.name}' is a memory — use get_mem() to read its contents")
            raise TypeError(f"Unknown signal kind: {s.kind}")

        self._cache_sig[sid] = bits
        return bits

    def _eval_expr_bits(self, e: Expr) -> int:
        """Evaluate expression *e* to a bit-pattern of width ``e.typ.width``."""
        return self._expr_eval.visit(e)

    # --- helpers to convert ---
    def _bits_to_int(self, bits):
        """Convert either an int or a list of 0/1 bits (LSB first) to Python int."""
        if isinstance(bits, int):
            return bits
        v = 0
        for i, b in enumerate(bits):
            if b & 1:
                v |= 1 << i
        return v

    def _resolve_expr(self, what):
        """Resolve string or Signal/Expr into an Expr. Avoid equality tests."""
        if isinstance(what, Signal):
            return what
        if isinstance(what, str):
            # find by name via identity (no __eq__)
            for s in self.m._signals:
                if s.name == what:
                    return s
            raise KeyError(f"No signal named '{what}' in module {self.m.name}.")
        # treat any Expr-like object (duck-typed)
        if hasattr(what, "to_verilog"):
            return what
        raise TypeError(f"peek/watch expects signal name or Expr, got {type(what)}")

    # --- public APIs ---
    def list_signals(self) -> list[str]:
        """All signal names (inputs, outputs, wires, regs)."""
        return [s.name for s in self.m._signals]

    def peek(self, what): # not converted to sign
        """Return current integer value of a Signal or Expr."""
        e = self._resolve_expr(what)

        if isinstance(e, Signal) and e.kind == "reg":
            bits = self._reg[_sid(e)]  # use the register state
        else:
            bits = self._eval_signal_bits(e) if isinstance(e, Signal) else self._eval_expr_bits(e)
        return self._bits_to_int(bits)

    def peek_next(self, reg_name):
        """Compute a register's next-state value."""

        # find reg by name using identity
        reg = None
        for s in self.m._signals:
            if s.name == reg_name and s.kind == "reg":
                reg = s
                break
        if reg is None:
            raise KeyError(f"{reg_name} is not a register.")
        if self.m.with_reset and self._in.get(_sid(self.m.rst), 0):
            # While reset is asserted, the next state is the init value, not the driver.
            init_bits = _to_unsigned(reg._init.value if reg._init is not None else 0, reg.typ.width)
            return self._bits_to_int(init_bits)
        drv = reg._driver
        if drv is None:
            raise ValueError(f"Register '{reg_name}' has no next-state.")
        # evaluate next expression and resize to reg width
        nxt_bits = self._eval_expr_bits(drv)
        nxt_bits = _resize_bits(nxt_bits, drv.typ.width, reg.typ.width, drv.typ.signed)
        return self._bits_to_int(nxt_bits)

    def watch(self, what, alias: str | None = None):
        """Register a probe; value captured at each eval()/tick()."""
        e = self._resolve_expr(what)
        name = alias
        if name is None:
            # Try to use the signal name; else make a unique label
            name = getattr(e, "name", None) or f"watch_{len(getattr(self, '_watches', {}))}"
        if not hasattr(self, "_watches"):
            self._watches = {}
            self._watch_values = {}
        self._watches[name] = e
        return self

    def get_watch(self, name: str) -> int:
        if not hasattr(self, "_watch_values") or name not in self._watch_values:
            raise KeyError(f"No watch named '{name}'.")
        return self._watch_values[name]

    def clear_watches(self):
        if hasattr(self, "_watches"):
            self._watches.clear()
            self._watch_values.clear()

    def _capture_watches(self):
        if not hasattr(self, "_watches"):
            return
        out = {}
        for name, e in self._watches.items():
            if isinstance(e, Signal) and e.kind == "input":
                bits = self._in[_sid(e)]
            elif isinstance(e, Signal) and e.kind == "reg":
                bits = self._reg[_sid(e)]  # use the register state
            elif isinstance(e, Signal):
                bits = self._eval_signal_bits(e)
            else:
                bits = self._eval_expr_bits(e)
            out[name] = self._bits_to_int(bits)
        self._watch_values = out

    # logging
    def log_expression_states(self, expr_list):
        return self._get_expr_snapshot(expr_list)

    def _get_expr_snapshot(self, expr_list):
        values = []
        for e in expr_list:
            v_bits = self._eval_signal_bits(e) if isinstance(e, Signal) else self._eval_expr_bits(e)
            values.append((e, self._bits_to_int(v_bits)))
        return values

    def record_expr_snapshot(self):
        if self.trace_enabled:
            state = self._get_expr_snapshot(self.traced_expressions)
            state_id_dict = dict([(id(e), v) for e, v in state])
            self.trace_history.append(state_id_dict)

    def get_traced_expr_names(self):
        names_dict = {}
        for e in self.traced_expressions:
            if hasattr(e, 'name'):
                names_dict[id(e)] = e.name
            else:
                names_dict[id(e)] = f"expr_{id(e)}"
        return names_dict

    def get_trace_by_names(self) -> dict[str, list[int]]:
        """
        Return trace history as a dict mapping signal/expression names to lists of values over time.
        """
        trace_names = self.get_traced_expr_names()
        history_by_name: dict[str, list[int]] = {name: [] for name in trace_names.values()}

        for snapshot in self.trace_history:
            for eid, value in snapshot.items():
                name = trace_names.get(eid, f"expr_{eid}")
                history_by_name[name].append(value)
        return history_by_name


# helpers
def _mask(w: int) -> int:
    return (1 << w) - 1 if w > 0 else 0


def _to_unsigned(v: int, w: int) -> int:
    return int(v) & _mask(w)


def _to_signed(bits: int, w: int) -> int:
    if w == 0:
        return 0
    sign = (bits >> (w - 1)) & 1
    return bits - (1 << w) if sign else bits


def _resize_bits(bits: int, from_w: int, to_w: int, signed: bool) -> int:
    """Truncate or extend a value in two's complement as needed."""
    bits = _to_unsigned(bits, from_w)
    if to_w == from_w:
        return bits
    if to_w < from_w:
        # Truncate LSBs kept (matches Verilog slicing)
        return _to_unsigned(bits, to_w)
    # Extend
    if signed:
        val = _to_signed(bits, from_w)
        return _to_unsigned(val, to_w)
    return _to_unsigned(bits, to_w)


def _sid(s: "Signal") -> int:
    return id(s)
