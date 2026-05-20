from spirehdl.spirehdl_module import Module
from spirehdl.spirehdl import *
from spirehdl.spirehdl import Signal, Expr, Const, Op1, Op2, Ternary, Concat, Slice, Resize, MemRead, Memory
from spirehdl.spirehdl_simulator_base import SimulatorBase
from spirehdl.spirehdl_visitor import ExprVisitor


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
        return _to_bits(e.value, e.typ.width)

    def visit_signal(self, e: Signal) -> int:
        return self._sim._eval_signal_bits(e)

    def visit_op1(self, e: Op1) -> int:
        a = self.visit(e.a)
        if e.op == "~":
            return _to_bits(~a, e.typ.width)
        raise NotImplementedError(f"Unary op '{e.op}' not implemented.")

    def visit_op2(self, e: Op2) -> int:
        op = e.op
        tw = e.typ.width

        if op in ("&", "|", "^"):
            av = self.visit(e.a)
            bv = self.visit(e.b)
            if op == "&":
                return _to_bits(av & bv, tw)
            elif op == "|":
                return _to_bits(av | bv, tw)
            else:
                return _to_bits(av ^ bv, tw)

        elif op == "nand":  # experimental feature
            av = self.visit(e.a)
            bv = self.visit(e.b)
            return _to_bits(~(av & bv), tw)

        elif op in ("+", "-"):
            aw = e.a.typ.width
            bw = e.b.typ.width
            av = _resize_bits(self.visit(e.a), aw, tw, e.a.typ.signed)
            bv = _resize_bits(self.visit(e.b), bw, tw, e.b.typ.signed)
            if op == "+":
                return _to_bits(av + bv, tw)
            else:
                return _to_bits(av - bv, tw)

        elif op == "*":
            aw = e.a.typ.width
            bw = e.b.typ.width
            a_raw = self.visit(e.a)
            b_raw = self.visit(e.b)
            a_int = _from_bits_signed(a_raw, aw) if e.a.typ.signed else _to_bits(a_raw, aw)
            b_int = _from_bits_signed(b_raw, bw) if e.b.typ.signed else _to_bits(b_raw, bw)
            prod = a_int * b_int
            return _to_bits(prod, tw)

        elif op in ("<<", ">>"):
            av = self.visit(e.a)
            bv = self.visit(e.b)
            shift = _to_bits(bv, max(e.b.typ.width, 32))
            if op == "<<":
                return _to_bits(av << shift, tw)
            else:
                src_w = e.a.typ.width
                av_src = _to_bits(av, src_w)
                return _to_bits(av_src >> shift, tw)

        elif op in ("==", "!=", "<", "<=", ">", ">="):
            cw = max(e.a.typ.width, e.b.typ.width)
            av_bits = _resize_bits(self.visit(e.a), e.a.typ.width, cw, e.a.typ.signed)
            bv_bits = _resize_bits(self.visit(e.b), e.b.typ.width, cw, e.b.typ.signed)
            if op in ("==", "!="):
                eq = av_bits == bv_bits
                val = 1 if (eq if op == "==" else not eq) else 0
            else:
                signed = e.a.typ.signed or e.b.typ.signed
                ai = _from_bits_signed(av_bits, cw) if signed else av_bits
                bi = _from_bits_signed(bv_bits, cw) if signed else bv_bits
                if op == "<":
                    val = 1 if ai < bi else 0
                elif op == "<=":
                    val = 1 if ai <= bi else 0
                elif op == ">":
                    val = 1 if ai > bi else 0
                else:
                    val = 1 if ai >= bi else 0
            return _to_bits(val, e.typ.width)

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
            acc |= _to_bits(pv, width) << shift
            shift += width
        return _to_bits(acc, e.typ.width)

    def visit_slice(self, e: Slice) -> int:
        av = self.visit(e.a)
        shifted = av >> e.lsb
        return _to_bits(shifted, e.typ.width)

    def visit_resize(self, e: Resize) -> int:
        av = self.visit(e.a)
        return _resize_bits(av, e.a.typ.width, e.to_width, e.a.typ.signed)

    def visit_memread(self, e: MemRead) -> int:
        addr_w = e.addr.typ.width
        addr = _to_bits(self.visit(e.addr), addr_w)
        arr = self._sim._mem_state.get(id(e.mem))
        if arr is None or addr >= len(arr):
            return 0
        return _to_bits(arr[addr], e.typ.width)


class Simulator(SimulatorBase):
    """
    Cycle-accurate simulator for a single Module.
    - step(): one rising edge of clk (if module has clock)
    - set(): set an input or (optionally) force a reg
    - get(): read any signal (inputs, wires, outputs, regs)
    - reset(): asynchronous reset (like posedge rst)
    - eval(): recompute combinational paths (lazy anyway)
    """

    def __init__(self, module: "Module"):
        self.m = module
        # Memory signals are not added to module._signals by the module API (Memory()
        # is constructed standalone). Walk from existing driver chains and pull any
        # discovered Memory instances in, additively — without resetting _signals
        # (which would drop user-added regs/wires when the module has no outputs).
        self._discover_memories()
        self.inputs = [s for s in self.m._ports if s.kind == "input"]
        self.outputs = [s for s in self.m._ports if s.kind == "output"]
        self.regs = [s for s in self.m._signals if s.kind == "reg"]
        self.wires = [s for s in self.m._signals if s.kind == "wire"]
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
            self._reg[_sid(r)] = _to_bits(init_bits, r.typ.width)

        # Memory contents: { id(mem) -> list[int] of length depth }. Initialise from
        # the memory's `init=` array if given, else zeros (matches verilog `initial begin
        # … end` semantics).
        self._mem_state: dict[int, list[int]] = {}
        for m in self.mems:
            w = m.typ.width
            if m.init is not None:
                self._mem_state[id(m)] = [_to_bits(v, w) for v in m.init]
            else:
                self._mem_state[id(m)] = [0] * m.depth

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
            self._in[_sid(s)] = _to_bits(value, s.typ.width)
        elif s.kind == "reg":
            self._reg[_sid(s)] = _to_bits(value, s.typ.width)
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
        return _from_bits_signed(bits, s.typ.width) if signed else bits

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
            mem_actions = self._compute_mem_actions()
            next_vals = self._compute_next_state()
            for sid, v in next_vals.items():
                self._reg[sid] = v
            for m, action in mem_actions:
                if action is None:
                    continue
                arr = self._mem_state.get(id(m))
                if arr is None:
                    continue
                if action[0] == "reset_all":
                    val = action[1]
                    for i in range(len(arr)):
                        arr[i] = val
                elif action[0] == "write":
                    _, addr, data = action
                    if 0 <= addr < len(arr):
                        arr[addr] = data
            if self.m.with_clock:
                self._in[_sid(self.m.clk)] = 1
                self._in[_sid(self.m.clk)] = 0
            self._time_steps += 1
            self._capture_watches()
            self.record_expr_snapshot()
            self._invalidate()
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
                self._reg[_sid(r)] = _to_bits(v, r.typ.width)
            self._invalidate()
        return self

    def deassert_reset(self):
        if self.m.with_reset:
            self._in[_sid(self.m.rst)] = 0
        return self

    # Peek, for raw outputs and inputs
    def peek_outputs(self, *e) -> dict[str, int]:
        return {y.name: self.peek(y) for y in self.outputs}

    def peek_inputs(self, *e) -> dict[str, int]:
        return {x.name: self.peek(x) for x in self.inputs}

    # -----------------------------
    # Internals
    # -----------------------------

    def _discover_memories(self) -> None:
        """Additively pull Memory signals (and their auto-created registered-read output
        regs) into ``self.m._signals``.

        Memory and its rdata register are constructed standalone (not via ``m.reg`` /
        ``m.input``), so they only enter ``_signals`` if a downstream collector finds
        them. We cannot call ``Module.collect_signals()`` (which would reset ``_signals``
        to ports and drop user-added regs/wires when the module has no outputs), so
        we do a targeted additive walk that only appends signals of kind ``mem`` and
        memory-managed registers. Auto-generated wires are intentionally NOT added —
        they aren't needed for simulation state and may have name collisions that
        would only show up in the duplicate-name check (uniquification happens in
        ``collect_signals``, not here).
        """
        existing = {id(s) for s in self.m._signals}
        seen_e: set = set()
        seen_s: set = set()
        stack: list = []
        for s in list(self.m._signals):
            drv = getattr(s, "_driver", None)
            if drv is not None:
                stack.append(drv)
        while stack:
            node = stack.pop()
            if node is None:
                continue
            nid = id(node)
            if isinstance(node, Signal):
                if nid in seen_s:
                    continue
                seen_s.add(nid)
                # Only add Memory and memory-managed register signals (others either
                # are already in _signals via m.reg/m.input or are auto-shared wires
                # the simulator doesn't need in _signals).
                if nid not in existing and (
                    node.kind == "mem"
                    or getattr(node, "_memory_managed_by", None) is not None
                ):
                    self.m._signals.append(node)
                    existing.add(nid)
                if node.kind == "mem":
                    for e in (node._write_addr, node._write_data, node._write_enable,
                              node._reset_enable, node._reset_value,
                              node._reg_read_addr, node._reg_read_enable):
                        if e is not None:
                            stack.append(e)
                    if node._reg_read_output is not None:
                        stack.append(node._reg_read_output)
                mem_owner = getattr(node, "_memory_managed_by", None)
                if mem_owner is not None:
                    stack.append(mem_owner)
                if node._driver is not None:
                    stack.append(node._driver)
                continue
            if nid in seen_e:
                continue
            seen_e.add(nid)
            if isinstance(node, MemRead):
                stack.append(node.mem)
                stack.append(node.addr)
                continue
            for attr in ("a", "b", "sel"):
                if hasattr(node, attr):
                    stack.append(getattr(node, attr))
            if hasattr(node, "parts"):
                for p in node.parts:
                    stack.append(p)

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
        """Compute next-state values for all regs (incl. memory-managed registered-read
        outputs) without committing. Memory write/reset actions are computed by the
        companion :meth:`_compute_mem_actions`.
        """
        res: dict[int, int] = {}
        rst_high = self.m.with_reset and self._in.get(_sid(self.m.rst), 0) != 0

        for r in self.regs:
            if getattr(r, "_memory_managed_by", None) is not None:
                # Memory-managed register: next-state comes from the parent memory's
                # registered-read port, computed below.
                if _sid(r) not in self._reg:
                    self._reg[_sid(r)] = 0
                continue
            if rst_high:
                if r._init is not None:
                    init_bits = self._eval_expr_bits(r._init)
                    v = _resize_bits(init_bits, r._init.typ.width, r.typ.width, r._init.typ.signed)
                else:
                    v = 0
                res[_sid(r)] = _to_bits(v, r.typ.width)
            else:
                drv = r._driver
                if drv is None:
                    raise ValueError(f"Register '{r.name}' has no next-state assignment.")
                nxt_bits = self._eval_expr_bits(drv)
                res[_sid(r)] = _resize_bits(nxt_bits, drv.typ.width, r.typ.width, drv.typ.signed)

        # Registered-read outputs: independent of module rst (memory's always block
        # has no async-rst sensitivity in emitted verilog).
        for m in self.mems:
            if m._reg_read_output is None:
                continue
            rout_sid = _sid(m._reg_read_output)
            re_bit = (self._eval_expr_bits(m._reg_read_enable) & 1) if m._reg_read_enable is not None else 1
            if re_bit:
                addr = _to_bits(self._eval_expr_bits(m._reg_read_addr), m._reg_read_addr.typ.width)
                arr = self._mem_state.get(id(m))
                val = arr[addr] if (arr is not None and addr < len(arr)) else 0
                res[rout_sid] = _to_bits(val, m.typ.width)
            else:
                res[rout_sid] = self._reg.get(rout_sid, 0)
        return res

    def _compute_mem_actions(self) -> list:
        """Compute memory write/reset actions for this cycle.

        Returns a list of (Memory, action) where action is one of:
          * ('reset_all', value) — reset enable high, clear all entries
          * ('write', addr, data) — write enable high
          * None — no change this cycle
        """
        out: list = []
        for m in self.mems:
            action = None
            if m._reset_enable is not None and self._eval_expr_bits(m._reset_enable) & 1:
                if m._reset_value is not None:
                    rv = self._eval_expr_bits(m._reset_value)
                    rv = _resize_bits(rv, m._reset_value.typ.width, m.typ.width, m._reset_value.typ.signed)
                else:
                    rv = 0
                action = ("reset_all", _to_bits(rv, m.typ.width))
            elif m._write_addr is not None:
                we_bit = (self._eval_expr_bits(m._write_enable) & 1) if m._write_enable is not None else 1
                if we_bit:
                    addr = _to_bits(self._eval_expr_bits(m._write_addr), m._write_addr.typ.width)
                    data = self._eval_expr_bits(m._write_data)
                    data = _resize_bits(data, m._write_data.typ.width, m.typ.width, m._write_data.typ.signed)
                    action = ("write", addr, _to_bits(data, m.typ.width))
            out.append((m, action))
        return out

    # ------- Expression evaluation (to bit patterns) -------

    def _eval_signal_bits(self, s: Signal) -> int:
        sid = id(s)
        if sid in self._cache_sig:
            return self._cache_sig[sid]

        if s.kind == "input":
            bits = _to_bits(self._in.get(_sid(s), 0), s.typ.width)

        elif s.kind == "reg":
            bits = _to_bits(self._reg.get(_sid(s), 0), s.typ.width)

        elif s.kind in ("wire", "output"):
            if s._driver is None:
                raise ValueError(f"Signal '{s.name}' ({s.kind}) has no driver.")
            visiting = self._expr_eval._visiting
            key = ("sig", sid)
            if key in visiting:
                raise RuntimeError(f"Combinational loop detected involving '{s.name}'.")
            visiting.add(key)
            drv_bits = self._expr_eval.visit(s._driver)
            visiting.remove(key)
            bits = _resize_bits(drv_bits, s._driver.typ.width, s.typ.width, s._driver.typ.signed)
        else:
            raise TypeError(f"Unknown signal kind: {s.kind}")

        self._cache_sig[sid] = bits
        return bits

    def _eval_expr_bits(self, e: Expr) -> int:
        """Evaluate expression *e* to a bit-pattern of width ``e.typ.width``."""
        return self._expr_eval.visit(e)

    # peek logic not teste yet -> not really working
    # spirehdl_simulator.py (patched parts)

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

    # In your eval()/step()/tick() method, after you've computed the current combinational state,
    # add this block to capture probe values (so they’re available immediately):
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
                bits = self._cache_sig[id(e)]
            else:
                bits = self._eval_signal_bits(e) if isinstance(e, Signal) else self._eval_expr_bits(e)
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


def _to_bits(v: int, w: int) -> int:
    return int(v) & _mask(w)


def _from_bits_signed(bits: int, w: int) -> int:
    if w == 0:
        return 0
    sign = (bits >> (w - 1)) & 1
    return bits - (1 << w) if sign else bits


def _resize_bits(bits: int, from_w: int, to_w: int, signed: bool) -> int:
    """Truncate or extend a value in two's complement as needed."""
    bits = _to_bits(bits, from_w)
    if to_w == from_w:
        return bits
    if to_w < from_w:
        # Truncate LSBs kept (matches Verilog slicing)
        return _to_bits(bits, to_w)
    # Extend
    if signed:
        val = _from_bits_signed(bits, from_w)
        return _to_bits(val, to_w)
    return _to_bits(bits, to_w)


def _sid(s: "Signal") -> int:
    return id(s)


def _clsname(o) -> str:
    return o.__class__.__name__
