"""Mini IEEE 1364-2005 expression evaluator for the restricted grammar emitted by spire.

Reference evaluator for the differential conformance harness (docs/README_semantics.md §5): `VlogModule` parses the
combinational part of spire's emitted Verilog and evaluates it the way a synthesis tool would, so tests can compare
"what the emitted text means per the LRM" against the Python simulator without external tools. Per the charter's
calibration rule, its verdicts count only because test_leaf_conformance.py pins it against the simulator on the
shapes where both must agree; it was additionally validated per-vector against real yosys elaboration when written.

Implements the LRM sizing/signedness rules (5.4.1 Table 5-22, 5.5.1 evaluation steps):
  - self-determined size/sign computed bottom-up
  - final expression size = max(RHS self-size, LHS size); sign = RHS self-sign
  - size+sign propagate down through context-determined operands; extension per the
    PROPAGATED sign (zero-extend if expression unsigned, replicate operand MSB if signed)
  - self-determined boundaries: concat/replication operands, $signed/$unsigned args,
    shift right operand, ternary condition, comparison operands (island: sized to max
    of the two, signed iff both signed), bit/part-select bases
  - comparison result: 1 bit unsigned; signed compare iff both operands signed
  - shifts: >> and << are logical; size/sign from left operand
All values are bit patterns (non-negative ints). 4-state X/Z not modeled (spire never
emits X/Z).
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field


def mask(w):
    return (1 << w) - 1


# ------------------------- Parsing -------------------------

TOK = re.compile(r"""
    (?P<lit>\d+'s?[bdh][0-9a-fA-F_xzXZ]+)
  | (?P<num>\d+)
  | (?P<id>[A-Za-z_$][A-Za-z0-9_$]*)
  | (?P<sym>\{\{|\}\}|[(){}\[\]?:,])
  | (?P<op><<<|>>>|<<|>>|<=|>=|==|!=|[+\-*/%&|^~<>])
  | (?P<ws>\s+)
""", re.X)


def tokenize(s):
    out = []
    i = 0
    while i < len(s):
        m = TOK.match(s, i)
        if not m:
            raise SyntaxError(f"tokenize error at {s[i:i+20]!r}")
        i = m.end()
        kind = m.lastgroup
        if kind == "ws":
            continue
        text = m.group()
        if kind == "sym" and text in ("{{", "}}"):
            # split doubled braces into two tokens
            out.append(text[0]); out.append(text[1])
            continue
        out.append(text)
    return out


@dataclass
class Node:
    kind: str                 # 'lit','id','bin','un','tern','concat','repl','sysfun','select'
    op: str = ""
    kids: list = field(default_factory=list)
    val: int = 0              # for lit: pattern value ; for select: (msb,lsb)
    w: int = 0                # for lit: width; select bounds
    signed: bool = False
    msb: int = -1
    lsb: int = -1
    name: str = ""


class Parser:
    def __init__(self, toks):
        self.t = toks
        self.i = 0

    def peek(self):
        return self.t[self.i] if self.i < len(self.t) else None

    def eat(self, tok=None):
        cur = self.peek()
        if tok is not None and cur != tok:
            raise SyntaxError(f"expected {tok!r}, got {cur!r} at {self.i}: ...{self.t[max(0,self.i-4):self.i+4]}")
        self.i += 1
        return cur

    BINOPS = {"+", "-", "*", "/", "%", "&", "|", "^", "<<", ">>", "<<<", ">>>",
              "<", "<=", ">", ">=", "==", "!="}

    def parse_expr(self):
        # The spire grammar: fully parenthesized Op1/Op2/Ternary; atoms otherwise.
        n = self.parse_atom()
        # binary op may follow ONLY inside parens (handled in parse_paren); at top level
        # of an assign RHS spire always emits either an atom or a parenthesized op —
        # but be permissive: allow one binary chain left-assoc (no precedence needed
        # since spire parenthesizes).
        while self.peek() in self.BINOPS:
            op = self.eat()
            rhs = self.parse_atom()
            n = Node("bin", op=op, kids=[n, rhs])
        if self.peek() == "?":
            self.eat("?")
            a = self.parse_expr()
            self.eat(":")
            b = self.parse_expr()
            n = Node("tern", kids=[n, a, b])
        return n

    def parse_atom(self):
        cur = self.peek()
        if cur == "(":
            self.eat("(")
            n = self.parse_expr()
            self.eat(")")
            return self.parse_select(n)
        if cur == "~":
            self.eat()
            return Node("un", op="~", kids=[self.parse_atom()])
        if cur == "-":
            self.eat()
            return Node("un", op="-", kids=[self.parse_atom()])
        if cur == "{":
            return self.parse_select(self.parse_concat())
        if cur is None:
            raise SyntaxError("unexpected end")
        if re.match(r"^-?\d+'", cur):
            self.eat()
            return self.parse_lit(cur)
        if cur == "$signed" or cur == "$unsigned":
            self.eat()
            self.eat("(")
            arg = self.parse_expr()
            self.eat(")")
            return self.parse_select(Node("sysfun", op=cur, kids=[arg]))
        if re.match(r"^\d+$", cur):
            # bare unsized decimal (spire should never emit; treat as 32-bit signed per LRM)
            self.eat()
            return Node("lit", val=int(cur) & mask(32), w=32, signed=True, op="UNSIZED")
        # identifier
        self.eat()
        return self.parse_select(Node("id", name=cur))

    def parse_select(self, base):
        while self.peek() == "[":
            self.eat("[")
            a = int(self.eat())
            if self.peek() == ":":
                self.eat(":")
                b = int(self.eat())
                base = Node("select", kids=[base], msb=a, lsb=b)
            else:
                base = Node("select", kids=[base], msb=a, lsb=a)
            self.eat("]")
        return base

    def parse_concat(self):
        self.eat("{")
        # replication? {N{expr}}
        first = self.parse_expr()
        if self.peek() == "{":
            # replication: first must be a literal count
            self.eat("{")
            inner = self.parse_expr()
            self.eat("}")
            self.eat("}")
            if first.kind != "lit" or first.op != "UNSIZED":
                # count could be an unsized number token parsed as lit UNSIZED
                if first.kind != "lit":
                    raise SyntaxError("replication count must be a literal")
            return Node("repl", val=first.val, kids=[inner])
        parts = [first]
        while self.peek() == ",":
            self.eat(",")
            parts.append(self.parse_expr())
        self.eat("}")
        return Node("concat", kids=parts)

    def parse_lit(self, text):
        m = re.match(r"^(-?)(\d+)'(s?)([bdh])([0-9a-fA-F_]+)$", text)
        if not m:
            raise SyntaxError(f"bad literal {text!r}")
        neg, w, s, basec, digits = m.groups()
        w = int(w)
        base = {"b": 2, "d": 10, "h": 16}[basec]
        v = int(digits.replace("_", ""), base)
        if v > mask(w):
            v &= mask(w)  # out-of-range literal truncates (lint error but defined)
        if neg:
            v = (-v) & mask(w)
        return Node("lit", val=v, w=w, signed=bool(s))


# ------------------------- Sizing (self-determined pass) -------------------------

def selfsize(n, env):
    """Returns (size, signed) per Table 5-22, memoized on the node."""
    if hasattr(n, "_ss"):
        return n._ss
    k = n.kind
    if k == "lit":
        r = (n.w, n.signed)
    elif k == "id":
        w, s = env.decl(n.name)
        r = (w, s)
    elif k == "select":
        selfsize(n.kids[0], env)
        r = (n.msb - n.lsb + 1, False)  # part/bit selects are unsigned
    elif k == "concat":
        r = (sum(selfsize(p, env)[0] for p in n.kids), False)
    elif k == "repl":
        r = (n.val * selfsize(n.kids[0], env)[0], False)
    elif k == "sysfun":
        w, _ = selfsize(n.kids[0], env)
        r = (w, n.op == "$signed")
    elif k == "un":
        r = selfsize(n.kids[0], env)  # ~ and unary -: size & sign of operand
    elif k == "bin":
        a, b = n.kids
        aw, asig = selfsize(a, env)
        bw, bsig = selfsize(b, env)
        if n.op in ("==", "!=", "<", "<=", ">", ">="):
            r = (1, False)
        elif n.op in ("<<", ">>", "<<<", ">>>"):
            r = (aw, asig)
        else:
            r = (max(aw, bw), asig and bsig)
    elif k == "tern":
        c, a, b = n.kids
        selfsize(c, env)
        aw, asig = selfsize(a, env)
        bw, bsig = selfsize(b, env)
        r = (max(aw, bw), asig and bsig)
    else:
        raise AssertionError(k)
    n._ss = r
    return r


# ------------------------- Evaluation with context propagation -------------------------

def ext(pattern, from_w, to_w, signed):
    """Extend pattern from from_w to to_w; sign-extend iff `signed` (the PROPAGATED sign)."""
    pattern &= mask(from_w)
    if to_w <= from_w:
        return pattern & mask(to_w)
    if signed and from_w > 0 and (pattern >> (from_w - 1)) & 1:
        return (mask(to_w) ^ mask(from_w)) | pattern
    return pattern


def evaluate(n, env, ctx_w=None, ctx_signed=None):
    """Evaluate node to a bit pattern at width ctx_w (context-determined) or self size."""
    sw, ssigned = selfsize(n, env)
    if ctx_w is None:
        ctx_w, ctx_signed = sw, ssigned
    k = n.kind
    if k == "lit":
        return ext(n.val, n.w, ctx_w, ctx_signed)
    if k == "id":
        w, s = env.decl(n.name)
        return ext(env.value(n.name), w, ctx_w, ctx_signed)
    if k == "select":
        basew, _ = selfsize(n.kids[0], env)
        bv = evaluate(n.kids[0], env)          # self-determined base
        v = (bv >> n.lsb) & mask(n.msb - n.lsb + 1)
        return ext(v, sw, ctx_w, ctx_signed)
    if k == "concat":
        acc = 0
        shift = 0
        for p in reversed(n.kids):             # rightmost part = LSBs
            pw, _ = selfsize(p, env)
            acc |= (evaluate(p, env) & mask(pw)) << shift
            shift += pw
        return ext(acc, sw, ctx_w, ctx_signed)
    if k == "repl":
        pw, _ = selfsize(n.kids[0], env)
        pv = evaluate(n.kids[0], env) & mask(pw)
        acc = 0
        for i in range(n.val):
            acc |= pv << (i * pw)
        return ext(acc, sw, ctx_w, ctx_signed)
    if k == "sysfun":
        aw, _ = selfsize(n.kids[0], env)
        av = evaluate(n.kids[0], env)          # self-determined arg
        return ext(av, aw, ctx_w, ctx_signed)
    if k == "un":
        av = evaluate(n.kids[0], env, ctx_w, ctx_signed)
        if n.op == "~":
            return (~av) & mask(ctx_w)
        if n.op == "-":
            return (-av) & mask(ctx_w)
    if k == "bin":
        a, b = n.kids
        op = n.op
        if op in ("==", "!=", "<", "<=", ">", ">="):
            # comparison island: operands sized to max(self sizes), signed iff both signed
            aw, asig = selfsize(a, env)
            bw, bsig = selfsize(b, env)
            w = max(aw, bw)
            s = asig and bsig
            av = evaluate(a, env, w, s)
            bv = evaluate(b, env, w, s)
            if s:
                ai = av - (1 << w) if (av >> (w - 1)) & 1 else av
                bi = bv - (1 << w) if (bv >> (w - 1)) & 1 else bv
            else:
                ai, bi = av, bv
            res = {"==": ai == bi, "!=": ai != bi, "<": ai < bi,
                   "<=": ai <= bi, ">": ai > bi, ">=": ai >= bi}[op]
            return ext(1 if res else 0, 1, ctx_w, ctx_signed)
        if op in ("<<", ">>", "<<<", ">>>"):
            av = evaluate(a, env, ctx_w, ctx_signed)   # left operand context-determined
            sh_w, sh_s = selfsize(b, env)
            sh = evaluate(b, env)                      # right operand self-determined
            if op == "<<" or op == "<<<":
                return (av << sh) & mask(ctx_w)
            if op == ">>":
                return (av >> sh) & mask(ctx_w)
            # >>> arithmetic iff expression signed
            if ctx_signed and (av >> (ctx_w - 1)) & 1:
                return ((av >> sh) | (mask(ctx_w) ^ mask(max(ctx_w - sh, 0)))) & mask(ctx_w)
            return (av >> sh) & mask(ctx_w)
        av = evaluate(a, env, ctx_w, ctx_signed)
        bv = evaluate(b, env, ctx_w, ctx_signed)
        if op == "+":
            return (av + bv) & mask(ctx_w)
        if op == "-":
            return (av - bv) & mask(ctx_w)
        if op == "*":
            return (av * bv) & mask(ctx_w)
        if op == "&":
            return av & bv
        if op == "|":
            return av | bv
        if op == "^":
            return av ^ bv
        if op in ("/", "%"):
            # signed division truncates toward zero if expression signed
            if ctx_signed:
                ai = av - (1 << ctx_w) if (av >> (ctx_w - 1)) & 1 else av
                bi = bv - (1 << ctx_w) if (bv >> (ctx_w - 1)) & 1 else bv
                if bi == 0:
                    raise ZeroDivisionError
                q = abs(ai) // abs(bi)
                if (ai < 0) != (bi < 0):
                    q = -q
                r = ai - q * bi
                return (q if op == "/" else r) & mask(ctx_w)
            if bv == 0:
                raise ZeroDivisionError
            return (av // bv if op == "/" else av % bv) & mask(ctx_w)
        raise AssertionError(op)
    if k == "tern":
        c, a, b = n.kids
        cv = evaluate(c, env)                      # condition self-determined
        chosen = a if cv != 0 else b
        return evaluate(chosen, env, ctx_w, ctx_signed)
    raise AssertionError(k)


# ------------------------- Netlist-level simulation -------------------------

class VlogModule:
    """Parses the exact text format spire emits and evaluates combinational nets."""

    DECL = re.compile(r"^\s*(input|output|wire|reg)\s+(signed\s+)?(\[(\d+):0\]\s+)?([A-Za-z_$][\w$\[\]]*)\s*;\s*$")
    ASSIGN = re.compile(r"^\s*assign\s+([A-Za-z_$][\w$\[\]]*)\s*=\s*(.*);\s*$")

    def __init__(self, text):
        self.decls = {}     # name -> (width, signed)
        self.assigns = {}   # name -> rhs ast
        self.values = {}    # name -> pattern
        self._memo = {}
        for line in text.splitlines():
            m = self.DECL.match(line)
            if m:
                kind, signed, _, wtxt, name = m.groups()
                w = int(wtxt) + 1 if wtxt else 1
                self.decls[name] = (w, bool(signed))
                continue
            m = self.ASSIGN.match(line)
            if m:
                name, rhs = m.groups()
                self.assigns[name] = Parser(tokenize(rhs)).parse_expr()

    # env protocol
    def decl(self, name):
        if name not in self.decls:
            raise KeyError(f"undeclared identifier {name!r} referenced in expression")
        return self.decls[name]

    def value(self, name):
        if name in self.values:
            return self.values[name] & mask(self.decls[name][0])
        if name in self.assigns:
            if name in self._memo:
                return self._memo[name]
            rhs = self.assigns[name]
            w, s = self.decls[name]
            # assignment context: RHS sized to max(self, LHS) with RHS's own sign
            sw, ssigned = selfsize(rhs, self)
            v = evaluate(rhs, self, max(sw, w), ssigned) & mask(w)
            self._memo[name] = v
            return v
        raise KeyError(f"no value for {name!r} (undriven?)")

    def set(self, name, v):
        self.values[name] = v & mask(self.decls[name][0])
        self._memo.clear()

    def get(self, name):
        return self.value(name)
