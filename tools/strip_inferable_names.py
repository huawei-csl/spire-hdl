#!/usr/bin/env python3
"""Remove redundant Signal/Wire/Register names that SpireHDL can infer.

A name argument is removed only when name inference would reproduce *exactly*
the same name at that call site, i.e. all of:

  * the call is on a single source line,
  * it sits in an inferrable position — ``lhs = Ctor(...)`` (assignment to a
    plain name/attribute) or ``field = Ctor(...)`` (a keyword-argument field,
    as in a dataclass IO bundle),
  * the explicit name string equals that ``lhs`` / field name.

This mirrors ``signal_name_inference`` exactly, so the transformed code builds
identical Signals. ``Signal("x", T, K)`` (positional) is rewritten to the
keyword form ``Signal(typ=T, kind=K)`` because, without the leading name, the
remaining positionals would bind to the wrong parameters.

Usage:
    python tools/strip_inferable_names.py            # dry run, prints summary
    python tools/strip_inferable_names.py --apply     # write changes
"""
from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

CTORS = {"Signal", "Wire", "Register"}

# Same pattern the runtime inference uses.
_ASSIGN_RE = re.compile(
    r"^\s*(?:self\.)?(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*(?::[^=]+)?=\s*(?:Signal|Wire|Register)\s*\("
)


def _sanitize(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_$]", "_", name)
    if not cleaned:
        return ""
    if cleaned[0].isdigit():
        cleaned = "sig_" + cleaned
    return cleaned


# --------------------------------------------------------------------------- #
# Python (AST-based, exact)
# --------------------------------------------------------------------------- #
def _seg(src: str, node: ast.AST) -> str:
    return ast.get_source_segment(src, node)


def _rebuild(src: str, node: ast.Call, ctor: str, removal) -> str | None:
    parts: list[str] = []
    pos = list(node.args)
    kws = list(node.keywords)

    if removal[0] == "kw":  # drop the name= keyword, keep everything else verbatim
        for a in pos:
            parts.append(_seg(src, a))
        for k in kws:
            if k.arg == "name":
                continue
            parts.append((f"{k.arg}=" if k.arg else "**") + _seg(src, k.value))
    else:  # positional name
        idx = removal[1]
        role = {1: "typ", 2: "kind"} if ctor == "Signal" else {}
        for i, a in enumerate(pos):
            if i == idx:
                continue
            # For Signal, the trailing positionals must become keywords once the
            # leading name is gone; for Wire/Register the name is the last
            # positional, so the rest stay positional.
            r = role.get(i)
            parts.append((f"{r}=" if r else "") + _seg(src, a))
        for k in kws:
            parts.append((f"{k.arg}=" if k.arg else "**") + _seg(src, k.value))

    return f"{ctor}(" + ", ".join(parts) + ")"


def transform_python(src: str):
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return src, 0

    lines = src.splitlines(keepends=True)
    starts, off = [], 0
    for ln in lines:
        starts.append(off)
        off += len(ln)

    def offset(lineno: int, col: int) -> int:
        return starts[lineno - 1] + col

    parents = {}
    for n in ast.walk(tree):
        for c in ast.iter_child_nodes(n):
            parents[c] = n

    edits = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in CTORS):
            continue
        if node.lineno != node.end_lineno:  # single line only
            continue
        ctor = node.func.id

        # inferrable context (assignment target or kwarg field)
        p = parents.get(node)
        ctx = None
        if isinstance(p, ast.Assign) and len(p.targets) == 1:
            t = p.targets[0]
            ctx = t.id if isinstance(t, ast.Name) else t.attr if isinstance(t, ast.Attribute) else None
        elif isinstance(p, ast.AnnAssign) and isinstance(p.target, ast.Name):
            ctx = p.target.id
        elif isinstance(p, ast.keyword):
            ctx = p.arg
        if not ctx:
            continue

        # locate the explicit name (string literal)
        kw = {k.arg: k for k in node.keywords if k.arg}
        name_str = removal = None
        if "name" in kw and isinstance(kw["name"].value, ast.Constant) and isinstance(kw["name"].value.value, str):
            name_str, removal = kw["name"].value.value, ("kw", "name")
        else:
            idx = {"Signal": 0, "Wire": 1, "Register": 2}[ctor]
            if len(node.args) > idx and isinstance(node.args[idx], ast.Constant) and isinstance(node.args[idx].value, str):
                name_str, removal = node.args[idx].value, ("pos", idx)
        if name_str is None:
            continue

        # safety gate: inference reproduces this exact name
        line = lines[node.lineno - 1]
        m = _ASSIGN_RE.match(line)
        if not m or _sanitize(m.group("name")) != _sanitize(name_str) or _sanitize(ctx) != _sanitize(name_str):
            continue

        new = _rebuild(src, node, ctor, removal)
        old = _seg(src, node)
        if not new or new == old:
            continue
        edits.append((offset(node.lineno, node.col_offset), offset(node.end_lineno, node.end_col_offset), new))

    if not edits:
        return src, 0
    out = src
    for start, end, txt in sorted(edits, key=lambda e: e[0], reverse=True):
        out = out[:start] + txt + out[end:]
    return out, len(edits)


# --------------------------------------------------------------------------- #
# Text (markdown + .py comments): conservative regex
# --------------------------------------------------------------------------- #
_T_POS = re.compile(
    r'(?P<pre>(?:self\.)?(?P<lhs>[A-Za-z_]\w*)\s*=\s*)Signal\(\s*"(?P<nm>[A-Za-z_]\w*)"\s*,\s*'
    r'(?P<typ>[A-Za-z_]\w*\([^(),]*\)|[A-Za-z_]\w*\(\))\s*,\s*(?P<kind>"[^"]+"|\'[^\']+\')\s*\)'
)
_T_KW_SIGNAL = re.compile(
    r'(?P<pre>(?:self\.)?(?P<lhs>[A-Za-z_]\w*)\s*=\s*Signal\()\s*name\s*=\s*"(?P<nm>[A-Za-z_]\w*)"\s*,\s*'
)
_T_KW_TAIL = re.compile(  # Wire/Register/Signal with a trailing `, name="x"`
    r'(?P<pre>(?:self\.)?(?P<lhs>[A-Za-z_]\w*)\s*=\s*(?:Wire|Register|Signal)\([^()\n]*?)\s*,\s*name\s*=\s*"(?P<nm>[A-Za-z_]\w*)"\s*(?P<post>\))'
)


def _txt_line(line: str) -> str:
    def pos(m):
        return f'{m.group("pre")}Signal(typ={m.group("typ")}, kind={m.group("kind")})' if m.group("lhs") == m.group("nm") else m.group(0)

    def kw_sig(m):
        return m.group("pre") if m.group("lhs") == m.group("nm") else m.group(0)

    def kw_tail(m):
        return f'{m.group("pre")}{m.group("post")}' if m.group("lhs") == m.group("nm") else m.group(0)

    line = _T_POS.sub(pos, line)
    line = _T_KW_SIGNAL.sub(kw_sig, line)
    line = _T_KW_TAIL.sub(kw_tail, line)
    return line


_PY_BLOCK = re.compile(r"(```python\n)(.*?)(```)", re.DOTALL)


def transform_markdown(src: str):
    total = 0

    def _block(m):
        nonlocal total
        body, a = transform_python(m.group(2))  # AST transform on the code block
        body, b = transform_py_comments(body)
        total += a + b
        return m.group(1) + body + m.group(3)

    src = _PY_BLOCK.sub(_block, src)
    # Conservative regex pass for prose / inline mentions outside code blocks (idempotent on already-stripped calls).
    out, n = [], 0
    for line in src.splitlines(keepends=True):
        new = _txt_line(line)
        n += new != line
        out.append(new)
    return "".join(out), total + n


def transform_py_comments(src: str):
    """Apply the text transform only to full-line ``#`` comments in a .py file."""
    out, n = [], 0
    for line in src.splitlines(keepends=True):
        if line.lstrip().startswith("#"):
            new = _txt_line(line)
            n += new != line
            out.append(new)
        else:
            out.append(line)
    return "".join(out), n


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="write changes (default: dry run)")
    ap.add_argument("roots", nargs="*", default=["src", "testing", "docs", "README.md"])
    args = ap.parse_args()

    files = []
    for r in args.roots:
        p = Path(r)
        if p.is_file():
            files.append(p)
        elif p.is_dir():
            files += sorted(p.rglob("*.py"))
            files += sorted(p.rglob("*.md"))

    total_sites = total_files = 0
    for f in files:
        src = f.read_text(encoding="utf-8")
        if f.suffix == ".py":
            new, a = transform_python(src)
            new, b = transform_py_comments(new)
            cnt = a + b
        else:
            new, cnt = transform_markdown(src)
        if cnt:
            total_sites += cnt
            total_files += 1
            print(f"{'WRITE' if args.apply else 'would change'}  {f}  ({cnt} site{'s' if cnt != 1 else ''})")
            if args.apply:
                f.write_text(new, encoding="utf-8")

    print(f"\n{total_sites} name(s) across {total_files} file(s){'' if args.apply else '  [dry run — pass --apply to write]'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
