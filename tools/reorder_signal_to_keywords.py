#!/usr/bin/env python3
"""Convert positional ``Signal(...)`` calls (old order: name, typ, kind) to the
explicit keyword form ``Signal(typ=..., kind=..., name=...)``.

Run together with the ``Signal.__init__`` reorder (name moved last, to align with
Wire/Register). Keyword keys make every call order-independent.

    python tools/reorder_signal_to_keywords.py            # dry run
    python tools/reorder_signal_to_keywords.py --apply
"""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

OLD_ORDER = ["name", "typ", "kind"]


def transform(src: str):
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return src, 0
    lines = src.splitlines(keepends=True)
    starts, off = [], 0
    for ln in lines:
        starts.append(off)
        off += len(ln)

    def offset(lineno, col):
        return starts[lineno - 1] + col

    edits = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "Signal"):
            continue
        if not node.args or node.lineno != node.end_lineno or len(node.args) > 3:
            continue
        roles = {}
        bad = False
        for i, a in enumerate(node.args):
            roles[OLD_ORDER[i]] = ast.get_source_segment(src, a)
        for k in node.keywords:
            if k.arg is None:  # **kwargs
                bad = True
                break
            roles[k.arg] = ast.get_source_segment(src, k.value)
        if bad or "typ" not in roles or "kind" not in roles:
            continue
        parts = [f"typ={roles['typ']}", f"kind={roles['kind']}"]
        if "name" in roles:
            parts.append(f"name={roles['name']}")
        for extra in roles:
            if extra not in ("typ", "kind", "name"):
                parts.append(f"{extra}={roles[extra]}")
        new = "Signal(" + ", ".join(parts) + ")"
        if new == ast.get_source_segment(src, node):
            continue
        edits.append((offset(node.lineno, node.col_offset), offset(node.end_lineno, node.end_col_offset), new))

    if not edits:
        return src, 0
    out = src
    for s, e, t in sorted(edits, key=lambda x: x[0], reverse=True):
        out = out[:s] + t + out[e:]
    return out, len(edits)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("roots", nargs="*", default=["src", "testing"])
    args = ap.parse_args()

    files = []
    for r in args.roots:
        files += sorted(Path(r).rglob("*.py"))

    total = nfiles = 0
    for f in files:
        src = f.read_text(encoding="utf-8")
        new, n = transform(src)
        if n:
            total += n
            nfiles += 1
            print(f"{'WRITE' if args.apply else 'would change'}  {f}  ({n})")
            if args.apply:
                f.write_text(new, encoding="utf-8")
    print(f"\n{total} call(s) across {nfiles} file(s){'' if args.apply else '  [dry run]'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
