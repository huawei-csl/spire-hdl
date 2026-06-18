#!/usr/bin/env python3
from __future__ import annotations
import argparse
import gzip
import io
import os
import sys
from dataclasses import dataclass, field
from typing import BinaryIO, Optional, List, Tuple

# ---------- tiny helpers ----------
def lit2var(lit: int) -> int:
    return lit // 2

def not_lit(lit: int) -> int:
    return lit ^ 1

def open_maybe_gz(path: str, mode: str) -> BinaryIO:
    """
    mode: 'rb' or 'wb'
    """
    if path == "-":
        if "r" in mode:
            return sys.stdin.buffer  # type: ignore[return-value]
        else:
            return sys.stdout.buffer  # type: ignore[return-value]
    if path.endswith(".gz"):
        return gzip.open(path, mode)  # type: ignore[return-value]
    return open(path, mode)  # type: ignore[return-value]

def open_text_maybe_gz(path: str, mode: str):
    """
    mode: 'rt' or 'wt'
    """
    if path.endswith(".gz"):
        return gzip.open(path, mode, encoding="utf-8", errors="replace")
    return open(path, mode, encoding="utf-8", errors="replace")

def _decode_ascii(line: bytes) -> str:
    # AIGER is ASCII; be lenient for names
    return line.decode("utf-8", errors="replace")

# ---------- core data model ----------
@dataclass
class Aiger:
    maxvar: int
    num_inputs: int
    num_latches: int
    num_outputs: int
    num_ands: int

    inputs: List[int] = field(default_factory=list)            # literal of each input (even numbers)
    latches: List[Tuple[int, int]] = field(default_factory=list)  # (lit, next)
    outputs: List[int] = field(default_factory=list)           # output literals
    ands: List[Tuple[int, int, int]] = field(default_factory=list)  # (lhs, rhs0, rhs1)

    # optional sections
    i_names: List[Optional[str]] = field(default_factory=list)     # index -> name
    l_names: List[Optional[str]] = field(default_factory=list)
    o_names: List[Optional[str]] = field(default_factory=list)
    comments: List[str] = field(default_factory=list)

    def recompute_maxvar(self) -> int:
        m = 0
        for lit in self.inputs:
            m = max(m, lit2var(lit))
        for lit, nxt in self.latches:
            m = max(m, lit2var(lit), lit2var(nxt))
        for lit in self.outputs:
            m = max(m, lit2var(lit))
        for lhs, r0, r1 in self.ands:
            m = max(m, lit2var(lhs), lit2var(r0), lit2var(r1))
        self.maxvar = m
        return m

# ---------- parser for header line ----------
def parse_header_line(first_line: bytes) -> Tuple[str, int, int, int, int, int]:
    text = _decode_ascii(first_line).strip()
    parts = text.split()
    if len(parts) != 6 or parts[0] not in ("aig", "aag"):
        raise ValueError(f"Invalid AIGER header: {text!r}")
    fmt = parts[0]
    M, I, L, O, A = map(int, parts[1:])
    return fmt, M, I, L, O, A

# ---------- varint decoding (7 bits per byte, MSB=continue) ----------
def read_varint(f: BinaryIO) -> int:
    res = 0
    shift = 0
    byte_count = 0
    while True:
        b = f.read(1)
        if not b:
            raise EOFError("Unexpected EOF while reading varint.")
        byte_count += 1
        if byte_count > 5:
            # 32-bit safety (matches original code's guard)
            raise ValueError("Invalid varint code (too long).")
        x = b[0]
        res |= (x & 0x7F) << shift
        if (x & 0x80) == 0:
            break
        shift += 7
    return res

# ---------- readers ----------
def read_ascii_int_line(f: BinaryIO) -> int:
    line = f.readline()
    if not line:
        raise EOFError("Unexpected EOF while reading ASCII integer line.")
    return int(_decode_ascii(line).strip())

def read_ascii_two_ints_line(f: BinaryIO) -> Tuple[int, int]:
    line = f.readline()
    if not line:
        raise EOFError("Unexpected EOF while reading two-int line.")
    toks = _decode_ascii(line).strip().split()
    if len(toks) < 2:
        raise ValueError(f"Expected two integers on latch line, got: {toks!r}")
    return int(toks[0]), int(toks[1])

def read_until_symbols_and_comments(aig: Aiger, f: BinaryIO) -> None:
    """
    After AND section: read optional symbol table lines ('i','l','o') and comments ('c' + lines).
    """
    # Switch to simple byte-by-byte parsing
    def read_line_bytes() -> bytes:
        buf = bytearray()
        while True:
            ch = f.read(1)
            if not ch:
                break
            if ch == b"\n":
                break
            buf.extend(ch)
        return bytes(buf)

    # Prepare name arrays
    if not aig.i_names:
        aig.i_names = [None] * aig.num_inputs
    if not aig.l_names:
        aig.l_names = [None] * aig.num_latches
    if not aig.o_names:
        aig.o_names = [None] * aig.num_outputs

    # Consume optional whitespace
    while True:
        pos = f.tell() if f.seekable() else None
        b = f.read(1)
        if not b:
            return
        if b in b" \t\r\n":
            continue
        # not whitespace -> put back one byte if possible
        if f.seekable() and pos is not None:
            f.seek(pos)
        else:
            # we can't un-read; instead remember it
            first = b
            def first_then_rest():
                yield first
                while True:
                    c = f.read(1)
                    if not c:
                        return
                    yield c
            gen = first_then_rest()
            # monkey patch f with a simple wrapper for this remainder
            f = io.BufferedReader(io.BytesIO(b"".join(gen)))  # type: ignore
        break

    # Now read symbol/comment section
    while True:
        b = f.read(1)
        if not b:
            break
        if b in (b"i", b"l", b"o"):
            line = read_line_bytes()
            text = _decode_ascii(line)
            text = text.strip()
            if not text:
                continue
            toks = text.split(" ", 1)
            if len(toks) == 1:
                pos_idx, name = toks[0], ""
            else:
                pos_idx, name = toks[0], toks[1]
            try:
                idx = int(pos_idx)
            except ValueError:
                continue
            if b == b"i":
                if 0 <= idx < len(aig.i_names) and name:
                    if aig.i_names[idx] is None:
                        aig.i_names[idx] = name
            elif b == b"l":
                if 0 <= idx < len(aig.l_names) and name:
                    if aig.l_names[idx] is None:
                        aig.l_names[idx] = name
            else:  # 'o'
                if 0 <= idx < len(aig.o_names) and name:
                    if aig.o_names[idx] is None:
                        aig.o_names[idx] = name
        elif b == b"c":
            # Expect newline, then comments to EOF
            nxt = f.read(1)
            if nxt != b"\n":
                # be lenient: skip to end of line
                _ = read_line_bytes()
            # Remaining lines are comments
            rest = f.read()
            if not rest:
                break
            text = _decode_ascii(rest)
            # Keep original line breaks; drop a trailing final newline if any
            lines = text.splitlines()
            aig.comments.extend(lines)
            break
        elif b in b" \t\r\n":
            continue
        else:
            # Unknown starter — skip remainder of line
            _ = read_line_bytes()

def read_aiger(path: str) -> Aiger:
    with open_maybe_gz(path, "rb") as f:
        header = f.readline()
        if not header:
            raise ValueError("Empty file.")
        fmt, M, I, L, O, A = parse_header_line(header)

        aig = Aiger(M, I, L, O, A)

        if fmt == "aag":
            # Inputs
            for _ in range(I):
                lit = read_ascii_int_line(f)
                aig.inputs.append(lit)
            # Latches (lit next)
            for _ in range(L):
                lit, nxt = read_ascii_two_ints_line(f)
                aig.latches.append((lit, nxt))
            # Outputs
            for _ in range(O):
                lit = read_ascii_int_line(f)
                aig.outputs.append(lit)
            # ANDs (lhs rhs0 rhs1)
            for _ in range(A):
                line = f.readline()
                if not line:
                    raise EOFError("Unexpected EOF while reading AND triples.")
                a, b, c = map(int, _decode_ascii(line).strip().split())
                aig.ands.append((a, b, c))
            read_until_symbols_and_comments(aig, f)
        else:
            # fmt == 'aig' (binary ands)
            # Inputs are implicit: 2,4,6,...,2I
            aig.inputs = [2 * (i + 1) for i in range(I)]
            # Latches: lit is implicit, we read ONLY next (ASCII), one per line
            for i in range(L):
                nxt = read_ascii_int_line(f)
                lit = 2 * (I + i + 1)
                aig.latches.append((lit, nxt))
            # Outputs (ASCII), one per line
            for _ in range(O):
                lit = read_ascii_int_line(f)
                aig.outputs.append(lit)
            # ANDs: 2 varints per AND -> deltas
            lhs = 2 * (I + L) + 2  # max input/latch + 2
            for _ in range(A):
                d0 = read_varint(f)  # lhs - rhs0
                d1 = read_varint(f)  # rhs0 - rhs1
                rhs0 = lhs - d0
                rhs1 = rhs0 - d1
                aig.ands.append((lhs, rhs0, rhs1))
                lhs += 2
            # Symbols + comments (ASCII)
            read_until_symbols_and_comments(aig, f)

        # Be safe: recompute maxvar (writers usually do this)
        aig.recompute_maxvar()
        return aig
    
# ---------------- New: AIGER map parsing & application ----------------
from typing import Dict, Tuple, List, Optional

def parse_aiger_map_file(map_path: str) -> Tuple[Dict[int, str], Dict[int, str], Dict[int, str], List[str]]:
    """
    Parse a 'map' file with lines like:
        input  0 0  operand_a_i[0]
        output 7 7  result_o[7]
    Returns (i_map, l_map, o_map, raw_lines_for_comments).
    - i_map/o_map/l_map map 0-based positions -> names
    - raw_lines_for_comments preserves the original non-empty lines (sans trailing newlines)
    """
    i_map: Dict[int, str] = {}
    l_map: Dict[int, str] = {}
    o_map: Dict[int, str] = {}
    raw: List[str] = []

    with open_text_maybe_gz(map_path, "rt") as tf:
        for raw_line in tf:
            line = raw_line.rstrip("\n")
            if not line.strip():
                continue
            raw.append(line)

            toks = line.strip().split(maxsplit=3)
            if not toks:
                continue
            kind = toks[0].lower()

            # We accept "i"/"input", "o"/"output", "l"/"latch"
            if kind.startswith("in"):   # input / inp / i
                # expected: input <pos> <...ignored...> <name or rest>
                if len(toks) >= 3:
                    try:
                        pos = int(toks[1])
                    except ValueError:
                        continue
                    name = toks[3] if len(toks) >= 4 else toks[-1]
                    i_map[pos] = name
            elif kind.startswith("out"):  # output / o
                if len(toks) >= 3:
                    try:
                        pos = int(toks[1])
                    except ValueError:
                        continue
                    name = toks[3] if len(toks) >= 4 else toks[-1]
                    o_map[pos] = name
            elif kind.startswith("l"):  # latch / l
                if len(toks) >= 3:
                    try:
                        pos = int(toks[1])
                    except ValueError:
                        continue
                    name = toks[3] if len(toks) >= 4 else toks[-1]
                    l_map[pos] = name
            else:
                # Unknown kind: ignore (but preserved in raw)
                pass

    return i_map, l_map, o_map, raw


def apply_symbol_maps(
    aig: Aiger,
    i_map: Dict[int, str],
    l_map: Dict[int, str],
    o_map: Dict[int, str],
    override_existing: bool = True,
) -> None:
    """
    Apply maps to Aiger symbol arrays. Extends the Aiger name arrays to the right size.
    If override_existing=False, only fills empty slots.
    """
    # ensure arrays exist and correct size
    if not aig.i_names or len(aig.i_names) < aig.num_inputs:
        aig.i_names = (aig.i_names or []) + [None] * (aig.num_inputs - len(aig.i_names or []))
    if not aig.l_names or len(aig.l_names) < aig.num_latches:
        aig.l_names = (aig.l_names or []) + [None] * (aig.num_latches - len(aig.l_names or []))
    if not aig.o_names or len(aig.o_names) < aig.num_outputs:
        aig.o_names = (aig.o_names or []) + [None] * (aig.num_outputs - len(aig.o_names or []))

    # inputs
    for pos, name in i_map.items():
        if 0 <= pos < aig.num_inputs:
            if override_existing or not aig.i_names[pos]:
                aig.i_names[pos] = name
    # latches
    for pos, name in l_map.items():
        if 0 <= pos < aig.num_latches:
            if override_existing or not aig.l_names[pos]:
                aig.l_names[pos] = name
    # outputs
    for pos, name in o_map.items():
        if 0 <= pos < aig.num_outputs:
            if override_existing or not aig.o_names[pos]:
                aig.o_names[pos] = name

# ---------- writer (ASCII AIGER) ----------
from typing import List

def get_aag_lines(aig: Aiger, strip: bool = False, extra_comments: Optional[List[str]] = None) -> List[str]:
    """
    Build the ASCII AIGER (.aag) file as a list of lines (no trailing newlines).
    extra_comments: appended (after existing aig.comments) in the comment block.
    """
    lines: List[str] = []

    # Ensure header matches current content
    M = aig.recompute_maxvar()
    I = len(aig.inputs)
    L = len(aig.latches)
    O = len(aig.outputs)
    A = len(aig.ands)

    # Header
    lines.append(f"aag {M} {I} {L} {O} {A}")

    # Inputs
    for lit in aig.inputs:
        lines.append(f"{lit}")

    # Latches
    for lit, nxt in aig.latches:
        lines.append(f"{lit} {nxt}")

    # Outputs
    for lit in aig.outputs:
        lines.append(f"{lit}")

    # ANDs
    for lhs, r0, r1 in aig.ands:
        lines.append(f"{lhs} {r0} {r1}")

    if not strip:
        # Symbols (emit only where names exist)
        if aig.i_names:
            for idx, name in enumerate(aig.i_names):
                if name:
                    lines.append(f"i{idx} {name}")
        if aig.l_names:
            for idx, name in enumerate(aig.l_names):
                if name:
                    lines.append(f"l{idx} {name}")
        if aig.o_names:
            for idx, name in enumerate(aig.o_names):
                if name:
                    lines.append(f"o{idx} {name}")

        # Comments block (existing + extra)
        all_comments: List[str] = []
        if aig.comments:
            all_comments.extend(aig.comments)
        if extra_comments:
            all_comments.extend(extra_comments)

        if all_comments:
            lines.append("c")
            for entry in all_comments:
                for ln in entry.splitlines():
                    lines.append(ln)

    return lines

def aig_file_to_aag_lines(aig_path: str, map_file: str|None = None) -> List[str]:
    """
    Read an AIGER file (binary or ASCII) and convert to ASCII AIGER lines.
    If map_file is given, apply it to set i/o/l symbols.
    """
    aig = read_aiger(aig_path)

    if map_file:
        i_map, l_map, o_map, _ = parse_aiger_map_file(map_file)
        apply_symbol_maps(aig, i_map, l_map, o_map, override_existing=True)

    aag_lines = get_aag_lines(aig)
    return aag_lines

       
def write_aag(aig: Aiger, path: str, strip: bool = False, extra_comments: Optional[List[str]] = None) -> None:
    """
    Write ASCII AIGER (.aag) to 'path' (supports '-' and '.gz').
    """
    lines = get_aag_lines(aig, strip=strip, extra_comments=extra_comments)
    payload = ("\n".join(lines) + "\n").encode("ascii", errors="replace")
    with open_maybe_gz(path, "wb") as f:
        f.write(payload)


# ---------------- Updated: CLI ----------------
def main():
    ap = argparse.ArgumentParser(
        description="Convert AIGER to ASCII AIGER (.aag). Supports binary 'aig' and ascii 'aag' input."
    )
    ap.add_argument("src", help="Input file ('.aig' / '.aag' / '.gz') or '-' for stdin")
    ap.add_argument("dst", help="Output file ('.aag' / '.aag.gz') or '-' for stdout")
    ap.add_argument("-s", "--strip", action="store_true", help="Strip symbols and comments")
    ap.add_argument("--map", dest="map_path", help="Path to an AIGER map file to set i/o/l symbols.")
    ap.add_argument("--embed-map-comment", action="store_true",
                    help="Also embed the raw map text inside the 'c' comment block.")
    ap.add_argument("--map-as-comments-only", action="store_true",
                    help="Do not set symbols from map; only embed the map text as comments.")
    ap.add_argument("--no-map-override", action="store_true",
                    help="When applying the map to symbols, do not overwrite existing names.")
    ap.add_argument("--comment", action="append", default=[],
                    help="Extra comment line(s) to append after symbols (can be given multiple times).")
    args = ap.parse_args()

    aig = read_aiger(args.src)

    extra_comments: List[str] = list(args.comment) if args.comment else []

    if args.map_path:
        i_map, l_map, o_map, map_raw = parse_aiger_map_file(args.map_path)

        if not args.map_as_comments_only:
            apply_symbol_maps(aig, i_map, l_map, o_map, override_existing=(not args.no_map_override))

        if args.embed_map_comment:
            # Put the raw map text into the comment block, bracketed for clarity.
            extra_comments.append("==== BEGIN AIGER MAP ====")
            extra_comments.extend(map_raw)
            extra_comments.append("==== END AIGER MAP ====")

    write_aag(aig, args.dst, strip=args.strip, extra_comments=extra_comments if extra_comments else None)


if __name__ == "__main__":
    main()