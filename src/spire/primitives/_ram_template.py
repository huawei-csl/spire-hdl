"""Shared Verilog emission for memory primitives.

`ram_block(...)` renders the Yosys-friendly clock-only RAM idiom for an arbitrary set of
write / read ports (plus an optional broadcast-reset arm and `$readmemh`-style init), so
`MemoryPrimitive` / `RamPrimitive` / `FIFOPrimitive` don't each hand-roll the same
`always @(posedge clk)` block. Storage is a single `reg [W-1:0] name[0:D-1];` array.

Port specs are plain dicts of signal *names* (the primitive resolves `.io.*.name` after
uniquification):

  write: {addr, data, en, mask=None, mask_chunks=0}
  read:  {addr, out, registered: bool, en=None,
          ruw='readFirst'|'writeFirst'|'dontCare', fwd=None|{en, addr, data}}
         `fwd` is the write port forwarded onto the read for `writeFirst`.
  reset: None | {en, val}

Read-under-write: `readFirst` (default) reads the pre-edge array; `writeFirst` muxes the
forwarded write data on a same-address collision; `dontCare` emits readFirst (synth may
pick the cheapest BRAM mode).
"""

from __future__ import annotations

from typing import Optional, Sequence


def _read_rhs(name: str, r: dict) -> str:
    """The value a read port observes: plain array index, or writeFirst forwarding mux."""
    base = f"{name}[{r['addr']}]"
    if r.get("ruw") == "writeFirst" and r.get("fwd"):
        f = r["fwd"]
        return f"({f['en']} && ({f['addr']} == {r['addr']})) ? {f['data']} : {base}"
    return base


def _emit_write(lines: list, indent: str, name: str, w: dict) -> None:
    en = w["en"]
    mask = w.get("mask")
    mask_chunks = w.get("mask_chunks", 0)
    if mask and mask_chunks > 1:
        lines.append(f"{indent}if ({en}) begin")
        # chunk width is implied by the data/array width; the caller guarantees divisibility.
        # Emit one guarded sub-write per chunk.
        cw = w["chunk_w"]
        for c in range(mask_chunks):
            hi = (c + 1) * cw - 1
            lo = c * cw
            lines.append(
                f"{indent}  if ({mask}[{c}]) {name}[{w['addr']}][{hi}:{lo}] <= {w['data']}[{hi}:{lo}];")
        lines.append(f"{indent}end")
    else:
        lines.append(f"{indent}if ({en}) {name}[{w['addr']}] <= {w['data']};")


def ram_block(*, name: str, depth: int, elem_w: int,
              writes: Sequence[dict], reads: Sequence[dict],
              reset: Optional[dict] = None,
              init: Optional[Sequence[int]] = None,
              clk: str = "clk", comment: str = "") -> str:
    L: list[str] = []
    if comment:
        L.append(f"  // {comment}")
    L.append(f"  reg [{elem_w-1}:0] {name}[0:{depth-1}];")
    for i, r in enumerate(reads):
        if r["registered"]:
            L.append(f"  reg [{elem_w-1}:0] {name}__rd{i};")

    if init is not None:
        L.append("  initial begin")
        for i, v in enumerate(init):
            L.append(f"    {name}[{i}] = {elem_w}'d{v};")
        L.append("  end")

    # Only emit the clocked always block if something clocked actually happens (writes,
    # reset arm, or a registered read). A pure async ROM (init + async reads only) needs no
    # always block — avoids an empty `always @(posedge clk) begin end`.
    has_clocked = bool(reset) or bool(writes) or any(r["registered"] for r in reads)
    if has_clocked:
        L.append(f"  always @(posedge {clk}) begin")
        # Write / reset section. Reset arm (broadcast clear) has priority over writes.
        if reset is not None:
            L.append(f"    if ({reset['en']}) begin")
            for i in range(depth):
                L.append(f"      {name}[{i}] <= {reset['val']};")
            if writes:
                L.append("    end else begin")
                for w in writes:
                    _emit_write(L, "      ", name, w)
                L.append("    end")
            else:
                L.append("    end")
        else:
            for w in writes:
                _emit_write(L, "    ", name, w)
        # Registered-read captures — separate statements, not gated by the reset arm.
        for i, r in enumerate(reads):
            if not r["registered"]:
                continue
            rhs = _read_rhs(name, r)
            if r.get("en"):
                L.append(f"    if ({r['en']}) {name}__rd{i} <= {rhs};")
            else:
                L.append(f"    {name}__rd{i} <= {rhs};")
        L.append("  end")

    # Read outputs.
    for i, r in enumerate(reads):
        if r["registered"]:
            L.append(f"  assign {r['out']} = {name}__rd{i};")
        else:
            L.append(f"  assign {r['out']} = {_read_rhs(name, r)};")
    return "\n".join(L)
