# aag_loader.py
# Minimal AIGER-ASCII (.aag) reader → builds graph via create_pi/create_po/create_and/create_buf

from __future__ import annotations
import abc
from typing import Dict, List, Tuple, Optional, Any, Type

class AagParseError(RuntimeError):
    pass


def _parse_header(h: str) -> Tuple[int, int, int, int, int]:
    parts = h.strip().split()
    if len(parts) < 6 or parts[0] != "aag":
        raise AagParseError("Header must start with: aag M I L O A")
    try:
        M = int(parts[1])
        I = int(parts[2])
        L = int(parts[3])
        O = int(parts[4])
        A = int(parts[5])
    except ValueError:
        raise AagParseError("Header counts must be integers")
    return M, I, L, O, A

def file_to_lines(path: str) -> List[str]:
    """
    Read a file and return its lines as a list of strings.
    Strips trailing newlines.
    """
    with open(path, "r", encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f]

def _read_aag(lines: List[str]) -> Dict[str, Any]:    

    if not lines:
        raise AagParseError("Empty file")

    M, I, L, O, A = _parse_header(lines[0])

    idx = 1
    # Input literals
    input_lits: List[int] = []
    for _ in range(I):
        if idx >= len(lines):
            raise AagParseError("Unexpected EOF in inputs")
        lit = int(lines[idx].split()[0])
        input_lits.append(lit)
        idx += 1

    # Latches (curr next [init]) — we only parse; minimal loader won’t build them
    latch_rows: List[Tuple[int, int, Optional[int]]] = []
    for _ in range(L):
        if idx >= len(lines):
            raise AagParseError("Unexpected EOF in latches")
        parts = lines[idx].split()
        if len(parts) < 2:
            raise AagParseError("Latch row needs at least two ints")
        curr = int(parts[0])
        nxt = int(parts[1])
        init = int(parts[2]) if len(parts) >= 3 else None
        latch_rows.append((curr, nxt, init))
        idx += 1

    # Outputs (one literal each)
    output_lits: List[int] = []
    for _ in range(O):
        if idx >= len(lines):
            raise AagParseError("Unexpected EOF in outputs")
        lit = int(lines[idx].split()[0])
        output_lits.append(lit)
        idx += 1

    # AND gates (lhs rhs0 rhs1)
    and_rows: List[Tuple[int, int, int]] = []
    for _ in range(A):
        if idx >= len(lines):
            raise AagParseError("Unexpected EOF in ANDs")
        a0 = lines[idx].split()
        if len(a0) < 3:
            raise AagParseError("AND row must have 3 ints: lhs rhs0 rhs1")
        lhs, r0, r1 = map(int, a0[:3])
        and_rows.append((lhs, r0, r1))
        idx += 1

    # Symbols (optional) and comments (‘c’)
    sym_i: Dict[int, str] = {}
    sym_l: Dict[int, str] = {}
    sym_o: Dict[int, str] = {}

    while idx < len(lines):
        line = lines[idx]
        if not line:
            idx += 1
            continue
        if line.startswith("c"):  # comment section begins
            break
        # lines like: i0 name, o3 name, l2 name
        tag = line[0]
        rest = line[1:].lstrip()
        try:
            n_str, name = rest.split(" ", 1)
        except ValueError:
            # allow empty name
            n_str, name = rest, ""
        try:
            n = int(n_str)
        except ValueError:
            # not a symbol line, ignore
            idx += 1
            continue

        if tag == "i":
            sym_i[n] = name
        elif tag == "o":
            sym_o[n] = name
        elif tag == "l":
            sym_l[n] = name
        # ignore others (b, cex, j, f…)
        idx += 1

    return {
        "M": M,
        "I": I,
        "L": L,
        "O": O,
        "A": A,
        "inputs": input_lits,
        "latches": latch_rows,
        "outputs": output_lits,
        "ands": and_rows,
        "sym_i": sym_i,
        "sym_l": sym_l,
        "sym_o": sym_o,
    }

def _get_aag_sym(lines: List[str]) -> List[str]:
    """
    Extract the symbol table from the AAG lines.
    Returns a list of symbols, one per line.
    """
    symbols = []
    for line in lines:
        if line.startswith("c") or not line.strip():
            continue  # skip comments and empty lines
        if line.startswith("i") or line.startswith("o") or line.startswith("l"):
            symbols.append(line)
    return symbols

# ---------- builder (duck-typed to your AIG object) ----------
class AbstractAdapter(abc.ABC):
    """Abstract adapter for AIG-like objects to be used with the AAG loader."""
    
    def __init__(self, graph: Any):
        self.graph = graph

    def pi(self, name: Optional[str] = None) -> Any:
        """Create a primary input node."""
        raise NotImplementedError

    def po(self, node: Any, name: Optional[str] = None) -> Any:
        """Create a primary output node."""
        raise NotImplementedError

    def buf(self, node: Any) -> Any:
        """Create a buffer (NOT gate)."""
        raise NotImplementedError

    def const(self, value: bool) -> Any:
        """Create a constant node (0 or 1)."""
        raise NotImplementedError

    def NOT(self, node: Any) -> Any:
        """Create a NOT gate for the given node."""
        raise NotImplementedError

    def AND(self, a: Any, b: Any) -> Any:
        """Create an AND gate for the two given nodes."""
        raise NotImplementedError

class _Adapter(AbstractAdapter):
    """Adapters around the target 'graph' (e.g. AIG) object to be forgiving about signatures."""

    def _call(self, name: str, *args, **kwargs):
        fn = getattr(self.graph, name, None)
        if fn is None:
            raise AttributeError(f"AIG object has no method '{name}'")
        try:
            return fn(*args, **kwargs)
        except TypeError:
            # try without kwargs like name=..., inverted=...
            if kwargs:
                return fn(*args)
            raise

    def pi(self, name: Optional[str] = None):
        if name is None:
            return self._call("create_pi")
        return self._call("create_pi") #, name)

    def po(self, node, name: Optional[str] = None):
        if name is None:
            return self._call("create_po", node)
        return self._call("create_po", node) #, name)

    def buf(self, node):
        return self._call("create_buf", node)

    def const(self, value: bool):
        """Create a constant node (0 or 1) in the AIG."""
        if value:
            return self._call("get_constant", True)
        return self._call("get_constant", False)

    def NOT(self, node):
        """Create a NOT gate for the given node."""
        return self._call("create_not", node)

    def AND(self, a, b):
        return self._call("create_and", a, b)


def read_aag_into_aig(path: str, aig: Any | None = None) -> Any:
    if aig is None:
        from aigverse import Aig

        aig = Aig()
    lines = file_to_lines(path)
    return conv_aag_into_aig(lines, aig)

def conv_aag_into_aig(lines: List[str], aig: Any | None = None) -> Any:
    if aig is None:
        from aigverse import Aig
        aig = Aig()
    return conv_aag_into_graph(lines, aig, _Adapter)

def conv_aag_into_graph(lines: List[str], aig: Any, adapter: Type[AbstractAdapter]) -> Any:
    """
    Parse an AIGER-ASCII (.aag) and build the AIG on `aig` using:
      - aig.create_pi(name?)
      - aig.create_po(node, name?)
      - aig.create_and(a, b)
      - aig.create_buf(node, inverted=False)   # used to build NOT
    Returns the same `aig` object for chaining.

    Limitations: Latches not supported unless you extend this to call a 'create_latch'.
    """
    data = _read_aag(lines)
    if data["L"] != 0:
        raise AagParseError("This minimal reader only supports combinational AAGs (L must be 0).")

    ad: AbstractAdapter = adapter(aig)

    # Helpers for literals → node refs
    var_to_node: Dict[int, Any] = {}  # maps AIGER variable index → node object
    nodes_in_order: List[Any] = []  # preserve PI order for names

    def lit_to_node(lit: int):
        if lit == 0:
            return ad.const(False)
        if lit == 1:
            return ad.const(True)
        v = lit >> 1
        inv = (lit & 1) == 1
        base = var_to_node.get(v)
        if base is None:
            raise AagParseError(f"Reference to undefined variable {v} (literal {lit})")
        return ad.NOT(base) if inv else base

    # 1) Create PIs in file order, remember names
    for i, lit in enumerate(data["inputs"]):
        if lit % 2 != 0:
            raise AagParseError(f"Input literal must be even (got {lit})")
        v = lit >> 1
        name = data["sym_i"].get(i, f"pi{i}")
        node = ad.pi(name)
        var_to_node[v] = node
        nodes_in_order.append(node)

    # 2) Build AND nodes in topological order
    # Each AND defines a new even literal 'lhs' (var index = lhs >> 1)
    for lhs, r0, r1 in data["ands"]:
        if lhs % 2 != 0:
            raise AagParseError(f"AND lhs must be even (got {lhs})")
        v = lhs >> 1
        a = lit_to_node(r0)
        b = lit_to_node(r1)

        node = ad.AND(a, b)

        var_to_node[v] = node

    # 3) Create POs (one per output)
    for i, lit in enumerate(data["outputs"]):
        node = lit_to_node(lit)
        name = data["sym_o"].get(i, f"po{i}")
        ad.po(node, name)

    return aig


def conv_aig_into_aag(aig, *, use_generic_symbols: bool = False, symbols: Optional[list]=None) -> List[str]:
    """
    Write an aigverse.Aig into an AIGER-ASCII (.aag) file.

    Limitations:
      - combinational only (no latches). Raises if aig.is_combinational() is False (when available).
      - no real names in the symbol table (optional generic names if include_symbols=True).

    Usage:
        write_aig_into_aag("f16mul_w.aag", aig)
    """
    # 0) Basic checks
    if hasattr(aig, "is_combinational") and not aig.is_combinational():
        raise ValueError("This writer supports combinational AIGs only (L=0).")

    pis = list(aig.pis())     # list[AigNode]
    pos = list(aig.pos())     # list[AigSignal]
    gates = list(aig.gates()) # list[AigNode]  (AND nodes)

    # 1) Build topo order of AND gates (Kahn)
    gate_set = set(gates)
    indeg = {g: 0 for g in gates}
    succs = {g: [] for g in gates}

    def _base_node(sig):
        return aig.get_node(sig)

    for g in gates:
        fins = [ _base_node(s) for s in aig.fanins(g) ]  # two fanins
        for c in fins:
            if c in gate_set:
                indeg[g] += 1
                succs[c].append(g)

    queue = [g for g in gates if indeg[g] == 0]
    topo = []
    while queue:
        n = queue.pop()
        topo.append(n)
        for h in succs[n]:
            indeg[h] -= 1
            if indeg[h] == 0:
                queue.append(h)

    if len(topo) != len(gates):
        raise RuntimeError("Cycle detected or unsupported node type in gates(); cannot topologically order.")

    # 2) Assign AIGER variable indices: PIs first, then ANDs in topo order
    var_index = {}  # AigNode -> int (1..M)
    for i, n in enumerate(pis, start=1):
        var_index[n] = i
    for j, n in enumerate(topo, start=1):
        var_index[n] = len(pis) + j

    I = len(pis)
    A = len(topo)
    L = 0
    O = len(pos)
    M = I + L + A

    # 3) Helpers to convert AigSignal -> AIGER literal
    def lit_of(sig) -> int:
        node = _base_node(sig)
        comp = aig.is_complemented(sig)
        if aig.is_constant(node):
            # AIGER has only const0 literal 0; const1 is 1 (inverted const0)
            return 1 if comp else 0
        # PI or AND node must have an assigned var index
        v = var_index.get(node)
        if v is None:
            # If you ever hit this, the AIG contains a node that isn't PI/AND/CONST.
            raise KeyError("Encountered unmapped node when encoding literal.")
        return (v << 1) | (1 if comp else 0)

    # 4) Emit lines
    lines = []
    lines.append(f"aag {M} {I} {L} {O} {A}")

    # inputs: in numbered order (2,4,6,...)
    for i in range(1, I + 1):
        lines.append(str(i << 1))

    # latches: none

    # outputs: arbitrary literals
    for s in pos:
        lines.append(str(lit_of(s)))

    # ANDs: lhs then rhs0 rhs1
    # lhs var indices are I+1 .. I+A  → lhs literal = 2*(I+k)
    for k, g in enumerate(topo, start=1):
        lhs = (I + k) << 1
        f0, f1 = aig.fanins(g)
        r0 = lit_of(f0)
        r1 = lit_of(f1)
        lines.append(f"{lhs} {r0} {r1}")

    # Optional symbol table with generic names
    if use_generic_symbols:
        for i in range(I):
            lines.append(f"i{i} pi{i}")
        for o in range(O):
            lines.append(f"o{o} po{o}")
    elif symbols is not None:
        lines.extend(symbols)

    lines.append("c")
    lines.append("generated by write_aig_into_aag")
    
    return lines

def write_aig_into_aag_file(filename: str, aig, *, include_symbols: bool = False) -> None:

    lines = conv_aig_into_aag(aig, use_generic_symbols=include_symbols)

    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# test
if __name__ == "__main__":

    # Then read into aigverse AIG:
    from aigverse import Aig  # or whatever your AIG class is called
    from spirehdl.aig.aig_aigerverse import conv_aag_into_aig
    from aigverse import equivalence_checking

    aig = Aig()
    # Create primary inputs
    x1 = aig.create_pi()
    x2 = aig.create_pi()

    # Create logic gates
    f_and = aig.create_and(x1, x2)  # AND gate
    f_or = aig.create_or(x1, x2)  # OR gate

    # Create primary outputs
    aig.create_po(f_and)
    aig.create_po(f_or)

    write_aig_into_aag_file("my.aag", aig, include_symbols=True)

    aig_load = Aig()
    read_aag_into_aig("my.aag", aig_load)

    print("PIs:", aig_load.num_pis() if hasattr(aig_load, "num_pis") else "<unknown>")
    print("POs:", aig.num_pos() if hasattr(aig, "num_pos") else "<unknown>")
    print("Nodes:", aig.size() if hasattr(aig, "size") else "<unknown>")

    # Check if the loaded AIG matches the original
    print("PIs loaded:", aig_load.num_pis() if hasattr(aig_load, "num_pis") else "<unknown>")
    print("POs loaded:", aig_load.num_pos() if hasattr(aig_load, "num_pos") else "<unknown>")
    print("Nodes loaded:", aig_load.size() if hasattr(aig_load, "size") else "<unknown>")

    # check logic equivalence

    if equivalence_checking(aig, aig_load):
        print("AIGs are equivalent.")
    else:
        print("AIGs differ.")