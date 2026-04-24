"""Clean prefix-adder builder with a minimal set of topologies.

Implemented topologies:
- Ripple carry
- Kogge-Stone
- Sklansky (with ``P_slansky`` alias)
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Set, Tuple

from spirehdl.spirehdl import Bool, Const, Expr, UInt, cat
from spirehdl.spirehdl_module import Module

Pair = Tuple[int, int]
PrefixNodes = Set[Pair]


def _normalize_P(n: int, nodes: PrefixNodes) -> PrefixNodes:
    """
    Accepts:
      - set of (i,j).
    Returns: set of (i,j) with 0 <= j <= i < n and j < i (combine nodes only).
    Leaves (i,i) are implicit and NOT stored.
    """
    normalized: PrefixNodes = set()
    for i, j in nodes:
        if not (0 <= j <= i < n):
            raise ValueError(f"Node {(i, j)} out of range for n={n}")
        if i != j:
            normalized.add((i, j))
    return normalized


def _exists(nodes: PrefixNodes, i: int, j: int) -> bool:
    """Does (i,j) exist as a node (combine) or leaf?"""
    return (i == j) or ((i, j) in nodes)


def _find_split(nodes: PrefixNodes, i: int, j: int) -> Optional[int]:
    """
    Find a k (j <= k < i) such that left=(i,k+1) and right=(k,j) both exist
    (either as combine nodes in 'nodes' or as leaves when left has k+1==i or right has k==j).
    Greedy from the top (largest k) to encourage shallow cones (typical for KS/Sklansky).
    """
    for k in range(i - 1, j - 1, -1):
        if _exists(nodes, i, k + 1) and _exists(nodes, k, j):
            return k
    return None


def _valid_splits(nodes: PrefixNodes, i: int, j: int) -> List[int]:
    """All k with j <= k < i such that left=(i,k+1) and right=(k,j) both exist."""
    ks: List[int] = []
    for k in range(j, i):
        if _exists(nodes, i, k + 1) and _exists(nodes, k, j):
            ks.append(k)
    return ks


def legalize_P(n: int, nodes: PrefixNodes) -> PrefixNodes:
    """
    Ensure that the nodes are legal for a prefix adder:
      - (i,0) exists for all i=1..n-1
      - (i,i) exists for all i=0..n-1 (implicit leaves)
      - (n-1,0) exists
    Returns: normalized set of nodes.
    """
    nodes = _normalize_P(n, nodes)

    # Ensure all carry prefixes (i,0) exist.
    for i in range(1, n):
        if (i, 0) not in nodes:
            nodes.add((i, 0))

    def fix_node(i: int, j: int) -> None:
        if i == j:
            return
        for k in range(j, i):
            if _exists(nodes, i, k + 1) and _exists(nodes, k, j):
                return

        # If no split exists, add minimal support and recurse.
        k_new = i - 1
        nodes.add((i, k_new))
        fix_node(i, k_new)
        if k_new - 1 != j:
            nodes.add((k_new - 1, j))
            fix_node(k_new - 1, j)

    for i, j in list(nodes):
        fix_node(i, j)
    for i in range(1, n):
        fix_node(i, 0)

    return nodes


def analyze_prefix_matrix(
    n: int,
    nodes: PrefixNodes,
    *,
    tie_break: str = "min_k",
) -> Tuple[Dict[Pair, int], Dict[Pair, int], Dict[int, int], int]:
    """
    Compute minimal combine-depth for every (i,j) in P (and leaves), choose best splits, and
    return:
      depth:   map (i,j) -> minimal black-cell depth
      best_k:  map (i,j) -> chosen split k that achieves depth[i,j] (only for i>j)
      carry_depth: map i -> depth(i,0)     (depth to compute G[i:0])
      worst_depth: max over i of depth(i,0)
    """
    nodes = legalize_P(n, nodes)
    depth: Dict[Pair, int] = {}
    best_k: Dict[Pair, int] = {}

    def node_depth(i: int, j: int) -> int:
        key = (i, j)
        if i == j:
            depth[key] = 0
            return 0
        if key in depth:
            return depth[key]
        if key not in nodes:
            raise ValueError(f"P does not declare combine node {(i, j)} but it is required.")

        candidates: List[Tuple[int, int, int, int]] = []
        for k in _valid_splits(nodes, i, j):
            dl = node_depth(i, k + 1)
            dr = node_depth(k, j)
            d = 1 + max(dl, dr)
            candidates.append((d, k, dl, dr))
        if not candidates:
            raise ValueError(f"No valid split for node {(i, j)}.")

        min_depth = min(c[0] for c in candidates)
        best = [c for c in candidates if c[0] == min_depth]
        if tie_break == "max_k":
            chosen = max(best, key=lambda t: t[1])
        elif tie_break == "min_k":
            chosen = min(best, key=lambda t: t[1])
        elif tie_break == "balanced":
            chosen = min(best, key=lambda t: (abs(t[2] - t[3]), -t[1]))
        else:
            raise ValueError(f"Unknown tie_break '{tie_break}'")

        depth[key] = chosen[0]
        best_k[key] = chosen[1]
        return depth[key]

    for i in range(n):
        node_depth(i, 0)
    for i, j in nodes:
        node_depth(i, j)

    carry_depth = {i: depth[(i, 0)] for i in range(n)}
    worst_depth = max(carry_depth.values()) if carry_depth else 0
    return depth, best_k, carry_depth, worst_depth


def build_prefix_adder_from_matrix(
    name: str,
    n: int,
    P: PrefixNodes,
    *,
    with_cin: bool = False,
    with_cout: bool = True,
    depth_optimize: bool = True,
) -> Module:
    """
    Build an n-bit SpireHDL adder using a prefix tree specified by P.

    P: set of (i,j) pairs (combine nodes).
       Leaves (i,i) are implicit and must NOT be listed.

    Ports:
      - a: UInt(n), b: UInt(n)
      - cin: UInt(1)  (optional)
      - y: UInt(n)
      - cout: UInt(1) (optional)

    Conventions:
      p_i = a_i ^ b_i, g_i = a_i & b_i
      combine: (G,P) o (G',P') = (G | (P & G'), P & P')
    """
    # Make sure the provided prefix nodes are structurally complete for carry-to-0.
    nodes = legalize_P(n, P)

    # Precompute best split points per node when depth optimization is enabled.
    best_k: Dict[Pair, int] = {}
    if depth_optimize and nodes:
        _, best_k, _, _ = analyze_prefix_matrix(n, nodes)

    # Build combinational adder module and declare ports.
    m = Module(name, with_clock=False, with_reset=False)
    a = m.input(UInt(n), "a")
    b = m.input(UInt(n), "b")
    cin: Expr = m.input(UInt(1), "cin") if with_cin else Const(False, Bool())
    y = m.output(UInt(n), "y")
    cout = m.output(UInt(1), "cout") if with_cout else None

    # Bit-level propagate/generate signals for each input bit.
    p = [a[i] ^ b[i] for i in range(n)]
    g = [a[i] & b[i] for i in range(n)]
    gp_cache: Dict[Pair, Tuple[Expr, Expr]] = {}

    def gp(i: int, j: int) -> Tuple[Expr, Expr]:
        # Recursively compute prefix pair (G[i:j], P[i:j]) with memoization.
        key = (i, j)
        if key in gp_cache:
            return gp_cache[key]
        if i == j:
            gp_cache[key] = (g[i], p[i])
            return gp_cache[key]
        if key not in nodes:
            raise ValueError(f"Prefix node {(i, j)} is required but missing.")

        if depth_optimize:
            k = best_k[key]
        else:
            k = _find_split(nodes, i, j)
            if k is None:
                raise ValueError(f"No valid split for node {(i, j)}.")

        # Prefix combine: (G,P) o (G',P') = (G | (P & G'), P & P').
        g_left, p_left = gp(i, k + 1)
        g_right, p_right = gp(k, j)
        gp_cache[key] = (g_left | (p_left & g_right), p_left & p_right)
        return gp_cache[key]

    # Compute carry-in for each bit from prefix against j=0 (plus optional global cin).
    carries: List[Expr] = [cin] * (n + 1)
    for i in range(n):
        g_i0, p_i0 = gp(i, 0)
        carries[i + 1] = g_i0 | (p_i0 & cin)

    # Sum bits use propagate xor carry-in; pack LSB..MSB via cat(*sums).
    sums = [p[i] ^ carries[i] for i in range(n)]
    y <<= cat(*sums)
    if cout is not None:
        cout <<= carries[n]

    return m


def P_ripple_carry(n: int) -> PrefixNodes:
    """Only (i,0) for i=1..n-1. Leaves (i,i) are implicit."""
    return {(i, 0) for i in range(1, n)}


def P_kogge_stone(n: int) -> PrefixNodes:
    """
    Kogge-Stone nodes over indices 0..n-1.
    At stage s, span = 2^{s+1} - 1 -> 1,3,7,15,...
    Add:
      - the KS pairs (i, i-span) for i >= span
      - (i, 0) for all i=1..n-1
    """
    nodes: PrefixNodes = {(i, 0) for i in range(1, n)}
    span = 1
    while span < n:
        for i in range(span, n):
            nodes.add((i, i - span))
        span = (span << 1) | 1
    return nodes


def P_sklansky(n: int) -> PrefixNodes:
    """
    Sklansky (divide&conquer): for stage s, block=2^{s+1}, half=2^s.
    For each block starting at g, connect all i in [g+half .. g+block-1] to j=g.
    """
    nodes: PrefixNodes = set()
    stage = 0
    while (1 << (stage + 1)) <= n:
        block = 1 << (stage + 1)
        half = 1 << stage
        for base in range(0, n, block):
            j = base
            upper = min(base + block - 1, n - 1)
            for i in range(base + half, upper + 1):
                nodes.add((i, j))
        stage += 1
    return nodes


def P_slansky(n: int) -> PrefixNodes:
    """Compatibility alias for the common misspelling."""
    return P_sklansky(n)


def build_ripple_carry_adder(
    name: str,
    n: int,
    *,
    with_cin: bool = False,
    with_cout: bool = True,
) -> Module:
    return build_prefix_adder_from_matrix(
        name, n, P_ripple_carry(n), with_cin=with_cin, with_cout=with_cout, depth_optimize=False
    )


def build_kogge_stone_adder(
    name: str,
    n: int,
    *,
    with_cin: bool = False,
    with_cout: bool = True,
) -> Module:
    return build_prefix_adder_from_matrix(
        name, n, P_kogge_stone(n), with_cin=with_cin, with_cout=with_cout, depth_optimize=True
    )


def build_sklansky_adder(
    name: str,
    n: int,
    *,
    with_cin: bool = False,
    with_cout: bool = True,
) -> Module:
    return build_prefix_adder_from_matrix(
        name, n, P_sklansky(n), with_cin=with_cin, with_cout=with_cout, depth_optimize=True
    )


def _build_adder_vectors(
    n: int,
    *,
    with_cin: bool,
    with_cout: bool,
    num_vectors: int,
    seed: int,
) -> List[Tuple[str, Dict[str, int], Dict[str, int]]]:
    """Build random simulation vectors in helpers.run_vectors format."""
    if num_vectors < 1:
        raise ValueError(f"num_vectors must be >= 1, got {num_vectors}")

    rng = random.Random(seed)
    mask = (1 << n) - 1
    vectors: List[Tuple[str, Dict[str, int], Dict[str, int]]] = []

    for idx in range(num_vectors):
        a_val = rng.randrange(1 << n)
        b_val = rng.randrange(1 << n)
        cin_val = rng.randrange(2) if with_cin else 0

        total = a_val + b_val + cin_val
        y_exp = total & mask
        cout_exp = (total >> n) & 1

        ins: Dict[str, int] = {"a": a_val, "b": b_val}
        if with_cin:
            ins["cin"] = cin_val

        outs: Dict[str, int] = {"y": y_exp}
        if with_cout:
            outs["cout"] = cout_exp

        vectors.append((f"vec_{idx:04d}", ins, outs))

    return vectors


def smoke_test_prefix_adder_simulation(
    *,
    topology: str = "kogge_stone",
    n: int = 8,
    with_cin: bool = False,
    with_cout: bool = True,
    num_vectors: int = 128,
    seed: int = 0xADDEF,
    print_on_pass: bool = False,
) -> None:
    """Build and simulate a clean prefix adder using helpers.run_vectors."""
    from spirehdl.helpers import run_vectors

    builders = {
        "ripple_carry": build_ripple_carry_adder,
        "kogge_stone": build_kogge_stone_adder,
        "sklansky": build_sklansky_adder,
        "slansky": build_sklansky_adder,
    }
    try:
        builder = builders[topology]
    except KeyError as exc:
        raise ValueError(f"Unknown topology '{topology}'. Expected one of: {sorted(builders.keys())}") from exc

    m = builder(
        name=f"PrefixAdder_{topology}_{n}",
        n=n,
        with_cin=with_cin,
        with_cout=with_cout,
    )
    vectors = _build_adder_vectors(
        n=n,
        with_cin=with_cin,
        with_cout=with_cout,
        num_vectors=num_vectors,
        seed=seed,
    )
    run_vectors(m, vectors, raise_on_fail=True, print_on_pass=print_on_pass)


if __name__ == "__main__":
    smoke_test_prefix_adder_simulation(topology="kogge_stone", n=8, num_vectors=64, with_cin=False, with_cout=True)
