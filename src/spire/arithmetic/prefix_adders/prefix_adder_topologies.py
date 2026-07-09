from typing import Dict, List, Optional, Set, Tuple

from spire.arithmetic.prefix_adders.prefix_adder_specials import get_multiscan_nodes_24, get_multiscan_nodes_32, prefix_nodes_to_ranges, zcg_24, zcg_32


Pair = Tuple[int, int]
PrefixNodes = Set[Pair]

# ------------------------- Utilities -------------------------


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


def legalize_P(n: int, nodes: PrefixNodes) -> PrefixNodes:
    """
    Ensure that the nodes are legal for a prefix adder:
      - (i,0) exists for all i=1..n-1
      - (i,i) exists for all i=0..n-1
      - (n-1,0) exists
    Returns: normalized set of nodes.
    """
    nodes = _normalize_P(n, nodes)

    # add all (i,0) for i=1..n-1
    for i in range(1, n):
        if not (i, 0) in nodes:
            nodes.add((i, 0))

    def exists(i, j):
        return (i == j) or ((i, j) in nodes)

    def fix_node(i: int, j: int):
        if i == j:
            return
        for k in range(j, i):  # look for a valid split
            if exists(i, k + 1) and exists(k, j):
                return
        # choose a k
        k_new = i - 1
        nodes.add((i, k_new))
        fix_node(i, k_new)  # recurse to fix the new node
        if k_new - 1 != j:
            nodes.add((k_new - 1, j))
            fix_node(k_new - 1, j)  # recurse to fix the new node

    for node in nodes.copy():
        fix_node(*node)

    for i in range(1, n):
        fix_node(i, 0)  # ensure (n-1,0) exists

    return nodes

# ------------------------- Famous topologies -------------------------


def P_ripple_carry(n: int) -> PrefixNodes:
    """Only (i,0) for i=1..n-1. Leaves (i,i) are implicit."""
    return {(i, 0) for i in range(1, n)}


def P_kogge_stone(n: int) -> PrefixNodes:
    """
    Kogge–Stone nodes over indices 0..n-1.
    At stage s, span = 2^{s+1} - 1 -> 1,3,7,15,...
    Add:
      - the KS pairs (i, i-span) for i >= span
      - (i, i) and (i, 0) for all i=1..n-1
    """
    nodes: PrefixNodes = set()

    # Ensure (i, i) and (i, 0) exist for i=1..n-1
    for i in range(1, n):
        # nodes.add((i, i)) # implicit in the tree
        nodes.add((i, 0))

    # Kogge–Stone spans: 1, 3, 7, 15, ...
    span = 1
    while span < n:
        for i in range(span, n):
            nodes.add((i, i - span))
        span = (span << 1) | 1  # 2*span + 1

    return nodes


def P_sklansky(n: int) -> PrefixNodes:
    """
    Sklansky (divide&conquer): for stage s, block=2^{s+1}, half=2^s.
    For each block starting at g, connect all i in [g+half .. g+block-1] to j=g.
    """
    nodes: PrefixNodes = set()
    s = 0
    while (1 << (s + 1)) <= n:
        block = 1 << (s + 1)
        half = 1 << s
        for g in range(0, n, block):
            j = g
            upper = min(g + block - 1, n - 1)
            for i in range(g + half, upper + 1):
                nodes.add((i, j))
        s += 1
    return nodes


def P_brent_kung(n: int) -> PrefixNodes:
    """
    Brent–Kung nodes as (i,j), where the node *outputs* the prefix over [j..i].
    We simulate upsweep (reduction) and downsweep (distribution) while tracking,
    for each bit index t, the current available prefix start j_at[t].
    """
    import math

    nodes: PrefixNodes = set()
    if n <= 1:
        return nodes

    L = math.ceil(math.log2(n))

    # j_at[t] = lowest index j such that we currently have prefix for [j..t]
    j_at = [i for i in range(n)]  # initially only leaves (i,i)

    # ---- Upsweep: compute group prefixes at block ends ----
    # At level s, step = 2^(s+1), half = 2^s.
    # For i = step-1, step-1+step, ... we combine (i, i-half+1) with prefix at r=i-half.
    for s in range(L):
        step = 1 << (s + 1)
        half = 1 << s
        for i in range(step - 1, n, step):
            r = i - half  # right index feeds the lower-half prefix
            j = j_at[r]  # right child's j determines the output j
            nodes.add((i, j))  # this black cell outputs prefix [j..i]
            j_at[i] = j  # update available prefix at i

    # ---- Downsweep: distribute prefixes to the missing positions ----
    # For level s = L-2 .. 0: positions i = 3*2^s - 1, then + step
    # Combine (i, i-half+1) with the prefix available at r=i-half (which already holds a wider j).
    for s in range(L - 2, -1, -1):
        step = 1 << (s + 1)
        half = 1 << s
        start = (3 << s) - 1  # 3*2^s - 1
        for i in range(start, n, step):
            r = i - half
            j = j_at[r]  # right child's j propagates to the output
            nodes.add((i, j))
            j_at[i] = j

    return nodes


def P_han_carlson(n: int) -> PrefixNodes:
    """
    Han–Carlson hybrid (sparse Kogge–Stone front-end with Brent–Kung backbone).
    Adds:
      - a sparse KS sweep with span=1,2,4,... placed every other block
      - BK-style bridging nodes to propagate prefixes to the missing positions
      - (i,0) for all i (carry to bit 0)
    """
    nodes: PrefixNodes = set()

    # Always allow direct carry to bit 0
    for i in range(1, n):
        nodes.add((i, 0))

    # Sparse Kogge–Stone sweep
    span = 1
    while span < n:
        for i in range(span, n, span * 2):
            nodes.add((i, i - span))
        span *= 2

    # Brent–Kung-style bridging sweep
    span = 2
    while span < n:
        for i in range(span - 1, n, span * 2):
            nodes.add((i, i - span + 1))
        span *= 2

    # Some widths (non–powers of two) miss interior support nodes; legalize to repair.
    return legalize_P(n, nodes)


def P_ladner_fischer(n: int) -> PrefixNodes:
    """
    Ladner–Fischer prefix network using odd/even recursive decomposition.
    This keeps depth near log2(n) while reducing fan-out versus Sklansky.
    """
    nodes: PrefixNodes = set()
    if n <= 1:
        return nodes

    # j_at[i] is the lowest bit currently covered by the prefix available at i.
    j_at = [i for i in range(n)]

    def build(indices: List[int]) -> None:
        if len(indices) <= 1:
            return

        odd_indices = indices[1::2]

        # Pre-combine odd positions with their preceding even positions.
        for pos, idx in enumerate(odd_indices):
            prev_even = indices[2 * pos]
            j = j_at[prev_even]
            nodes.add((idx, j))
            j_at[idx] = j

        # Recurse over odd outputs to build the upper prefix spine.
        build(odd_indices)

        # Distribute computed odd prefixes to remaining even positions.
        for local_pos in range(2, len(indices), 2):
            idx = indices[local_pos]
            prev_odd = indices[local_pos - 1]
            j = j_at[prev_odd]
            nodes.add((idx, j))
            j_at[idx] = j

    build(list(range(n)))
    return legalize_P(n, nodes)


def _P_sparse_kogge_stone(n: int, sparsity: int) -> PrefixNodes:
    """
    Sparse Kogge–Stone helper.
    Builds a dense KS tree over block endpoints and relies on legalization to
    complete intra-block carry coverage.
    """
    if sparsity < 2:
        raise ValueError(f"sparsity must be >= 2, got {sparsity}")

    nodes: PrefixNodes = set()
    if n <= 1:
        return nodes

    # Block endpoints: sparsity-1, 2*sparsity-1, ...
    endpoints = list(range(sparsity - 1, n, sparsity))
    if not endpoints or endpoints[-1] != n - 1:
        endpoints.append(n - 1)

    j_at: Dict[int, int] = {}

    # Local block prefix at each endpoint.
    for endpoint in endpoints:
        group_start = max(0, endpoint - (sparsity - 1))
        nodes.add((endpoint, group_start))
        j_at[endpoint] = group_start

    # KS sweep over endpoints in endpoint index-space.
    span = 1
    while span < len(endpoints):
        for endpoint_pos in range(span, len(endpoints)):
            endpoint = endpoints[endpoint_pos]
            prev_endpoint = endpoints[endpoint_pos - span]
            j = j_at[prev_endpoint]
            nodes.add((endpoint, j))
            j_at[endpoint] = j
        span <<= 1

    return legalize_P(n, nodes)


def P_sparse_kogge_stone_2(n: int) -> PrefixNodes:
    """Sparse-2 Kogge–Stone (carry tree at every 2nd bit)."""
    return _P_sparse_kogge_stone(n, sparsity=2)


def P_sparse_kogge_stone_4(n: int) -> PrefixNodes:
    """Sparse-4 Kogge–Stone (carry tree at every 4th bit)."""
    return _P_sparse_kogge_stone(n, sparsity=4)


# Analysis ---------------------------------------------------------------------------------------------------------


def _exists(nodes: PrefixNodes, i: int, j: int) -> bool:
    """Does (i,j) exist as a node (combine) or leaf?"""
    return (i == j) or ((i, j) in nodes)


def _find_split(nodes: PrefixNodes, i: int, j: int) -> Optional[int]:
    """
    Find a k (j ≤ k < i) such that left=(i,k+1) and right=(k,j) both exist
    (either as combine nodes in 'nodes' or as leaves when left has k+1==i or right has k==j).
    Greedy from the top (largest k) to encourage shallow cones (typical for KS/BK/Sklansky).
    """
    for k in reversed(range(i - 1, j - 1, -1)): # actually needs to be reversed
        left_ok  = _exists(nodes, i, k + 1)
        right_ok = _exists(nodes, k, j)
        if left_ok and right_ok:
            return k
    return None

def _valid_splits(nodes: PrefixNodes, i: int, j: int) -> List[int]:
    """All k with j ≤ k < i such that left=(i,k+1) and right=(k,j) both exist."""
    Ks: List[int] = []
    for k in range(j, i):
        if _exists(nodes, i, k + 1) and _exists(nodes, k, j):
            Ks.append(k)
    return Ks

def analyze_prefix_matrix(
    n: int,
    nodes: PrefixNodes,
    *,
    tie_break: str = "min_k",  # among equal-min-depth splits: "max_k" | "min_k" | "balanced"
) -> Tuple[Dict[Pair, int], Dict[Pair, int], Dict[int, int], int]:
    """
    Compute minimal combine-depth for every (i,j) in P (and leaves), choose best splits, and
    return:
      nodes:   normalized set of (i,j)
      depth:   map (i,j) -> minimal black-cell depth
      best_k:  map (i,j) -> chosen split k that achieves depth[i,j] (only for i>j)
      carry_depth: map i -> depth(i,0)     (depth to compute G[i:0])
      worst_depth: max over i of depth(i,0)
    """
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
            raise ValueError(f"P does not declare combine node {(i,j)} but it is required.")
        Ks = _valid_splits(nodes, i, j)
        if not Ks:
            raise ValueError(f"No valid split for node {(i,j)}. Check your P matrix.")
        # Evaluate all splits: depth = 1 + max(depth(left), depth(right))
        cand: List[Tuple[int, int, int, int]] = []  # (depth, k, dl, dr)
        for k in Ks:
            dl = node_depth(i, k + 1)
            dr = node_depth(k, j)
            d = 1 + max(dl, dr)
            cand.append((d, k, dl, dr))
        # Choose min depth; tie-break as requested
        minD = min(c[0] for c in cand)
        best = [c for c in cand if c[0] == minD]
        if tie_break == "max_k":
            chosen = max(best, key=lambda t: t[1])
        elif tie_break == "min_k":
            chosen = min(best, key=lambda t: t[1])
        elif tie_break == "balanced":
            # minimize subtree imbalance; tie-break to larger k
            def imbalance(t):
                _, k, dl, dr = t
                return (abs(dl - dr), -k)
            chosen = min(best, key=imbalance)
        else:
            raise ValueError(f"Unknown tie_break '{tie_break}'")
        depth[key] = chosen[0]
        best_k[key] = chosen[1]
        return depth[key]

    # Ensure carries to bit 0 exist; compute depths for all required (i,0)
    for i in range(n):
        node_depth(i, 0)

    # Optionally compute depth for every declared node (fills depth map)
    for (i, j) in nodes:
        node_depth(i, j)

    carry_depth = {i: depth[(i, 0)] for i in range(n)}
    worst_depth = max(carry_depth.values()) if carry_depth else 0
    return depth, best_k, carry_depth, worst_depth