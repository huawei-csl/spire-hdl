from typing import Iterable, List, Tuple, Set, Dict


# Other custom topologies

Pair = Tuple[int, int]
PrefixNodes = Set[Pair]


def ParallelScan_8_a(n: int) -> PrefixNodes:

    assert n == 8

    nodes: PrefixNodes = set()

    for i in range(1, n):
        nodes.add((i, 0))

    for i in range(0, n):
        nodes.add((i, i))

    # nodes.add((1,0))
    nodes.add((3, 2))
    nodes.add((5, 4))
    nodes.add((7, 6))
    # nodes.add((3, 0))
    # nodes.add((5,0))

    return nodes


def ParallelScan_8_b(n: int) -> PrefixNodes:

    assert n == 8

    nodes: PrefixNodes = set()

    for i in range(1, n):
        nodes.add((i, 0))

    for i in range(0, n):
        nodes.add((i, i))

    nodes.add((1, 0))
    # nodes.add((3,0))
    nodes.add((4, 3))
    nodes.add((5, 3))
    nodes.add((7, 6))
    # nodes.add((5,0))

    return nodes


def ParallelScan_16_a(n: int) -> PrefixNodes:

    assert n == 16

    nodes: PrefixNodes = set()

    for i in range(1, n):
        nodes.add((i, 0))

    for i in range(0, n):
        nodes.add((i, i))

    # nodes.add((1,0))
    nodes.add((3, 2))
    nodes.add((5, 4))
    nodes.add((7, 6))
    nodes.add((9, 8))
    nodes.add((11, 10))
    nodes.add((13, 12))
    nodes.add((15, 14))

    nodes.add((7, 4))
    nodes.add((11, 8))

    return nodes


def ParallelScan_16_b(n: int) -> PrefixNodes:

    assert n == 16

    nodes: PrefixNodes = set()

    for i in range(1, n):
        nodes.add((i, 0))

    for i in range(0, n):
        nodes.add((i, i))

    nodes.add((1, 0))
    nodes.add((3, 0))
    nodes.add((4, 3))
    nodes.add((5, 3))
    nodes.add((7, 6))
    nodes.add((8, 6))
    nodes.add((10, 9))
    nodes.add((11, 9))
    nodes.add((13, 12))
    nodes.add((14, 12))
    # nodes.add((5,0))
    nodes.add((11, 6))
    # nodes.add((11,0))

    return nodes


# Widths for which a hand-crafted multi-scan / ZCG prefix graph exists.
_MULTI_SCAN_SIZES = (8, 16, 24, 32)
_ZCG_SIZES = (24, 32)


def multi_scan_n(n: int) -> PrefixNodes:
    """Hand-crafted multi-scan prefix graph.

    When ``n`` is one of the explicitly-supported sizes (8, 16, 24, 32),
    returns that template directly. Otherwise falls back to the next
    *lower* supported size and returns its nodes untouched: the caller is
    expected to run ``legalize_P(n, nodes)`` on the result, which
    extends the smaller template to cover the extra bits by adding the
    missing ``(i, 0)`` carry prefixes (and any splits they pull in).
    Raises ``ValueError`` only when no lower supported size exists
    (i.e. ``n < 8``).
    """
    if n == 8:
        return ParallelScan_8_b(n)
    if n == 16:
        return ParallelScan_16_b(n)
    if n == 24:
        return get_multiscan_nodes_24()
    if n == 32:
        return get_multiscan_nodes_32()
    lower = [s for s in _MULTI_SCAN_SIZES if s < n]
    if not lower:
        raise ValueError(
            f"FSA option PREFIX_MULTI_SCAN is not available for {n}-bit adders: its hand-crafted "
            f"templates start at {min(_MULTI_SCAN_SIZES)} bits (supported: {_MULTI_SCAN_SIZES}); "
            f"pick another --fsa-opt / fsa_opt for narrow widths"
        )
    return multi_scan_n(max(lower))


def ZCG_n(n: int) -> PrefixNodes:
    """Hand-crafted ZCG prefix graph.

    Same fallback policy as :func:`multi_scan_n`: when ``n`` is not in
    ``{24, 32}``, return the next lower supported template and let the
    caller's ``legalize_P(n, nodes)`` extend it. Raises ``ValueError``
    only when ``n < 24``.
    """
    if n == 24:
        return prefix_nodes_to_ranges(zcg_24)
    if n == 32:
        return prefix_nodes_to_ranges(zcg_32)
    lower = [s for s in _ZCG_SIZES if s < n]
    if not lower:
        raise ValueError(
            f"FSA option PREFIX_ZCG is not available for {n}-bit adders: its hand-crafted "
            f"templates start at {min(_ZCG_SIZES)} bits (supported: {_ZCG_SIZES}); "
            f"pick another --fsa-opt / fsa_opt for narrow widths"
        )
    return ZCG_n(max(lower))


def prefix_nodes_to_ranges(nodes: Iterable[Tuple[int, int, int]]) -> Set[Tuple[int, int]]:
    """
    Transform a prefix-sum/adder node list into the set of (upper, lower) ranges each node covers.

    Each input item is (p0, p1, level). The node’s identity is the rightmost index max(p0, p1).
    If a parent hasn't been produced by an earlier node, it's treated as a singleton range (i, i).

    Returns:
        A set of (upper, lower) tuples for all nodes created.
    """
    # Normalize to list and process by level, then by parents for determinism (not required for correctness).
    nodes_list: List[Tuple[int, int, int]] = list(nodes)
    nodes_list.sort(key=lambda x: (x[2], min(x[0], x[1]), max(x[0], x[1])))

    # Map "node id" -> (upper, lower) range. Base case: unseen parent i means (i, i).
    id_to_range: Dict[int, Tuple[int, int]] = {}

    out: Set[Tuple[int, int]] = set()
    for p0, p1, _lvl in nodes_list:
        r0 = id_to_range.get(p0, (p0, p0))
        r1 = id_to_range.get(p1, (p1, p1))

        upper = max(r0[0], r1[0])  # same as max(p0, p1)
        lower = min(r0[1], r1[1])

        node_id = max(p0, p1)  # rightmost index labels the produced node
        id_to_range[node_id] = (upper, lower)
        out.add((upper, lower))

    return out


# parallel scan patterns for 24 and 32 bits
ps_24 = [[0, 1, 1],
        [2, 3, 1],
        [4, 5, 1],
        [6, 7, 1],
        [8, 9, 1],
        [10, 11, 1],
        [12, 13, 1],
        [14, 15, 1],
        [16, 17, 1],
        [18, 19, 1],
        [20, 21, 1],
        [22, 23, 1],
        [1, 3, 2],
        [5, 7, 2],
        [9, 11, 2],
        [13, 15, 2],
        [17, 19, 2],
        [3, 7, 3],
        [11, 15, 3],
        [7, 15, 4],
        [7, 11, 5],
        [15, 19, 5],
        [3, 5, 6],
        [7, 9, 6],
        [11, 13, 6],
        [15, 17, 6],
        [19, 21, 6],
        [1, 2, 7],
        [3, 4, 7],
        [5, 6, 7],
        [7, 8, 7],
        [9, 10, 7],
        [11, 12, 7],
        [13, 14, 7],
        [15, 16, 7],
        [17, 18, 7],
        [19, 20, 7],
        [21, 22, 7],
        [21, 23, 7]]

ps_32 = [[0, 1, 1],
        [2, 3, 1],
        [4, 5, 1],
        [6, 7, 1],
        [8, 9, 1],
        [10, 11, 1],
        [12, 13, 1],
        [14, 15, 1],
        [16, 17, 1],
        [18, 19, 1],
        [20, 21, 1],
        [22, 23, 1],
        [24, 25, 1],
        [26, 27, 1],
        [28, 29, 1],
        [30, 31, 1],
        [1, 3, 2],
        [3, 5, 3],
        [7, 9, 2],
        [9, 11, 3],
        [13, 15, 2],
        [15, 17, 3],
        [19, 21, 2],
        [21, 23, 3],
        [25, 27, 2],
        [27, 29, 3],
        [5, 11, 4],
        [17, 23, 4],
        [11, 17, 5],
        [11, 23, 5],
        [5, 7, 6],
        [5, 9, 6],
        [11, 13, 6],
        [11, 15, 6],
        [17, 19, 6],
        [17, 21, 6],
        [23, 25, 6],
        [23, 27, 6],
        [23, 29, 6],
        [1, 2, 7],
        [3, 4, 7],
        [5, 6, 7],
        [7, 8, 7],
        [9, 10, 7],
        [11, 12, 7],
        [13, 14, 7],
        [15, 16, 7],
        [17, 18, 7],
        [19, 20, 7],
        [21, 22, 7],
        [23, 24, 7],
        [25, 26, 7],
        [27, 28, 7],
        [29, 30, 7],
        [29, 31, 7]]

def get_multiscan_nodes_24() -> Set[Tuple[int, int]]:
    inp = ps_24
    nodes = prefix_nodes_to_ranges(inp)
    return nodes

def get_multiscan_nodes_32() -> Set[Tuple[int, int]]:
    inp = ps_32
    nodes = prefix_nodes_to_ranges(inp)
    return nodes

zcg_24 = [[0, 1, 1],
        [2, 3, 1],
        [4, 5, 1],
        [6, 7, 1],
        [8, 9, 1],
        [10, 11, 1],
        [12, 13, 1],
        [14, 15, 1],
        [16, 17, 1],
        [18, 19, 1],
        [20, 21, 1],
        [1, 2, 2],
        [1, 3, 2],
        [5, 7, 2],
        [9, 11, 2],
        [13, 15, 2],
        [17, 19, 2],
        [21, 22, 2],
        [3, 4, 3],
        [3, 5, 3],
        [3, 7, 3],
        [11, 15, 3],
        [19, 22, 3],
        [5, 6, 4],
        [7, 8, 4],
        [7, 9, 4],
        [7, 11, 4],
        [7, 15, 4],
        [9, 10, 5],
        [11, 12, 5],
        [11, 13, 5],
        [15, 16, 5],
        [15, 17, 5],
        [15, 19, 5],
        [15, 22, 5],
        [13, 14, 6],
        [17, 18, 6],
        [19, 20, 6],
        [19, 21, 6],
        [22, 23, 6]]

zcg_32 = [[0, 1, 1],
[2, 3, 1],
[4, 5, 1],
[6, 7, 1],
[8, 9, 1],
[10, 11, 1],
[12, 13, 1],
[14, 15, 1],
[16, 17, 1],
[18, 19, 1],
[20, 21, 1],
[23, 24, 1],
[27, 28, 1],
[1, 2, 2],
[1, 3, 2],
[5, 7, 2],
[9, 11, 2],
[13, 15, 2],
[17, 19, 2],
[21, 22, 2],
[24, 25, 2],
[28, 29, 2],
[3, 4, 3],
[3, 5, 3],
[3, 7, 3],
[11, 15, 3],
[19, 22, 3],
[25, 26, 3],
[29, 30, 3],
[5, 6, 4],
[7, 8, 4],
[7, 9, 4],
[7, 11, 4],
[7, 15, 4],
[22, 26, 4],
[30, 31, 4],
[9, 10, 5],
[11, 12, 5],
[11, 13, 5],
[15, 16, 5],
[15, 17, 5],
[15, 19, 5],
[15, 22, 5],
[15, 26, 5],
[13, 14, 6],
[17, 18, 6],
[19, 20, 6],
[19, 21, 6],
[22, 23, 6],
[22, 24, 6],
[22, 25, 6],
[26, 27, 6],
[26, 28, 6],
[26, 29, 6],
[26, 30, 6],
[26, 31, 6]]


# data_list = [[0, 1, 1], [2, 3, 1], [4, 5, 1], [6, 7, 1], [1, 3, 2], [3, 5, 3], [1, 2, 4], [3, 4, 4], [5, 6, 4], [5, 7, 4]]


def test_prefix_merge_ranges():

    inp = ps_24
    nodes = prefix_nodes_to_ranges(inp)
    print(nodes)
    print(len(ps_24))
    print(len(nodes))
    # -> {(4, 0), (7, 0), (2, 0), (5, 4), (3, 0), (5, 0), (7, 6), (6, 0), (1, 0), (3, 2)}

if __name__ == "__main__":
    test_prefix_merge_ranges()