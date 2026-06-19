import random
import time
from typing import List, Optional, Tuple

from aigverse import DepthAig, DepthAig, aig_cut_rewriting, aig_resubstitution, balancing, sop_refactoring
import numpy as np
from tqdm import tqdm
from spire.aig.aig_aigerverse import conv_aag_into_aig, conv_aig_into_aag
from testing.low_level_arithmetic.compressor_tree.compressor_tree_multiplier import Graph, build_wallace_compressor_graph, get_node_kind_counts, random_compressor_tree
from spire.arithmetic.prefix_adders.prefix_adder_analysis import Vec
from spire.helpers import get_yosys_transistor_count
from spire.expr import Bool, HDLType, Op2
from spire.aiger import AigerExporter
from spire.component import Netlist
from spire.simulator import Simulator


def build_multiplier_from_compressor_graph(name: str, A, nodes):
    """
    Build a Spire combinational multiplier module from a compressor-tree graph.

    Inputs inferred from graph:
      - n is inferred from pp[i,j] signal names (square n×n multiplier).

    Graph assumptions (as used in this chat):
      - Node(kind) ∈ {'sig','FA','HA','sink'}
      - For 'sig' nodes: port ∈ {'pp','sum','carry'}; weight = i+j for pp; carry has weight+1
      - Edges go: sig(w) -> (FA/HA at w), and (FA/HA at w) -> sum(w) / carry(w+1), and sig(w)->sink(w)
      - Final two rows either via explicit 'sink' nodes named "final_row_[A|B]_w{w}"
        or implicitly as signals with no outgoing edge to adders.

    Returns:
      - spire.component.Netlist with ports:
          a: UInt(n), b: UInt(n), y: UInt(2*n)
    """
    import re
    import numpy as np
    from collections import defaultdict

    from spire.component import Netlist
    from spire.expr import UInt, cat

    # --- helpers ---------------------------------------------------------------
    def infer_n():
        pp_nodes = [nd for nd in nodes if nd.kind == "sig" and nd.port == "pp"]
        if not pp_nodes:
            raise ValueError("No partial products found (sig/pp).")
        mi = mj = 0
        for nd in pp_nodes:
            m = re.search(r"pp\[(\d+),(\d+)\]", nd.name)
            if not m:
                raise ValueError(f"Bad pp name: {nd.name}")
            i, j = int(m.group(1)), int(m.group(2))
            mi = max(mi, i); mj = max(mj, j)
        if mi != mj:
            raise ValueError(f"Inferred non-square pp grid: max_i={mi}, max_j={mj}")
        return mi + 1

    def parse_pp_indices(name: str):
        m = re.search(r"pp\[(\d+),(\d+)\]", name)
        if not m:
            raise ValueError(f"Cannot parse pp indices from '{name}'")
        return int(m.group(1)), int(m.group(2))

    # Build predecessor/successor lists once
    rows, cols = np.nonzero(A)
    succ = {i: [] for i in range(A.shape[0])}
    pred = {i: [] for i in range(A.shape[0])}
    for u, v in zip(rows.tolist(), cols.tolist()):
        succ[u].append(v)
        pred[v].append(u)

    n = infer_n()
    W = 2 * n  # product width

    # --- Spire module skeleton ----------------------------------------------
    m = Netlist(name, with_clock=False, with_reset=False)
    a = m.input(UInt(n), "a")
    b = m.input(UInt(n), "b")
    y = m.output(UInt(W), "y")

    # One-bit expression for each signal node
    sig_expr = {}  # node_id -> Spire bit expr

    # Initialize partial products
    for nd in nodes:
        if nd.kind == "sig" and nd.port == "pp":
            i, j = parse_pp_indices(nd.name)
            sig_expr[nd.idx] = (a[i] & b[j])

    # Compute adder outputs stage-by-stage to respect dataflow
    max_stage = max((nd.stage or 0) for nd in nodes)
    for st in range(1, max_stage + 1):
        for nd in nodes:
            if nd.kind in ("FA", "HA") and nd.stage == st:
                # gather inputs (signals) in any order; graph guarantees correct wiring by weight
                ins = [u for u in pred[nd.idx] if nodes[u].kind == "sig"]
                if nd.kind == "FA":
                    if len(ins) != 3:
                        raise ValueError(f"FA at idx={nd.idx} does not have 3 signal inputs")
                    x, yb, z = (sig_expr[ins[0]], sig_expr[ins[1]], sig_expr[ins[2]])

                    # fast version
                    # s_bit = (x ^ yb) ^ z
                    # c_bit = (x & yb) | (x & z) | (yb & z)

                    # with nands
                    #def nand(u, v):
                    #   return ~(u & v)

                    def nand(u, v):
                        return Op2(u, v, "nand", Bool())  # experimental feature

                    s1 = x ^ yb
                    s_bit = s1 ^ z
                    c_bit = nand(nand(s1, z), nand(x, yb))

                    # low transistor count version
                    # s1 = x ^ yb
                    # s_bit = s1 ^ z
                    # c_bit = (s1 & z) | (x & yb)
                    
                else:  # HA
                    if len(ins) != 2:
                        raise ValueError(f"HA at idx={nd.idx} does not have 2 signal inputs")
                    x, yb = (sig_expr[ins[0]], sig_expr[ins[1]])
                    s_bit = x ^ yb
                    c_bit = x & yb

                # drive successor signal nodes
                outs = [v for v in succ[nd.idx] if nodes[v].kind == "sig"]
                for v in outs:
                    if nodes[v].port == "sum":
                        sig_expr[v] = s_bit
                    elif nodes[v].port == "carry":
                        sig_expr[v] = c_bit

    # Collect final two rows per weight
    rowA = [0] * W
    rowB = [0] * W

    # Prefer explicit sinks if present
    sinks = [nd for nd in nodes if nd.kind == "sink"]
    if sinks:
        for sk in sinks:
            mname = re.search(r"final_row_(A|B)_w(\d+)", sk.name)
            if not mname:
                continue
            row_name, w = mname.group(1), int(mname.group(2))
            # find the single signal predecessor
            p_sig = [u for u in pred[sk.idx] if nodes[u].kind == "sig"]
            if not p_sig:
                continue
            bit = sig_expr[p_sig[0]]
            if w < W:
                if row_name == "A":
                    rowA[w] = bit
                else:
                    rowB[w] = bit
    else:
        # Implicit: signals that no longer feed adders are final bits
        finals_by_w = defaultdict(list)
        for nd in nodes:
            if nd.kind == "sig" and nd.port in ("sum", "carry"):
                if not any(nodes[v].kind in ("FA", "HA") for v in succ[nd.idx]):
                    finals_by_w[nd.weight].append(nd.idx)
        for w, ids in finals_by_w.items():
            ids_sorted = sorted(ids)
            if w < W and len(ids_sorted) >= 1:
                rowA[w] = sig_expr[ids_sorted[0]]
            if w < W and len(ids_sorted) >= 2:
                rowB[w] = sig_expr[ids_sorted[1]]

        # Also include any passthrough pp bits that were never consumed and land directly in final columns
        # (e.g., w=0 LSB)
        for nd in nodes:
            if nd.kind == "sig" and nd.port == "pp":
                if not any(nodes[v].kind in ("FA", "HA") for v in succ[nd.idx]):
                    w = nd.weight
                    if w < W and rowA[w] == 0:
                        rowA[w] = sig_expr[nd.idx]
                    elif w < W and rowB[w] == 0:
                        rowB[w] = sig_expr[nd.idx]

    # Ripple-carry adder for the two rows -> product bits
    carry = 0
    bits = []
    for w in range(W):
        a_bit = rowA[w] if w < len(rowA) else 0
        b_bit = rowB[w] if w < len(rowB) else 0

        #s_bit = (a_bit ^ b_bit) ^ carry
        #c_out = (a_bit & b_bit) | (a_bit & carry) | (b_bit & carry)
        
        # optimized version:
        # if b_bit is integer and zero
        if isinstance(b_bit, int) and b_bit == 0:
            if isinstance(carry, int) and carry == 0:
                s_bit = a_bit
                c_out = 0
            else:
                s_bit = a_bit ^ carry
                c_out = a_bit & carry
        else:
            s1 = a_bit ^ b_bit
            s_bit = s1 ^ carry
            c_out = (a_bit & b_bit) | (s1 & carry)
            
        bits.append(s_bit)
        carry = c_out
    # (product fits in W bits; overflow carry is discarded by design)

    y <<= cat(*bits)
    return m


def gen_compressor_tree_graph_and_spire_module(n_bits: int, policy: str = "dadda", name: Optional[str] = None) -> Tuple[Graph, Netlist]:

    """
    Convenience function to build a compressor-tree multiplier graph and
    corresponding Spire module for an n-bit unsigned multiplier.

    Args:
      name: module name
      n: input bitwidth (n x n multiplier)
      policy: "dadda" or "wallace"

    Returns:
      spire.component.Netlist
    """

    N = n_bits  # change to try other bitwidths; for plotting, keep <= 6
    if policy == "wallace":
        A, nodes = build_wallace_compressor_graph(N)
    elif policy == "random":
        A, nodes = random_compressor_tree(n=N, seed=42, shrink_range=(0.6, 0.95), p_fa=0.65)
    else:
        raise ValueError(f"Unknown policy '{policy}'; must be 'dadda', 'wallace', or 'random'")
    m = build_multiplier_from_compressor_graph(policy if name is None else name, A, nodes)
    return Graph(nodes, A), m

def build_mul_verctor_rand(n: int, num_random: int = 512, seed: int = 0xADDEF) -> List[Vec]:
    M = (1 << n) - 1
    V: List[Vec] = []

        
    def mul(a: int, b: int) -> int:
        total = (a & M) * (b & M)
        y = total & ((1 << (2*n)) - 1)
        V.append((f"mul{a:02d}x{b:02d}", a & M, b & M, y))
    

    rng = random.Random(seed)
    for i in range(num_random):
        a = rng.randrange(1 << n)
        b = rng.randrange(1 << n)
        c = 0 #c = rng.randrange(2)
        mul(a, b)
        
    return V

def run_vectors_local(mod, vectors, *, label="") -> bool:
    sim = Simulator(mod)
    print(f"\n== {label} ==")
    ok = 0
    for name, a, b, y in vectors:
        sim.set("a", a).set("b", b).eval()
        goty = sim.get("y")
        cout_available = True
        pass_fail = "PASS" if goty == y else "FAIL"

        #print(f"{pass_fail:4s}  {name:25s}  a=0x{a:04X}  b=0x{b:04X}  -> y=0x{goty:04X}  (exp 0x{y:04X})")
        if goty == y:
            ok += 1
    print(f"Summary: {ok}/{len(vectors)} passed.\n")
    all_ok = ok == len(vectors)  # True if all passed
    if not all_ok:
        raise ValueError("Some vectors failed!")
    return all_ok


def main():



    def get_size_and_depth(name: str, m: Netlist):
        aag = AigerExporter(m).get_aag()
        aig = conv_aag_into_aig(aag)

        # Clone the AIG network for size comparison
        aig_clone = aig.clone()
        # Optimize the AIG with several optimization algorithms
        n_iter_optimizations = 10
        for i in range(n_iter_optimizations):
            for optimization in [aig_resubstitution, sop_refactoring, aig_cut_rewriting, balancing]:
                optimization(aig)

        print(f"Results for {name}")
        print(f"Original AIG Size:  {aig_clone.size()}")
        print(f"Optimized AIG Size: {aig.size()}")
        print(f"Original AIG Depth: {DepthAig(aig_clone).num_levels()}")
        print(f"Optimized AIG Depth: {DepthAig(aig).num_levels()}")

        return aig.size(), DepthAig(aig).num_levels()

    n_bits = 4

    g, m = gen_compressor_tree_graph_and_spire_module(n_bits, policy="wallace")
    run_vectors_local(m, build_mul_verctor_rand(n_bits), label="8x8 Wallace Multiplier")
    s, d = get_size_and_depth("8x8 Wallace Multiplier", m)

    c_n = get_node_kind_counts(g.nodes)

    # aag_lines = AigerExporter(m).get_aag()
    # stat = extract_yosys_metrics(aag_lines)
    # print(f"Yosys stats: {stat}")

    def get_transistor_count(node_counts: dict, n_bits) -> int:
        # Rough estimates based on typical implementations
        # use yosys notech count
        fa_count = node_counts.get("FA", 0) # 2 XOR, 3 NAND -> 2*12 + 3*4 = 24 + 12 = 36 # or 2 XOR + MUX = 2*12 + 12 = 36
        ha_count = node_counts.get("HA", 0) # 1 XOR, 1 AND -> 12 +  6 = 18
        sig_count = node_counts.get("sig", 0)
        pp_count = node_counts.get("pp", 0) # 1 AND = 6

        # s_bit = (a_bit ^ b_bit) ^ carry
        # c_out = (a_bit & b_bit) | (a_bit & carry) | (b_bit & carry)
        # 2 xor + 3 and + 2 or = 2*12 + 3*6 + 2*6 = 24 + 18 + 12 = 54
        n_ripple = 2/2* n_bits * 54  # rough estimate for final ripple-carry adder

        # Estimate: 1 transistor per signal, 4 per FA, 2 per HA
        return sig_count * 0 + 36 * fa_count + 18 * ha_count + 6 * pp_count + n_ripple

    def get_transistor_count_from_m(m: Netlist) -> int:

        gr = m.module_analyze()
        transistor_count_dict = {'Op2<&>': 6, 'Op2<|>': 6, 'Op2<^>': 12, 'Op1<~>': 2, "Op2<nand>":4,
                                 'Op1<-:>': None, 'Op1<+:>': None}

        total_transistor_count = 0
        for cls_op, count in gr.by_class_incl_typ.items():
            if cls_op in transistor_count_dict and transistor_count_dict[cls_op] is not None:
                total_transistor_count += count * transistor_count_dict[cls_op]
            else:
                raise ValueError(f"Unknown operation type for transistor count estimation: {cls_op}")
        return total_transistor_count



    transistor_count = get_transistor_count(c_n, n_bits)
    print(f"Node counts: {c_n}, estimated transistor count: {transistor_count}")
    print(f"Transistor count from module analysis: {get_transistor_count_from_m(m)}")

    # Wallace: https://de.wikipedia.org/wiki/Wallace-Tree-Multiplizierer
    # For 8 bit:
    # 14 HA + 38 FA = 14*14 + 38*28 -->  196 + 1064 = 1260 transistors
    # we get 1740

    g, m = gen_compressor_tree_graph_and_spire_module(n_bits, policy="random")
    run_vectors_local(m, build_mul_verctor_rand(n_bits), label="8x8 Random Compressor Multiplier")
    s, d = get_size_and_depth("8x8 Random Compressor Multiplier", m)

    # Generate and compare n random compressor trees
    import matplotlib.pyplot as plt

    def compare_random_compressor_trees(n_bits, n_realizations=50, shrink_range=(0.6, 0.95), p_fa=0.65):
        """Generate n_realizations random compressor trees and plot size vs depth."""
        sizes = []
        depths = []
        transistor_counts = []

        # Generate test vectors once
        vectors = build_mul_verctor_rand(n_bits, num_random=50)

        for i in tqdm(range(n_realizations), desc=f"Generating {n_bits}x{n_bits} compressor trees"):
            seed = int(time.time()) + i  # Different seed for each tree
            try:
                # Generate random compressor tree
                A, nodes = random_compressor_tree(n=n_bits, seed=seed, shrink_range=shrink_range, p_fa=p_fa)

                # Build the multiplier module
                name = f"random_mul_{i}"
                m = build_multiplier_from_compressor_graph(name, A, nodes)

                # Verify correctness with test vectors
                if run_vectors_local(m, vectors, label=f"Tree {i}"):
                    # Get metrics
                    size, depth = get_size_and_depth(name, m)
                    sizes.append(size)
                    depths.append(depth)
                    #transistor_count = get_transistor_count_from_m(m)
                    transistor_count = get_yosys_transistor_count(m, n_iter_optimizations=2)
                    transistor_counts.append(transistor_count)
                    m_depth = m.module_analyze().max_depth
                else:
                    print(f"Tree {i} failed verification")
            except Exception as e:
                print(f"Error with tree {i}: {str(e)}")

        # Plot results
        if sizes:
            plt.figure(figsize=(10, 6))
            plt.scatter(sizes, depths, alpha=0.7)
            plt.xlabel('AIG Size')
            plt.ylabel('AIG Depth')
            plt.title(f'Size vs Depth for {len(sizes)} Random {n_bits}x{n_bits} Compressor Trees')
            plt.grid(True)

            # Add trendline
            # if len(sizes) > 1:
            #     z = np.polyfit(sizes, depths, 1)
            #     p = np.poly1d(z)
            #     plt.plot(sizes, p(sizes), "r--", alpha=0.8,
            #             label=f"Trend: y={z[0]:.4f}x+{z[1]:.4f}")
            #     plt.legend()

            plt.savefig(f"random_compressor_tree_{n_bits}x{n_bits}_n{len(sizes)}.png")
            # plt.show()

            # Print statistics
            print(f"\nStatistics for {len(sizes)} successful realizations:")
            print(f"Size: min={min(sizes)}, max={max(sizes)}, avg={sum(sizes)/len(sizes):.2f}")
            print(f"Depth: min={min(depths)}, max={max(depths)}, avg={sum(depths)/len(depths):.2f}")
        else:
            print("No successful trees to plot.")

        if transistor_counts:
            plt.figure(figsize=(10, 6))
            plt.scatter(transistor_counts, depths, alpha=0.7)
            plt.xlabel('Transistor Count')
            plt.ylabel('AIG Depth')
            plt.title(f'Transistor Count vs Depth for {len(transistor_counts)} Random {n_bits}x{n_bits} Compressor Trees')
            plt.grid(True)
            plt.savefig(f"random_compressor_tree_{n_bits}x{n_bits}__n{len(sizes)}_transistor_count.png")
            # plt.show()
            print(f"\nTransistor Count Statistics for {len(transistor_counts)} successful realizations:")

        return sizes, depths

    # Run the comparison with multiple random compressor trees
    n_realizations = 1000

    print(f"\nGenerating {n_realizations} random compressor trees...")
    sizes, depths = compare_random_compressor_trees(n_bits, n_realizations)


if __name__ == "__main__":
    main()