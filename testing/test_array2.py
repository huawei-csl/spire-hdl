from typing import Optional

import numpy as np
from spirehdl.composite.array import Array  # your renamed class
from spirehdl.spirehdl import Expr, ExprLike, Const, UInt, Wire, fit_width, reset_shared_cache
from spirehdl.spirehdl_module import Module
from spirehdl.spirehdl_simulator import Simulator


def inner_product(vec_a: Array, vec_b: Array) -> Expr:
    """Compute ⟨a, b⟩ = Σ_k a[k] * b[k] for 1D Arrays."""
    assert len(vec_a) == len(vec_b), "inner_product: length mismatch"
    # Assume elements are Exprs (Wire/Const/etc.)
    acc = vec_a[0] * vec_b[0]
    for i in range(1, len(vec_a)):
        acc = acc + (vec_a[i] * vec_b[i])
    return acc


def matmul(A: Array, B: Array) -> Array:
    """
    Matrix multiplication C = A * B, keeping everything as Arrays.

    A: shape (M, K) as Array[Array]
    B: shape (K, N) as Array[Array]
    Returns:
        C: shape (M, N) as Array[Array]
    """
    M = len(A)
    K = len(A[0])  # number of columns in A
    assert len(B) == K, "matmul: inner dimensions mismatch"
    N = len(B[0])  # number of columns in B

    rows = []
    for i in range(M):
        a_row = A[i, :]  # row i, shape (K,)
        row_out = []
        for j in range(N):
            b_col = B[:, j]  # column j, shape (K,)
            row_out.append(inner_product(a_row, b_col))
        rows.append(Array(row_out))
    return Array(rows)


def matmul_accumulate_comb(C_prev: Array, A: Array, B: Array, acc_w: Optional[int] = None) -> Array:
    """
    Purely combinational:

      C_next[i,j] = C_prev[i,j] + (A * B)[i,j]

    C_prev is not mutated; a new Array is returned.
    """
    M = len(C_prev)
    N = len(C_prev[0])

    # Sanity checks on shapes
    assert len(A) == M
    assert len(B[0]) == N

    rows = []
    for i in range(M):
        a_row = A[i, :]
        row_out = []
        for j in range(N):
            b_col = B[:, j]
            dot = inner_product(a_row, b_col)
            acc = C_prev[i, j] + dot
            if acc_w is not None:
                acc = acc[:acc_w]  # fit to acc_w if specified, take LSBs
            row_out.append(acc)
        rows.append(Array(row_out))
    return Array(rows)


def test_matmul_accumulate_once():
    """
    C_next = C_init + A*B (single combinational step).
    C_init is arbitrary non-zero matrix.
    """
    w = 8
    acc_w = 16  # wider for the sum

    # A 2x2
    A = Array(
        [
            Array([Const(1, UInt(w)), Const(2, UInt(w))]),
            Array([Const(3, UInt(w)), Const(4, UInt(w))]),
        ]
    )

    # B 2x2
    B = Array(
        [
            Array([Const(5, UInt(w)), Const(6, UInt(w))]),
            Array([Const(7, UInt(w)), Const(8, UInt(w))]),
        ]
    )

    # C_init arbitrary (not zero)
    C_init = Array(
        [
            Array([Const(10, UInt(acc_w)), Const(20, UInt(acc_w))]),
            Array([Const(30, UInt(acc_w)), Const(40, UInt(acc_w))]),
        ]
    )

    # A*B once:
    # [[19, 22],
    #  [43, 50]]
    C_ab = matmul(A, B)

    # Our comb accumulate
    C_next = matmul_accumulate_comb(C_init, A, B, acc_w)

    # numpy matmul
    c_numpy = [
        [10 + 19, 20 + 22],
        [30 + 43, 40 + 50],
    ]

    # Check numerically: C_next = C_init + C_ab
    assert eval_expr(C_next[0, 0]) == 10 + eval_expr(C_ab[0, 0])  # 10 + 19
    assert eval_expr(C_next[0, 1]) == 20 + eval_expr(C_ab[0, 1])  # 20 + 22
    assert eval_expr(C_next[1, 0]) == 30 + eval_expr(C_ab[1, 0])  # 30 + 43
    assert eval_expr(C_next[1, 1]) == 40 + eval_expr(C_ab[1, 1])  # 40 + 50


def extract_C_from_bits(bits_expr: Expr, acc_w: int) -> Array:
    """
    Utility: given a flat Expr representing C bits (from c_reg),
    reconstruct C as Array[Array] of wires and return it.
    """
    # Here we assume 2x2 just for the test; generalise as needed.
    C = Array(
        [
            Array([Wire(UInt(acc_w)), Wire(UInt(acc_w))]),
            Array([Wire(UInt(acc_w)), Wire(UInt(acc_w))]),
        ]
    )
    C <<= bits_expr
    return C


def build_matmul_accum_reg_module():
    """
    Build a Module that implements:

        c_next = c_prev + A*B

    with constant A,B and C_init. C is stored in a register as bits,
    and unpacked/packed via Array.to_bits/from_bits.
    """
    m = Module("matmul_accum_reg")

    w = 8
    acc_w = 16

    # Constant A, B (hard-wired in the datapath)
    A = Array(
        [
            Array([Const(1, UInt(w)), Const(2, UInt(w))]),
            Array([Const(3, UInt(w)), Const(4, UInt(w))]),
        ]
    )

    # or use variable widths:
    # A = Array([
    #         Array([1, 2]),
    #         Array([3, 4])
    #     ]
    # )

    B = Array(
        [
            Array([Const(5, UInt(w)), Const(6, UInt(w))]),
            Array([Const(7, UInt(w)), Const(8, UInt(w))]),
        ]
    )

    a_np = np.array([[1, 2], [3, 4]])
    b_np = np.array([[5, 6], [7, 8]])
    ab_np = a_np @ b_np  # for reference

    # Initial C (arbitrary)
    C_init = Array(
        [
            Array([Const(10, UInt(acc_w)), Const(20, UInt(acc_w))]),
            Array([Const(30, UInt(acc_w)), Const(40, UInt(acc_w))]),
        ]
    )

    C_width = C_init.width

    # Register holding C as a flat bit-vector
    c_reg = m.reg(UInt(C_width), "c_reg", init=C_init.to_bits())

    # --- Combinational unpack: C_prev from c_reg bits ---
    # Build an Array of wires with same shape as C_init
    C_prev = Array(
        [
            Array([Wire(UInt(acc_w)), Wire(UInt(acc_w))]),
            Array([Wire(UInt(acc_w)), Wire(UInt(acc_w))]),
        ]
    )
    # Hook them up to the reg output bits
    C_prev <<= c_reg

    # --- Combinational accumulate: C_next = C_prev + A*B ---
    C_next = matmul_accumulate_comb(C_prev, A, B, acc_w)

    # Pack back into bits and drive the reg
    c_reg <<= C_next.to_bits()

    return m, c_reg, C_init, A, B, C_prev, C_next, ab_np


def test_matmul_accumulate_twice_with_reg():
    """
    Two cycles of C_next = C_prev + A*B using a register, with to_bits/from_bits.

    We don't directly do C <<= C + dot; instead:
      - c_reg holds C bits
      - we unpack into C_prev (Array of wires)
      - compute C_next = C_prev + A*B (combinational)
      - drive c_reg with C_next bits
    """
    reset_shared_cache()
    m, c_reg, C_init, A, B, C_prev, C_next, ab_np = build_matmul_accum_reg_module()
    sim = Simulator(m)
    acc_w = 16

    # Helper to evaluate a C matrix from current c_reg value
    def eval_C_matrix(sim: Simulator) -> list[list[int]]:
        C_arr = extract_C_from_bits(c_reg, acc_w)
        return [
            [sim.peek(C_arr[0, 0]), sim.peek(C_arr[0, 1])],
            [sim.peek(C_arr[1, 0]), sim.peek(C_arr[1, 1])],
        ]

    # Cycle 0: before any clock edge, reg outputs C_init
    C0_vals = eval_C_matrix(sim)
    assert C0_vals == [
        [10, 20],
        [30, 40],
    ]

    # Manually compute A*B once (as integers)
    C_ab = matmul(A, B)
    sim.eval()
    AB_vals = [
        [sim.peek(C_ab[0, 0]), sim.peek(C_ab[0, 1])],
        [sim.peek(C_ab[1, 0]), sim.peek(C_ab[1, 1])],
    ]
    # [[19, 22], [43, 50]]
    assert AB_vals == ab_np.tolist()

    # --- First clock: C1 = C0 + A*B ---
    sim.step()

    C1_vals = eval_C_matrix(sim)
    assert C1_vals == [
        [10 + AB_vals[0][0], 20 + AB_vals[0][1]],
        [30 + AB_vals[1][0], 40 + AB_vals[1][1]],
    ]

    # --- Second clock: C2 = C1 + A*B ---
    sim.step()

    C2_vals = eval_C_matrix(sim)
    assert C2_vals == [
        [10 + 2 * AB_vals[0][0], 20 + 2 * AB_vals[0][1]],
        [30 + 2 * AB_vals[1][0], 40 + 2 * AB_vals[1][1]],
    ]


def eval_expr(expr: Expr) -> int:
    """
    Evaluate an Expr numerically by wiring it to a temporary Signal
    and asking the Simulator to compute its bits.
    """
    sig = Wire(expr.typ)
    sig <<= expr
    sim = Simulator(Module("eval_expr"))
    return sim.peek(sig)


def test_inner_product_3d_const():
    # a = [1, 2, 3]
    # b = [4, 5, 6]
    # expected = 1*4 + 2*5 + 3*6 = 32
    width = 8
    a = Array(
        [
            Const(1, UInt(width)),
            Const(2, UInt(width)),
            Const(3, UInt(width)),
        ]
    )
    b = Array(
        [
            Const(4, UInt(width)),
            Const(5, UInt(width)),
            Const(6, UInt(width)),
        ]
    )

    res = inner_product(a, b)
    assert eval_expr(res) == 1 * 4 + 2 * 5 + 3 * 6


def test_matmul_2x2_const():
    # A = [[1, 2],
    #      [3, 4]]
    #
    # B = [[5, 6],
    #      [7, 8]]
    #
    # C = A * B =
    #   [[1*5 + 2*7, 1*6 + 2*8],
    #    [3*5 + 4*7, 3*6 + 4*8]]
    #  = [[19, 22],
    #     [43, 50]]
    w = 8
    A = Array(
        [
            Array([Const(1, UInt(w)), Const(2, UInt(w))]),
            Array([Const(3, UInt(w)), Const(4, UInt(w))]),
        ]
    )
    B = Array(
        [
            Array([Const(5, UInt(w)), Const(6, UInt(w))]),
            Array([Const(7, UInt(w)), Const(8, UInt(w))]),
        ]
    )

    C = matmul(A, B)

    assert eval_expr(C[0, 0]) == 19
    assert eval_expr(C[0, 1]) == 22
    assert eval_expr(C[1, 0]) == 43
    assert eval_expr(C[1, 1]) == 50


if __name__ == "__main__":
    test_inner_product_3d_const()
    test_matmul_2x2_const()
    test_matmul_accumulate_once()
    test_matmul_accumulate_twice_with_reg()
