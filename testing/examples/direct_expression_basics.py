"""
Simple arithmetic expression example (no intermediate wires).

Shows:
- `y = a + b`
- `+`, `-`, `*`, unary `-`
- `Const(..., Int(...))` constants
- `Const(False, Bool())` and plain `False`
- Recursive expression generation with Horner polynomial form
"""

from spire.expr import Bool, Const, SInt, Signal, mux


def recursive_horner(x, coeffs):
    """
    Build a polynomial recursively in Horner form.
    Example for coeffs [2, -3, 1]: (2*x - 3)*x + 1 = 2*x^2 - 3*x + 1
    """
    if not coeffs:
        raise ValueError("recursive_horner requires at least one coefficient")
    if len(coeffs) == 1:
        return Const(coeffs[0], SInt(8))
    return recursive_horner(x, coeffs[:-1]) * x + Const(coeffs[-1], SInt(8))


if __name__ == "__main__":
    a = Signal(typ=SInt(8), kind="input")
    b = Signal(typ=SInt(8), kind="input")

    # Example 1: smallest direct form: no wire for y, just an expression.
    y_add = a + b
    print("=== Example 1: y_add = a + b ===")
    print(f"verilog: {y_add.to_verilog()}")
    print(f"type   : width={y_add.typ.width}, signed={y_add.typ.signed}")
    # Captured output:
    # === Example 1: y_add = a + b ===
    # verilog: (a + b)
    # type   : width=9, signed=True
    print()

    # Example 2: subtraction.
    y_sub = a - b
    print("=== Example 2: y_sub = a - b ===")
    print(f"verilog: {y_sub.to_verilog()}")
    print(f"type   : width={y_sub.typ.width}, signed={y_sub.typ.signed}")
    # Captured output:
    # === Example 2: y_sub = a - b ===
    # verilog: (a - b)
    # type   : width=9, signed=True
    print()

    # Example 3: constants (including negative constants).
    y_consts = a + b - Const(3, SInt(8)) - Const(-1, SInt(8))
    print("=== Example 3: y_consts = a + b - Const(3, SInt(8)) - Const(-1, SInt(8)) ===")
    print(f"verilog: {y_consts.to_verilog()}")
    print(f"type   : width={y_consts.typ.width}, signed={y_consts.typ.signed}")
    # Captured output:
    # === Example 3: y_consts = a + b - Const(3, SInt(8)) - Const(-1, SInt(8)) ===
    # verilog: (((a + b) - 8'sd3) - -8'sd1)
    # type   : width=11, signed=True
    print()

    # Example 4: unary minus on a constant.
    y_neg_const = -Const(2, SInt(8))
    print("=== Example 4: y_neg_const = -Const(2, SInt(8)) ===")
    print(f"verilog: {y_neg_const.to_verilog()}")
    print(f"type   : width={y_neg_const.typ.width}, signed={y_neg_const.typ.signed}")
    # Captured output:
    # === Example 4: y_neg_const = -Const(2, SInt(8)) ===
    # verilog: (1'd0 - 8'sd2)
    # type   : width=9, signed=True
    print()

    # Example 5: typed False in mux select.
    y_mux_typed_false = mux(Const(False, Bool()), a, b)
    print("=== Example 5: y_mux_typed_false = mux(Const(False, Bool()), a, b) ===")
    print(f"verilog: {y_mux_typed_false.to_verilog()}")
    print(f"type   : width={y_mux_typed_false.typ.width}, signed={y_mux_typed_false.typ.signed}")
    # Captured output:
    # === Example 5: y_mux_typed_false = mux(Const(False, Bool()), a, b) ===
    # verilog: (1'b0 ? a : b)
    # type   : width=8, signed=True
    print()

    # Example 6: plain False in mux select.
    y_mux_plain_false = mux(False, a, b)
    print("=== Example 6: y_mux_plain_false = mux(False, a, b) ===")
    print(f"verilog: {y_mux_plain_false.to_verilog()}")
    print(f"type   : width={y_mux_plain_false.typ.width}, signed={y_mux_plain_false.typ.signed}")
    # Captured output:
    # === Example 6: y_mux_plain_false = mux(False, a, b) ===
    # verilog: (1'b0 ? a : b)
    # type   : width=8, signed=True
    print()

    # Example 7: recursive Horner-form polynomial.
    y_recursive_poly = recursive_horner(a, [2, -3, 1])
    print("=== Example 7: y_recursive_poly = recursive_horner(a, [2, -3, 1]) ===")
    print(f"verilog: {y_recursive_poly.to_verilog()}")
    print(f"type   : width={y_recursive_poly.typ.width}, signed={y_recursive_poly.typ.signed}")
    # Captured output:
    # === Example 7: y_recursive_poly = recursive_horner(a, [2, -3, 1]) ===
    # verilog: ((((8'sd2 * a) + -8'sd3) * a) + 8'sd1)
    # type   : width=26, signed=True

    # Example 8: multiplication.
    y_mul = a * b
    print()
    print("=== Example 8: y_mul = a * b ===")
    print(f"verilog: {y_mul.to_verilog()}")
    print(f"type   : width={y_mul.typ.width}, signed={y_mul.typ.signed}")
    # Captured output:
    # === Example 8: y_mul = a * b ===
    # verilog: (a * b)
    # type   : width=16, signed=True
