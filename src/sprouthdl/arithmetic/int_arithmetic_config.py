from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

from sprouthdl.arithmetic.int_multipliers.eval.multiplier_stage_options_demo_lib import (
    FSAOption,
    MultiplierOption,
    PPAOption,
    PPGOption,
    TwoInputAritEncodings,
)
from sprouthdl.arithmetic.int_multipliers.eval.testvector_generation import Encoding, is_signed
from sprouthdl.arithmetic.prefix_adders.adders import StageBasedPrefixAdder, StageBasedSubtractor
from sprouthdl.sprouthdl import Expr, Op2, SInt, Signal, UInt, cast, reset_shared_cache


@dataclass
class MultiplierConfig:
    """Configuration for choosing between Sprout operator and explicit multiplier."""

    use_operator: bool = False
    multiplier_opt: MultiplierOption | None = None
    encodings: TwoInputAritEncodings | None = None
    ppg_opt: PPGOption | None = None
    ppa_opt: PPAOption | None = None
    fsa_opt: FSAOption | None = None
    optim_type: Literal["area", "speed"] = "area"


@dataclass
class AdderConfig:
    """Configuration for choosing between Sprout operator and explicit adder."""

    use_operator: bool = False
    encoding: Encoding = Encoding.unsigned
    optim_type: Literal["area", "speed"] = "area"
    fsa_opt: FSAOption | None = None
    full_output_bit: bool = True


def build_multiplier(a: Expr, b: Expr, mult_cfg: MultiplierConfig) -> Expr:
    if mult_cfg.use_operator:
        return a * b

    assert mult_cfg.multiplier_opt is not None, "multiplier_opt must be provided for explicit multipliers"
    assert mult_cfg.encodings is not None, "encodings must be provided for explicit multipliers"
    assert mult_cfg.ppg_opt is not None and mult_cfg.ppa_opt is not None and mult_cfg.fsa_opt is not None

    multiplier = mult_cfg.multiplier_opt.value(
        a_w=a.typ.width,
        b_w=b.typ.width,
        a_encoding=mult_cfg.encodings.a,
        b_encoding=mult_cfg.encodings.b,
        ppg_cls=mult_cfg.ppg_opt.value,
        ppa_cls=mult_cfg.ppa_opt.value,
        fsa_cls=mult_cfg.fsa_opt.value,
        optim_type=mult_cfg.optim_type,
    ).make_internal()
    multiplier.io.a <<= a
    multiplier.io.b <<= b
    return multiplier.io.y


def build_adder(a: Expr, b: Expr, adder_cfg: AdderConfig) -> Expr:
    if adder_cfg.use_operator:
        return a + b

    assert adder_cfg.fsa_opt is not None, "fsa_opt must be provided for explicit adders"
    signed = is_signed(adder_cfg.encoding)

    adder = StageBasedPrefixAdder(
        a_w=a.typ.width,
        b_w=b.typ.width,
        signed_a=signed,
        signed_b=signed,
        optim_type=adder_cfg.optim_type,
        fsa_cls=adder_cfg.fsa_opt.value,
        full_output_bit=adder_cfg.full_output_bit,
    ).make_internal()
    adder.io.a <<= a
    adder.io.b <<= b
    return adder.io.y


@dataclass
class SubtractorConfig:
    """Configuration for choosing between Sprout operator and explicit subtractor."""

    use_operator: bool = False
    encoding: Encoding = Encoding.unsigned
    optim_type: Literal["area", "speed"] = "area"
    fsa_opt: FSAOption | None = None
    full_output_bit: bool = True


def build_subtractor(a: Expr, b: Expr, sub_cfg: SubtractorConfig) -> Expr:
    if sub_cfg.use_operator:
        return a - b

    assert sub_cfg.fsa_opt is not None, "fsa_opt must be provided for explicit subtractors"
    signed = is_signed(sub_cfg.encoding)

    sub = StageBasedSubtractor(
        a_w=a.typ.width,
        b_w=b.typ.width,
        signed_a=signed,
        signed_b=signed,
        optim_type=sub_cfg.optim_type,
        fsa_cls=sub_cfg.fsa_opt.value,
        full_output_bit=sub_cfg.full_output_bit,
    ).make_internal()
    sub.io.a <<= a
    sub.io.b <<= b
    return sub.io.y


def adder_tree(values: Sequence[Expr], adder_cfg: AdderConfig) -> Expr:
    if len(values) == 0:
        raise ValueError("Adder tree requires at least one value")
    if len(values) == 1:
        return values[0]

    mid = len(values) // 2
    left = adder_tree(values[:mid], adder_cfg)
    right = adder_tree(values[mid:], adder_cfg)
    return build_adder(left, right, adder_cfg)


# ---------------------------------------------------------------------------
# Graph transformation: replace arithmetic Op2 nodes with StageBased components
# ---------------------------------------------------------------------------

@dataclass
class ArithmeticConfig:
    """Unified configuration for replacing arithmetic operators with StageBased components."""

    encoding: Encoding = Encoding.unsigned
    optim_type: Literal["area", "speed"] = "area"
    fsa_opt: FSAOption = FSAOption.PREFIX_BRENT_KUNG
    full_output_bit: bool = True
    # multiplier-specific
    multiplier_opt: MultiplierOption = MultiplierOption.STAGE_BASED_MULTIPLIER
    ppg_opt: PPGOption = PPGOption.AND
    ppa_opt: PPAOption = PPAOption.CARRY_SAVE_TREE


@dataclass
class ArithmeticAutoConfig:
    """Auto-selects the best per-operation config from the evaluation database.

    Parameters
    ----------
    objective : "area" | "delay" | "adp"
        - ``"area"``:  minimize yosys transistor count
        - ``"delay"``: minimize AIG depth (proxy for critical-path delay)
        - ``"adp"``:   minimize area-delay product (transistor_count * aig_depth)
    """

    objective: Literal["area", "delay", "adp"] = "area"
    full_output_bit: bool = True


def replace_arithmetic_ops(component, config: ArithmeticConfig | ArithmeticAutoConfig) -> None:
    """Walk the component's expression DAG and replace arithmetic and
    comparison Op2 nodes with optimized subgraphs.

    Replaces ``+``, ``-``, ``*`` with StageBased adder/subtractor/multiplier
    components, ``==``/``!=`` with XOR + balanced NOR-tree, and detects
    ``a * b + c`` patterns for fused multiply-accumulate (MAC) replacement
    when using ``ArithmeticAutoConfig``.

    Supports operands with different bit-widths.

    Modifies the expression graph in-place. Call before to_module().
    """
    from sprouthdl.sprouthdl_module import iter_values

    # Prevent stale id() collisions in the global shared-wire cache
    reset_shared_cache()

    # Collect output signals as walk starting points
    outputs = [sig for sig in iter_values(component.io)
               if isinstance(sig, Signal) and sig.kind == "output"]

    # Post-order DFS: children appear before parents in `order`
    visited: set[int] = set()
    order: list[Expr] = []

    def walk(node: Expr | None) -> None:
        if node is None or id(node) in visited:
            return
        visited.add(id(node))
        for attr in ("a", "b", "sel"):
            child = getattr(node, attr, None)
            if isinstance(child, Expr):
                walk(child)
        if hasattr(node, "parts"):
            for part in node.parts:
                walk(part)
        if isinstance(node, Signal) and node._driver is not None:
            walk(node._driver)
        order.append(node)

    for sig in outputs:
        walk(sig)

    # Build reference count map to detect single-consumer * nodes (for MAC fusion)
    ref_count: dict[int, int] = {}
    for node in order:
        for attr in ("a", "b", "sel"):
            child = getattr(node, attr, None)
            if child is not None:
                ref_count[id(child)] = ref_count.get(id(child), 0) + 1
        if hasattr(node, "parts"):
            for part in node.parts:
                ref_count[id(part)] = ref_count.get(id(part), 0) + 1
        if isinstance(node, Signal) and node._driver is not None:
            ref_count[id(node._driver)] = ref_count.get(id(node._driver), 0) + 1

    # Track which * nodes have been consumed by a MAC fusion
    mac_consumed: set[int] = set()

    # Build replacement map bottom-up
    replacements: dict[int, Expr] = {}

    def get(expr: Expr) -> Expr:
        return replacements.get(id(expr), expr)

    for node in order:
        if not isinstance(node, Op2) or node.op not in ("+", "-", "*", "==", "!="):
            continue

        a_expr = get(node.a)
        b_expr = get(node.b)

        a_w = a_expr.typ.width
        b_w = b_expr.typ.width
        signed_a = getattr(a_expr.typ, "signed", False)
        signed_b = getattr(b_expr.typ, "signed", False)

        # Equality: replace with XOR + NOR-tree (no config needed)
        if node.op in ("==", "!="):
            max_w = max(a_w, b_w)
            if a_w < max_w:
                a_expr = cast(a_expr, UInt(max_w))
            if b_w < max_w:
                b_expr = cast(b_expr, UInt(max_w))
            xor_bits = [a_expr[i] ^ b_expr[i] for i in range(max_w)]
            level = xor_bits
            while len(level) > 1:
                next_level = []
                for i in range(0, len(level), 2):
                    if i + 1 < len(level):
                        next_level.append(level[i] | level[i + 1])
                    else:
                        next_level.append(level[i])
                level = next_level
            result = ~level[0] if node.op == "==" else level[0]

            if result.typ.width != node.typ.width or result.typ.signed != node.typ.signed:
                result = cast(result, node.typ)
            replacements[id(node)] = result
            continue

        # MAC detection: Op2<+> where one child is a single-consumer Op2<*>
        if node.op == "+" and isinstance(config, ArithmeticAutoConfig):
            mul_node = None
            c_expr = None
            # Check if a is a * that's only used here
            orig_a = node.a  # use original (pre-replacement) to check structure
            orig_b = node.b
            if isinstance(orig_a, Op2) and orig_a.op == "*" and ref_count.get(id(orig_a), 0) == 1 and id(orig_a) not in mac_consumed:
                mul_node = orig_a
                c_expr = b_expr
            elif isinstance(orig_b, Op2) and orig_b.op == "*" and ref_count.get(id(orig_b), 0) == 1 and id(orig_b) not in mac_consumed:
                mul_node = orig_b
                c_expr = a_expr

            if mul_node is not None:
                from sprouthdl.arithmetic.eval.auto_config import lookup_best_mac_config
                ma = get(mul_node.a)
                mb = get(mul_node.b)
                ma_w = ma.typ.width
                mb_w = mb.typ.width
                mul_signed = getattr(ma.typ, "signed", False) or getattr(mb.typ, "signed", False)
                mac_cfg = lookup_best_mac_config(max(ma_w, mb_w), mul_signed, config.objective)

                if mac_cfg is not None:
                    from sprouthdl.cores.matmul_accumulate.matmul_accumulate_core_fused import (
                        MultiplierConfig as FusedMultiplierConfig,
                        fused_inner_product,
                    )
                    encoding = Encoding.twos_complement if mul_signed else Encoding.unsigned
                    fused_cfg = FusedMultiplierConfig(
                        ppg_opt=PPGOption[mac_cfg["ppg_opt"]],
                        ppa_opt=PPAOption[mac_cfg["ppa_opt"]],
                        fsa_opt=FSAOption[mac_cfg["fsa_opt"]],
                        optim_type=mac_cfg.get("optim_type", "area") or "area",
                    )
                    # Pad to symmetric widths for fused MAC
                    max_w = max(ma_w, mb_w)
                    if ma_w < max_w:
                        ma = cast(ma, SInt(max_w) if mul_signed else UInt(max_w))
                    if mb_w < max_w:
                        mb = cast(mb, SInt(max_w) if mul_signed else UInt(max_w))

                    result = fused_inner_product([ma], [mb], c_expr, fused_cfg, encoding)
                    mac_consumed.add(id(mul_node))
                    replacements[id(mul_node)] = result  # prevent standalone * replacement

                    if result.typ.width != node.typ.width or result.typ.signed != node.typ.signed:
                        result = cast(result, node.typ)
                    replacements[id(node)] = result
                    continue

        # Skip * nodes that were already consumed by MAC fusion
        if node.op == "*" and id(node) in mac_consumed:
            continue

        # Resolve per-node config when using auto mode
        if isinstance(config, ArithmeticAutoConfig):
            from sprouthdl.arithmetic.eval.auto_config import lookup_best_arithmetic_config
            node_cfg, swap = lookup_best_arithmetic_config(
                node.op, a_w, b_w, signed_a or signed_b,
                config.objective, config.full_output_bit,
            )
            if swap:
                a_expr, b_expr = b_expr, a_expr
                a_w, b_w = b_w, a_w
                signed_a, signed_b = signed_b, signed_a
        else:
            node_cfg = config

        if node.op == "+":
            repl = StageBasedPrefixAdder(
                a_w=a_w, b_w=b_w,
                signed_a=signed_a, signed_b=signed_b,
                optim_type=node_cfg.optim_type,
                fsa_cls=node_cfg.fsa_opt.value,
                full_output_bit=node_cfg.full_output_bit,
            ).make_internal()
            repl.io.a <<= a_expr
            repl.io.b <<= b_expr
            result = repl.io.y

        elif node.op == "-":
            repl = StageBasedSubtractor(
                a_w=a_w, b_w=b_w,
                signed_a=signed_a, signed_b=signed_b,
                optim_type=node_cfg.optim_type,
                fsa_cls=node_cfg.fsa_opt.value,
                full_output_bit=node_cfg.full_output_bit,
            ).make_internal()
            repl.io.a <<= a_expr
            repl.io.b <<= b_expr
            result = repl.io.y

        elif node.op == "*":
            enc_a = Encoding.twos_complement if signed_a else Encoding.unsigned
            enc_b = Encoding.twos_complement if signed_b else Encoding.unsigned

            # Karatsuba requires equal widths — pad if needed
            eff_a_w, eff_b_w = a_w, b_w
            eff_a, eff_b = a_expr, b_expr
            if eff_a_w != eff_b_w and node_cfg.multiplier_opt in (
                MultiplierOption.KARATSUBA_MULTIPLIER,
                MultiplierOption.KARATSUBA_MULTIPLIER_FROM_OPTIMIZED_4BIT_BLOCKS,
            ):
                max_w = max(eff_a_w, eff_b_w)
                if eff_a_w < max_w:
                    eff_a = cast(eff_a, SInt(max_w) if signed_a else UInt(max_w))
                    eff_a_w = max_w
                if eff_b_w < max_w:
                    eff_b = cast(eff_b, SInt(max_w) if signed_b else UInt(max_w))
                    eff_b_w = max_w

            repl = node_cfg.multiplier_opt.value(
                a_w=eff_a_w, b_w=eff_b_w,
                a_encoding=enc_a,
                b_encoding=enc_b,
                ppg_cls=node_cfg.ppg_opt.value,
                ppa_cls=node_cfg.ppa_opt.value,
                fsa_cls=node_cfg.fsa_opt.value,
                optim_type=node_cfg.optim_type,
            ).make_internal()
            repl.io.a <<= eff_a
            repl.io.b <<= eff_b
            result = repl.io.y

        # Match the original Op2 type (width/signedness)
        if result.typ.width != node.typ.width or result.typ.signed != node.typ.signed:
            result = cast(result, node.typ)

        replacements[id(node)] = result

    if not replacements:
        return

    # Rewrite all references to replaced nodes
    for node in order:
        if id(node) in replacements:
            continue
        for attr in ("a", "b", "sel", "_driver"):
            old = getattr(node, attr, None)
            if old is not None and id(old) in replacements:
                setattr(node, attr, replacements[id(old)])
        if hasattr(node, "parts"):
            node.parts = [replacements.get(id(p), p) for p in node.parts]
