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
    """Walk the component's expression DAG and replace +/-/* Op2 nodes
    with StageBased component subgraphs.

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

    # Build replacement map bottom-up
    replacements: dict[int, Expr] = {}

    def get(expr: Expr) -> Expr:
        return replacements.get(id(expr), expr)

    for node in order:
        if not isinstance(node, Op2) or node.op not in ("+", "-", "*"):
            continue

        a_expr = get(node.a)
        b_expr = get(node.b)

        a_w = a_expr.typ.width
        b_w = b_expr.typ.width
        signed_a = getattr(a_expr.typ, "signed", False)
        signed_b = getattr(b_expr.typ, "signed", False)

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
