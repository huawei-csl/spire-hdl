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
from sprouthdl.arithmetic.eval.auto_config import lookup_best_config
from sprouthdl.arithmetic.prefix_adders.adders import StageBasedPrefixAdder, StageBasedSubtractor
from sprouthdl.sprouthdl import Const, Expr, Op2, SInt, Signal, UInt, fit_type


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


def build_multiplier(a: Expr, b: Expr, mult_cfg: MultiplierConfig | ArithmeticAutoConfig) -> Expr:
    if isinstance(mult_cfg, ArithmeticAutoConfig):

        signed = getattr(a.typ, "signed", False) or getattr(b.typ, "signed", False)
        arith_cfg, swap = lookup_best_arithmetic_config(
            "*", a.typ.width, b.typ.width, signed, mult_cfg.objective,
        )
        if swap:
            a, b = b, a
        enc = Encoding.twos_complement if signed else Encoding.unsigned
        mult_cfg = MultiplierConfig(
            multiplier_opt=arith_cfg.multiplier_opt,
            encodings=TwoInputAritEncodings.with_enc(enc),
            ppg_opt=arith_cfg.ppg_opt,
            ppa_opt=arith_cfg.ppa_opt,
            fsa_opt=arith_cfg.fsa_opt,
            optim_type=arith_cfg.optim_type,
        )
    if mult_cfg.use_operator:
        signed_a = (mult_cfg.encodings is not None and is_signed(mult_cfg.encodings.a)) or getattr(a.typ, "signed", False)
        signed_b = (mult_cfg.encodings is not None and is_signed(mult_cfg.encodings.b)) or getattr(b.typ, "signed", False)
        if signed_a:
            a = fit_type(a, SInt(a.typ.width))
        if signed_b:
            b = fit_type(b, SInt(b.typ.width))
        result = a * b
        # Structural multipliers always return UInt; match that interface.
        if result.typ.signed and mult_cfg.encodings is not None:
            result = fit_type(result, UInt(result.typ.width))
        return result

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


def build_adder(a: Expr, b: Expr, adder_cfg: AdderConfig | ArithmeticAutoConfig) -> Expr:
    if isinstance(adder_cfg, ArithmeticAutoConfig):

        signed = getattr(a.typ, "signed", False) or getattr(b.typ, "signed", False)
        arith_cfg, swap = lookup_best_arithmetic_config(
            "+", a.typ.width, b.typ.width, signed, adder_cfg.objective,
        )
        if swap:
            a, b = b, a
        encoding = Encoding.twos_complement if signed else Encoding.unsigned
        adder_cfg = AdderConfig(
            encoding=encoding,
            optim_type=arith_cfg.optim_type,
            fsa_opt=arith_cfg.fsa_opt,
        )
    if adder_cfg.use_operator:
        if is_signed(adder_cfg.encoding) or getattr(a.typ, "signed", False):
            a = fit_type(a, SInt(a.typ.width))
        if is_signed(adder_cfg.encoding) or getattr(b.typ, "signed", False):
            b = fit_type(b, SInt(b.typ.width))
        result = a + b
        # Structural adders always return UInt; match that interface.
        if result.typ.signed and adder_cfg.encoding is not None:
            result = fit_type(result, UInt(result.typ.width))
        return result

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


def build_subtractor(a: Expr, b: Expr, sub_cfg: SubtractorConfig | ArithmeticAutoConfig) -> Expr:
    if isinstance(sub_cfg, ArithmeticAutoConfig):

        signed = getattr(a.typ, "signed", False) or getattr(b.typ, "signed", False)
        arith_cfg, _ = lookup_best_arithmetic_config(
            "-", a.typ.width, b.typ.width, signed, sub_cfg.objective,
        )
        encoding = Encoding.twos_complement if signed else Encoding.unsigned
        sub_cfg = SubtractorConfig(
            encoding=encoding,
            optim_type=arith_cfg.optim_type,
            fsa_opt=arith_cfg.fsa_opt,
        )
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


def adder_tree(values: Sequence[Expr], adder_cfg: AdderConfig | ArithmeticAutoConfig) -> Expr:
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
    """Unified configuration for replacing arithmetic operators with StageBased components.

    The output width of the replacement is inferred from each Op2 node's
    declared type during :func:`replace_arithmetic_ops`, so there is no
    ``full_output_bit`` knob on this struct — use :class:`AdderConfig` /
    :class:`SubtractorConfig` directly if you need to construct a
    truncated adder via :func:`build_adder` / :func:`build_subtractor`.
    """

    encoding: Encoding = Encoding.unsigned
    optim_type: Literal["area", "speed"] = "area"
    fsa_opt: FSAOption = FSAOption.PREFIX_BRENT_KUNG
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


def lookup_best_arithmetic_config(
    op: Literal["+", "-", "*"],
    a_w: int,
    b_w: int,
    signed: bool,
    objective: Literal["area", "delay", "adp"] = "area",
):
    """Return ``(ArithmeticConfig, swap)`` for the empirically best configuration."""
    entry, swap = lookup_best_config(op, a_w, b_w, signed, objective)
    encoding = Encoding.twos_complement if signed else Encoding.unsigned

    if entry is None:
        return ArithmeticConfig(encoding=encoding), False

    optim_type = entry.get("optim_type", "area") or "area"

    if op == "*":
        cfg = ArithmeticConfig(
            encoding=encoding,
            optim_type=optim_type,
            fsa_opt=FSAOption[entry["fsa_opt"]],
            multiplier_opt=MultiplierOption.STAGE_BASED_MULTIPLIER,
            ppg_opt=PPGOption[entry["ppg_opt"]],
            ppa_opt=PPAOption[entry["ppa_opt"]],
        )
    else:
        cfg = ArithmeticConfig(
            encoding=encoding,
            optim_type=optim_type,
            fsa_opt=FSAOption[entry["fsa_opt"]],
        )
    return cfg, swap


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

    # --- Inner product / MAC chain detection (pre-pass) ---
    # Collect chains of + nodes where each has a single-consumer * child.
    # A chain like ((a0*b0 + a1*b1) + a2*b2) + a3*b3 is detected from the
    # root + downward.  We only collect from top-level + nodes (those not
    # themselves consumed as a child of another + in the chain).

    def _unwrap_mul(expr: Expr) -> Op2 | None:
        """If expr is an Op2<*> (possibly wrapped in a Signal wire), return it."""
        if isinstance(expr, Op2) and expr.op == "*":
            return expr
        if isinstance(expr, Signal) and expr._driver is not None:
            drv = expr._driver
            if isinstance(drv, Op2) and drv.op == "*":
                return drv
        return None

    def _collect_mul_chain(node: Op2) -> tuple[list[Op2], Expr] | None:
        """Walk a + chain collecting single-consumer * children.
        Returns (mul_nodes, c_term) or None."""
        muls: list[Op2] = []
        current: Expr = node
        while isinstance(current, Op2) and current.op == "+":
            a, b = current.a, current.b
            mul_a = _unwrap_mul(a)
            mul_b = _unwrap_mul(b)
            if mul_a is not None and ref_count.get(id(a), 0) == 1:
                muls.append(mul_a)
                current = b
            elif mul_b is not None and ref_count.get(id(b), 0) == 1:
                muls.append(mul_b)
                current = a
            else:
                break
        # Check if the remaining term is also a single-consumer *
        mul_c = _unwrap_mul(current)
        if mul_c is not None and ref_count.get(id(current), 0) == 1:
            muls.append(mul_c)
            current = Const(0, UInt(1))  # zero accumulate
        return (muls, current) if len(muls) >= 1 else None

    # Find root + nodes of chains (a + that is NOT a child-* in another chain)
    # We only record the chain structure here; actual marking of consumed nodes
    # happens during replacement when we know the strategy.
    chain_roots: dict[int, tuple[list[Op2], Expr]] = {}  # id(root+) -> (muls, c)
    chain_member_candidates: set[int] = set()  # tentative members (used to prevent nested detection)

    if isinstance(config, ArithmeticAutoConfig):
        for node in order:
            if not isinstance(node, Op2) or node.op != "+":
                continue
            if id(node) in chain_member_candidates:
                continue
            result = _collect_mul_chain(node)
            if result is None or len(result[0]) < 1:
                continue
            muls, c_term = result
            chain_roots[id(node)] = (muls, c_term)
            # Tentatively mark to prevent nested detection
            chain_member_candidates.add(id(node))
            for m in muls:
                chain_member_candidates.add(id(m))
            current_walk: Expr = node
            while isinstance(current_walk, Op2) and current_walk.op == "+":
                chain_member_candidates.add(id(current_walk))
                a, b = current_walk.a, current_walk.b
                ma = _unwrap_mul(a)
                mb = _unwrap_mul(b)
                if ma is not None and id(ma) in chain_member_candidates:
                    chain_member_candidates.add(id(a))
                    current_walk = b
                elif mb is not None and id(mb) in chain_member_candidates:
                    chain_member_candidates.add(id(b))
                    current_walk = a
                else:
                    break

    # Actual consumed nodes — populated during replacement when fusion succeeds
    chain_consumed: set[int] = set()

    # Build replacement map bottom-up
    replacements: dict[int, Expr] = {}

    def get(expr: Expr) -> Expr:
        return replacements.get(id(expr), expr)

    for node in order:
        if not isinstance(node, Op2) or node.op not in ("+", "-", "*", "==", "!="):
            continue

        # Skip nodes consumed by inner product / MAC chains
        if id(node) in chain_consumed:
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
                a_expr = fit_type(a_expr, UInt(max_w))
            if b_w < max_w:
                b_expr = fit_type(b_expr, UInt(max_w))
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
                result = fit_type(result, node.typ)
            replacements[id(node)] = result
            continue

        # Inner product / MAC chain replacement
        if id(node) in chain_roots and isinstance(config, ArithmeticAutoConfig):
            from sprouthdl.arithmetic.eval.auto_config import (
                pick_best_dot_strategy, pick_best_mac_strategy,
            )
            from sprouthdl.cores.matmul_accumulate.matmul_accumulate_core_fused import (
                MultiplierConfig as FusedMultiplierConfig,
                fused_inner_product,
            )
            muls, c_term_orig = chain_roots[id(node)]
            # Resolve expressions through replacement map
            mul_pairs = [(get(m.a), get(m.b)) for m in muls]
            c_resolved = get(c_term_orig)

            n_terms = len(mul_pairs)
            max_mul_w = max(max(ma.typ.width, mb.typ.width) for ma, mb in mul_pairs)
            any_signed = any(
                getattr(ma.typ, "signed", False) or getattr(mb.typ, "signed", False)
                for ma, mb in mul_pairs
            )

            if n_terms >= 2:
                strategy, dot_cfg = pick_best_dot_strategy(
                    n_terms, max_mul_w, any_signed, config.objective,
                )
            else:
                # Single MAC
                strategy, dot_cfg = pick_best_mac_strategy(
                    max_mul_w, c_resolved.typ.width, any_signed, config.objective,
                )

            if strategy in ("dot", "mac") and dot_cfg is not None:
                encoding = Encoding.twos_complement if any_signed else Encoding.unsigned
                fused_cfg = FusedMultiplierConfig(
                    ppg_opt=PPGOption[dot_cfg["ppg_opt"]],
                    ppa_opt=PPAOption[dot_cfg["ppa_opt"]],
                    fsa_opt=FSAOption[dot_cfg["fsa_opt"]],
                    optim_type=dot_cfg.get("optim_type", "area") or "area",
                )
                # Pad all operands to max width (fused_inner_product needs uniform widths)
                vec_a, vec_b = [], []
                for ma, mb in mul_pairs:
                    if ma.typ.width < max_mul_w:
                        ma = fit_type(ma, SInt(max_mul_w) if any_signed else UInt(max_mul_w))
                    if mb.typ.width < max_mul_w:
                        mb = fit_type(mb, SInt(max_mul_w) if any_signed else UInt(max_mul_w))
                    vec_a.append(ma)
                    vec_b.append(mb)

                result = fused_inner_product(vec_a, vec_b, c_resolved, fused_cfg, encoding)

                if result.typ.width != node.typ.width or result.typ.signed != node.typ.signed:
                    result = fit_type(result, node.typ)
                replacements[id(node)] = result
                # Mark all consumed nodes so they don't get replaced individually
                for m in muls:
                    chain_consumed.add(id(m))
                    replacements[id(m)] = result
                # Mark intermediate + nodes and Signal wrappers
                cur = node
                while isinstance(cur, Op2) and cur.op == "+":
                    chain_consumed.add(id(cur))
                    a, b = cur.a, cur.b
                    ma = _unwrap_mul(a)
                    mb = _unwrap_mul(b)
                    if ma is not None and id(ma) in chain_consumed:
                        chain_consumed.add(id(a))
                        cur = b
                    elif mb is not None and id(mb) in chain_consumed:
                        chain_consumed.add(id(b))
                        cur = a
                    else:
                        break
                continue
            # else: fall through to separate replacement below

        # Resolve per-node config when using auto mode
        if isinstance(config, ArithmeticAutoConfig):
            node_cfg, swap = lookup_best_arithmetic_config(
                node.op, a_w, b_w, signed_a or signed_b, config.objective,
            )
            if swap:
                a_expr, b_expr = b_expr, a_expr
                a_w, b_w = b_w, a_w
                signed_a, signed_b = signed_b, signed_a
        else:
            node_cfg = config

        # Structurally infer whether the replacement needs the extra top
        # (carry-out) bit: sprouthdl's Op2<+/-> always produces
        # ``max(a_w, b_w) + 1``, but hand-constructed DAGs may use the
        # truncated width. Reading it from the node type is always right.
        effective_full_bit = node.typ.width > max(a_w, b_w)

        if node.op == "+":
            repl = StageBasedPrefixAdder(
                a_w=a_w, b_w=b_w,
                signed_a=signed_a, signed_b=signed_b,
                optim_type=node_cfg.optim_type,
                fsa_cls=node_cfg.fsa_opt.value,
                full_output_bit=effective_full_bit,
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
                full_output_bit=effective_full_bit,
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
                    eff_a = fit_type(eff_a, SInt(max_w) if signed_a else UInt(max_w))
                    eff_a_w = max_w
                if eff_b_w < max_w:
                    eff_b = fit_type(eff_b, SInt(max_w) if signed_b else UInt(max_w))
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
            result = fit_type(result, node.typ)

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
