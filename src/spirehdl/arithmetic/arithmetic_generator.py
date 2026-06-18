from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from spirehdl.arithmetic.int_multipliers.eval.multiplier_stage_options_demo_lib import (
    FSAOption,
    MultiplierOption,
    PPAOption,
    PPGOption,
    TwoInputAritEncodings,
    encoding_for_multiplier,
    supports_stages,
)
from spirehdl.arithmetic.int_multipliers.eval.testvector_generation import (
    AdderTestVectors,
    Encoding,
    MultiplierTestVectors,
    is_signed,
)
from spirehdl.arithmetic.int_mac_fused import (
    FusedMacComponent,
    MacBuildConfig,
    MacTestVectors,
    MAC_SUPPORTED_INPUT_ENCODINGS,
    resolve_mac_c_bits,
    resolve_mac_output_encoding,
)
from spirehdl.arithmetic.int_arithmetic_config import (
    AdderConfig as MatmulAdderConfig,
    MultiplierConfig as MatmulMultiplierConfig,
)
from spirehdl.cores.matmul_accumulate.matmul_accumulate_core import (
    MMAcCfg,
    MMAcDims,
    MMAcWidths,
    MatmulAccumulateComponent,
    max_y_width_unsigned,
)
from spirehdl.cores.matmul_accumulate.matmul_accumulate_core_fused import (
    MMAcFusedCfg,
    MatmulAccumulateComponent as MatmulAccumulateFusedComponent,
    MultiplierConfig as MatmulFusedMultiplierConfig,
)
from spirehdl.arithmetic.int_multipliers.multipliers.multiplier_stage_core import OptimType, SelectionMode, SplitMode
from spirehdl.cores.matmul_accumulate.matmul_accumulate_core_float import (
    FpMMAcCfg,
    FpMMAcDims,
    FpMatmulAccumulateComponent,
)
from spirehdl.composite.floating_point import FloatingPointType
from spirehdl.arithmetic.floating_point.spire_hdl_float_mult_sn import FpMulSN
from spirehdl.arithmetic.floating_point.spire_hdl_float_add import FpAdd
from spirehdl.arithmetic.floating_point.fp_mul_testvectors import FpMulTestVectors, build_targeted_mul_vectors
from spirehdl.arithmetic.floating_point.fp_add_testvectors import FpAddTestVectors, build_targeted_add_vectors
from spirehdl.cores.matmul_accumulate.matmul_test_vectors import (
    generate_fp_matmul_vectors,
    generate_matmul_vectors,
)
from spirehdl.arithmetic.int_multipliers.multipliers.multipliers_ext import StageBasedMultiplierBase
from spirehdl.arithmetic.prefix_adders.adders import StageBasedPrefixAdder
from spirehdl.helpers import get_aig_stats, get_yosys_metrics, run_vectors_on_simulator
from spirehdl.spirehdl_verilog_testbench import TestbenchGenSimulator
from spirehdl.spirehdl_aiger import AigerExporter
from spirehdl.spirehdl_module import Module
from spirehdl.spirehdl_simulator import Simulator


# Configs: Generator configuration dataclasses ######################################################################


@dataclass(frozen=True)
class MultiplierGeneratorConfig:
    n_bits: int
    multiplier_opt: MultiplierOption = MultiplierOption.STAGE_BASED_MULTIPLIER
    ppg_opt: PPGOption = PPGOption.AND
    ppa_opt: PPAOption = PPAOption.ACCUMULATOR_TREE
    fsa_opt: FSAOption = FSAOption.RIPPLE_CARRY
    input_encoding: Encoding = Encoding.unsigned
    output_encoding: Encoding | None = None
    optim_type: OptimType = "area"
    selection_mode: SelectionMode | None = None
    split_mode: SplitMode | None = None
    module_name: str | None = None
    with_clock: bool = False
    with_reset: bool = False


@dataclass(frozen=True)
class AdderGeneratorConfig:
    n_bits: int
    fsa_opt: FSAOption = FSAOption.RIPPLE_CARRY
    input_encoding: Encoding = Encoding.unsigned
    output_encoding: Encoding | None = None
    optim_type: OptimType = "area"
    full_output_bit: bool = True
    module_name: str | None = None
    with_clock: bool = False
    with_reset: bool = False


@dataclass(frozen=True)
class MacGeneratorConfig:
    n_bits: int
    c_bits: int | None = None
    use_operator: bool = False
    ppg_opt: PPGOption = PPGOption.AND
    ppa_opt: PPAOption = PPAOption.ACCUMULATOR_TREE
    fsa_opt: FSAOption = FSAOption.RIPPLE_CARRY
    input_encoding: Encoding = Encoding.unsigned
    output_encoding: Encoding | None = None
    optim_type: OptimType = "area"
    selection_mode: SelectionMode | None = None
    split_mode: SplitMode | None = None
    module_name: str | None = None
    with_clock: bool = False
    with_reset: bool = False


@dataclass(frozen=True)
class MatmulAccumulateGeneratorConfig:
    dim_m: int
    dim_n: int
    dim_k: int
    a_width: int
    c_width: int | None = None
    use_operator: bool = False
    multiplier_opt: MultiplierOption = MultiplierOption.STAGE_BASED_MULTIPLIER
    ppg_opt: PPGOption = PPGOption.AND
    ppa_opt: PPAOption = PPAOption.ACCUMULATOR_TREE
    fsa_opt: FSAOption = FSAOption.RIPPLE_CARRY
    input_encoding: Encoding = Encoding.unsigned
    output_encoding: Encoding | None = None
    optim_type: OptimType = "area"
    module_name: str | None = None
    with_clock: bool = False
    with_reset: bool = False


@dataclass(frozen=True)
class FpMatmulAccumulateGeneratorConfig:
    dim_m: int
    dim_n: int
    dim_k: int
    exponent_width: int
    fraction_width: int
    subnormal_support: bool = False
    always_subnormal_rounding: bool = False
    use_operator: bool = False
    multiplier_opt: MultiplierOption = MultiplierOption.STAGE_BASED_MULTIPLIER
    ppg_opt: PPGOption = PPGOption.AND
    ppa_opt: PPAOption = PPAOption.ACCUMULATOR_TREE
    fsa_opt: FSAOption = FSAOption.RIPPLE_CARRY
    optim_type: OptimType = "area"
    module_name: str | None = None
    with_clock: bool = False
    with_reset: bool = False


@dataclass(frozen=True)
class FpMultiplierGeneratorConfig:
    exponent_width: int
    fraction_width: int
    subnormals: bool = True
    always_subnormal_rounding: bool = False
    mult_cfg: MatmulMultiplierConfig | None = None
    module_name: str | None = None
    with_clock: bool = False
    with_reset: bool = False


@dataclass(frozen=True)
class FpAdderGeneratorConfig:
    exponent_width: int
    fraction_width: int
    subnormals: bool = True
    adder_cfg: MatmulAdderConfig | None = None
    module_name: str | None = None
    with_clock: bool = False
    with_reset: bool = False


@dataclass(frozen=True)
class MatmulAccumulateFusedGeneratorConfig:
    dim_m: int
    dim_n: int
    dim_k: int
    a_width: int
    c_width: int | None = None
    ppg_opt: PPGOption = PPGOption.AND
    ppa_opt: PPAOption = PPAOption.ACCUMULATOR_TREE
    fsa_opt: FSAOption = FSAOption.RIPPLE_CARRY
    input_encoding: Encoding = Encoding.unsigned
    optim_type: OptimType = "area"
    module_name: str | None = None
    with_clock: bool = False
    with_reset: bool = False


@dataclass(frozen=True)
class GenerationActions:
    verilog_out: str | Path | None = None
    aag_out: str | Path | None = None
    testbench_out: str | Path | None = None
    data_driven_testbench: bool = False
    simulate: bool = True
    num_vectors: int = 64
    targeted_test_vectors: bool = False
    tb_sigma: float | None = None
    yosys_stats: bool = False
    yosys_deepsyn: bool = False
    yosys_opt_iterations: int | None = None
    yosys_via_aig: bool = False
    aig_stats: bool = False
    aig_opt_iterations: int | None = None
    aig_simple: bool = False


@dataclass
class GenerationResult:
    module: Module
    component: Any
    input_encoding: Encoding | None = None
    output_encoding: Encoding | None = None
    vectors: list[tuple[str, dict[str, int], dict[str, int]]] | None = None
    simulation_failures: int | None = None
    verilog_out: Path | None = None
    aag_out: Path | None = None
    testbench_out: Path | None = None
    testbench_data_out: Path | None = None
    yosys_stats: dict[str, Any] | None = None
    aig_stats: dict[str, Any] | None = None

    @property
    def transistor_count(self) -> int | None:
        if self.yosys_stats is None:
            return None
        return int(self.yosys_stats["estimated_num_transistors"])

    @property
    def aig_num_gates(self) -> int | None:
        if self.aig_stats is None:
            return None
        return int(self.aig_stats["num_gates"])

    @property
    def aig_depth(self) -> int | None:
        if self.aig_stats is None:
            return None
        return int(self.aig_stats["depth"])


# Helpers: Internal utilities and shared logic #######################################################################


def _enum_type(enum_cls):
    def _parse(raw: str):
        for item in enum_cls:
            if raw.upper() == item.name.upper() or raw.lower() == str(item.value).lower():
                return item
        valid = ", ".join(item.name for item in enum_cls)
        raise argparse.ArgumentTypeError(f"Invalid {enum_cls.__name__}: '{raw}'. Valid values: {valid}")

    return _parse


def _resolve_path(path_like: str | Path) -> Path:
    return path_like if isinstance(path_like, Path) else Path(path_like)


def _ensure_parent(path: Path) -> None:
    if path.parent:
        path.parent.mkdir(parents=True, exist_ok=True)


def _resolve_multiplier_encodings(cfg: MultiplierGeneratorConfig) -> TwoInputAritEncodings:
    supported = encoding_for_multiplier(cfg.multiplier_opt.value)
    matching_inputs = [enc for enc in supported if enc.a == cfg.input_encoding and enc.b == cfg.input_encoding]
    if not matching_inputs:
        supported_inputs = sorted({enc.a.name for enc in supported if enc.a == enc.b})
        raise ValueError(
            f"{cfg.multiplier_opt.name} does not support input encoding {cfg.input_encoding.name}. "
            f"Supported (a=b) encodings: {supported_inputs}"
        )

    if cfg.output_encoding is None:
        return matching_inputs[0]

    for enc in matching_inputs:
        if enc.y == cfg.output_encoding:
            return enc

    supported_outputs = sorted({enc.y.name for enc in matching_inputs})
    raise ValueError(
        f"{cfg.multiplier_opt.name} with input encoding {cfg.input_encoding.name} does not support "
        f"output encoding {cfg.output_encoding.name}. Supported outputs: {supported_outputs}"
    )


def _resolve_adder_output_encoding(cfg: AdderGeneratorConfig) -> Encoding:
    if cfg.output_encoding is not None:
        return cfg.output_encoding
    if cfg.full_output_bit:
        return cfg.input_encoding
    return Encoding.twos_complement_overflow if is_signed(cfg.input_encoding) else Encoding.unsigned_overflow


def _validate_clock_reset(with_clock: bool, with_reset: bool) -> None:
    if with_reset and not with_clock:
        raise ValueError("with_reset=True requires with_clock=True")


def _apply_actions(
    module: Module,
    vectors: list[tuple[str, dict[str, int], dict[str, int]]] | None,
    *,
    actions: GenerationActions,
    with_clock: bool,
) -> tuple[int | None, Path | None, Path | None, Path | None, Path | None, dict[str, Any] | None, dict[str, Any] | None]:
    sim_failures = None
    verilog_out = None
    aag_out = None
    testbench_out = None
    testbench_data_out = None
    yosys_stats = None
    aig_stats = None

    if actions.simulate or actions.testbench_out is not None:
        if vectors is None:
            raise ValueError("Simulation/testbench generation was requested but no vectors were generated")
        sim = TestbenchGenSimulator(module) if actions.testbench_out is not None else Simulator(module)
        sim_failures = run_vectors_on_simulator(
            sim,
            vectors,
            use_signed=False,
            raise_on_fail=True,
            print_on_pass=False,
            with_clk=with_clock,
        )
        if actions.testbench_out is not None:
            testbench_out = _resolve_path(actions.testbench_out)
            _ensure_parent(testbench_out)
            if actions.data_driven_testbench:
                data_path = testbench_out.with_suffix(".dat")
                sim.to_data_driver_testbench_file_incl_dat(
                    str(testbench_out), vectors, str(data_path), dump_vcd=False,
                )
                testbench_data_out = data_path
            else:
                sim.to_testbench_file(str(testbench_out), dump_vcd=False)

    if actions.verilog_out is not None:
        verilog_out = _resolve_path(actions.verilog_out)
        _ensure_parent(verilog_out)
        module.to_verilog_file(str(verilog_out))

    if actions.aag_out is not None:
        aag_out = _resolve_path(actions.aag_out)
        _ensure_parent(aag_out)
        AigerExporter(module).write_aag(str(aag_out))

    if actions.yosys_stats:
        yosys_stats = get_yosys_metrics(
            module,
            n_iter_optimizations=actions.yosys_opt_iterations,
            deepsyn=actions.yosys_deepsyn,
            via_aig=actions.yosys_via_aig,
        )

    if actions.aig_stats:
        aig_stats = get_aig_stats(
            module,
            n_iter_optimizations=actions.aig_opt_iterations,
            simple=actions.aig_simple,
        )

    return sim_failures, verilog_out, aag_out, testbench_out, testbench_data_out, yosys_stats, aig_stats


def _finalize(
    module: Module,
    component: Any,
    vectors: list[tuple[str, dict[str, int], dict[str, int]]] | None,
    actions: GenerationActions,
    with_clock: bool,
    input_encoding: Encoding | None = None,
    output_encoding: Encoding | None = None,
) -> GenerationResult:
    sim_failures, verilog_out, aag_out, testbench_out, testbench_data_out, yosys_stats, aig_stats = _apply_actions(
        module, vectors, actions=actions, with_clock=with_clock,
    )
    return GenerationResult(
        module=module,
        component=component,
        input_encoding=input_encoding,
        output_encoding=output_encoding,
        vectors=vectors,
        simulation_failures=sim_failures,
        verilog_out=verilog_out,
        aag_out=aag_out,
        testbench_out=testbench_out,
        testbench_data_out=testbench_data_out,
        yosys_stats=yosys_stats,
        aig_stats=aig_stats,
    )


# Generators: Public API for building and exporting arithmetic modules ################################################


def generate_multiplier(
    cfg: MultiplierGeneratorConfig,
    actions: GenerationActions | None = None,
) -> GenerationResult:
    _validate_clock_reset(cfg.with_clock, cfg.with_reset)
    if cfg.n_bits <= 0:
        raise ValueError("n_bits must be > 0")
    actions = GenerationActions() if actions is None else actions
    if actions.num_vectors <= 0:
        raise ValueError("num_vectors must be > 0")

    encodings = _resolve_multiplier_encodings(cfg)
    use_stage_options = supports_stages(cfg.multiplier_opt)
    # selection_mode/split_mode only apply to the stage-based PPA/FSA; gate them like ppg/ppa/fsa_cls so non-stage
    # multiplier options never receive an unexpected kwarg.
    stage_mode_kwargs = dict(selection_mode=cfg.selection_mode, split_mode=cfg.split_mode) if use_stage_options else {}
    component = cfg.multiplier_opt.value(
        a_w=cfg.n_bits,
        b_w=cfg.n_bits,
        a_encoding=encodings.a,
        b_encoding=encodings.b,
        ppg_cls=cfg.ppg_opt.value if use_stage_options else None,
        ppa_cls=cfg.ppa_opt.value if use_stage_options else None,
        fsa_cls=cfg.fsa_opt.value if use_stage_options else None,
        optim_type=cfg.optim_type,
        **stage_mode_kwargs,
    )
    if not isinstance(component, StageBasedMultiplierBase):
        raise TypeError(f"Expected StageBasedMultiplierBase, got {type(component)}")

    module_name = cfg.module_name or f"mul_{cfg.n_bits}_{cfg.multiplier_opt.name.lower()}"
    module = component.to_module(module_name, with_clock=cfg.with_clock, with_reset=cfg.with_reset)

    vectors = MultiplierTestVectors(
        a_w=cfg.n_bits,
        b_w=cfg.n_bits,
        y_w=component.io.y.typ.width,
        num_vectors=actions.num_vectors,
        tb_sigma=actions.tb_sigma,
        a_encoding=encodings.a,
        b_encoding=encodings.b,
        y_encoding=encodings.y,
    ).generate()

    return _finalize(module, component, vectors, actions, cfg.with_clock,
                     input_encoding=encodings.a, output_encoding=encodings.y)


def generate_adder(
    cfg: AdderGeneratorConfig,
    actions: GenerationActions | None = None,
) -> GenerationResult:
    _validate_clock_reset(cfg.with_clock, cfg.with_reset)
    if cfg.n_bits <= 0:
        raise ValueError("n_bits must be > 0")
    if cfg.input_encoding not in {Encoding.unsigned, Encoding.twos_complement}:
        raise ValueError(
            f"Adder input encoding {cfg.input_encoding.name} is not supported. "
            "Use Encoding.unsigned or Encoding.twos_complement."
        )
    actions = GenerationActions() if actions is None else actions
    if actions.num_vectors <= 0:
        raise ValueError("num_vectors must be > 0")

    adder_output_encoding = _resolve_adder_output_encoding(cfg)
    signed = is_signed(cfg.input_encoding)

    component = StageBasedPrefixAdder(
        a_w=cfg.n_bits,
        b_w=cfg.n_bits,
        signed_a=signed,
        signed_b=signed,
        optim_type=cfg.optim_type,
        fsa_cls=cfg.fsa_opt.value,
        full_output_bit=cfg.full_output_bit,
    )

    module_name = cfg.module_name or f"add_{cfg.n_bits}_{cfg.fsa_opt.name.lower()}"
    module = component.to_module(module_name, with_clock=cfg.with_clock, with_reset=cfg.with_reset)

    vectors = AdderTestVectors(
        a_w=cfg.n_bits,
        b_w=cfg.n_bits,
        y_w=component.io.y.typ.width,
        num_vectors=actions.num_vectors,
        tb_sigma=actions.tb_sigma,
        a_encoding=cfg.input_encoding,
        b_encoding=cfg.input_encoding,
        y_encoding=adder_output_encoding,
    ).generate()

    return _finalize(module, component, vectors, actions, cfg.with_clock,
                     input_encoding=cfg.input_encoding, output_encoding=adder_output_encoding)


def generate_mac(
    cfg: MacGeneratorConfig,
    actions: GenerationActions | None = None,
) -> GenerationResult:
    _validate_clock_reset(cfg.with_clock, cfg.with_reset)
    if cfg.n_bits <= 0:
        raise ValueError("n_bits must be > 0")
    if cfg.input_encoding not in MAC_SUPPORTED_INPUT_ENCODINGS:
        raise ValueError(
            f"MAC input encoding {cfg.input_encoding.name} is not supported. "
            "Use Encoding.unsigned or Encoding.twos_complement."
        )

    resolved_c_bits = resolve_mac_c_bits(cfg.n_bits, cfg.c_bits)
    mac_output_encoding = resolve_mac_output_encoding(cfg.input_encoding, cfg.output_encoding)
    actions = GenerationActions() if actions is None else actions
    if actions.num_vectors <= 0:
        raise ValueError("num_vectors must be > 0")

    component = FusedMacComponent(
        MacBuildConfig(
            n_bits=cfg.n_bits,
            c_bits=resolved_c_bits,
            use_operator=cfg.use_operator,
            ppg_opt=cfg.ppg_opt,
            ppa_opt=cfg.ppa_opt,
            fsa_opt=cfg.fsa_opt,
            encoding=cfg.input_encoding,
            optim_type=cfg.optim_type,
            selection_mode=cfg.selection_mode,
            split_mode=cfg.split_mode,
        )
    )

    module_name = cfg.module_name or f"mac_{cfg.n_bits}_{resolved_c_bits}_{cfg.ppg_opt.name.lower()}"
    module = component.to_module(module_name, with_clock=cfg.with_clock, with_reset=cfg.with_reset)

    vectors = MacTestVectors(
        a_w=cfg.n_bits,
        b_w=cfg.n_bits,
        c_w=resolved_c_bits,
        y_w=component.io.y.typ.width,
        num_vectors=actions.num_vectors,
        tb_sigma=actions.tb_sigma,
        input_encoding=cfg.input_encoding,
        output_encoding=mac_output_encoding,
    ).generate()

    return _finalize(module, component, vectors, actions, cfg.with_clock,
                     input_encoding=cfg.input_encoding, output_encoding=mac_output_encoding)


def generate_matmul_accumulate(
    cfg: MatmulAccumulateGeneratorConfig,
    actions: GenerationActions | None = None,
) -> GenerationResult:
    _validate_clock_reset(cfg.with_clock, cfg.with_reset)
    if cfg.input_encoding not in MAC_SUPPORTED_INPUT_ENCODINGS:
        raise ValueError(
            f"matmul input encoding {cfg.input_encoding.name} is not supported. "
            "Use Encoding.unsigned or Encoding.twos_complement."
        )
    actions = GenerationActions() if actions is None else actions
    if actions.num_vectors <= 0:
        raise ValueError("num_vectors must be > 0")

    resolved_c_width = (
        cfg.c_width
        if cfg.c_width is not None
        else max_y_width_unsigned(cfg.a_width, cfg.a_width, cfg.dim_k, include_carry_from_add=False)
    )
    output_encoding = resolve_mac_output_encoding(cfg.input_encoding, cfg.output_encoding)

    if cfg.use_operator:
        mult_cfg = MatmulMultiplierConfig(use_operator=True)
        add_cfg = MatmulAdderConfig(use_operator=True, encoding=cfg.input_encoding)
    else:
        encodings = TwoInputAritEncodings.with_enc(cfg.input_encoding)
        mult_cfg = MatmulMultiplierConfig(
            use_operator=False,
            multiplier_opt=cfg.multiplier_opt,
            encodings=encodings,
            ppg_opt=cfg.ppg_opt,
            ppa_opt=cfg.ppa_opt,
            fsa_opt=cfg.fsa_opt,
            optim_type=cfg.optim_type,
        )
        add_cfg = MatmulAdderConfig(
            use_operator=False,
            encoding=cfg.input_encoding,
            optim_type=cfg.optim_type,
            fsa_opt=cfg.fsa_opt,
            full_output_bit=True,
        )

    core_cfg = MMAcCfg(
        dims=MMAcDims(dim_m=cfg.dim_m, dim_n=cfg.dim_n, dim_k=cfg.dim_k),
        widths=MMAcWidths(a_width=cfg.a_width, b_width=cfg.a_width, c_width=resolved_c_width),
        mult_cfg=mult_cfg,
        add_cfg=add_cfg,
    )

    signed_io_type = True if cfg.use_operator else False
    component = MatmulAccumulateComponent(core_cfg, signed_io_type=signed_io_type)
    module_name = cfg.module_name or f"matmul_{cfg.dim_m}x{cfg.dim_n}x{cfg.dim_k}_{cfg.a_width}b"
    module = component.to_module(module_name, with_clock=cfg.with_clock, with_reset=cfg.with_reset)

    vectors = generate_matmul_vectors(
        component, encoding=cfg.input_encoding, num_vectors=actions.num_vectors, sigma=actions.tb_sigma,
    )

    return _finalize(module, component, vectors, actions, cfg.with_clock,
                     input_encoding=cfg.input_encoding, output_encoding=output_encoding)


def generate_matmul_accumulate_fused(
    cfg: MatmulAccumulateFusedGeneratorConfig,
    actions: GenerationActions | None = None,
) -> GenerationResult:
    _validate_clock_reset(cfg.with_clock, cfg.with_reset)
    if cfg.input_encoding not in MAC_SUPPORTED_INPUT_ENCODINGS:
        raise ValueError(
            f"matmul-fused input encoding {cfg.input_encoding.name} is not supported. "
            "Use Encoding.unsigned or Encoding.twos_complement."
        )
    actions = GenerationActions() if actions is None else actions
    if actions.num_vectors <= 0:
        raise ValueError("num_vectors must be > 0")

    resolved_c_width = (
        cfg.c_width
        if cfg.c_width is not None
        else max_y_width_unsigned(cfg.a_width, cfg.a_width, cfg.dim_k, include_carry_from_add=False)
    )
    output_encoding = resolve_mac_output_encoding(cfg.input_encoding, None)

    mult_cfg = MatmulFusedMultiplierConfig(
        ppg_opt=cfg.ppg_opt,
        ppa_opt=cfg.ppa_opt,
        fsa_opt=cfg.fsa_opt,
        optim_type=cfg.optim_type,
    )

    core_cfg = MMAcFusedCfg(
        dims=MMAcDims(dim_m=cfg.dim_m, dim_n=cfg.dim_n, dim_k=cfg.dim_k),
        widths=MMAcWidths(a_width=cfg.a_width, b_width=cfg.a_width, c_width=resolved_c_width),
        mult_cfg=mult_cfg,
        encoding=cfg.input_encoding,
    )

    component = MatmulAccumulateFusedComponent(core_cfg)
    module_name = cfg.module_name or f"matmul_fused_{cfg.dim_m}x{cfg.dim_n}x{cfg.dim_k}_{cfg.a_width}b"
    module = component.to_module(module_name, with_clock=cfg.with_clock, with_reset=cfg.with_reset)

    vectors = generate_matmul_vectors(
        component, encoding=cfg.input_encoding, num_vectors=actions.num_vectors, sigma=actions.tb_sigma,
    )

    return _finalize(module, component, vectors, actions, cfg.with_clock,
                     input_encoding=cfg.input_encoding, output_encoding=output_encoding)


def generate_fp_multiplier(
    cfg: FpMultiplierGeneratorConfig,
    actions: GenerationActions | None = None,
) -> GenerationResult:
    _validate_clock_reset(cfg.with_clock, cfg.with_reset)
    actions = GenerationActions() if actions is None else actions

    if not cfg.subnormals and not cfg.always_subnormal_rounding:
        warnings.warn(
            "subnormals is disabled without always_subnormal_rounding: "
            "products that round up to min_normal will be incorrectly flushed to zero. "
            "Consider enabling always_subnormal_rounding for correct FTZ boundary behaviour.",
            stacklevel=2,
        )

    component = FpMulSN(
        cfg.exponent_width, cfg.fraction_width,
        subnormals=cfg.subnormals,
        always_subnormal_rounding=cfg.always_subnormal_rounding,
        mult_cfg=cfg.mult_cfg,
    )
    module_name = cfg.module_name or f"fp_mul_e{cfg.exponent_width}f{cfg.fraction_width}"
    module = component.to_module(module_name, with_clock=cfg.with_clock, with_reset=cfg.with_reset)

    vectors = None
    if actions.simulate or actions.testbench_out is not None:
        if actions.targeted_test_vectors:
            targeted = build_targeted_mul_vectors(cfg.exponent_width, cfg.fraction_width, subnormals=cfg.subnormals)
            num_random = max(0, actions.num_vectors - len(targeted))
            random_vecs = FpMulTestVectors(
                EW=cfg.exponent_width, FW=cfg.fraction_width,
                num_vectors=num_random,
                subnormals=cfg.subnormals,
                always_subnormal_rounding=cfg.always_subnormal_rounding,
            ).generate() if num_random > 0 else []
            vectors = targeted + random_vecs
        else:
            vectors = FpMulTestVectors(
                EW=cfg.exponent_width, FW=cfg.fraction_width,
                num_vectors=actions.num_vectors,
                subnormals=cfg.subnormals,
                always_subnormal_rounding=cfg.always_subnormal_rounding,
            ).generate()

    return _finalize(module, component, vectors, actions, cfg.with_clock)


def generate_fp_adder(
    cfg: FpAdderGeneratorConfig,
    actions: GenerationActions | None = None,
) -> GenerationResult:
    _validate_clock_reset(cfg.with_clock, cfg.with_reset)
    actions = GenerationActions() if actions is None else actions

    component = FpAdd(
        cfg.exponent_width, cfg.fraction_width,
        adder_cfg=cfg.adder_cfg,
        subnormals=cfg.subnormals,
    )
    module_name = cfg.module_name or f"fp_add_e{cfg.exponent_width}f{cfg.fraction_width}"
    module = component.to_module(module_name, with_clock=cfg.with_clock, with_reset=cfg.with_reset)

    vectors = None
    if actions.simulate or actions.testbench_out is not None:
        if actions.targeted_test_vectors:
            targeted = build_targeted_add_vectors(cfg.exponent_width, cfg.fraction_width, subnormals=cfg.subnormals)
            num_random = max(0, actions.num_vectors - len(targeted))
            random_vecs = FpAddTestVectors(
                EW=cfg.exponent_width, FW=cfg.fraction_width,
                num_vectors=num_random,
                subnormals=cfg.subnormals,
            ).generate() if num_random > 0 else []
            vectors = targeted + random_vecs
        else:
            vectors = FpAddTestVectors(
                EW=cfg.exponent_width, FW=cfg.fraction_width,
                num_vectors=actions.num_vectors,
                subnormals=cfg.subnormals,
            ).generate()

    return _finalize(module, component, vectors, actions, cfg.with_clock)


def generate_fp_matmul_accumulate(
    cfg: FpMatmulAccumulateGeneratorConfig,
    actions: GenerationActions | None = None,
) -> GenerationResult:
    _validate_clock_reset(cfg.with_clock, cfg.with_reset)
    actions = GenerationActions() if actions is None else actions
    if actions.num_vectors <= 0:
        raise ValueError("num_vectors must be > 0")

    if not cfg.subnormal_support and not cfg.always_subnormal_rounding:
        warnings.warn(
            "subnormal_support is disabled without always_subnormal_rounding: "
            "products that round up to min_normal will be incorrectly flushed to zero. "
            "Consider enabling always_subnormal_rounding for correct FTZ boundary behaviour.",
            stacklevel=2,
        )

    ft = FloatingPointType(
        exponent_width=cfg.exponent_width,
        fraction_width=cfg.fraction_width,
        subnormal_support=cfg.subnormal_support,
        always_subnormal_rounding=cfg.always_subnormal_rounding,
    )

    if cfg.use_operator:
        adder_cfg = None
        mult_cfg = None
    else:
        mult_cfg = MatmulMultiplierConfig(
            use_operator=False,
            multiplier_opt=cfg.multiplier_opt,
            encodings=TwoInputAritEncodings.with_enc(Encoding.unsigned),
            ppg_opt=cfg.ppg_opt,
            ppa_opt=cfg.ppa_opt,
            fsa_opt=cfg.fsa_opt,
            optim_type=cfg.optim_type,
        )
        adder_cfg = MatmulAdderConfig(
            use_operator=False,
            encoding=Encoding.unsigned,
            optim_type=cfg.optim_type,
            fsa_opt=cfg.fsa_opt,
            full_output_bit=True,
        )

    core_cfg = FpMMAcCfg(
        dims=FpMMAcDims(dim_m=cfg.dim_m, dim_n=cfg.dim_n, dim_k=cfg.dim_k),
        ftype=ft,
        adder_cfg=adder_cfg,
        mult_cfg=mult_cfg,
    )

    component = FpMatmulAccumulateComponent(core_cfg)
    module_name = cfg.module_name or f"fp_matmul_{cfg.dim_m}x{cfg.dim_n}x{cfg.dim_k}_e{cfg.exponent_width}f{cfg.fraction_width}"
    module = component.to_module(module_name, with_clock=cfg.with_clock, with_reset=cfg.with_reset)

    vectors = None
    if actions.simulate or actions.testbench_out is not None:
        vectors = generate_fp_matmul_vectors(component, actions.num_vectors)

    return _finalize(module, component, vectors, actions, cfg.with_clock)


# CLI: Argument parsing and entry point ##############################################################################


def _add_common_action_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--verilog-out", type=str, default=None, help="Optional path for generated Verilog")
    parser.add_argument("--aag-out", type=str, default=None, help="Optional path for generated .aag")
    parser.add_argument("--testbench-out", type=str, default=None, help="Optional path for generated Verilog testbench")
    parser.add_argument("--data-driven-testbench", action="store_true", help="Generate data-driven testbench with separate .dat file instead of inline vectors")
    parser.add_argument("--skip-simulate", action="store_true", help="Skip vector simulation after generation (simulation runs by default to verify the generated core)")
    parser.add_argument("--num-vectors", type=int, default=64, help="Number of vectors for simulation")
    parser.add_argument("--targeted-test-vectors", action="store_true", help="Prepend targeted edge-case vectors (FP mul/add only)")
    parser.add_argument("--tb-sigma", type=float, default=None, help="Optional sigma for normal-distributed vectors")
    parser.add_argument("--yosys-stats", action="store_true", help="Collect Yosys stats")
    parser.add_argument("--yosys-deepsyn", action="store_true", help="Use Yosys/ABC deepsyn flow")
    parser.add_argument(
        "--yosys-opt-iterations",
        type=int,
        default=None,
        help="AIG optimization iterations before stats (default uses internal project default)",
    )
    parser.add_argument(
        "--yosys-via-aig",
        action="store_true",
        help="Collect Yosys stats via an intermediate AIG export (default: measure on the Verilog directly)",
    )
    parser.add_argument("--aig-stats", action="store_true", help="Collect AIG stats (num gates, depth, size, ...)")
    parser.add_argument(
        "--aig-opt-iterations",
        type=int,
        default=None,
        help="AIG optimization iterations before stats (default uses internal project default)",
    )
    parser.add_argument(
        "--aig-simple",
        action="store_true",
        help="Use the simple AIG optimization pass for stats (default: elaborate pass)",
    )
    parser.add_argument("--json-out", type=str, default=None, help="Optional path to save result JSON")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate integer adders/multipliers/MACs with optional exports/stats")
    sub = parser.add_subparsers(dest="kind", required=True)

    multiplier_parser = sub.add_parser("multiplier", help="Generate multiplier module")
    multiplier_parser.add_argument("--n-bits", type=int, required=True)
    multiplier_parser.add_argument("--module-name", type=str, default=None)
    multiplier_parser.add_argument(
        "--multiplier-opt",
        type=_enum_type(MultiplierOption),
        default=MultiplierOption.STAGE_BASED_MULTIPLIER,
    )
    multiplier_parser.add_argument("--ppg-opt", type=_enum_type(PPGOption), default=PPGOption.AND)
    multiplier_parser.add_argument("--ppa-opt", type=_enum_type(PPAOption), default=PPAOption.ACCUMULATOR_TREE)
    multiplier_parser.add_argument("--fsa-opt", type=_enum_type(FSAOption), default=FSAOption.RIPPLE_CARRY)
    multiplier_parser.add_argument("--encoding", type=_enum_type(Encoding), default=Encoding.unsigned)
    multiplier_parser.add_argument("--output-encoding", type=_enum_type(Encoding), default=None)
    multiplier_parser.add_argument("--optim-type", choices=["area", "speed"], default="area")
    multiplier_parser.add_argument("--with-clock", action="store_true", help="Generate module with clock input")
    multiplier_parser.add_argument("--with-reset", action="store_true", help="Generate module with reset input")
    _add_common_action_args(multiplier_parser)

    adder_parser = sub.add_parser("adder", help="Generate adder module")
    adder_parser.add_argument("--n-bits", type=int, required=True)
    adder_parser.add_argument("--module-name", type=str, default=None)
    adder_parser.add_argument("--fsa-opt", type=_enum_type(FSAOption), default=FSAOption.RIPPLE_CARRY)
    adder_parser.add_argument("--encoding", type=_enum_type(Encoding), default=Encoding.unsigned)
    adder_parser.add_argument("--output-encoding", type=_enum_type(Encoding), default=None)
    adder_parser.add_argument("--optim-type", choices=["area", "speed"], default="area")
    adder_parser.add_argument(
        "--full-output-bit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When false, generate overflow-width output",
    )
    adder_parser.add_argument("--with-clock", action="store_true", help="Generate module with clock input")
    adder_parser.add_argument("--with-reset", action="store_true", help="Generate module with reset input")
    _add_common_action_args(adder_parser)

    mac_parser = sub.add_parser("mac", help="Generate fused multiply-accumulate module (y = a*b + c)")
    mac_parser.add_argument("--n-bits", type=int, required=True)
    mac_parser.add_argument("--c-bits", type=int, default=None)
    mac_parser.add_argument("--module-name", type=str, default=None)
    mac_parser.add_argument(
        "--use-operator",
        action="store_true",
        help="Use * and + operators directly instead of explicit fused stage decomposition",
    )
    mac_parser.add_argument("--ppg-opt", type=_enum_type(PPGOption), default=PPGOption.AND)
    mac_parser.add_argument("--ppa-opt", type=_enum_type(PPAOption), default=PPAOption.ACCUMULATOR_TREE)
    mac_parser.add_argument("--fsa-opt", type=_enum_type(FSAOption), default=FSAOption.RIPPLE_CARRY)
    mac_parser.add_argument("--encoding", type=_enum_type(Encoding), default=Encoding.unsigned)
    mac_parser.add_argument("--output-encoding", type=_enum_type(Encoding), default=None)
    mac_parser.add_argument("--optim-type", choices=["area", "speed"], default="area")
    mac_parser.add_argument("--with-clock", action="store_true", help="Generate module with clock input")
    mac_parser.add_argument("--with-reset", action="store_true", help="Generate module with reset input")
    _add_common_action_args(mac_parser)

    matmul_parser = sub.add_parser("matmulacc", help="Generate matrix multiply-accumulate module (Y = A @ B + C)")
    matmul_parser.add_argument("--dim-m", type=int, required=True, help="Rows of A, C, Y")
    matmul_parser.add_argument("--dim-n", type=int, required=True, help="Columns of B, C, Y")
    matmul_parser.add_argument("--dim-k", type=int, required=True, help="Shared dimension of A and B")
    matmul_parser.add_argument("--a-width", type=int, required=True, help="Bit width of A and B elements")
    matmul_parser.add_argument("--c-width", type=int, default=None, help="Bit width of C elements (auto if omitted)")
    matmul_parser.add_argument("--module-name", type=str, default=None)
    matmul_parser.add_argument(
        "--use-operator",
        action="store_true",
        help="Use * and + operators directly instead of explicit stage-based multiplier/adder",
    )
    matmul_parser.add_argument(
        "--multiplier-opt",
        type=_enum_type(MultiplierOption),
        default=MultiplierOption.STAGE_BASED_MULTIPLIER,
    )
    matmul_parser.add_argument("--ppg-opt", type=_enum_type(PPGOption), default=PPGOption.AND)
    matmul_parser.add_argument("--ppa-opt", type=_enum_type(PPAOption), default=PPAOption.ACCUMULATOR_TREE)
    matmul_parser.add_argument("--fsa-opt", type=_enum_type(FSAOption), default=FSAOption.RIPPLE_CARRY)
    matmul_parser.add_argument("--encoding", type=_enum_type(Encoding), default=Encoding.unsigned)
    matmul_parser.add_argument("--output-encoding", type=_enum_type(Encoding), default=None)
    matmul_parser.add_argument("--optim-type", choices=["area", "speed"], default="area")
    matmul_parser.add_argument("--with-clock", action="store_true", help="Generate module with clock input")
    matmul_parser.add_argument("--with-reset", action="store_true", help="Generate module with reset input")
    _add_common_action_args(matmul_parser)

    fp_matmul_parser = sub.add_parser("fpmatmulacc", help="Generate floating-point matrix multiply-accumulate module (Y = A @ B + C)")
    fp_matmul_parser.add_argument("--dim-m", type=int, required=True, help="Rows of A, C, Y")
    fp_matmul_parser.add_argument("--dim-n", type=int, required=True, help="Columns of B, C, Y")
    fp_matmul_parser.add_argument("--dim-k", type=int, required=True, help="Shared dimension of A and B")
    fp_matmul_parser.add_argument("--exponent-width", type=int, required=True, help="Exponent bit width (e.g. 5 for float16)")
    fp_matmul_parser.add_argument("--fraction-width", type=int, required=True, help="Fraction bit width (e.g. 10 for float16)")
    fp_matmul_parser.add_argument("--subnormal-support", action="store_true", help="Enable subnormal support in FP multiplier")
    fp_matmul_parser.add_argument("--always-subnormal-rounding", action="store_true", help="Enable correct FTZ boundary rounding (requires subnormal rounding logic without full subnormal output support)")
    fp_matmul_parser.add_argument("--module-name", type=str, default=None)
    fp_matmul_parser.add_argument(
        "--use-operator",
        action="store_true",
        help="Use * and + operators for mantissa arithmetic instead of explicit stage-based configs",
    )
    fp_matmul_parser.add_argument("--multiplier-opt", type=_enum_type(MultiplierOption), default=MultiplierOption.STAGE_BASED_MULTIPLIER)
    fp_matmul_parser.add_argument("--ppg-opt", type=_enum_type(PPGOption), default=PPGOption.AND)
    fp_matmul_parser.add_argument("--ppa-opt", type=_enum_type(PPAOption), default=PPAOption.ACCUMULATOR_TREE)
    fp_matmul_parser.add_argument("--fsa-opt", type=_enum_type(FSAOption), default=FSAOption.RIPPLE_CARRY)
    fp_matmul_parser.add_argument("--optim-type", choices=["area", "speed"], default="area")
    fp_matmul_parser.add_argument("--with-clock", action="store_true")
    fp_matmul_parser.add_argument("--with-reset", action="store_true")
    _add_common_action_args(fp_matmul_parser)

    fp_mul_parser = sub.add_parser("fpmul", help="Generate standalone floating-point multiplier")
    fp_mul_parser.add_argument("--exponent-width", type=int, required=True, help="Exponent bit width (e.g. 5 for float16)")
    fp_mul_parser.add_argument("--fraction-width", type=int, required=True, help="Fraction bit width (e.g. 10 for float16)")
    fp_mul_parser.add_argument("--subnormal-support", action="store_true", help="Enable subnormal support")
    fp_mul_parser.add_argument("--always-subnormal-rounding", action="store_true", help="Enable correct FTZ boundary rounding")
    fp_mul_parser.add_argument("--module-name", type=str, default=None)
    fp_mul_parser.add_argument("--use-operator", action="store_true", help="Use * operator for mantissa multiplication")
    fp_mul_parser.add_argument("--multiplier-opt", type=_enum_type(MultiplierOption), default=MultiplierOption.STAGE_BASED_MULTIPLIER)
    fp_mul_parser.add_argument("--ppg-opt", type=_enum_type(PPGOption), default=PPGOption.AND)
    fp_mul_parser.add_argument("--ppa-opt", type=_enum_type(PPAOption), default=PPAOption.ACCUMULATOR_TREE)
    fp_mul_parser.add_argument("--fsa-opt", type=_enum_type(FSAOption), default=FSAOption.RIPPLE_CARRY)
    fp_mul_parser.add_argument("--optim-type", choices=["area", "speed"], default="area")
    fp_mul_parser.add_argument("--with-clock", action="store_true")
    fp_mul_parser.add_argument("--with-reset", action="store_true")
    _add_common_action_args(fp_mul_parser)

    fp_add_parser = sub.add_parser("fpadd", help="Generate standalone floating-point adder")
    fp_add_parser.add_argument("--exponent-width", type=int, required=True, help="Exponent bit width (e.g. 5 for float16)")
    fp_add_parser.add_argument("--fraction-width", type=int, required=True, help="Fraction bit width (e.g. 10 for float16)")
    fp_add_parser.add_argument("--subnormal-support", action="store_true", help="Enable subnormal support")
    fp_add_parser.add_argument("--module-name", type=str, default=None)
    fp_add_parser.add_argument("--use-operator", action="store_true", help="Use + operator for internal addition")
    fp_add_parser.add_argument("--fsa-opt", type=_enum_type(FSAOption), default=FSAOption.RIPPLE_CARRY)
    fp_add_parser.add_argument("--with-clock", action="store_true")
    fp_add_parser.add_argument("--with-reset", action="store_true")
    _add_common_action_args(fp_add_parser)

    matmul_fused_parser = sub.add_parser(
        "matmulacc-fused", help="Generate fused matrix multiply-accumulate module (Y = A @ B + C)"
    )
    matmul_fused_parser.add_argument("--dim-m", type=int, required=True, help="Rows of A, C, Y")
    matmul_fused_parser.add_argument("--dim-n", type=int, required=True, help="Columns of B, C, Y")
    matmul_fused_parser.add_argument("--dim-k", type=int, required=True, help="Shared dimension of A and B")
    matmul_fused_parser.add_argument("--a-width", type=int, required=True, help="Bit width of A and B elements")
    matmul_fused_parser.add_argument("--c-width", type=int, default=None, help="Bit width of C elements (auto if omitted)")
    matmul_fused_parser.add_argument("--module-name", type=str, default=None)
    matmul_fused_parser.add_argument("--ppg-opt", type=_enum_type(PPGOption), default=PPGOption.AND)
    matmul_fused_parser.add_argument("--ppa-opt", type=_enum_type(PPAOption), default=PPAOption.ACCUMULATOR_TREE)
    matmul_fused_parser.add_argument("--fsa-opt", type=_enum_type(FSAOption), default=FSAOption.RIPPLE_CARRY)
    matmul_fused_parser.add_argument("--encoding", type=_enum_type(Encoding), default=Encoding.unsigned)
    matmul_fused_parser.add_argument("--optim-type", choices=["area", "speed"], default="area")
    matmul_fused_parser.add_argument("--with-clock", action="store_true", help="Generate module with clock input")
    matmul_fused_parser.add_argument("--with-reset", action="store_true", help="Generate module with reset input")
    _add_common_action_args(matmul_fused_parser)

    return parser


def _actions_from_args(args: argparse.Namespace) -> GenerationActions:
    return GenerationActions(
        verilog_out=args.verilog_out,
        aag_out=args.aag_out,
        testbench_out=args.testbench_out,
        data_driven_testbench=args.data_driven_testbench,
        simulate=not args.skip_simulate,
        num_vectors=args.num_vectors,
        targeted_test_vectors=args.targeted_test_vectors,
        tb_sigma=args.tb_sigma,
        yosys_stats=args.yosys_stats,
        yosys_deepsyn=args.yosys_deepsyn,
        yosys_opt_iterations=args.yosys_opt_iterations,
        yosys_via_aig=args.yosys_via_aig,
        aig_stats=args.aig_stats,
        aig_opt_iterations=args.aig_opt_iterations,
        aig_simple=args.aig_simple,
    )


def _result_to_dict(result: GenerationResult) -> dict[str, Any]:
    data: dict[str, Any] = {
        "module_name": result.module.name,
        "input_encoding": result.input_encoding.name if result.input_encoding is not None else None,
        "output_encoding": result.output_encoding.name if result.output_encoding is not None else None,
        "vector_count": len(result.vectors) if result.vectors is not None else 0,
        "simulation_failures": result.simulation_failures,
        "verilog_out": str(result.verilog_out) if result.verilog_out is not None else None,
        "aag_out": str(result.aag_out) if result.aag_out is not None else None,
        "testbench_out": str(result.testbench_out) if result.testbench_out is not None else None,
        "testbench_data_out": str(result.testbench_data_out) if result.testbench_data_out is not None else None,
        "transistor_count": result.transistor_count,
        "aig_num_gates": result.aig_num_gates,
        "aig_depth": result.aig_depth,
    }
    if result.yosys_stats is not None:
        data["yosys_stats"] = result.yosys_stats
    if result.aig_stats is not None:
        data["aig_stats"] = result.aig_stats
    return data


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    actions = _actions_from_args(args)

    if args.kind == "multiplier":
        cfg = MultiplierGeneratorConfig(
            n_bits=args.n_bits,
            multiplier_opt=args.multiplier_opt,
            ppg_opt=args.ppg_opt,
            ppa_opt=args.ppa_opt,
            fsa_opt=args.fsa_opt,
            input_encoding=args.encoding,
            output_encoding=args.output_encoding,
            optim_type=args.optim_type,
            module_name=args.module_name,
            with_clock=args.with_clock,
            with_reset=args.with_reset,
        )
        result = generate_multiplier(cfg, actions=actions)
    elif args.kind == "adder":
        cfg = AdderGeneratorConfig(
            n_bits=args.n_bits,
            fsa_opt=args.fsa_opt,
            input_encoding=args.encoding,
            output_encoding=args.output_encoding,
            optim_type=args.optim_type,
            full_output_bit=args.full_output_bit,
            module_name=args.module_name,
            with_clock=args.with_clock,
            with_reset=args.with_reset,
        )
        result = generate_adder(cfg, actions=actions)
    elif args.kind == "mac":
        cfg = MacGeneratorConfig(
            n_bits=args.n_bits,
            c_bits=args.c_bits,
            use_operator=args.use_operator,
            ppg_opt=args.ppg_opt,
            ppa_opt=args.ppa_opt,
            fsa_opt=args.fsa_opt,
            input_encoding=args.encoding,
            output_encoding=args.output_encoding,
            optim_type=args.optim_type,
            module_name=args.module_name,
            with_clock=args.with_clock,
            with_reset=args.with_reset,
        )
        result = generate_mac(cfg, actions=actions)
    elif args.kind == "matmulacc":
        cfg = MatmulAccumulateGeneratorConfig(
            dim_m=args.dim_m,
            dim_n=args.dim_n,
            dim_k=args.dim_k,
            a_width=args.a_width,
            c_width=args.c_width,
            use_operator=args.use_operator,
            multiplier_opt=args.multiplier_opt,
            ppg_opt=args.ppg_opt,
            ppa_opt=args.ppa_opt,
            fsa_opt=args.fsa_opt,
            input_encoding=args.encoding,
            output_encoding=args.output_encoding,
            optim_type=args.optim_type,
            module_name=args.module_name,
            with_clock=args.with_clock,
            with_reset=args.with_reset,
        )
        result = generate_matmul_accumulate(cfg, actions=actions)
    elif args.kind == "fpmul":
        mult_cfg = None
        if not args.use_operator:
            mult_cfg = MatmulMultiplierConfig(
                use_operator=False,
                multiplier_opt=args.multiplier_opt,
                encodings=TwoInputAritEncodings.with_enc(Encoding.unsigned),
                ppg_opt=args.ppg_opt,
                ppa_opt=args.ppa_opt,
                fsa_opt=args.fsa_opt,
                optim_type=args.optim_type,
            )
        cfg = FpMultiplierGeneratorConfig(
            exponent_width=args.exponent_width,
            fraction_width=args.fraction_width,
            subnormals=args.subnormal_support,
            always_subnormal_rounding=args.always_subnormal_rounding,
            mult_cfg=mult_cfg,
            module_name=args.module_name,
            with_clock=args.with_clock,
            with_reset=args.with_reset,
        )
        result = generate_fp_multiplier(cfg, actions=actions)
    elif args.kind == "fpadd":
        adder_cfg = None
        if not args.use_operator:
            adder_cfg = MatmulAdderConfig(
                use_operator=False,
                encoding=Encoding.unsigned,
                fsa_opt=args.fsa_opt,
            )
        cfg = FpAdderGeneratorConfig(
            exponent_width=args.exponent_width,
            fraction_width=args.fraction_width,
            subnormals=args.subnormal_support,
            adder_cfg=adder_cfg,
            module_name=args.module_name,
            with_clock=args.with_clock,
            with_reset=args.with_reset,
        )
        result = generate_fp_adder(cfg, actions=actions)
    elif args.kind == "fpmatmulacc":
        cfg = FpMatmulAccumulateGeneratorConfig(
            dim_m=args.dim_m,
            dim_n=args.dim_n,
            dim_k=args.dim_k,
            exponent_width=args.exponent_width,
            fraction_width=args.fraction_width,
            subnormal_support=args.subnormal_support,
            always_subnormal_rounding=args.always_subnormal_rounding,
            use_operator=args.use_operator,
            multiplier_opt=args.multiplier_opt,
            ppg_opt=args.ppg_opt,
            ppa_opt=args.ppa_opt,
            fsa_opt=args.fsa_opt,
            optim_type=args.optim_type,
            module_name=args.module_name,
            with_clock=args.with_clock,
            with_reset=args.with_reset,
        )
        result = generate_fp_matmul_accumulate(cfg, actions=actions)
    else:
        cfg = MatmulAccumulateFusedGeneratorConfig(
            dim_m=args.dim_m,
            dim_n=args.dim_n,
            dim_k=args.dim_k,
            a_width=args.a_width,
            c_width=args.c_width,
            ppg_opt=args.ppg_opt,
            ppa_opt=args.ppa_opt,
            fsa_opt=args.fsa_opt,
            input_encoding=args.encoding,
            optim_type=args.optim_type,
            module_name=args.module_name,
            with_clock=args.with_clock,
            with_reset=args.with_reset,
        )
        result = generate_matmul_accumulate_fused(cfg, actions=actions)

    result_json = json.dumps(_result_to_dict(result), indent=2, sort_keys=True)
    print(result_json)
    if args.json_out is not None:
        with open(args.json_out, "w") as f:
            f.write(result_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
