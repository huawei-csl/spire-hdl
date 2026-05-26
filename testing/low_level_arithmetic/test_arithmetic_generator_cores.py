import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from spirehdl.arithmetic.arithmetic_generator import (
    FpMatmulAccumulateGeneratorConfig,
    GenerationActions,
    MatmulAccumulateGeneratorConfig,
    MatmulAccumulateFusedGeneratorConfig,
    generate_fp_matmul_accumulate,
    generate_matmul_accumulate,
    generate_matmul_accumulate_fused,
)
from spirehdl.arithmetic.int_multipliers.eval.multiplier_stage_options_demo_lib import (
    FSAOption,
    PPAOption,
    PPGOption,
)
from spirehdl.arithmetic.int_multipliers.eval.testvector_generation import Encoding


# --- Unsigned matmul-accumulate core ---


def test_generate_matmulacc_unsigned_with_artifacts_and_sim(tmp_path: Path):
    cfg = MatmulAccumulateGeneratorConfig(
        dim_m=2,
        dim_n=2,
        dim_k=2,
        a_width=4,
        ppg_opt=PPGOption.AND,
        ppa_opt=PPAOption.CARRY_SAVE_TREE,
        fsa_opt=FSAOption.RIPPLE_CARRY,
        input_encoding=Encoding.unsigned,
    )
    tb_path = tmp_path / "matmulacc_tb.v"
    actions = GenerationActions(
        verilog_out=tmp_path / "matmulacc.v",
        aag_out=tmp_path / "matmulacc.aag",
        testbench_out=tb_path,
        simulate=True,
        num_vectors=16,
    )

    result = generate_matmul_accumulate(cfg, actions=actions)

    assert result.module.name.startswith("matmul_")
    assert result.verilog_out == tmp_path / "matmulacc.v"
    assert result.aag_out == tmp_path / "matmulacc.aag"
    assert result.verilog_out.exists()
    assert result.aag_out.exists()
    assert result.testbench_out == tb_path
    assert tb_path.exists()
    assert result.simulation_failures == 0
    assert result.input_encoding == Encoding.unsigned
    assert result.output_encoding == Encoding.unsigned

    tb_text = tb_path.read_text()
    assert "$dumpvars" not in tb_text
    assert "$dumpfile" not in tb_text


# --- Signed matmul-accumulate core ---


def test_generate_matmulacc_signed_twos_complement_sim():
    cfg = MatmulAccumulateGeneratorConfig(
        dim_m=2,
        dim_n=2,
        dim_k=2,
        a_width=4,
        ppg_opt=PPGOption.BAUGH_WOOLEY,
        ppa_opt=PPAOption.WALLACE_TREE,
        fsa_opt=FSAOption.RIPPLE_CARRY,
        input_encoding=Encoding.twos_complement,
    )
    actions = GenerationActions(simulate=True, num_vectors=16)

    result = generate_matmul_accumulate(cfg, actions=actions)

    assert result.simulation_failures == 0
    assert result.input_encoding == Encoding.twos_complement
    assert result.output_encoding == Encoding.twos_complement


def test_generate_matmulacc_unsigned_use_operator_sim():
    cfg = MatmulAccumulateGeneratorConfig(
        dim_m=2,
        dim_n=2,
        dim_k=2,
        a_width=4,
        use_operator=True,
        input_encoding=Encoding.unsigned,
    )
    actions = GenerationActions(simulate=True, num_vectors=16)

    result = generate_matmul_accumulate(cfg, actions=actions)

    assert result.simulation_failures == 0
    assert result.input_encoding == Encoding.unsigned
    assert result.output_encoding == Encoding.unsigned


# --- Fused matmul-accumulate core ---


def test_generate_matmulacc_fused_unsigned_sim(tmp_path: Path):
    cfg = MatmulAccumulateFusedGeneratorConfig(
        dim_m=2,
        dim_n=2,
        dim_k=2,
        a_width=4,
        ppg_opt=PPGOption.AND,
        ppa_opt=PPAOption.WALLACE_TREE,
        fsa_opt=FSAOption.RIPPLE_CARRY,
        input_encoding=Encoding.unsigned,
    )
    actions = GenerationActions(
        verilog_out=tmp_path / "matmulacc_fused.v",
        simulate=True,
        num_vectors=16,
    )

    result = generate_matmul_accumulate_fused(cfg, actions=actions)

    assert result.verilog_out == tmp_path / "matmulacc_fused.v"
    assert result.verilog_out.exists()
    assert result.simulation_failures == 0
    assert result.input_encoding == Encoding.unsigned
    assert result.output_encoding == Encoding.unsigned


def test_generate_matmulacc_fused_signed_sim():
    cfg = MatmulAccumulateFusedGeneratorConfig(
        dim_m=2,
        dim_n=2,
        dim_k=2,
        a_width=4,
        ppg_opt=PPGOption.BAUGH_WOOLEY,
        ppa_opt=PPAOption.WALLACE_TREE,
        fsa_opt=FSAOption.RIPPLE_CARRY,
        input_encoding=Encoding.twos_complement,
    )
    actions = GenerationActions(simulate=True, num_vectors=16)

    result = generate_matmul_accumulate_fused(cfg, actions=actions)

    assert result.simulation_failures == 0
    assert result.input_encoding == Encoding.twos_complement
    assert result.output_encoding == Encoding.twos_complement


# --- Floating-point matmul-accumulate core ---


def test_generate_fp_matmulacc_sim(tmp_path: Path):
    cfg = FpMatmulAccumulateGeneratorConfig(
        dim_m=2,
        dim_n=2,
        dim_k=2,
        exponent_width=5,
        fraction_width=10,
        use_operator=True,
    )
    actions = GenerationActions(
        verilog_out=tmp_path / "fp_matmulacc.v",
        simulate=True,
        num_vectors=16,
    )

    result = generate_fp_matmul_accumulate(cfg, actions=actions)

    assert result.module.name.startswith("fp_matmul_")
    assert result.verilog_out == tmp_path / "fp_matmulacc.v"
    assert result.verilog_out.exists()
    assert result.simulation_failures == 0


def test_generate_fp_matmulacc_staged_sim():
    cfg = FpMatmulAccumulateGeneratorConfig(
        dim_m=2,
        dim_n=2,
        dim_k=2,
        exponent_width=5,
        fraction_width=10,
        ppg_opt=PPGOption.AND,
        ppa_opt=PPAOption.WALLACE_TREE,
        fsa_opt=FSAOption.RIPPLE_CARRY,
    )
    actions = GenerationActions(simulate=True, num_vectors=16)

    result = generate_fp_matmul_accumulate(cfg, actions=actions)

    assert result.simulation_failures == 0


# --- CLI smoke tests ---


def test_cli_matmulacc_smoke(tmp_path: Path):
    verilog_path = tmp_path / "cli_matmulacc.v"

    env = os.environ.copy()
    src_path = str(Path.cwd() / "src")
    env["PYTHONPATH"] = src_path if not env.get("PYTHONPATH") else src_path + os.pathsep + env["PYTHONPATH"]

    cmd = [
        sys.executable,
        "-m",
        "spirehdl.arithmetic.arithmetic_generator",
        "matmulacc",
        "--dim-m", "2",
        "--dim-n", "2",
        "--dim-k", "2",
        "--a-width", "4",
        "--ppg-opt", "AND",
        "--ppa-opt", "ACCUMULATOR_TREE",
        "--fsa-opt", "RIPPLE_CARRY",
        "--encoding", "unsigned",
        "--num-vectors", "16",
        "--verilog-out", str(verilog_path),
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
    stdout = proc.stdout.strip()
    json_start = stdout.find("{")
    json_end = stdout.rfind("}")
    payload = json.loads(stdout[json_start : json_end + 1])

    assert payload["simulation_failures"] == 0
    assert payload["input_encoding"] == "unsigned"
    assert payload["output_encoding"] == "unsigned"
    assert payload["verilog_out"] == str(verilog_path)
    assert verilog_path.exists()


def test_cli_matmulacc_fused_smoke(tmp_path: Path):
    verilog_path = tmp_path / "cli_matmulacc_fused.v"

    env = os.environ.copy()
    src_path = str(Path.cwd() / "src")
    env["PYTHONPATH"] = src_path if not env.get("PYTHONPATH") else src_path + os.pathsep + env["PYTHONPATH"]

    cmd = [
        sys.executable,
        "-m",
        "spirehdl.arithmetic.arithmetic_generator",
        "matmulacc-fused",
        "--dim-m", "2",
        "--dim-n", "2",
        "--dim-k", "2",
        "--a-width", "4",
        "--ppg-opt", "BAUGH_WOOLEY",
        "--ppa-opt", "WALLACE_TREE",
        "--fsa-opt", "RIPPLE_CARRY",
        "--encoding", "twos_complement",
        "--num-vectors", "16",
        "--verilog-out", str(verilog_path),
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
    stdout = proc.stdout.strip()
    json_start = stdout.find("{")
    json_end = stdout.rfind("}")
    payload = json.loads(stdout[json_start : json_end + 1])

    assert payload["simulation_failures"] == 0
    assert payload["input_encoding"] == "twos_complement"
    assert payload["output_encoding"] == "twos_complement"
    assert payload["verilog_out"] == str(verilog_path)
    assert verilog_path.exists()


def test_cli_fpmatmulacc_smoke(tmp_path: Path):
    verilog_path = tmp_path / "cli_fp_matmulacc.v"

    env = os.environ.copy()
    src_path = str(Path.cwd() / "src")
    env["PYTHONPATH"] = src_path if not env.get("PYTHONPATH") else src_path + os.pathsep + env["PYTHONPATH"]

    cmd = [
        sys.executable,
        "-m",
        "spirehdl.arithmetic.arithmetic_generator",
        "fpmatmulacc",
        "--dim-m", "2",
        "--dim-n", "2",
        "--dim-k", "2",
        "--exponent-width", "5",
        "--fraction-width", "10",
        "--use-operator",
        "--num-vectors", "16",
        "--verilog-out", str(verilog_path),
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
    stdout = proc.stdout.strip()
    json_start = stdout.find("{")
    json_end = stdout.rfind("}")
    payload = json.loads(stdout[json_start : json_end + 1])

    assert payload["simulation_failures"] == 0
    assert payload["verilog_out"] == str(verilog_path)
    assert verilog_path.exists()


if __name__ == "__main__":
    pytest.main([__file__])
