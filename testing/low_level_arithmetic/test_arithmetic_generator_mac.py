import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from spirehdl.arithmetic.arithmetic_generator import (
    GenerationActions,
    MacGeneratorConfig,
    generate_mac,
)
from spirehdl.arithmetic.int_multipliers.eval.multiplier_stage_options_demo_lib import (
    FSAOption,
    PPAOption,
    PPGOption,
)
from spirehdl.arithmetic.int_multipliers.eval.testvector_generation import Encoding


def test_generate_mac_api_with_artifacts_and_sim(tmp_path: Path):
    tb_path = tmp_path / "mac_tb.v"
    cfg = MacGeneratorConfig(
        n_bits=4,
        c_bits=8,
        ppg_opt=PPGOption.AND,
        ppa_opt=PPAOption.CARRY_SAVE_TREE,
        fsa_opt=FSAOption.RIPPLE_CARRY,
        input_encoding=Encoding.unsigned,
    )
    actions = GenerationActions(
        verilog_out=tmp_path / "mac.v",
        aag_out=tmp_path / "mac.aag",
        testbench_out=tb_path,
        simulate=True,
        num_vectors=24,
    )

    result = generate_mac(cfg, actions=actions)

    assert result.module.name.startswith("mac_")
    assert result.verilog_out == tmp_path / "mac.v"
    assert result.aag_out == tmp_path / "mac.aag"
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


def test_generate_mac_signed_twos_complement_sim():
    cfg = MacGeneratorConfig(
        n_bits=4,
        c_bits=8,
        ppg_opt=PPGOption.BAUGH_WOOLEY,
        ppa_opt=PPAOption.WALLACE_TREE,
        fsa_opt=FSAOption.RIPPLE_CARRY,
        input_encoding=Encoding.twos_complement,
    )
    actions = GenerationActions(simulate=True, num_vectors=20)

    result = generate_mac(cfg, actions=actions)

    assert result.simulation_failures == 0
    assert result.input_encoding == Encoding.twos_complement
    assert result.output_encoding == Encoding.twos_complement


def test_generate_mac_default_c_bits_behavior():
    n_bits = 5
    cfg = MacGeneratorConfig(
        n_bits=n_bits,
        c_bits=None,
        ppg_opt=PPGOption.AND,
        ppa_opt=PPAOption.ACCUMULATOR_TREE,
        fsa_opt=FSAOption.RIPPLE_CARRY,
        input_encoding=Encoding.unsigned,
    )

    result = generate_mac(cfg)

    expected_c_width = 2 * n_bits
    expected_y_width = max(
        expected_c_width,
        (((1 << (2 * n_bits)) - 1) + ((1 << expected_c_width) - 1)).bit_length(),
    )
    assert result.component.io.c.typ.width == expected_c_width
    assert result.component.io.y.typ.width == expected_y_width


def test_cli_mac_smoke(tmp_path: Path):
    verilog_path = tmp_path / "cli_mac.v"
    aag_path = tmp_path / "cli_mac.aag"

    env = os.environ.copy()
    src_path = str(Path.cwd() / "src")
    env["PYTHONPATH"] = src_path if not env.get("PYTHONPATH") else src_path + os.pathsep + env["PYTHONPATH"]

    cmd = [
        sys.executable,
        "-m",
        "spirehdl.arithmetic.arithmetic_generator",
        "mac",
        "--n-bits",
        "4",
        "--c-bits",
        "8",
        "--ppg-opt",
        "AND",
        "--ppa-opt",
        "ACCUMULATOR_TREE",
        "--fsa-opt",
        "RIPPLE_CARRY",
        "--encoding",
        "unsigned",
        "--num-vectors",
        "16",
        "--verilog-out",
        str(verilog_path),
        "--aag-out",
        str(aag_path),
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
    assert payload["aag_out"] == str(aag_path)
    assert verilog_path.exists()
    assert aag_path.exists()


def test_generate_mac_validation_errors():
    with pytest.raises(ValueError, match="n_bits must be > 0"):
        generate_mac(MacGeneratorConfig(n_bits=0))

    with pytest.raises(ValueError, match="c_bits must be > 0"):
        generate_mac(MacGeneratorConfig(n_bits=4, c_bits=0))

    with pytest.raises(ValueError, match="MAC input encoding"):
        generate_mac(MacGeneratorConfig(n_bits=4, input_encoding=Encoding.sign_magnitude))

    with pytest.raises(ValueError, match="MAC output encoding"):
        generate_mac(
            MacGeneratorConfig(
                n_bits=4,
                input_encoding=Encoding.unsigned,
                output_encoding=Encoding.twos_complement,
            )
        )
        
if __name__ == "__main__":
    pytest.main([__file__])
