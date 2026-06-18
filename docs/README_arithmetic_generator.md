
## Arithmetic generators

The `spire/arithmetic` package collects reusable datapath blocks:

- Integer multipliers include configurable stage-based designs and optimized AIG-backed implementations ([`int_multipliers/multipliers`](../src/spire/arithmetic/int_multipliers/multipliers)).
- Prefix adders cover several topologies for depth/area exploration ([`prefix_adders`](../src/spire/arithmetic/prefix_adders)).
- Floating-point implementations related utilities ([`floating_point`](../src/spire/arithmetic/floating_point)). *Note: floating point arithmetic might have some rounding errors for some settings of exponent and mantissa*

Each module ships with small vector generators or evaluators so you can integrate them into regression tests quickly.

### Unified adder/multiplier/mac/matmul generator

For integer adders, multipliers, and MACs (`y = a*b + c`), as well as matmul-accumulate there is a unified generator with both Python API and CLI frontend:
[`arithmetic_generator.py`](../src/spire/arithmetic/arithmetic_generator.py).

It can optionally:
- write Verilog
- write AAG
- write a Verilog testbench (`--testbench-out`) generated from vectors via `TestbenchGenSimulator`
- write a data-driven Verilog testbench (`--testbench-out --data-driven-testbench`) that reads stimulus from a separate `.dat` file instead of inlining vectors; the `.dat` path is reported in the result JSON as `testbench_data_out`
- run vector simulation
- collect Yosys metrics (including `estimated_num_transistors`)
- save the result summary JSON to a file (`--json-out`)

Multiplier/Adder Python API usage reference:
[`testing/low_level_arithmetic/test_arithmetic_generator.py`](../testing/low_level_arithmetic/test_arithmetic_generator.py).

Python API example (integer multiplier):

```python
from spire.arithmetic.arithmetic_generator import (
    GenerationActions,
    MultiplierGeneratorConfig,
    generate_multiplier,
)
from spire.arithmetic.int_multipliers.eval.multiplier_stage_options_demo_lib import (
    FSAOption, MultiplierOption, PPAOption, PPGOption,
)
from spire.arithmetic.int_multipliers.eval.testvector_generation import Encoding

cfg = MultiplierGeneratorConfig(
    n_bits=8,
    multiplier_opt=MultiplierOption.STAGE_BASED_MULTIPLIER,
    ppg_opt=PPGOption.BAUGH_WOOLEY,
    ppa_opt=PPAOption.WALLACE_TREE,
    fsa_opt=FSAOption.RIPPLE_CARRY,
    input_encoding=Encoding.twos_complement,
)
actions = GenerationActions(
    verilog_out="out/mul8.v",
    aag_out="out/mul8.aag",
    simulate=True,
    num_vectors=128,
    yosys_stats=True,
)

result = generate_multiplier(cfg, actions=actions)
print(result.simulation_failures)  # 0
print(result.transistor_count)     # estimated transistor count from Yosys
```

MAC Python API usage reference:
[`testing/low_level_arithmetic/test_arithmetic_generator_mac.py`](../testing/low_level_arithmetic/test_arithmetic_generator_mac.py).

CLI examples:

```bash
python -m spire.arithmetic.arithmetic_generator multiplier \
  --n-bits 8 \
  --multiplier-opt STAGE_BASED_MULTIPLIER \
  --ppg-opt BAUGH_WOOLEY \
  --ppa-opt WALLACE_TREE \
  --fsa-opt RIPPLE_CARRY \
  --encoding twos_complement \
  --num-vectors 128 \
  --verilog-out out/mul8.v \
  --aag-out out/mul8.aag \
  --testbench-out out/mul8_tb.v \
  --yosys-stats \
  --json-out out/mul8_result.json

python -m spire.arithmetic.arithmetic_generator adder \
  --n-bits 16 \
  --fsa-opt PREFIX_BRENT_KUNG \
  --encoding twos_complement \
  --num-vectors 128 \
  --verilog-out out/add16.v

python -m spire.arithmetic.arithmetic_generator mac \
  --n-bits 8 \
  --c-bits 16 \
  --ppg-opt BAUGH_WOOLEY \
  --ppa-opt WALLACE_TREE \
  --fsa-opt RIPPLE_CARRY \
  --encoding twos_complement \
  --num-vectors 128 \
  --verilog-out out/mac8.v \
  --aag-out out/mac8.aag \
  --testbench-out out/mac8_tb.v

# Matrix multiply-accumulate (Y = A @ B + C), explicit multiplier and adder stages
python -m spire.arithmetic.arithmetic_generator matmulacc \
  --dim-m 4 --dim-n 4 --dim-k 4 \
  --a-width 8 \
  --ppg-opt BAUGH_WOOLEY \
  --ppa-opt WALLACE_TREE \
  --fsa-opt RIPPLE_CARRY \
  --encoding twos_complement \
  --num-vectors 16 \
  --verilog-out out/matmulacc_4x4x4_8b.v \
  --json-out out/matmulacc_4x4x4_8b.json

# Matrix multiply-accumulate using * and + operators directly (compact Verilog output)
python -m spire.arithmetic.arithmetic_generator matmulacc \
  --dim-m 4 --dim-n 4 --dim-k 4 \
  --a-width 8 \
  --use-operator \
  --encoding twos_complement \
  --num-vectors 16 \
  --verilog-out out/matmulacc_4x4x4_8b.v \
  --json-out out/matmulacc_4x4x4_8b.json

# Fused matrix multiply-accumulate: partial products from all cells merged before final addition
python -m spire.arithmetic.arithmetic_generator matmulacc-fused \
  --dim-m 4 --dim-n 4 --dim-k 4 \
  --a-width 8 \
  --ppg-opt BAUGH_WOOLEY \
  --ppa-opt WALLACE_TREE \
  --fsa-opt RIPPLE_CARRY \
  --encoding twos_complement \
  --num-vectors 16 \
  --verilog-out out/matmulacc_fused_4x4x4_8b.v \
  --json-out out/matmulacc_fused_4x4x4_8b.json

# Data-driven testbench: vectors stored in a separate .dat file
python -m spire.arithmetic.arithmetic_generator multiplier \
  --n-bits 8 \
  --num-vectors 128 \
  --verilog-out out/mul8.v \
  --testbench-out out/mul8_tb.v \
  --data-driven-testbench \
  --json-out out/mul8_result.json

# Floating-point matrix multiply-accumulate (Y = A @ B + C), operator-based mantissa arithmetic
python -m spire.arithmetic.arithmetic_generator fpmatmulacc \
  --dim-m 4 --dim-n 4 --dim-k 4 \
  --exponent-width 5 --fraction-width 10 \
  --use-operator \
  --num-vectors 16 \
  --verilog-out out/fp_matmulacc_4x4x4_f16.v \
  --json-out out/fp_matmulacc_4x4x4_f16.json

# Floating-point matrix multiply-accumulate with explicit stage-based mantissa multiplier and adder
python -m spire.arithmetic.arithmetic_generator fpmatmulacc \
  --dim-m 4 --dim-n 4 --dim-k 4 \
  --exponent-width 5 --fraction-width 10 \
  --ppg-opt AND \
  --ppa-opt WALLACE_TREE \
  --fsa-opt RIPPLE_CARRY \
  --num-vectors 16 \
  --verilog-out out/fp_matmulacc_4x4x4_f16_staged.v \
  --json-out out/fp_matmulacc_4x4x4_f16_staged.json

# Standalone floating-point multiplier (bfloat16) with simulation, data-driven testbench, and Yosys stats
python -m spire.arithmetic.arithmetic_generator fpmul \
  --exponent-width 8 --fraction-width 7 \
  --subnormal-support \
  --num-vectors 128 \
  --verilog-out out/fp_mul_bf16.v \
  --testbench-out out/fp_mul_bf16_tb.v \
  --data-driven-testbench \
  --yosys-stats \
  --json-out out/fp_mul_bf16.json

# Standalone floating-point adder (float16) with simulation, data-driven testbench, and Yosys stats
python -m spire.arithmetic.arithmetic_generator fpadd \
  --exponent-width 5 --fraction-width 10 \
  --subnormal-support \
  --num-vectors 128 \
  --verilog-out out/fp_add_f16.v \
  --testbench-out out/fp_add_f16_tb.v \
  --data-driven-testbench \
  --yosys-stats \
  --json-out out/fp_add_f16.json
```

## Arithmetic Evaluations
### Integers
Run the evaluation script 
```bash
python -m spire.arithmetic.int_multipliers.eval.run_multiplier_stage_options_eval_ext_stat
```
This will generate a parquet file in the folder `data`. Visualization can be done with the plotly app via 
```bash
python -m spire.arithmetic.int_multipliers.eval.plot.plotly_app --file data/data_file.parquet
``` 
or with the script 
```bash
python -m spire.arithmetic.int_multipliers.eval.plot.multiplier_stage_plot --file data/data_file.parquet
```
Replace `data_file.parquet` with the file produced by the evaluate script.

If desired, new multiplier options can be added here: `spire/arithmetic/int_multipliers/eval/multiplier_stage_options_demo_lib.py`.

#### Optimized Multipliers
The multipliers in `spire/arithmetic/int_multipliers/multipliers/multipliers_ext_optimized.py` rely on precomputed AIGs (and map files). By default they load packaged assets from `spire/arithmetic/int_multipliers/data/optimized/` using the following filenames:

- `unsigned_3b|4b|8b`
- `signed_3b|4b|8b`
- `unsigned_4b_strong`

To point at your own artifacts, set `SPIRE_OPT_MULT_DIR=/path/to/optimized` (keep the same subdirectory names/files) or pass a custom `f_aag_lines` callable when constructing the multiplier. Clear errors are raised if neither packaged nor user-supplied assets are found.

## References
- HiFloat8 specification: [1] Luo, Y., Zhang, Z., Wu, R., Liu, H., Jin, Y., Zheng, K., ... & Huang, Z. (2024). Ascend hifloat8 format for deep learning. arXiv preprint arXiv:2409.16626.
- Winograd inner product: [2] S. Winograd. A new algorithm for inner product. IEEE Trans. Comput., C-18: 693–694, 1968.
