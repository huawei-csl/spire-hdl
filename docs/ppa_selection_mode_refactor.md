# PPA Selection Mode Refactoring

## Summary

Replaced the boolean `canonical_bit_selection` flag on `PartialProductAccumulatorBase`
with an explicit `SelectionMode = Literal["fifo", "lifo", "canonical"]` parameter.
The "legacy" terminology is removed; each historical bit-selection strategy now has
a clear name.

## Motivation

The original design toggled between two code paths using a boolean:
- `canonical_bit_selection = False` -- "legacy" (either FIFO or LIFO depending on the PPA)
- `canonical_bit_selection = True` -- "canonical" (earliest-arrival-first)

This was confusing because "legacy" could mean either FIFO (CompressorTree) or LIFO
(all others), and the boolean provided no way to switch between FIFO and LIFO.

A prior attempt (commit 9f833e2) introduced `SelectionMode` with unified helpers but
was reverted (commit e03466a) because it also unified the **loop schedules**, changing
the circuit output. This refactoring adopts the naming and helpers from 9f833e2 while
preserving the separate loop schedules that produce correct hardware.

## The three selection modes

| Mode | Bit-picking rule | Description |
|------|-----------------|-------------|
| `"fifo"` | Left-to-right (`pop(0)`) | Historical CompressorTree default |
| `"lifo"` | Right-to-left (`pop()`) | Historical Wallace/Dadda/CarrySave/FourTwo default |
| `"canonical"` | Earliest-arrival-first by `(level, ord_)` | Matches `PPAEnv._canonical_bits` |

## Per-PPA defaults

| PPA Class | `default_selection_mode` | Notes |
|-----------|-------------------------|-------|
| `WallaceTreeAccumulator` | `"canonical"` | Tie at widths 6-16; canonical wins at width 4 |
| `BalancedDelayWallaceAccumulator` | `"canonical"` | Only supports canonical (priority-queue scheduling) |
| `EagerWallaceAccumulator` | `"canonical"` | Only supports canonical |
| `DaddaTreeAccumulator` | `"canonical"` | 60% depth reduction on average |
| `CarrySaveAccumulator` | `"lifo"` | Canonical loses (+2-3 depth) |
| `FourTwoCompressorAccumulator` | `"canonical"` | Pareto improvement at all widths |
| `FourTwoCompressorParallelAccumulator` | `"lifo"` | Canonical bypasses the parallel gate override |
| `FiveTwoCompressorAccumulator` | `"lifo"` | Only LIFO implemented |
| `CompressorTreeAccumulator` | `"fifo"` | FIFO is 37% shallower than canonical at widths 12-16 |

## API changes

### Before

```python
from spirehdl.arithmetic.int_multipliers.multipliers.multiplier_stage_core import (
    PartialProductAccumulatorBase,
)

class MyPPA(PartialProductAccumulatorBase):
    canonical_bit_selection: ClassVar[bool] = True

    def __init__(self, config, *, canonical_bit_selection=None):
        super().__init__(config, canonical_bit_selection=canonical_bit_selection)
```

### After

```python
from spirehdl.arithmetic.int_multipliers.multipliers.multiplier_stage_core import (
    PartialProductAccumulatorBase,
    SelectionMode,
)

class MyPPA(PartialProductAccumulatorBase):
    default_selection_mode: ClassVar[SelectionMode] = "canonical"

    def __init__(self, config, *, selection_mode=None):
        super().__init__(config, selection_mode=selection_mode)
```

### Overriding the mode through the multiplier builder

The builder classes (`StageBasedMultiplierBase`, `StageBasedMultiplierBasic`, etc.)
don't carry a `selection_mode` parameter — they just call `ppa_cls(config)`.
To override the PPA's default mode, use `functools.partial`:

```python
from functools import partial
from spirehdl.arithmetic.int_multipliers.multipliers.mutipliers_ext import (
    StageBasedMultiplier,
)

mult = StageBasedMultiplier(
    8, 8,
    ppg_cls=SomePPG,
    ppa_cls=partial(DaddaTreeAccumulator, selection_mode="lifo"),
)
```

## Base class helpers

`PartialProductAccumulatorBase` provides unified helpers that all subclasses use:

- `_wrap_columns(columns)` -- convert `Dict[int, List[Expr]]` to `_LeveledBit` columns
- `_unwrap_columns(wrapped)` -- strip metadata back to plain `Expr` columns
- `_take_bits(bits, k)` -- dispatch to the active selection mode's bit-picking rule
- `_take_earliest(bits, k)` -- canonical: by `(level, ord_)`
- `_take_lifo(bits, k)` -- `pop()` from end
- `_take_fifo(bits, k)` -- `pop(0)` from front
- `_apply_fa(col_lower, col_upper, full_adder)` -- full adder using `_take_bits`
- `_apply_ha(col_lower, col_upper)` -- half adder using `_take_bits`
- `_apply_c42(col_lower, col_upper, full_adder, zero)` -- 4:2 compressor using `_take_bits`

## Loop schedules

Different PPAs use different loop structures for FIFO/LIFO vs canonical modes.
This is why the prior unification attempt was reverted -- the schedule affects
the circuit even when the same bits are selected.

| PPA | FIFO/LIFO schedule | Canonical schedule |
|-----|-------------------|-------------------|
| Wallace | next_cols buffer swap | Snapshot-based in-place |
| Dadda | Threshold in-place (same for all modes) | Threshold in-place |
| CarrySave | next_cols buffer swap | Snapshot + cleanup |
| FourTwo | next_cols buffer swap | Snapshot-based |
| CompressorTree | next_cols buffer swap | In-place greedy |

**Dadda is the only PPA with a single `accumulate()` method** for all modes,
because its threshold-based loop structure is identical regardless of bit selection.

## Files modified

- `src/.../multipliers/multiplier_stage_core.py` -- `SelectionMode` type, base class, unified helpers
- `src/.../stages/ppa_stages.py` -- All PPA subclasses
- `src/.../multipliers/mutipliers_ext.py` -- `selection_mode` kwarg forwarding
- `src/.../stages/ppa_fsa_util.py` -- `compressor_sum()` accepts `selection_mode`
- `src/.../multipliers/multipliers_ext_optimized.py` -- Forwards `selection_mode`
- `testing/.../test_ppa_selection_modes.py` -- Regression tests (19 tests)

## Verification

All tests pass:
- Existing `test_stage_multipliers` (functional correctness via simulation) -- PASS
- New `test_ppa_selection_modes`:
  - `test_default_selection_mode` -- 7/7 PASS
  - `test_modes_produce_different_trees` -- 5/5 PASS (confirms mode switch is effective)
  - `test_tree_matches_reference` -- 7/7 PASS (exact Expr tree identity with pre-refactor code)

The Dadda unification (single `accumulate()` for all modes) was verified to produce
identical trees for both canonical and LIFO modes compared to the original separate
`_accumulate_canonical()` and `_accumulate_legacy()` methods.
