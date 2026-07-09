# Todo

- remove is_bool flag, probably not necessary, just use length of 1
- add better hierarchy capablities / all in graph.
- Uint(value), optional length bit?
- simulation: get any signal in graph wich is there implicitly, run simulation just on function and after setting starting nodes to a value
- unify get and peek (same thing but one is with sign conversion, the other is not)
- unify log_expression_states and watches in the Simulator.
- in module there is all_exprs and _collect_signals_from_outputs maybe this can be merged, especially by removing the _signal attribute (maybe still retain it as a cache)
- ~~add subnormal support for fp add~~ (done: FpAdd(subnormals=True) is bit-exact incl. subnormal outputs)
- unify run_vectors_local and run_vectors
- unify testvector_generation_fp.py and testing/floating_point/fp_testvectors_general.py and  testvector_generation.py
- rename aggregate types, composite types / the others expr.py (formerly spirehdl.py) should be base type
- probabliy not a nice pattern: type(elem).wire_like(elem), better do -> elem.get_wire_clone()
- in testing/test_matmul_accumulate_core.py, etc use vec.to_list() to generate io dict -> to dataclass/named tuple in a wrapper componnet
- new synthax of control strucutres is _if, _else. maybe change to when and otherwise or elsewhen, so we can drop the underscore.
- document aggregate types
- content-addressed cache for `optimized_fsm` / `optimized_encoding` (same idea as `@abc_optimized` / `@flowy_optimized`) — key on `(state_cls, module-hash, objective, strategy, width)` and store the winning assignment, so re-runs of the encoding search are instant.
- merge inner product and mia and allow for mixed fusion of the ppa, e.g. a+b*c+d, see eg run_arithmetic_eval.py.
- rename canonical to depth_driven in the ppas (make sure the name is correct), (lifo is the other option)

- Revisit `_to_composite()` dynamic dataclass generation. It is acceptable for Patch 2, but the generated `"DynIO"` classes are opaque in stack traces, IDEs, and debug output. A later cleanup could return `CompositeRecord(**fields)` directly for dict/plain-object inputs, or generate source-specific names/metadata when dynamic classes are still useful.
- Tighten `Component.io` after migration. The temporary `CompositeRecord | Any` annotation should eventually become `CompositeRecord` only, and existing components using dict/dataclass/plain-object IO containers should be rewritten to construct `CompositeRecord` / `IORecord` explicitly.
- Restore old class-template ergonomics in the unified `CompositeRecord`. Patch 2 rewrites fixed-shape records like `class _Bus(CompositeRecord): data = Wire(UInt(8)); valid = Wire(UInt(1))` into explicit `__init__` methods, which is more verbose and loses a useful declaration style. Later, make class-level `Signal` templates clone per instance while preserving `kind` (`wire`, `input`, `output`, `reg`) and field-key names. Files touched by this migration include `testing/memory/test_primitive_fifo.py`, `testing/memory/test_primitive_memory.py`, `testing/memory/test_primitive_rom.py`, `testing/test_composite_record.py`, and `src/spire/primitives/primitive_memory.py`.