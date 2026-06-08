# Todo

- remove is_bool flag, probably not necessary, just use length of 1
- add better hierarchy capablities / all in graph.
- Uint(value), optional length bit?
- simulation: get any signal in graph wich is there implicitly, run simulation just on function and after setting starting nodes to a value
- unify get and peek (same thing but one is with sign conversion, the other is not)
- unify log_expression_states and watches in the Simulator.
- in module there is all_exprs and _collect_signals_from_outputs maybe this can be merged, especially by removing the _signal attribute (maybe still retain it as a cache)
- add subnormal support for fp add
- unify run_vectors_local and run_vectors
- unify testvector_generation_fp.py and testing/floating_point/fp_testvectors_general.py and  testvector_generation.py
- rename aggregate types, composite types / the others spirehdl.py should be base type
- probabliy not a nice pattern: type(elem).wire_like(elem), better do -> elem.get_wire_clone()
- in testing/test_matmul_accumulate_core.py, etc use vec.to_list() to generate io dict -> to dataclass/named tuple in a wrapper componnet
- new synthax of control strucutres is _if, _else. maybe change to when and otherwise or elsewhen, so we can drop the underscore.
- document aggregate types
- content-addressed cache for `optimized_fsm` / `optimized_encoding` (same idea as `@abc_optimized` / `@flowy_optimized`) — key on `(state_cls, module-hash, objective, strategy, width)` and store the winning assignment, so re-runs of the encoding search are instant.
- merge inner product and mia and allow for mixed fusion of the ppa, e.g. a+b*c+d, see eg run_arithmetic_eval.py.
