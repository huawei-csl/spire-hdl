# roundtrip_and_group.py
from collections import OrderedDict
from typing import Dict, Tuple

from spire.aig.aig_aigerverse import _get_aag_sym
from spire.aiger import AigerExporter, AigerImporter
from spire.component import IOCollector

# If these live elsewhere, fix the imports:
# from spire_hdl import UInt, Bool, Module
# from your_aiger_bindings import AigerExporter, AigerImporter
# from your_aag_helpers import conv_aag_into_aig, _get_aag_sym
# from your_collectors import IOCollector


def _as_uint_width(w: int):
    # map Bool → UInt(1), UInt/SInt(w) → UInt(w)
    from spire.expr import UInt

    return UInt(int(w))


def _build_group_spec_from_ports(m) -> "OrderedDict[str, object]":
    """
    Look at the original module 'm' and produce a name→type spec like:
      {"vec": UInt(W), "idx": UInt(IW), "foo": UInt(1), ...}
    """
    spec = OrderedDict()
    for p in m._ports:
        # skip clk/rst if present
        if p.name in ("clk", "rst"):
            continue
        spec[p.name] = _as_uint_width(p.typ.width if not p.typ.is_bool else 1)
    return spec


def roundtrip_and_group(
    m,
    *,
    keep_symbols: bool = True,
    include_scalars: bool = True,
) -> Tuple[object, Dict[str, object]]:
    """
    Export `m` → AAG → import as `m2`, then group bit-level I/Os in `m2`
    to match the widths from the original `m`.

    Returns: (m2, group_spec_used)
    """
    # 1) Export to AAG
    aag = AigerExporter(m).get_aag()  # typically a list[str]

    # 2) (Optional) run through any AIG step you want; no-op here
    #    If you have a conv_aag_into_aig(...) that returns an AIG object you optimize,
    #    do it here and then regenerate AAG if desired.

    # 3) Preserve names via symbol lines if available
    aag_for_import = aag
    if keep_symbols:
        try:
            aag_sym = _get_aag_sym(aag)
            # In your flow you did: aag[:-2] + aag_sym
            # Keep that exact trick to preserve symbols
            aag_for_import = aag[:-2] + aag_sym
        except NameError:
            # _get_aag_sym not available; proceed without it
            pass

    # 4) Import back to SpireHDL
    m2 = AigerImporter(aag_for_import).get_spire_module()

    # 5) Build grouping spec from original ports
    group_spec = _build_group_spec_from_ports(m)

    # Optionally drop scalars; otherwise keep them (works for 'name' or 'name[0]' cases)
    if not include_scalars:
        group_spec = OrderedDict((k, t) for k, t in group_spec.items() if t.width > 1)

    # 6) Group the bit-level I/Os on m2
    IOCollector().group(m2, group_spec)

    return m2, group_spec