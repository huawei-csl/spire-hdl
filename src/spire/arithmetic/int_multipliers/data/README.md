# Optimized multiplier assets

Place precomputed AIG and map files for optimized multipliers under `optimized/`.

Expected layout (all files are required):
- `optimized/unsigned_3b/{out_aiger.aig,aiger_map_cleaned.map}`
- `optimized/unsigned_4b/{out_aiger.aig,aiger_map_cleaned.map}`
- `optimized/unsigned_8b/{out_aiger.aig,aiger_map_cleaned.map}`
- `optimized/signed_3b/{out_aiger.aig,aiger_map_cleaned.map}`
- `optimized/signed_4b/{out_aiger.aig,aiger_map_cleaned.map}`
- `optimized/signed_8b/{out_aiger.aig,aiger_map_cleaned.map}`
- `optimized/unsigned_4b_strong/{out_aiger_strong.aig,out_aiger_map_strong.map}` (alternate 4-bit unsigned design)

You can rename your existing artifacts to match these filenames or adjust the loader by passing a custom `f_aag_lines`.
