# MLX vs Metal cross-checks

Tests whose ids name both paths (`-k "mlx and metal"`), run separately after the benchmarks because the per-path test runs excluded them.

| project | result | passed | failed |
|---|---|---|---|
| 01-point-sphere-gravity | PASS | 3 | 0 |
| 02-prism-gravity | PASS | 24 | 0 |
| 03-prism-magnetic | PASS | 6 | 0 |
| 04-tesseroid-gravity | PASS | 12 | 0 |
| 05-talwani-2d | PASS | 7 | 0 |
| 06-terrain-bouguer | PASS | 3 | 0 |
| 09-equivalent-sources | PASS | 4 | 0 |
| 10-density-inversion | PASS | 3 | 0 |
| 12-acoustic-1d | PASS | 5 | 0 |
| 13-acoustic-2d | PASS | 4 | 0 |
| 14-born-2d | PASS | 7 | 0 |
| 15-elastic-psv | PASS | 14 | 0 |
| 16-fwi-marmousi | PASS | 2 | 0 |
| 17-rtm | none | 0 | 0 |
| 18-lsrtm | PASS | 3 | 0 |
| 21-checkpointing | PASS | 24 | 0 |
| 24-fast-marching | PASS | 32 | 0 |
| 26-ray-tracing | PASS | 14 | 0 |
| 27-traveltime-tomo | PASS | 9 | 0 |
| 28-diff-eikonal | PASS | 3 | 0 |
| 30-eq-location | PASS | 4 | 0 |
| 32-mt-1d | PASS | 7 | 0 |
| 36-gpr-fdtd | PASS | 5 | 0 |
| 39-ambient-noise | PASS | 3 | 0 |
| 40-cmp-processing | PASS | 3 | 0 |
| 41-deconvolution | PASS | 4 | 0 |
| 42-fk-radon | PASS | 4 | 0 |
| 44-okada | PASS | 13 | 0 |
| 47-geotherm | PASS | 4 | 0 |
| 48-mantle-convection | PASS | 20 | 0 |
| 49-tsunami | PASS | 8 | 0 |

## Follow-up run (after the 06/49 re-locks; projects missed by the first sweep)

| project | selector | result | passed | failed |
|---|---|---|---|---|
| 06-terrain-bouguer | `mlx and metal` | PASS | 7 | 0 |
| 34-dc-2d | `mlx and metal` | PASS | 3 | 0 |
| 46-plate-flexure | `mlx and metal` | PASS | 6 | 0 |
| 49-tsunami | `mlx and metal` | PASS | 12 | 0 |
| 17-rtm | `cross_backend` | PASS | 4 | 0 |
