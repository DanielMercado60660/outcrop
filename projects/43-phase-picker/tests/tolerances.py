"""Numeric criteria; F1 gates calibrated from the sealed MLX GPU reference (documented deviation)."""

# Project-wide float32 parity against a float64 computation.
PARITY_RTOL = 1e-4
# Absolute floor for probabilities/gradients near zero; no relative accuracy exists at zero.
PARITY_ATOL = 1e-6
# Project-wide centered finite-difference gradient requirement, measured in vector L2 norm.
GRAD_RTOL = 1e-3
# h=.02 makes the loss difference about 1e-3 (loss ~1.33): float32 rounding
# contributes <2e-4 relative, while central truncation stays <1e-4. The judge
# verifies NumPy and PyTorch float32 arithmetic as well as float64. Endpoints are
# saved after float32 rounding; tests project along their effective secants.
FD_STEP = 0.02
# Specified limits, inclusive; errors measured in seconds.
P_MEDIAN_MAX_S = 0.05
S_MEDIAN_MAX_S = 0.1
# Default F1 threshold, retained until stronger MLX GPU evidence is available.
F1_MIN = 0.9
# The earlier relaxed gates were withdrawn: their runtime premise was wrong.
# These are provisional default gates, NOT a claim of reference feasibility.
# No lower gate can be locked until three strengthened MLX GPU seeds finish.
# APPROVED DEVIATION (2026-10-28): the strengthened, sealed MLX GPU reference (36000 steps,
# ~326 s training per seed on an M5 Max) reaches test F1 P/S/micro of
#   43043: 0.8884 / 0.7491 / 0.8214
#   43044: 0.8934 / 0.7536 / 0.8259
#   43045: 0.8880 / 0.7504 / 0.8214
# No seed reaches the default 0.9, so each gate is the worst seed minus REFERENCE_F1_MARGIN.
# S is limited by analyst label spread (~0.05-0.1 s inter-analyst disagreement on STEAD S picks).
P_F1_MIN = 0.8880203908241291 - 0.02  # worst seed minus REFERENCE_F1_MARGIN
S_F1_MIN = 0.749133369823025 - 0.02  # worst seed minus REFERENCE_F1_MARGIN
MICRO_F1_MIN = 0.8213815500746072 - 0.02  # worst seed minus REFERENCE_F1_MARGIN
# Historical self-checks from the earlier, retired gates only. These must never grade an implementation.
# Kept exactly to preserve the original broken-model sensitivity audit.
RETIRED_ROUND3_F1_GATES = (0.8372888283378747, 0.7034121122599705, 0.7749286284245057)
# Adopted rule: retain .9 wherever worst seed >=.9; otherwise propose
# worst seed minus .02, explicitly disclosed as an accuracy deviation. Do NOT
# subtract .02 before checking .9 (e.g. a worst seed of .91 keeps the .9 gate).
REFERENCE_F1_MARGIN = 0.02
# Budget: <=20-30 GPU minutes per strengthened seed. This bounds the
# synchronized training loop including augmentation, but excludes data loading,
# checkpoint I/O and test inference, whose measured times are reported separately.
REFERENCE_MAX_TRAINING_S = 1800.0
# A broken picker must miss every active F1 gate by at least 0.20 absolute F1;
# this is an explicit judge-strength audit, not slack on an implementation pass.
BROKEN_F1_GAP = 0.2
MATCH_TOLERANCE_S = 0.1
# Decoder operating point fixed before training; a contiguous >= threshold region
# produces one pick at its earliest maximum. No truth-assisted windowing.
PICK_THRESHOLD = 0.5
# Probability normalization uses the same float32 error budget as numeric parity.
PROB_SUM_ATOL = 1e-4
