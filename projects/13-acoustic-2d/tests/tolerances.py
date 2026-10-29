"""Numerical thresholds and their rationale."""
# Float32 accumulated time-stepping error normalized by the reference peak.
PARITY = 1e-4
# Independent float32 rounding in two implementations.
CROSS_PATH = 2e-4
# Coarse interval has cancellation between temporal and spatial error at order 4;
# expected combined order is 2, with 0.75 allowance for pre-asymptotic effects.
CONVERGENCE_COARSE_MIN = 1.25
# Finer interval is closer to the combined second-order asymptote; allow 0.35.
CONVERGENCE_FINE_MIN = 1.65
# Upper bound also detects accidental cancellation or an inconsistent experiment.
CONVERGENCE_MAX = 2.75
METADATA = [
    ('PARITY', PARITY, 'float32 time-stepping error relative to reference peak'),
    ('CROSS_PATH', CROSS_PATH, 'independent accumulated float32 rounding'),
    ('CONVERGENCE_COARSE_MIN', CONVERGENCE_COARSE_MIN, 'combined order 2 minus 0.75, coarse cancellation'),
    ('CONVERGENCE_FINE_MIN', CONVERGENCE_FINE_MIN, 'combined order 2 minus 0.35'),
    ('CONVERGENCE_MAX', CONVERGENCE_MAX, 'combined order 2 plus 0.75'),
]
