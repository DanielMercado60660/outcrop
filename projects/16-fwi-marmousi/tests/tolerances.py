"""Locked numeric criteria; all relative errors use float64 test arithmetic."""

# Project-wide float32 parity against the identical float64 discrete operator.
PARITY = 1e-4
# Judge-only f64 equation audit: independently ordered operations over 360 steps
# should agree far below float32 parity; this is not a new implementation gate.
EQUATION_AUDIT_REL = 1e-10
# Required Deepwave gradient direction agreement; magnitude is also tested.
GRADIENT_COSINE = 0.999
# Project-wide centered finite differences; h=1 m/s times RMS-one directions.
FD_REL = 1e-3
FD_STEP = 1.0
# Required recovery criterion after exactly this many accepted-or-stationary steps.
MODEL_ERROR_RATIO = 0.70
ITERATIONS = 40
# Fixed optimization protocol, not adjustable implementation parameters.
UPDATE_MAX = 20.0  # m/s maximum component of initial trial update
BACKTRACKS = 10  # halve UPDATE_MAX at most nine times; otherwise stay put
VELOCITY_MIN = 1400.0  # m/s physical projection bounds
VELOCITY_MAX = 2600.0
# Frozen contract's project golden size ceiling, in bytes.
MAX_GOLDEN_BYTES = 20_000_000

REASONS = {
    "PARITY": "1e-4 project-wide float32 parity, matched grid/stencil/dt/PML; global max norm",
    "GRADIENT_COSINE": "0.999 project requirement; gradient magnitude separately at PARITY",
    "FD_REL": "1e-3 project-wide centered FD; RMS-one directions and 1 m/s step avoid float32 cancellation",
    "MODEL_ERROR_RATIO": "0.70 project minimum 30% relative model L2 error reduction",
    "EQUATION_AUDIT_REL": "1e-10 judge-only f64 forward equation audit; independent operation order, 360 steps",
}
