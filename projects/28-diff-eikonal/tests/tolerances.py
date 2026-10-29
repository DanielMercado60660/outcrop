"""All numerical acceptance thresholds, including judge self-checks."""

# Project-wide float32 parity at a matched discrete problem; global infinity norm.
PARITY_REL = 1e-4
# The project's explicit tomography gradient direction requirement.
GRADIENT_COSINE_MIN = 0.999
# Project-wide centered finite differences, also checked using implementation loss.
FD_REL = 1e-3
# Absolute slowness increments (s/m), binary-exact: ~0.05% for dense probes,
# ~0.2% for coordinate probes. Coordinates need larger steps because their
# loss differences have less signal. These pass both the rounded-loss audit
# and the full float32 stable-increment sweep/adjoint audit. The stable root
# is REQUIRED in tests/api.py; absolute-time quadratic roots amplify FD noise.
FD_STEPS = (2.0**-22, 2.0**-22, 2.0**-20, 2.0**-20, 2.0**-20)
# Float64 independent heap-marching versus converged sweeping at the same stencil.
ORACLE_FORWARD_REL = 2e-12
# Float64 adjoint vs centered differences; finite-step truncation is ~2.2e-5
# for the signed dense layered probe. This remains 20x tighter than FD_REL.
ORACLE_FD_REL = 5e-5
# Float64 nonlinear residual and analytic axis travel times, rounding only.
ORACLE_RESIDUAL_REL = 2e-12
# Sweep stop in seconds, far below float32 requirements on these <2 s grids.
ORACLE_SWEEP_ATOL = 1e-14
# Pure zero-loss case must be exact, including zero-weight observations.
ZERO_ATOL = 0.0
