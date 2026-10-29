"""All numeric acceptance thresholds for the locked checkpointing judge."""

TOLERANCES = {
    # Project 21's explicit requirement: replay must preserve the same path's
    # full-storage gradient, measured with max-norm relative error.
    "checkpoint_gradient": 1.0e-6,
    # Recomputed forward results must also preserve the full-storage answer.
    "checkpoint_forward": 1.0e-6,
    # Project-wide float32 vs float64 limit at the SAME discrete operator, not a
    # continuum solution. This also bounds independent MLX/Metal arithmetic.
    "reference": 1.0e-4,
    "cross_path": 1.0e-4,
    # Project-wide centered FD rule. The saved directional steps below avoid
    # float32 loss cancellation while remaining small vs ~1800 m/s.
    "finite_difference": 1.0e-3,
    # Judge-only check that the float64 FD truncation is much smaller than
    # the implementation allowance. No implementation is held to this value.
    "oracle_finite_difference": 2.0e-5,
    # Full-storage float64 eager-autodiff vs compiled Deepwave adjoint;
    # both evaluate precisely the same grid, stencil, PML and sampling.
    "oracle_adjoint": 1.0e-10,
    # Recompute the loss from returned float32 receiver samples in float64.
    "loss_from_receivers": 2.0e-6,
    # Zero-source propagation is exactly zero, including its model gradient.
    "zero_absolute": 0.0,
    # Allocator test: permit working arrays/adjoints to occupy 40% of the
    # full-storage peak. This deliberately generous allowance covers lazy
    # execution, VJP temporaries and different stencil implementations.
    "mlx_memory_workspace_fraction": 0.4,
    # A full-storage implementation can share successive wavefield levels
    # and store compact PML slabs. Allow six times the normalized payload
    # ratio (six dense fields per restart) to cover such representation
    # differences. Combined limit on this fixture is 0.509375, far below 1
    # for an unchanged full-trajectory implementation with a fabricated log.
    "mlx_memory_payload_factor": 6.0,
}

# This is a DIFFERENCING PARAMETER, not an acceptance threshold. Saved in
# goldens along with the actual rounded float32 plus/minus velocity arrays.
FD_STEPS_M_PER_S = (0.625, 1.25, 2.0)

# Work limits and storage counts are exact integers, not approximate timing
# claims. The exact minimum recursive-split costs defined in api.py supply
# the work limits, with no slack. Optimality under that convention IS needed.
# For the curve, at least one saved state must save work; full storage must
# require no extra propagation.
