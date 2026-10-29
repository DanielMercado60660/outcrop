"""Locked numerical criteria. All error measures are evaluated in float64."""

# Each reason is also copied verbatim into golden/meta.json by the oracle.
CRITERIA = {
    "parity": (1e-4, "Project-wide float32 parity at the identical Deepwave grid, stencil, dt and PML; max absolute error divided by reference max amplitude."),
    "gradient_fd": (1e-3, "Project-wide centered finite differences; h=1 m/s balances float32 objective subtraction and O(h^2) truncation."),
    "oracle_fd": (2e-5, "Float64 adjoint versus centered differences at h=1 m/s; finite-step truncation is measured, not treated as rounding."),
    "history_atol": (1e-8, "Absolute allowance for near-zero normalized objectives in an iterative optimizer; relative parity alone is undefined there."),
    "single_flat_velocity": (1.0, "Absolute m/s allowance ONLY for the free coefficient in late single-scale states within 1 m/s of the float64 endpoint, with the other coefficient already on its bound. Curvature about 3.6e-6 per (m/s)^2 and float32 objective scatter of several 1e-7 relative make roughly 0.3-0.5 m/s unresolved; 1 m/s provides margin across arithmetic implementations. Early single-scale states, bound coordinates and all multiscale states retain 1e-4 parity."),
    "improvement": (0.6, "Unchanged project pass criterion: multiscale final grid-model L2 error is strictly below 0.6 times the single-scale error."),
    "baseline_error_fraction": (0.3, "Fixture validity: the high-frequency inversion must retain at least 30% of the starting grid-model error, demonstrating a failed baseline."),
    "half_cycle": (0.5, "Fixture validity: the initial direct-arrival travel-time error exceeds half the source peak-frequency period."),
    "baseline_stationarity": (5e-6, "Projected normalized-objective derivative per m/s; the 1 m/s single-scale uncertainty band has derivatives below 4e-6 in both saved sweeps, with 1e-6 margin for float32 arithmetic. Retains a convergence guard without demanding sub-resolution placement in the flat valley; endpoint parameter and objective parity also remain enforced."),
    "zero_objective": (1e-8, "Noiseless truth: squared float32 propagation noise divided by strongly suppressed 3 Hz observed energy; cancellation makes the absolute objective less accurate than unfiltered waveform parity."),
    "zero_gradient": (1e-7, "Absolute derivative per m/s at the noiseless truth, where relative gradient error has zero denominator."),
    "optimizer_stop": (1e-10, "Locked optimizer objective stopping threshold: freeze converged stages before normalized descent amplifies roundoff-sized gradients."),
}

TOL = {name: value for name, (value, _) in CRITERIA.items()}
