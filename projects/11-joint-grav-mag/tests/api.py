"""Frozen project 11 API. MLX only. No truth enters these functions.

All arrays in/out are NumPy float32, C contiguous, unless stated otherwise.
Models are C-flattened (east, north, depth), depth fastest. Density u is in
units of 500 kg/m^3, susceptibility v in units of 0.05 SI. Coordinates of the
saved physical prisms/receivers are east, north, up (metres). Gravity is g_z
positive down (mGal); magnetic data is the ENU projection onto the inducing
field (nT), the linear total-field approximation, no remanence. Data and
matrix rows are divided by their property's saved physical_data_scales.
"""


def predict(density, susceptibility, gravity_matrix, magnetic_matrix):
    """Apply linear, normalized prism sensitivities to both properties.

    Args:
        density: (N,) float32, normalized density u (500 kg/m^3 per unit).
        susceptibility: (N,) float32, normalized susceptibility v (0.05 SI/unit).
        gravity_matrix: (Ng,N) float32, normalized g_z per unit u.
        magnetic_matrix: (Nm,N) float32, normalized projected anomaly per unit v.
    Returns:
        dict gravity: (Ng,) float32 Ag@u, magnetic: (Nm,) float32 Am@v.
        Both outputs are dimensionless normalized survey amplitudes.
    """
    raise NotImplementedError


def objective_and_grad(density, susceptibility, gravity_matrix, magnetic_matrix,
                       gravity_data, magnetic_data, derivatives, alpha, smooth, coupling):
    """Evaluate the sum objective and autodiff gradients in normalized units.

    Args:
        density, susceptibility: (N,) float32 u,v, units in module docstring.
        gravity_matrix: (Ng,N) float32 Ag, normalized gravity sensitivity.
        magnetic_matrix: (Nm,N) float32 Am, normalized magnetic sensitivity.
        gravity_data: (Ng,) float32 dg, normalized observed down gravity.
        magnetic_data: (Nm,) float32 dm, normalized total-field projection.
        derivatives: (3,N,N) float32 D; axis order east,north,depth.
            Forward differences in units of one 100 m cell; outward boundary
            rows zero; no wrapping. Gradients and cross products are collocated
            at each cell. The supplied operators define the discretization.
        alpha: Python float >0, dimensionless ridge coefficient.
        smooth: Python float >=0, dimensionless squared-gradient coefficient.
        coupling: Python float >=0, dimensionless cross-gradient coefficient.

    F = ||Ag u-dg||²/2 + ||Am v-dm||²/2 + alpha*(||u||²+||v||²)/2
        + smooth*(||Du||²+||Dv||²)/2 + coupling*||Du x Dv||²/2.
    Sums, never means; no cell volume weighting or hidden normalization. Cross
    product uses the stated east,north,depth axis order, including its sign.
    Returns dict (all float32): objective scalar shape (); terms (4,) in order
    data,ridge,smooth,cross; grad_density (N,), grad_susceptibility (N,), the
    derivatives of F w.r.t. the normalized u,v; cross_gradient (3,N), Du x Dv,
    component first. Objective, terms and cross_gradient are dimensionless.
    MLX must obtain objective gradients by autodiff.
    """
    raise NotImplementedError


def invert(gravity_matrix, magnetic_matrix, gravity_data, magnetic_data,
           derivatives, alpha, smooth, coupling, sweeps):
    """Deterministic alternating quadratic minimization, unconstrained models.

    Args:
        gravity_matrix: (Ng,N) float32 Ag, normalized g_z sensitivity.
        magnetic_matrix: (Nm,N) float32 Am, normalized total-field sensitivity.
        gravity_data: (Ng,) float32 dg, normalized observed gravity.
        magnetic_data: (Nm,) float32 dm, normalized observed magnetic anomaly.
        derivatives: (3,N,N) float32 D, east,north,depth forward difference per
            100 m cell with zero outward rows, as in objective_and_grad.
        alpha: Python float >0, dimensionless ridge weight.
        smooth: Python float >=0, dimensionless smoothness weight.
        coupling: Python float >=0, dimensionless cross-gradient weight.
        sweeps: Python int >=0, exact count, no early stopping.

    R=alpha I+smooth sum_a D[a].T D[a], Hg=Ag.T Ag+R, Hm=Am.T Am+R.
    Initialize u=solve(Hg,Ag.T dg), v=solve(Hm,Am.T dm), independently.
    Define C(w)q = Dq x Dw, stacked by component (3N,N).
    Each sweep first solves (Hg+coupling C(v).T C(v))u=Ag.T dg,
    then (Hm+coupling C(u).T C(u))v=Am.T dm using the NEW u.
    Zero sweeps gives separate inversion. No clipping, truth, noise generation,
    added damping, renormalization, random initialization, or hidden schedules.
    This fixed schedule locks numerical parity; no global minimum is claimed.
    Float32 model parity allows 3e-3 global relative error because the separate
    normal matrices have condition numbers about 1.18e4 and 7.80e3. Predicted
    data from every returned model must still match its float64 iterate to
    1e-4. These bounds are centralized in tests/tolerances.py.
    Returns dict density (N,), susceptibility (N,), float32 normalized models
    in units of 500 kg/m^3 and 0.05 SI respectively, C order east,north,depth.
    """
    raise NotImplementedError
