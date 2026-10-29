"""Locked public interface for one-dimensional shallow-water propagation."""


def propagate(initial, dx, dt, steps, mode, depth):
    """Advance a periodic, uniform, cell-centered grid with Rusanov and Euler.

    initial: float32 ndarray (n, 2), cells in increasing x. In linear mode,
        columns are surface displacement eta (m) and depth-integrated flux
        q (m^2/s); depth is the positive mean water depth H. In nonlinear
        mode columns are total water depth h (m) and discharge hu (m^2/s),
        with strictly positive h. Values are cell averages.
    dx: Python float, cell width in m. dt: Python float, timestep in s.
    steps: Python int, number of forward Euler steps. mode: 'linear' or
        'nonlinear'. depth: Python float, H in m for linear, ignored for
        nonlinear. Gravity is exactly 9.81 m/s^2. Positive q travels +x.

    Return {'state': float32 ndarray (n, 2)} in the same order and units.
    Periodic faces connect the last and first cells. With state U and flux F,
    face i+1/2 is (F(U_i)+F(U_i+1)-a*(U_i+1-U_i))/2. Linear F is
    (q, g*H*eta) and a=sqrt(g*H); nonlinear F is (q,q^2/h+g*h^2/2)
    and a=max(|q_i/h_i|+sqrt(g*h_i), |q_j/h_j|+sqrt(g*h_j)). Each
    step uses the old state for every face and U_i -= dt/dx*(flux_i+1/2
    - flux_i-1/2). Tests use stable steps and do not require dry cells.
    At ocean depths, implementations must retain perturbation accuracy in
    float32: for example, evolve h-H around a constant reference H and
    remove the constant hydrostatic pressure g*H**2/2 from the flux.
    Nonlinear depth remains ignored; any such reference is internal.
    """
    raise NotImplementedError


def objective_and_grad(initial, weights, dx, dt, steps, mode, depth):
    """Return a linear final-state objective and its initial-state gradient.

    initial, dx, dt, steps, mode, depth have the conventions of propagate.
    weights: float32 ndarray (n, 2), fixed dimensionless coefficients aligned
    with state cells and columns. Objective is sum(weights * final_state),
    without normalization. Return {'objective': float32 scalar ndarray (),
    'gradient': float32 ndarray (n, 2)}. The gradient is the derivative of
    that scalar with respect to each entry of initial, including derivatives
    of nonlinear flux and Rusanov speed. Objective and derivative carry the
    units implied by this weighted sum. MLX uses autodiff; Metal uses an
    adjoint. Test inputs avoid ties in nonlinear face wave speeds.
    """
    raise NotImplementedError
