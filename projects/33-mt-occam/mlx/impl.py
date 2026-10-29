"""One-dimensional magnetotellurics and bounded Occam inversion."""

import math

import mlx.core as mx

_LN10 = math.log(10.0)
_MU0 = 4.0 * math.pi * 1.0e-7


def _response(model, thickness, frequencies):
    if model.shape[0] == 1:
        # The halfspace response is already in log10 units. Avoid converting
        # to linear resistivity: even finite log10 models can overflow or
        # underflow float32 there. Broadcasting also preserves autodiff.
        logrho = mx.broadcast_to(model[0], frequencies.shape)
        phase = mx.full_like(frequencies, math.pi / 4.0)
        return mx.stack([logrho, phase])

    # Z = (1+i)*sqrt(mu0*omega/2)*q; halfspace q is sqrt(rho).
    omega = (2.0 * math.pi) * frequencies
    qr = mx.ones_like(frequencies) * mx.exp(model[-1] * (0.5 * _LN10))
    qi = mx.zeros_like(frequencies)
    for i in range(model.shape[0] - 2, -1, -1):
        s = mx.exp(model[i] * (0.5 * _LN10))
        x = thickness[i] * mx.sqrt((_MU0 * 0.5) * omega) / s
        # tanh(x+ix), divided through by cosh(2x), never overflows.
        e = mx.exp(-2.0 * x)
        inv_cosh = 2.0 * e / (1.0 + e * e)
        denom = 1.0 + mx.cos(2.0 * x) * inv_cosh
        tr = mx.tanh(2.0 * x) / denom
        ti = mx.sin(2.0 * x) * inv_cosh / denom
        nr = qr + s * tr
        ni = qi + s * ti
        dr = s + qr * tr - qi * ti
        di = qr * ti + qi * tr
        norm = dr * dr + di * di
        qr, qi = s * (nr * dr + ni * di) / norm, s * (ni * dr - nr * di) / norm
    logrho = mx.log(qr * qr + qi * qi) / _LN10
    phase = (math.pi / 4.0) + mx.arctan2(qi, qr)
    return mx.stack([logrho, phase])


def forward_mt(model, thickness, frequencies):
    with mx.stream(mx.gpu):
        m, h, f = (_float32(a) for a in (model, thickness, frequencies))
        return {"response": _response(m, h, f)}


def objective_and_grad(model, thickness, frequencies, observed, sigma, alpha):
    with mx.stream(mx.gpu):
        h, f, obs, std = (
            _float32(a) for a in (thickness, frequencies, observed, sigma)
        )

        def objective(m):
            residual = (_response(m, h, f) - obs) / std
            diff = m[1:] - m[:-1]
            return 0.5 * mx.mean(residual * residual) + 0.5 * alpha * mx.sum(diff * diff)

        value, gradient = mx.value_and_grad(objective)(_float32(model))
        return {"objective": value, "gradient": gradient}


def _float32(a):
    return mx.array(a, dtype=mx.float32)


def _roughness(model):
    differences = model[1:] - model[:-1]
    return 0.5 * mx.sum(differences * differences)


def _damped_solve(matrix, rhs, damping, identity):
    """Diagonally equilibrated LDL^T solve, entirely in float32 on the GPU.

    The matrix is a Gauss--Newton normal matrix plus first-difference
    regularization. Relative diagonal damping makes it positive definite;
    scaling its diagonal to one avoids mixing very different sensitivities.
    Explicit elimination uses GPU array operations, with no LAPACK/CPU solve
    or float64 optimizer state. Damping changes the step, not the objective.
    """
    scale = mx.sqrt(mx.maximum(mx.diag(matrix), 1.0e-12))
    work = matrix / (scale[:, None] * scale[None, :]) + damping * identity
    reduced_rhs = rhs / scale
    n = rhs.shape[0]
    indices = mx.arange(n)
    pivots = []
    for k in range(n):
        column = mx.where(indices > k, work[:, k], 0.0)
        pivot = mx.maximum(work[k, k], 1.0e-12)
        pivots.append(pivot)
        multiplier = column / pivot
        reduced_rhs = reduced_rhs - multiplier * reduced_rhs[k]
        work = work - multiplier[:, None] * column[None, :]
    solution = mx.zeros_like(rhs)
    for k in range(n - 1, -1, -1):
        # Use the same guarded pivot as elimination; dividing by the raw
        # diagonal here would undo the safeguard for a near-singular system.
        value = (reduced_rhs[k] - mx.sum(work[k] * solution)) / pivots[k]
        solution = solution + identity[k] * value
    return solution / scale


def invert_occam(initial_model, thickness, frequencies, observed, sigma,
                 target_rms, max_iterations):
    # Only scalar scheduling/comparisons are performed by Python. Forward
    # physics, autodiff, reductions, normal equations, solves, and all model
    # updates stay float32 on the GPU; there is no float64 CPU exception.
    with mx.stream(mx.gpu):
        return _invert_occam(initial_model, thickness, frequencies, observed,
                             sigma, target_rms, max_iterations)


def _invert_occam(initial_model, thickness, frequencies, observed, sigma,
                   target_rms, max_iterations):
    h, f, obs, std = (
        _float32(a) for a in (thickness, frequencies, observed, sigma)
    )
    model = mx.clip(_float32(initial_model), 0.0, 4.0)
    normalizer = math.sqrt(obs.size)
    limit = float(target_rms) ** 2
    remaining = max(0, int(max_iterations))
    best = None
    best_roughness = math.inf
    least_misfit = math.inf
    least_model = model

    def residual(m):
        return (((_response(m, h, f) - obs) / std) / normalizer).reshape(-1)

    def evaluate(m):
        nonlocal best, best_roughness, least_misfit, least_model
        r = residual(m)
        mse = mx.sum(r * r)
        rough = _roughness(m)
        mx.eval(m, r, mse, rough)
        mse_value, rough_value = mse.item(), rough.item()
        if mse_value < least_misfit:
            least_misfit, least_model = mse_value, m
        # Feasibility uses the requested target, without changing it by a
        # fixed safety factor. Every candidate is already bounded float32.
        if mse_value <= limit and rough_value < best_roughness:
            best, best_roughness = m, rough_value
        return r, mse, rough

    def result(m):
        response = _response(m, h, f)
        r = (response - obs) / std
        return {"model": m, "response": response,
                "rms": mx.sqrt(mx.mean(r * r)), "roughness": _roughness(m)}

    evaluate(model)
    if remaining == 0:
        return result(model)

    # The constant model has an analytic weighted least-squares amplitude:
    # phase is pi/4 independently of that amplitude. If feasible, R=0 is the
    # global constrained minimum, including for the single-halfspace case.
    weights = 1.0 / (std[0] * std[0])
    level = mx.clip(mx.sum(weights * obs[0]) / mx.sum(weights), 0.0, 4.0)
    constant = mx.broadcast_to(level, model.shape)
    _, constant_mse, _ = evaluate(constant)
    if constant_mse.item() <= limit or model.size == 1:
        return result(constant)

    identity = mx.eye(model.size, dtype=mx.float32)
    difference = identity[1:] - identity[:-1]
    roughness_hessian = difference.T @ difference

    def minimize_regularized(start, alpha):
        """Bounded, damped Gauss--Newton for mean(r^2)/2 + alpha*R."""
        nonlocal remaining
        m = start
        r, mse, rough = evaluate(m)
        damping = 1.0e-5
        for _ in range(min(25, remaining)):
            remaining -= 1
            # One forward-mode autodiff direction per model parameter. The
            # frequency/row flattening matches residual exactly, including
            # duplicate frequencies and both independent uncertainty rows.
            columns = []
            for k in range(m.size):
                _, tangent = mx.jvp(lambda x: [residual(x)], [m], [identity[k]])
                columns.append(tangent[0])
            jacobian = mx.stack(columns, axis=1)
            gradient = jacobian.T @ r + alpha * (roughness_hessian @ m)
            curvature = jacobian.T @ jacobian + alpha * roughness_hessian
            blocked = ((m <= 0.0) & (gradient > 0.0)) | (
                (m >= 4.0) & (gradient < 0.0)
            )
            free = (~blocked).astype(mx.float32)
            projected_gradient = free * gradient
            curvature = (curvature * free[:, None] * free[None, :]
                         + mx.diag(1.0 - free))
            mx.eval(gradient, projected_gradient, curvature)
            if mx.max(mx.abs(projected_gradient)).item() <= 1.0e-5:
                break
            objective = 0.5 * mse + alpha * rough
            accepted = False
            step_size = math.inf
            for _ in range(4):
                step = _damped_solve(curvature, -projected_gradient,
                                     damping, identity)
                # Limit early nonlinear steps to one log10 resistivity unit.
                step = step / mx.maximum(1.0, mx.max(mx.abs(step)))
                mx.eval(step)
                fraction = 1.0
                for _ in range(12):
                    trial = mx.clip(m + fraction * step, 0.0, 4.0)
                    displacement = trial - m
                    slope = mx.sum(gradient * displacement)
                    trial_r, trial_mse, trial_rough = evaluate(trial)
                    trial_objective = 0.5 * trial_mse + alpha * trial_rough
                    sufficient_decrease = ((slope < 0.0)
                                           & (trial_objective <= objective + 1.0e-4 * slope))
                    if sufficient_decrease.item():
                        step_size = mx.max(mx.abs(displacement)).item()
                        m, r, mse, rough = trial, trial_r, trial_mse, trial_rough
                        accepted = True
                        damping = max(1.0e-6, 0.3 * damping)
                        break
                    fraction *= 0.5
                if accepted:
                    break
                damping *= 10.0
            # At float32 resolution the objective may stop decreasing before
            # the gradient vanishes. Do not spend the whole target-search
            # budget on sub-ulp updates at a single regularization weight.
            if not accepted or step_size <= 1.0e-5:
                break
        return m, mse.item()

    # Increasing alpha trades fit for smoothness. Bracket the requested RMS
    # with converged regularized solutions, then bisect in log(alpha).
    # All subproblems share the one nonlinear iteration budget.
    alpha = 1.0
    low = None
    high = None
    while remaining > 0:
        model, mse = minimize_regularized(model, alpha)
        if mse <= limit:
            low = (alpha, model, mse)
            if high is not None:
                break
            alpha *= 4.0
        else:
            high = (alpha, model, mse)
            if low is not None:
                break
            alpha *= 0.25
        if alpha < 1.0e-10 or alpha > 1.0e10:
            break

    while low is not None and high is not None and remaining > 0:
        # A feasible endpoint within 0.05% of the RMS target is sufficient;
        # it remains inside the exact requested constraint, never above it.
        if low[2] >= limit * (1.0 - 1.0e-3):
            break
        if high[0] / low[0] <= 1.001:
            break
        alpha = math.sqrt(low[0] * high[0])
        model, mse = minimize_regularized(low[1], alpha)
        if mse <= limit:
            low = (alpha, model, mse)
        else:
            high = (alpha, model, mse)

    return result(best if best is not None else least_model)
