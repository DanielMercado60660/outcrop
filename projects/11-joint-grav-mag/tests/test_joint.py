"""Physical forward, full objective/gradient parity, fixed solver, recovery.

This is an MLX-only project, so it has no Metal or cross-path tests.
All probes, operators, noisy observations, FD perturbations and truth come
from judge goldens. No reference library imports or random test generation.
"""
import numpy as np
import pytest
from outcrop_harness import call, golden

from .tolerances import (
    FD_STEP,
    GRAD_RTOL,
    MAX_ERROR_RATIO,
    PARITY_RTOL,
    SOLVER_MODEL_RTOL,
    ZERO_ATOL,
)

PATHS = ['mlx']
PROPERTIES = ['density', 'susceptibility']


def common(p):
    result = {k: p[k].astype(np.float32) for k in
              ('gravity_matrix', 'magnetic_matrix', 'gravity_data', 'magnetic_data', 'derivatives')}
    result.update({k: float(p[k]) for k in ('alpha', 'smooth', 'coupling')})
    return result


def check_array(actual, reference, rtol=PARITY_RTOL, zero=False):
    actual, reference = np.asarray(actual), np.asarray(reference)
    assert actual.shape == reference.shape
    assert actual.dtype == np.dtype('float32')
    assert np.all(np.isfinite(actual))
    scale = float(np.max(np.abs(reference)))
    error = float(np.max(np.abs(actual.astype(np.float64)-reference)))
    bound = ZERO_ATOL if zero or scale == 0 else rtol*scale
    assert error <= bound, (error, bound)


@pytest.mark.parametrize('path', PATHS)
def test_forward(path):
    p, probe = golden('inversion'), golden('forward')
    kwargs = {k: p[k].astype(np.float32) for k in ('gravity_matrix', 'magnetic_matrix')}
    kwargs.update({k: probe[k].astype(np.float32) for k in PROPERTIES})
    out = call(path, 'predict', **kwargs)
    assert set(out) == {'gravity', 'magnetic'}
    for key in out:
        check_array(out[key], probe[key])


@pytest.mark.parametrize('path', PATHS)
@pytest.mark.parametrize('case', ['coupled', 'uncoupled', 'affine'])
def test_objective_and_gradient(path, case):
    p, probe = golden('inversion'), golden(f'objective_{case}')
    kwargs = common(p)
    kwargs['coupling'] = float(probe['coupling'])
    kwargs.update({k: probe[k].astype(np.float32) for k in PROPERTIES})
    out = call(path, 'objective_and_grad', **kwargs)
    keys = {'objective', 'terms', 'grad_density', 'grad_susceptibility', 'cross_gradient'}
    assert set(out) == keys
    for key in keys:
        check_array(out[key], probe[key], zero=(case == 'affine' and key == 'cross_gradient'))
    # Affine dependence has mathematically zero cross penalty, despite arbitrary
    # nonzero data residuals and gradients. No whole-objective zero exemption.
    if case == 'affine':
        assert abs(float(out['terms'][3])) <= ZERO_ATOL
    if case != 'affine':
        for i, key in enumerate(PROPERTIES):
            check_array(out[f'grad_{key}'], probe['finite_difference'][i], rtol=GRAD_RTOL)


@pytest.mark.parametrize('path', PATHS)
@pytest.mark.parametrize('case', ['coupled', 'uncoupled'])
def test_centered_finite_differences(path, case):
    p, probe = golden('inversion'), golden(f'objective_{case}')
    kwargs = common(p)
    kwargs['coupling'] = float(probe['coupling'])
    kwargs.update({k: probe[k].astype(np.float32) for k in PROPERTIES})
    out = call(path, 'objective_and_grad', **kwargs)
    grad = np.stack([out['grad_density'], out['grad_susceptibility']]).astype(np.float64)
    analytic = np.einsum('kpn,pn->k', probe['directions'], grad)
    # h=0.002 normalized units, perturbations saved as float32 by the judge;
    # the independent judge also evaluates these perturbations in float64.
    observed = []
    for plus, minus in zip(probe['fd_plus'], probe['fd_minus']):
        values = []
        for models in (plus, minus):
            args = dict(kwargs, density=models[0], susceptibility=models[1])
            value = call(path, 'objective_and_grad', **args)['objective']
            assert np.asarray(value).shape == ()
            assert np.all(np.isfinite(value))
            values.append(float(value))
        observed.append((values[0]-values[1])/(2*FD_STEP))
    scale = float(np.max(np.abs(probe['directional_fd'])))
    for candidate in (analytic, np.array(observed)):
        assert np.max(np.abs(candidate-probe['directional_fd'])) <= GRAD_RTOL*scale


@pytest.mark.parametrize('path', PATHS)
@pytest.mark.parametrize('sweeps,coupled,prefix', [
    (0, False, 'separate'), (3, False, 'separate'),
    (1, True, 'sweep1'), (7, True, 'sweep7'), (150, True, 'joint')])
def test_solver_parity(path, sweeps, coupled, prefix):
    p = golden('inversion')
    kwargs = common(p)
    if not coupled:
        kwargs['coupling'] = 0.
    out = call(path, 'invert', **kwargs, sweeps=sweeps)
    assert set(out) == set(PROPERTIES)
    for key, matrix_key in zip(PROPERTIES, ('gravity_matrix', 'magnetic_matrix')):
        reference = p[f'{prefix}_{key}']
        check_array(out[key], reference, rtol=SOLVER_MODEL_RTOL)
        # Model-space errors are amplified by the ill-conditioned inverse.
        # Require the usual tight parity in data space, evaluated here in
        # float64 so this check tests invert independently of predict.
        matrix = p[matrix_key]
        predicted = matrix @ out[key].astype(np.float64)
        expected = matrix @ reference
        scale = float(np.max(np.abs(expected)))
        assert np.max(np.abs(predicted-expected)) <= PARITY_RTOL*scale


@pytest.mark.parametrize('path', PATHS)
@pytest.mark.parametrize('property_name', PROPERTIES)
def test_joint_improves_each_property_by_ten_percent(path, property_name):
    p = golden('inversion')
    kwargs = common(p)
    separate_args = dict(kwargs, coupling=0.)
    separate = call(path, 'invert', **separate_args, sweeps=0)
    joint = call(path, 'invert', **kwargs, sweeps=int(p['sweeps']))
    # Score the returned models ourselves. Neither truth nor golden solver
    # results enter the implementation API. Relative L2 errors over ALL cells.
    truth = p[f'true_{property_name}']
    for result in (separate, joint):
        assert result[property_name].shape == truth.shape
        assert np.all(np.isfinite(result[property_name]))
    separate_error = np.linalg.norm(separate[property_name].astype(float)-truth)/np.linalg.norm(truth)
    joint_error = np.linalg.norm(joint[property_name].astype(float)-truth)/np.linalg.norm(truth)
    assert separate_error > 0
    assert joint_error <= MAX_ERROR_RATIO*separate_error, (joint_error, separate_error)
