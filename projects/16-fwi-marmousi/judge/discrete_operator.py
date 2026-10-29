"""NumPy evaluation of tests/DISCRETE_OPERATOR.md, for oracle audit only.

No reference-library internals are called here. This is a forward-only judge
check, not either implementation or a test-time reference.
"""
import numpy as np


def derivative(field, axis, spacing, order):
    """Fourth-order first/second derivative with two zero output guards."""
    f = np.moveaxis(field, axis, -1)
    out = np.zeros_like(f)
    if order == 1:
        out[..., 2:-2] = ((2/3)*(f[..., 3:-1]-f[..., 1:-3])
                          - (1/12)*(f[..., 4:]-f[..., :-4]))/spacing
    else:
        center = f[..., 2:-2]
        out[..., 2:-2] = ((4/3)*(f[..., 3:-1]+f[..., 1:-3]-2*center)
                          - (1/12)*(f[..., 4:]+f[..., :-4]-2*center))/spacing**2
    return np.moveaxis(out, -1, axis)


def profiles(length, width, spacing, dt, frequency, speed):
    """Nodal polynomial C-PML law; returns a, b, and D b."""
    index = np.arange(length, dtype=np.float64)
    left = np.clip((2+width-index)/width, 0, 1)
    right = np.clip((index-(length-3-width))/width, 0, 1)
    fraction = np.maximum(left, right)
    sigma = -3*speed*np.log(0.001)/(2*width*spacing)*(left**2+right**2)
    alpha = np.pi*frequency*(1-fraction)
    total = sigma+alpha
    ratio = np.divide(sigma, total, out=np.zeros_like(sigma), where=total != 0)
    a = np.where(fraction > 0, np.exp(-total*dt), 0)
    b = ratio*np.expm1(-total*dt)
    return a, b, derivative(b, 0, spacing, 1)


def predict(velocity, sources, sl, rl, settings):
    """Evaluate the written equations in float64 at exact saved API inputs."""
    assert settings['accuracy'] == 4
    width, h, dt = (settings[k] for k in ('pml_width', 'grid_spacing', 'dt'))
    offset = width+2
    v = np.pad(np.asarray(velocity, np.float64), offset, mode='edge')
    shots, _, nt = sources.shape
    shape = (shots, *v.shape)
    profile = []
    for axis, length in enumerate(v.shape, start=1):
        broadcast = [1, 1, 1]
        broadcast[axis] = length
        profile.append(tuple(p.reshape(broadcast) for p in profiles(
            length, width, h, dt, settings['pml_freq'], settings['max_vel'])))
    current, previous = np.zeros(shape), np.zeros(shape)
    psi = [np.zeros(shape) for _ in range(2)]
    zeta = [np.zeros(shape) for _ in range(2)]
    predicted = np.empty((shots, rl.shape[1], nt))
    shot = np.arange(shots)[:, None]
    source_index = (shot, sl[..., 0]+offset, sl[..., 1]+offset)
    receiver_index = (shot, rl[..., 0]+offset, rl[..., 1]+offset)

    def zero_guards(field):
        field[:, :2, :] = field[:, -2:, :] = 0
        field[:, :, :2] = field[:, :, -2:] = 0
        return field

    for k in range(nt):
        predicted[..., k] = current[receiver_index]
        spatial = np.zeros(shape)
        for q, (a, b, c) in enumerate(profile):
            du = derivative(current, q+1, h, 1)
            inner = ((1+b)*derivative(current, q+1, h, 2) + c*du
                     + derivative(a*psi[q], q+1, h, 1))
            spatial += (1+b)*inner+a*zeta[q]
            psi[q] = zero_guards(b*du+a*psi[q])
            zeta[q] = zero_guards(b*inner+a*zeta[q])
        following = 2*current-previous+(v*dt)**2*spatial
        np.add.at(following, source_index,
                  -(v[source_index[1:]]*dt)**2*sources[..., k])
        previous, current = current, zero_guards(following)
    return predicted
