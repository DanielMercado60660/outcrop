"""EikoNet contract (MLX only).

Both transport paths can represent this API; this project's locked tests select
only MLX. No reference travel times, FMM grid, test receivers, or golden filenames
are supplied to training. The implementation must train a physics-informed
network, not read goldens or substitute a numerical eikonal solver.

The deliberately small, smooth EikoNet variant is a factored all-source network
in D=2 or D=3, with two tanh hidden layers of width H. For source s and receiver r,

    x = concatenate((s-origin)/length_scale, (r-origin)/length_scale)
    a = tanh(x @ W1 + b1)
    b = tanh(a @ W2 + b2)
    f = b @ w3 + b3
    T(s,r) = ||r-s|| / reference_velocity * exp(f).

Flatten parameters in this order, each C/row-major:
W1 (2D,H), b1 (H,), W2 (H,H), b2 (H,), w3 (H,), b3 (1,).
P = 2*D*H + H*H + 3*H + 1. Coordinates are Cartesian metres, axis
order (x,z) in 2D and (x,y,z) in 3D; z increases downwards. Time is seconds,
velocity m/s, receiver derivatives s/m. Boundaries are open/outgoing; no
reflections or periodic wrap. All array inputs/outputs are finite NumPy float32.
"""


def evaluate(parameters, sources, receivers, origin, length_scale,
             reference_velocity, width):
    """Evaluate the specified network and its receiver-coordinate derivative.

    Args:
        parameters: (P,) float32, dimensionless flattened network parameters.
        sources: (N,D) float32, source positions in metres; rows form pairs.
        receivers: (N,D) float32, receiver positions in metres; one per source row.
        origin: (D,) float32, normalization origin in metres, not a source.
        length_scale: positive Python float, metres; a single isotropic scale.
        reference_velocity: positive Python float, m/s; ansatz scaling only.
        width: positive Python int H, hidden-layer width (tests use 4 and 32).

    Returns:
        dict with exactly 'time': (N,) float32 seconds, and
        'receiver_gradient': (N,D) float32 dT/dr in s/m, with other inputs fixed.
        T is exactly zero at coincident pairs. Its derivative there is undefined;
        this API returns the zero vector there. Else use the full product rule,
        including the Euclidean distance factor. N may vary between calls.
    """
    raise NotImplementedError


def physics_loss_and_grad(parameters, sources, receivers, velocities, origin,
                          length_scale, reference_velocity, width):
    """Return a dimensionless eikonal objective and its parameter gradient.

    Args:
        parameters, sources, receivers, origin, length_scale,
        reference_velocity, width: as in evaluate; all pairs are noncoincident.
        velocities: (N,) float32 positive m/s, sampled at receiver positions.
            Velocities are held constant when differentiating parameters.

    Objective:
        q_i = velocities[i]**2 * sum_j (dT_i/dr_j)**2 - 1
        loss = mean_i(q_i**2).
        No data term, regularization, boundary penalty, factor of 1/2, or
        differentiation through velocity. Coordinate derivatives are physical
        derivatives, so normalization contributes 1/length_scale. MLX must use
        autodiff for parameter gradients, including the mixed derivative through
        dT/dr (no stop-gradient). This checks the core PINN training operation.

    Returns:
        dict with exactly 'loss': scalar shape () float32, dimensionless, and
        'gradient': (P,) float32 d(loss)/d(parameters), same flattening order.
    """
    raise NotImplementedError


def train(initial_parameters, sources, receivers, velocities, origin,
          length_scale, reference_velocity, width, steps, learning_rate):
    """Fit one medium using ONLY the supplied physics collocation pairs.

    Args:
        initial_parameters: (P,) float32, supplied deterministic initialization;
            same layout as evaluate.parameters. Must not be mutated in place.
        sources, receivers: (N,D) float32 metres, fixed collocation pairs.
        velocities: (N,) float32 m/s at the collocation receivers.
        origin, length_scale, reference_velocity, width: as in evaluate.
        steps: positive Python int, maximum optimizer updates (tests: 10000).
        learning_rate: positive Python float, initial optimizer step (0.001).

    Minimize exactly physics_loss_and_grad's objective with MLX autodiff.
    Optimizer/schedule are implementation choices within the update budget;
    full-batch or deterministic batching of supplied rows is permitted. Do not
    generate random data, use supervised travel times, call FMM, read goldens,
    or initialize from an external trained checkpoint. No randomness is needed.
    Each call starts independently from initial_parameters; no persistent state.

    Returns:
        dict with exactly 'parameters': (P,) float32 trained parameters usable
        by evaluate for arbitrary held-out pairs. No custom model objects.
    """
    raise NotImplementedError
