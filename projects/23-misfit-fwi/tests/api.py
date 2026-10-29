"""Project 23 API. MLX only. All returned arrays must be float32."""


def misfit_and_grad(predicted, observed, dt, kind, epsilon):
    """Return a trace objective and its derivative with respect to predicted data.

    Parameters
    ----------
    predicted, observed : numpy.ndarray, float32, shape (ntrace, ntime)
        Real seismograms, equal shape, ntrace >= 1, ntime >= 2. Axes are
        (receiver, time), uniform sampling; amplitude units are arbitrary but
        identical. These are finite records, including any DC/Nyquist content.
    dt : float
        Positive sampling interval, seconds. Time integral uses a rectangular
        sum over ALL samples, then averages over receivers, not time samples.
    kind : str
        Exactly 'l2' or 'envelope'. Let H be the periodic, discrete Hilbert
        transform along the last axis: Fourier multiplier -i for positive
        frequencies, +i for negative, 0 at DC and even-length Nyquist.
        No padding, tapering, detrending, or amplitude normalization is applied.
        For l2, J = dt/(2*ntrace) * sum((predicted-observed)**2).
        For envelope, E(x)=sqrt(x*x + H(x)*H(x) + epsilon**2),
        J = dt/(2*ntrace) * sum((E(predicted)-E(observed))**2).
        The literal envelope subtraction is allowed; no cancellation-free
        algebraic rewrite is required. Evaluate both envelopes consistently:
        exact-fit and polarity-flipped envelope pairs have zero objective and
        gradient, including when predicted and observed are batched differently.
    epsilon : float
        Positive envelope smoothing, in trace amplitude units; ignored for l2.

    Returns
    -------
    dict with exactly:
        objective : numpy.ndarray, float32, shape ()
            Scalar nonnegative J, units amplitude squared times seconds.
        gradient : numpy.ndarray, float32, shape (ntrace, ntime)
            Euclidean derivative dJ/dpredicted (NOT divided by dt), units
            amplitude times seconds. Observed is held fixed. MLX uses autodiff.
            Differentiation includes H and envelope smoothing. Sign is positive
            for increasing J. No derivative with respect to dt is requested.
    """
    raise NotImplementedError
