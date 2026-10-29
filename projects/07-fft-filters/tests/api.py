"""Public numerical API for this MLX-only project."""


def filter_grid(grid, dx, dy, operation, height, inclination, declination,
                magnetization_inclination, magnetization_declination):
    """Apply a transformation and return {'result': array}.

    grid: finite float32 array (ny, nx), arbitrary field units; rows increase
        northing and columns increase easting, with an arbitrary origin. No
        padding or taper is applied; the FFT treats the grid as periodic.
    dx, dy: positive Python floats, spacing in metres along easting, northing.
    operation: 'upward', 'rtp', 'vertical', or 'tilt'.
    height: nonnegative Python float, upward displacement in metres.
    inclination, declination: Python floats in degrees, inducing field
        inclination positive downward, declination clockwise from north.
    magnetization_inclination, magnetization_declination: Python floats in
        degrees, same convention, describing source magnetization. Angle
        arguments apply only to RTP; height applies only to continuation.

    Returns {'result': float32 array (ny, nx)}. Upward and RTP have input
    units, vertical has input units per metre, tilt is radians. Vertical is
    the first upward derivative (Fourier factor -|k|). RTP uses Harmonica's
    factor and sets zero frequency to zero, including at the pole. Return the
    real part of the inverse FFT. Tilt is atan2(vertical, hypot(easting
    derivative, northing derivative)), with centered horizontal differences
    in the interior and first-order one-sided differences at the edges.
    No RTP regularization is applied. Zero inducing or magnetization
    inclination and zero total tilt gradient are outside this contract.
    """
    raise NotImplementedError
