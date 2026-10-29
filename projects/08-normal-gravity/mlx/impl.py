"""WGS84 normal gravity in ellipsoidal-harmonic coordinates."""

import math
import struct

import mlx.core as mx

_A = 6378137.0
_F = 1.0 / 298.257223563
_GM = 398600441800000.0
_OMEGA = 7.292115e-5
_E2 = _F * (2.0 - _F)
_E = math.sqrt(_E2)
_B = 1.0 - _F
_X0 = _B / _E
_Q0 = 0.5 * ((1.0 + 3.0 * _X0 * _X0) * math.atan(1.0 / _X0) - 3.0 * _X0)
_G = _GM / (_A * _A)
_C = _OMEGA * _OMEGA * _A
_H = _C * _E / (2.0 * _Q0)


def _equatorial_value_and_slope(height):
    """Evaluate the equatorial expression and its height derivative in float64."""
    r = 1.0 + height / _A
    u = math.sqrt(r * r - _E2)
    t = _E2 / (u * u)
    q = t * (2.0 / 5.0 + t * (-6.0 / 35.0 + t * (
        2.0 / 21.0 + t * (-2.0 / 33.0 + t * (6.0 / 143.0))
    )))
    dq = 2.0 / 5.0 + t * (-12.0 / 35.0 + t * (
        6.0 / 21.0 + t * (-8.0 / 33.0 + t * (30.0 / 143.0))
    ))
    inv_denom = 1.0 / (u * u + _E2)
    numerator = _G - _H * q / 3.0
    value = numerator * inv_denom - _C * u
    du_dh = r / (u * _A)
    dt_du = -2.0 * t / u
    slope = ((-_H / 3.0 * dq * dt_du) * inv_denom
             - 2.0 * u * numerator * inv_denom * inv_denom - _C) * du_dh
    return value, slope


# Split the equatorial zero into two float32 constants to preserve the small
# distance to the zero when height itself is a large float32 number.
_lo, _hi = 30_000_000.0, 40_000_000.0
for _ in range(60):
    _mid = (_lo + _hi) * 0.5
    if _equatorial_value_and_slope(_mid)[0] > 0.0:
        _lo = _mid
    else:
        _hi = _mid
_ROOT = (_lo + _hi) * 0.5
_ROOT_HI = struct.unpack("f", struct.pack("f", _ROOT))[0]
_ROOT_LO = _ROOT - _ROOT_HI
_ROOT_SLOPE = _equatorial_value_and_slope(_ROOT)[1]


def normal_gravity(latitude, height):
    """Return signed normal gravity in the ellipsoidal u direction, in m/s²."""
    latitude = mx.array(latitude, dtype=mx.float32)
    height = mx.array(height, dtype=mx.float32)
    lat = latitude * (math.pi / 180.0)
    h = height / _A
    sin_lat = mx.sin(lat)
    cos_lat = mx.cos(lat)

    # Normalize before squaring so even the largest finite input height works.
    n = mx.rsqrt(1.0 - _E2 * sin_lat * sin_lat)
    scale = n + h
    z_scaled = (((1.0 - _E2) * n + h) / scale) * sin_lat
    r_scaled = mx.sqrt(cos_lat * cos_lat + z_scaled * z_scaled)
    r = scale * r_scaled
    z_over_r = z_scaled / r_scaled
    inv_r2 = (1.0 / r) * (1.0 / r)
    d = 1.0 - _E2 * inv_r2
    u_over_r = mx.sqrt(0.5 * (d + mx.sqrt(
        d * d + 4.0 * _E2 * inv_r2 * z_over_r * z_over_r
    )))
    u = r * u_over_r
    sin_beta = z_over_r / u_over_r
    sin_beta2 = sin_beta * sin_beta
    inv_u = 1.0 / u
    t = _E2 * inv_u * inv_u
    inv_denom = (inv_u * inv_u) / (1.0 + t)
    w = mx.sqrt((1.0 + t * sin_beta2) / (1.0 + t))

    # q'(u), expanded in (E/u)^2 to avoid cancellation.
    qprime = t * (2.0 / 5.0 + t * (-6.0 / 35.0 + t * (
        2.0 / 21.0 + t * (-2.0 / 33.0 + t * (6.0 / 143.0))
    )))
    gravitational = _G * inv_denom
    harmonic = _H * qprime * inv_denom * (sin_beta2 - 1.0 / 3.0)
    centrifugal = _C * u * (1.0 - sin_beta2)
    gravity = (gravitational + harmonic - centrifugal) / w

    # Near the equatorial zero, evaluate a local Taylor expansion using the
    # input height before dividing by the semi-major axis.
    distance = (height - _ROOT_HI) - _ROOT_LO
    near_equatorial_zero = (latitude == 0.0) & (mx.abs(distance) < 64.0)
    gravity = mx.where(near_equatorial_zero, distance * _ROOT_SLOPE, gravity)
    return {"gravity": gravity}
