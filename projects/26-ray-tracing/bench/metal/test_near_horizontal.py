"""Independent regressions, separate from the locked test suite."""

from decimal import Decimal, localcontext
from math import atan2, copysign
from pathlib import Path

import numpy as np
import pytest

from outcrop_harness import call


def slowness_reference(interfaces, velocities, depth, offsets):
    """Independent Snell solve in horizontal slowness with 60-digit arithmetic.

    High precision lets this reference resolve p below critical incidence even
    where float32 (and, for sufficiently large offsets, float64) cannot.
    """
    times, angles = [], []
    with localcontext() as ctx:
        ctx.prec = 60
        one = Decimal(1)
        depth = Decimal(depth)
        top = Decimal(0)
        layers = []
        for j, velocity in enumerate(velocities):
            bottom = min(depth, Decimal(float(interfaces[j]))) if j < len(interfaces) else depth
            layers.append((bottom - top, Decimal(float(velocity))))
            top = bottom
            if top == depth:
                break

        for offset in offsets:
            target = Decimal(abs(float(offset)))
            lo, hi = Decimal(0), one / max(v for _, v in layers)
            if target:
                for _ in range(256):
                    p = (lo + hi) / 2
                    x = sum(h * p * v / (one - (p * v) ** 2).sqrt() for h, v in layers)
                    if x < target:
                        lo = p
                    else:
                        hi = p
                p = (lo + hi) / 2
            else:
                p = Decimal(0)
            times.append(float(sum(h / (v * (one - (p * v) ** 2).sqrt()) for h, v in layers)))
            s0 = p * layers[0][1]
            angles.append(copysign(atan2(float(s0), float((one - s0 * s0).sqrt())), float(offset)))
    return np.array(times), np.array(angles)


@pytest.mark.parametrize(
    "interfaces,velocities",
    [
        ([], [2200]),
        ([200, 500], [2200, 2200, 2200]),
        ([1500, 3000], [2200, 5000, 10000]),
        ([200, 800, 2000], [1800, 2600, 2200, 9000]),
        ([300, 800], [2600, 1800, 2100]),
        ([300, 800], [2200, 2200.000244140625, 1800]),
    ],
    ids=["empty-interfaces", "equal-speeds", "untraversed-fast", "buried-fast", "surface-fast", "close-speeds"],
)
def test_near_horizontal_metal(interfaces, velocities, monkeypatch):
    monkeypatch.setenv("OUTCROP_PROJECT", str(Path(__file__).resolve().parents[2]))
    interfaces = np.array(interfaces, dtype=np.float32)
    velocities = np.array(velocities, dtype=np.float32)
    offsets = np.array([-1e10, -1e7, -1000, -0.001, 0, 0.001, 1000, 1e7, 1e10], dtype=np.float32)
    depth = 1000.0
    expected_times, expected_angles = slowness_reference(interfaces, velocities, depth, offsets)
    actual = call("metal", "bend_layered", interfaces=interfaces, velocities=velocities, offsets=offsets, depth=depth)
    assert set(actual) == {"endpoints", "times", "angles"}
    for name, shape in [("endpoints", (len(offsets), 2)), ("times", offsets.shape), ("angles", offsets.shape)]:
        assert actual[name].dtype == np.float32
        assert actual[name].shape == shape
        assert np.all(np.isfinite(actual[name]))
    np.testing.assert_array_equal(actual["endpoints"][:, 0], offsets)
    np.testing.assert_array_equal(actual["endpoints"][:, 1], np.full_like(offsets, depth))
    np.testing.assert_allclose(actual["times"], expected_times, rtol=1e-4, atol=0)
    np.testing.assert_allclose(actual["angles"], expected_angles, rtol=0, atol=1e-4)
