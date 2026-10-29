"""Check the signed layer's surface derivative at zero relief.

Run on an Apple silicon GPU with ``python mlx/check_flat_gradient.py``.
"""

import mlx.core as mx
import numpy as np
from impl import terrain_bouguer


def main():
    centers = np.array([0.0, 100.0], dtype=np.float32)
    reference = np.zeros((2, 2), dtype=np.float32)
    density = np.full((2, 2), 2670.0, dtype=np.float32)
    direction = mx.array([[1.0, 0.0], [0.0, 0.0]], dtype=mx.float32)

    for height in (150.0, 500.0):
        coordinates = np.array([[0.0, 0.0, height]], dtype=np.float32)

        def gravity(surface, observation=coordinates):
            return terrain_bouguer(
                centers, centers, surface, reference, density, observation
            )["layer_gz"].sum()

        flat = mx.zeros((2, 2), dtype=mx.float32)
        gradient = mx.grad(gravity)(flat)
        step = 1.0
        finite_difference = (gravity(flat + step * direction)
                             - gravity(flat - step * direction)) / (2.0 * step)
        actual = float(gradient[0, 0].item())
        expected = float(finite_difference.item())
        print(f"height={height:g} m: autodiff={actual:.8g}, finite difference={expected:.8g} mGal/m")
        assert expected > 0.0
        np.testing.assert_allclose(actual, expected, rtol=0.03, atol=1e-7)


if __name__ == "__main__":
    main()
