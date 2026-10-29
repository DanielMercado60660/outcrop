"""Frozen project 27 API, shared by MLX and Rust/Metal.

All array inputs are C-contiguous NumPy float32 arrays, with finite values.
All outputs are dicts of finite NumPy float32 arrays, including rank-zero arrays
for scalar outputs. No data generation belongs in implementations.
"""

import numpy as np


def ray_matrix(
    x_edges: np.ndarray,
    z_edges: np.ndarray,
    sources: np.ndarray,
    receivers: np.ndarray,
) -> dict[str, np.ndarray]:
    """Integrate straight source-to-receiver segments through a 2-D cell grid.

    x_edges: float32 (nx+1,), strictly increasing cell boundaries in metres.
    z_edges: float32 (nz+1,), strictly increasing boundaries in metres, z down.
    sources, receivers: float32 (nrays, 2), paired (x, z) endpoints in metres,
        all inside the closed grid rectangle. Each row defines one ray, not a
        Cartesian product. The coordinate origin is arbitrary and shared.

    Return {"matrix": float32 (nrays, nz*nx)}: nonnegative path lengths in
    metres, row-major cells j = iz*nx + ix. Travel times are matrix @ slowness.
    Internal boundaries belong to the cell on the positive-coordinate side;
    the outer maximum boundary belongs to the final cell. This convention
    matters for rays lying along a grid line. A zero-length ray is a zero row.
    Oblique crossings have no double counting, including crossings at corners.
    Reversing endpoints must leave the matrix unchanged. No bent-ray tracing,
    eikonal solve, interpolation or quadrature approximation is part of this API.
    """
    raise NotImplementedError


def solve_tomography(
    matrix: np.ndarray,
    travel_times: np.ndarray,
    reference_slowness: np.ndarray,
    damping: float,
) -> dict[str, np.ndarray]:
    """Solve fixed-ray, unweighted linear slowness tomography.

    matrix: float32 (nrays, ncells), nonnegative ray path lengths in metres.
    travel_times: float32 (nrays,), observed one-way travel times in seconds.
    reference_slowness: float32 (ncells,), prior absolute slowness in s/m.
    damping: nonnegative Python float in metres, NOT its square. Minimize

        F(s) = 0.5*||matrix@s - travel_times||_2**2
             + 0.5*damping**2*||s - reference_slowness||_2**2.

    There is no division by nrays, positivity constraint, smoothness penalty,
    weighting, or velocity conversion. Damping=0 fixtures are full column rank;
    positive damping makes even rank-deficient cases uniquely solvable. Solve
    the reference-centered update and return absolute slowness, not the update.

    Return {"slowness": float32 (ncells,) in s/m,
            "predicted_times": float32 (nrays,) in seconds}, where predictions
    equal matrix @ returned slowness. Matrix columns use the ray_matrix order,
    but this solver also accepts arbitrary supplied linear ray matrices.
    """
    raise NotImplementedError


def misfit_and_grad(
    matrix: np.ndarray,
    travel_times: np.ndarray,
    slowness: np.ndarray,
    reference_slowness: np.ndarray,
    damping: float,
) -> dict[str, np.ndarray]:
    """Evaluate the same objective as solve_tomography and its slowness gradient.

    matrix: float32 (nrays, ncells), lengths in metres.
    travel_times: float32 (nrays,), observations in seconds.
    slowness: float32 (ncells,), evaluation point, absolute slowness in s/m.
    reference_slowness: float32 (ncells,), prior absolute slowness in s/m.
    damping: nonnegative Python float in metres, with the exact objective and
        no normalization defined in solve_tomography.

    Return {"objective": float32 rank-zero array in s**2,
            "gradient": float32 (ncells,) in s*m}.
    Gradient is dF/ds = matrix.T@(matrix@s-travel_times)
    + damping**2*(s-reference_slowness). Differentiate with respect to absolute
    slowness, not velocity or the matrix. MLX uses autodiff; Metal a hand adjoint.
    """
    raise NotImplementedError
