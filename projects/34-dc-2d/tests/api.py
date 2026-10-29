"""Public contract for the 2.5D DC resistivity forward solver."""


def dc_forward(hx, hz, origin, conductivity, ab, mn, currents, ky, weights):
    """Solve point-electrode DC on a cell-centered x-z finite-volume mesh.

    All arrays are float32. Coordinates and dimensions are in metres; x is
    horizontal and z increases upwards, with the flat surface at z=0. The
    conductivity is invariant along the horizontal out-of-plane y axis.

    Args:
        hx: (nx,) positive cell widths along x, ordered left to right, m.
        hz: (nz,) positive cell widths along z, ordered bottom to top, m.
        origin: (2,) x,z coordinates of the lower-left mesh node, m.
        conductivity: (nx, nz) cell conductivities in S/m, x first.
        ab: (n, 2) x coordinates of source electrodes A and B, m; both
            lie at (x, y, z)=(x, 0, 0). A injects and B withdraws current.
        mn: (n, 2) x coordinates of potential electrodes M and N, m; both
            lie at y=z=0. Row i belongs to source row i.
        currents: (n,) source current magnitudes in amperes.
        ky: (nk,) positive Fourier wavenumbers in inverse metres.
        weights: (nk,) inverse-transform quadrature weights in inverse metres,
            including the 1/pi normalization.

    Returns:
        Dict of float32 arrays, each (n,): 'voltage' is phi(M)-phi(N) in V;
        'apparent_resistivity' is voltage / (current * G) in ohm-m, where
        G=(1/|A-M|-1/|B-M|-1/|A-N|+1/|B-N|)/(2*pi).

    Use SimPEG's cell-centered 2.5D finite-volume discretization: sources
    inject at the closest cell centers; receiver potentials use multilinear
    cell-center interpolation, clamped at the surface. Apply zero normal flux
    at the top and its Robin condition on sides and bottom. The Robin decay is
    ky*K1(ky*r)/K0(ky*r) times the boundary-normal projection of the radial
    unit vector from the midpoint of the mesh top. Integrate Fourier modes
    with the supplied weights. Resistivity on faces uses the finite-volume
    resistivity inner product (harmonic conductivity across interfaces).
    """
    raise NotImplementedError
