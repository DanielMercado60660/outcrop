"""Contract for terrain and Bouguer corrections."""


def terrain_bouguer(easting, northing, surface, reference, density, coordinates):
    """Compute the signed DEM prism layer and its Bouguer slab approximation.

    All arrays are float32. ``easting`` (nx,) and ``northing`` (ny,) are
    regularly spaced ascending cell centers in metres; both have at least two
    entries. ``surface``, ``reference``, and ``density`` are (ny,nx) arrays
    in metres upward, metres upward, and kg/m^3 respectively. Cells extend
    half a grid spacing from each center. Density is a positive contrast;
    cells below reference have negative effective density and zero thickness
    cells contribute zero. ``coordinates`` is (n,3), easting, northing,
    upward height in metres. Observations are strictly above the entire layer.

    Return float32 arrays (n,) in a dict: ``layer_gz``, the downward gravity
    in mGal from the finite signed prism layer; ``slab_gz``, the infinite
    Bouguer slab approximation 2*pi*G*density*(surface-reference)*1e5,
    evaluated using the cell under each observation; ``terrain_gz`` equal to
    layer_gz minus slab_gz; and ``correction_gz`` equal to minus layer_gz.
    Observations are within the horizontal extent of the grid. Cell selection
    uses nearest center (no observations lie on cell boundaries). G is
    6.67430e-11 m^3 kg^-1 s^-2. The correction is added to observed gravity.
    """
    raise NotImplementedError
