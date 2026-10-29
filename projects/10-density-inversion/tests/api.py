"""Frozen array API for both paths (MLX autodiff, Metal hand adjoint).

All array inputs/outputs are contiguous NumPy float32. Integers, bools, scalar
floats and lists are ordinary Python values. Scalar outputs have shape ().
No file access, random data generation, or reference libraries in implementations.
The dense sensitivity comes from dependency 02, frozen here using SimPEG; this
project does not reimplement prism quadrature. The mesh is uniform, all cells
active. ENU coordinates, z up, mesh top at z=0, bottom at -nz*cell_size.
Flatten (nx,ny,nz) in Fortran order: x fastest, then y, then z (bottom to top).
Density contrast is g/cm^3; data are upward gz in mGal (1 mGal=1e-5 m/s^2).
Positive density below a receiver produces negative gz.
"""


def apply_operator(sensitivity, vector, transpose=False):
    """Apply the fixed dependency-02 forward operator or its exact transpose.

    sensitivity: float32 (n_data,n_cells), mGal/(g/cm^3), receiver rows and
        x-fastest model columns. No noise or uncertainty scaling in this call.
    vector: float32 (n_cells,) density g/cm^3 when transpose=False;
        float32 (n_data,) arbitrary data-space dual when transpose=True.
    transpose: Python bool, default False; True applies G.T, without weights.
    Returns {"result": float32 (n_data,) or (n_cells,)} respectively.
    """
    raise NotImplementedError


def misfit_and_grad(sensitivity, data, uncertainty, model, reference, shape,
                    cell_size, depth_offset, alpha_xyz, beta):
    r"""Return the full quadratic objective, its model gradient and depth weights.

    sensitivity: float32 (n_data,n_cells), same conventions as apply_operator.
    data: float32 (n_data,), observed gz, mGal.
    uncertainty: float32 (n_data,), strictly positive one-sigma errors, mGal.
    model, reference: float32 (n_cells,), density contrast, g/cm^3.
    shape: Python list [nx,ny,nz] of ints, product equals n_cells.
    cell_size: positive Python float, uniform cubic cell width, metres.
    depth_offset: positive Python float, metres; depth-weight threshold.
    alpha_xyz: float32 (3,), nonnegative dimensionless x,y,z smoothing strengths.
    beta: nonnegative Python float, regularization multiplier (fixed, not fitted).

    Cell depth d_k=(nz-k-0.5)*cell_size, k=0 at the bottom. Normalized
    depth weights w_i=(cell_size/2+depth_offset)/(d_i+depth_offset).
    This is SimPEG depth_weighting(exponent=2, reference_locs=0).
    Put delta=model-reference. With each unordered pair of face-adjacent
    cells (i,j) counted once in axis a:

      phi_d = sum(((G@model-data)/uncertainty)**2)
      phi_m = sum(w_i**2*delta_i**2)
            + sum_a alpha_xyz[a]*sum_pairs (w_i**2+w_j**2)/2*(delta_j-delta_i)**2
      objective = phi_d + beta*phi_m

    No exterior-face penalties (homogeneous Neumann), no cross-axis edges,
    no bounds, no half factors. Reference is subtracted in BOTH smallness
    and smoothness. This equals SimPEG WeightedLeastSquares divided by
    cell_size**3, alpha_s=1, alpha_a=cell_size**2*alpha_xyz[a], second-order
    strengths zero, weights={'depth': w**2}, reference_model_in_smooth=True.
    The normalization removes a common volume factor; beta uses these units.

    Returns dict: 'objective', 'phi_d', 'phi_m' float32 scalar arrays;
    'gradient' float32 (n_cells,) derivative with respect to density;
    'weights' float32 (n_cells,) dimensionless w (NOT w squared).
    MLX uses autodiff; Metal uses the explicit hand adjoint.
    """
    raise NotImplementedError


def invert(sensitivity, data, uncertainty, reference, shape, cell_size,
           depth_offset, alpha_xyz, beta):
    """Minimize exactly the unconstrained objective specified by misfit_and_grad.

    Inputs and units match misfit_and_grad, excluding model. In particular
    beta is supplied by the judge's discrepancy-principle search; do not tune
    beta, impose positivity, or normalize data/model internally in the result.
    Use any convergent direct or iterative solver for the unique SPD problem.
    Return {'model': float32 (n_cells,) in g/cm^3,
            'predicted': float32 (n_data,) G@model in mGal}.
    The reference is a penalty center, not an initial model or fixed background;
    data prediction always uses the full model. There are no random starts.
    """
    raise NotImplementedError
