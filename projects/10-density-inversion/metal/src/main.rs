//! Density inversion on Metal. The host only validates, transfers, and dispatches.
//! All operator, regularization, adjoint, and linear-solve arithmetic is in MSL.

use outcrop_common::gpu::{Arg, Buffer, Library};
use outcrop_common::{Array, Gpu, Inputs, Outputs, Result};

fn main() {
    outcrop_common::run(&[
        ("apply_operator", apply_operator),
        ("misfit_and_grad", misfit_and_grad),
        ("invert", invert),
    ]);
}

fn checked_array(inp: &Inputs, name: &str, shape: &[usize]) -> Result<Array<f32>> {
    let a = inp.f32(name)?;
    if a.shape != shape || a.data.iter().any(|x| !x.is_finite()) {
        return Err(format!("{name}: expected finite array of shape {shape:?}").into());
    }
    Ok(a)
}

fn sensitivity(inp: &Inputs) -> Result<Array<f32>> {
    let g = inp.f32("sensitivity")?;
    if g.shape.len() != 2
        || g.shape.contains(&0)
        || g.shape.iter().any(|&n| n > u32::MAX as usize)
        || g.data.len() > u32::MAX as usize
        || g.data.iter().any(|x| !x.is_finite())
    {
        return Err("sensitivity must be a finite, nonempty matrix with uint32 indices".into());
    }
    Ok(g)
}

fn library(gpu: &Gpu) -> Result<Library> {
    gpu.compile(include_str!("../kernels/density.metal"))
}

fn apply_operator(inp: &Inputs, out: &mut Outputs) -> Result<()> {
    let g = sensitivity(inp)?;
    let (nd, nc) = (g.shape[0], g.shape[1]);
    let transpose = inp.has_arg("transpose") && inp.flag("transpose")?;
    let (ni, no) = if transpose { (nd, nc) } else { (nc, nd) };
    let v = checked_array(inp, "vector", &[ni])?;
    let gpu = Gpu::new()?;
    let lib = library(&gpu)?;
    let kernel = gpu.pipeline(&lib, "apply_matrix")?;
    let bg = gpu.buffer_from(&g.data);
    let bv = gpu.buffer_from(&v.data);
    let result = gpu.buffer_zeroed::<f32>(no);
    gpu.dispatch(
        &kernel,
        &[
            Arg::Buf(&bg), Arg::Buf(&bv), Arg::Buf(&result),
            Arg::U32(nd as u32), Arg::U32(nc as u32), Arg::U32(u32::from(transpose)),
        ],
        [no, 1, 1],
    )?;
    out.f32("result", &[no], &gpu.read::<f32>(&result, no))
}

// Split Python scalar parameters without reducing their precision before transfer.
// Each MSL float2 is a high/low pair; its arithmetic uses float32 operations only.
fn pair(x: f64) -> [f32; 2] {
    let hi = x as f32;
    [hi, (x - hi as f64) as f32]
}

struct Problem {
    nd: usize,
    nc: usize,
    g: Buffer,
    data: Buffer,
    sigma: Buffer,
    reference: Buffer,
    dims: Buffer,
    scalars: Buffer,
    alpha: Buffer,
    weights: Buffer,
}

impl Problem {
    fn load(inp: &Inputs, gpu: &Gpu, lib: &Library) -> Result<Self> {
        let g = sensitivity(inp)?;
        let (nd, nc) = (g.shape[0], g.shape[1]);
        let data = checked_array(inp, "data", &[nd])?;
        let sigma = checked_array(inp, "uncertainty", &[nd])?;
        if sigma.data.iter().any(|&s| s <= 0.0) {
            return Err("uncertainty must be strictly positive".into());
        }
        let reference = checked_array(inp, "reference", &[nc])?;
        let alpha = checked_array(inp, "alpha_xyz", &[3])?;
        if alpha.data.iter().any(|&a| a < 0.0) {
            return Err("alpha_xyz must be nonnegative".into());
        }
        let shape = inp.nums("shape")?;
        if shape.len() != 3
            || shape.iter().any(|&n| n < 1.0 || n.fract() != 0.0 || n > u32::MAX as f64)
        {
            return Err("shape must contain three positive integers".into());
        }
        let mesh: Vec<usize> = shape.iter().map(|&n| n as usize).collect();
        if mesh.iter().try_fold(1usize, |p, &n| p.checked_mul(n)) != Some(nc) {
            return Err("shape product differs from sensitivity column count".into());
        }
        let h = inp.num("cell_size")?;
        let offset = inp.num("depth_offset")?;
        let beta = inp.num("beta")?;
        if !h.is_finite() || h <= 0.0 || !offset.is_finite() || offset <= 0.0
            || !beta.is_finite() || beta < 0.0
            || !(beta as f32).is_finite()
        {
            return Err("cell_size/depth_offset must be positive; beta nonnegative; all finite".into());
        }
        let dims = [nd as u32, nc as u32, mesh[0] as u32, mesh[1] as u32, mesh[2] as u32];
        // Only their ratio enters the depth weights. Scale before converting
        // to float32 pairs so lengths below float32 range (e.g. 1e-50 m)
        // cannot both round to zero. One scaled length is exactly 1; depths
        // are bounded by nz rather than the dimensional length's magnitude.
        let length_scale = h.max(offset);
        let scalars = [pair(h / length_scale), pair(offset / length_scale), pair(beta)];
        let p = Self {
            nd, nc,
            g: gpu.buffer_from(&g.data), data: gpu.buffer_from(&data.data),
            sigma: gpu.buffer_from(&sigma.data), reference: gpu.buffer_from(&reference.data),
            dims: gpu.buffer_from(&dims), scalars: gpu.buffer_from(&scalars),
            alpha: gpu.buffer_from(&alpha.data), weights: gpu.buffer_zeroed::<[f32; 2]>(nc),
        };
        gpu.dispatch(
            &gpu.pipeline(lib, "depth_weights")?,
            &[Arg::Buf(&p.dims), Arg::Buf(&p.scalars), Arg::Buf(&p.weights)],
            [nc, 1, 1],
        )?;
        Ok(p)
    }
}

fn misfit_and_grad(inp: &Inputs, out: &mut Outputs) -> Result<()> {
    let gpu = Gpu::new()?;
    let lib = library(&gpu)?;
    let p = Problem::load(inp, &gpu, &lib)?;
    let model = checked_array(inp, "model", &[p.nc])?;
    let model = gpu.buffer_from(&model.data);
    let delta = gpu.buffer_zeroed::<[f32; 2]>(p.nc);
    let dual = gpu.buffer_zeroed::<[f32; 2]>(p.nd);
    let data_terms = gpu.buffer_zeroed::<[f32; 2]>(p.nd);
    let reg_terms = gpu.buffer_zeroed::<[f32; 2]>(p.nc);
    let reg_action = gpu.buffer_zeroed::<[f32; 2]>(p.nc);
    let gradient = gpu.buffer_zeroed::<f32>(p.nc);
    let weights = gpu.buffer_zeroed::<f32>(p.nc);
    let objectives = gpu.buffer_zeroed::<f32>(3);
    let kd = gpu.pipeline(&lib, "model_delta")?;
    let kdata = gpu.pipeline(&lib, "data_objective")?;
    let kr = gpu.pipeline(&lib, "regularization")?;
    let kg = gpu.pipeline(&lib, "hand_gradient")?;
    let ks = gpu.pipeline(&lib, "sum_objective")?;
    gpu.batch(|b| {
        b.dispatch(&kd, &[Arg::Buf(&model), Arg::Buf(&p.reference), Arg::Buf(&delta),
            Arg::U32(p.nc as u32)], [p.nc, 1, 1])?;
        b.dispatch(&kdata, &[Arg::Buf(&p.g), Arg::Buf(&model), Arg::Buf(&p.data),
            Arg::Buf(&p.sigma), Arg::Buf(&p.dims), Arg::Buf(&dual), Arg::Buf(&data_terms)],
            [p.nd, 1, 1])?;
        b.dispatch(&kr, &[Arg::Buf(&delta), Arg::Buf(&p.weights), Arg::Buf(&p.dims),
            Arg::Buf(&p.alpha), Arg::Buf(&reg_action), Arg::Buf(&reg_terms)], [p.nc, 1, 1])?;
        b.dispatch(&kg, &[Arg::Buf(&p.g), Arg::Buf(&dual), Arg::Buf(&reg_action),
            Arg::Buf(&p.weights), Arg::Buf(&p.dims), Arg::Buf(&p.scalars),
            Arg::Buf(&gradient), Arg::Buf(&weights)], [p.nc, 1, 1])?;
        b.dispatch(&ks, &[Arg::Buf(&data_terms), Arg::Buf(&reg_terms), Arg::Buf(&p.dims),
            Arg::Buf(&p.scalars), Arg::Buf(&objectives)], [1, 1, 1])
    })?;
    let values = gpu.read::<f32>(&objectives, 3);
    out.f32("objective", &[], &values[0..1])?;
    out.f32("phi_d", &[], &values[1..2])?;
    out.f32("phi_m", &[], &values[2..3])?;
    out.f32("gradient", &[p.nc], &gpu.read::<f32>(&gradient, p.nc))?;
    out.f32("weights", &[p.nc], &gpu.read::<f32>(&weights, p.nc))
}

fn invert(inp: &Inputs, out: &mut Outputs) -> Result<()> {
    let gpu = Gpu::new()?;
    let lib = library(&gpu)?;
    let p = Problem::load(inp, &gpu, &lib)?;
    // Only inversion allocates and indexes an nc-by-nc dense normal matrix.
    // The objective and hand adjoint need only linear-size model buffers.
    if p.nc.checked_mul(p.nc).is_none_or(|n| n > u32::MAX as usize) {
        return Err("normal matrix exceeds uint32 indexing".into());
    }
    // Solve for delta = model - reference:
    // (G^T U^-2 G + beta R) delta = G^T U^-2 (data - G reference).
    // Pair accumulation preserves float32 output accuracy through conditioning.
    // No damping, positivity bounds, or beta fitting.
    let shifted_dual = gpu.buffer_zeroed::<[f32; 2]>(p.nd);
    let factor = gpu.buffer_zeroed::<[f32; 2]>(p.nc * p.nc);
    let rhs = gpu.buffer_zeroed::<[f32; 2]>(p.nc);
    let work = gpu.buffer_zeroed::<[f32; 2]>(p.nc);
    let delta = gpu.buffer_zeroed::<[f32; 2]>(p.nc);
    let model = gpu.buffer_zeroed::<f32>(p.nc);
    let predicted = gpu.buffer_zeroed::<f32>(p.nd);
    let kshift = gpu.pipeline(&lib, "shifted_data")?;
    let ksystem = gpu.pipeline(&lib, "normal_system")?;
    let kpivot = gpu.pipeline(&lib, "cholesky_pivot")?;
    let kcolumn = gpu.pipeline(&lib, "cholesky_column")?;
    let ksolve = gpu.pipeline(&lib, "triangular_solve")?;
    let kfinish = gpu.pipeline(&lib, "finish_model")?;
    let kpredict = gpu.pipeline(&lib, "apply_matrix")?;
    gpu.batch(|b| {
        b.dispatch(&kshift, &[Arg::Buf(&p.g), Arg::Buf(&p.data), Arg::Buf(&p.sigma),
            Arg::Buf(&p.reference), Arg::Buf(&p.dims), Arg::Buf(&shifted_dual)], [p.nd, 1, 1])?;
        b.dispatch(&ksystem, &[Arg::Buf(&p.g), Arg::Buf(&p.sigma), Arg::Buf(&p.weights),
            Arg::Buf(&p.dims), Arg::Buf(&p.scalars), Arg::Buf(&p.alpha), Arg::Buf(&shifted_dual),
            Arg::Buf(&factor), Arg::Buf(&rhs)], [p.nc, p.nc, 1])?;
        for k in 0..p.nc {
            b.dispatch(&kpivot, &[Arg::Buf(&factor), Arg::U32(p.nc as u32), Arg::U32(k as u32)],
                [1, 1, 1])?;
            if k + 1 < p.nc {
                b.dispatch(&kcolumn, &[Arg::Buf(&factor), Arg::U32(p.nc as u32), Arg::U32(k as u32)],
                    [p.nc - k - 1, 1, 1])?;
            }
        }
        b.dispatch(&ksolve, &[Arg::Buf(&factor), Arg::Buf(&rhs), Arg::Buf(&work),
            Arg::Buf(&delta), Arg::U32(p.nc as u32)], [1, 1, 1])?;
        b.dispatch(&kfinish, &[Arg::Buf(&delta), Arg::Buf(&p.reference), Arg::Buf(&model),
            Arg::U32(p.nc as u32)], [p.nc, 1, 1])?;
        // Prediction uses the rounded, returned model, matching the public API.
        b.dispatch(&kpredict, &[Arg::Buf(&p.g), Arg::Buf(&model), Arg::Buf(&predicted),
            Arg::U32(p.nd as u32), Arg::U32(p.nc as u32), Arg::U32(0)], [p.nd, 1, 1])
    })?;
    let model = gpu.read::<f32>(&model, p.nc);
    let predicted = gpu.read::<f32>(&predicted, p.nd);
    if model.iter().chain(&predicted).any(|x| !x.is_finite()) {
        return Err("normal system is not numerically positive definite".into());
    }
    out.f32("model", &[p.nc], &model)?;
    out.f32("predicted", &[p.nd], &predicted)
}
