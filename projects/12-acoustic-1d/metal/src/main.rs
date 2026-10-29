use outcrop_common::gpu::Arg;
use outcrop_common::{Gpu, Inputs, Outputs, Result};

mod buffer;

fn propagate(inp: &Inputs, out: &mut Outputs) -> Result<()> {
    let initial = inp.f32("u0")?;
    let previous = inp.f32("u_prev")?;
    if initial.shape.len() != 1 || initial.shape != previous.shape {
        return Err("u0 and u_prev must have the same one-dimensional shape".into());
    }
    let n = initial.data.len();
    let steps = usize::try_from(inp.int("steps")?).map_err(|_| "steps must be nonnegative")?;
    let c = inp.num("c")?;
    let dx = inp.num("dx")?;
    let dt = inp.num("dt")?;
    let r = c * dt / dx;
    if !c.is_finite()
        || !dx.is_finite()
        || !dt.is_finite()
        || c <= 0.0
        || dx <= 0.0
        || dt <= 0.0
        || !(r > 0.0 && r <= 1.0)
    {
        return Err("expected positive finite c, dx, dt and Courant number in (0, 1]".into());
    }
    let rows = steps.checked_add(1).ok_or("too many steps")?;
    let count = rows.checked_mul(n).ok_or("wavefield too large")?;
    if steps == 0 {
        // Preserve row zero exactly, including for empty and short grids.
        return out.f32("wavefield", &[rows, n], &initial.data);
    }
    if n < 3 {
        return Err("at least 3 points are needed for the Mur boundary update".into());
    }
    let n32 = u32::try_from(n).map_err(|_| "grid too large for Metal")?;
    if count > u32::MAX as usize {
        return Err("wavefield too large for Metal indexing".into());
    }
    let gpu = Gpu::new()?;
    let library = gpu.compile(include_str!("../kernels/acoustic.metal"))?;
    let kernel = gpu.pipeline(&library, "advance")?;
    // Every later row is fully written by advance before it is read.
    let history = buffer::allocate_f32(&gpu, count, "wavefield")?;
    gpu.write(&history, &initial.data);
    // The API supplies t=-dt explicitly; never reconstruct it from u0.
    let prev = buffer::allocate_f32(&gpu, n, "u_prev")?;
    gpu.write(&prev, &previous.data);
    // Scalars arrive as f64. Compute coefficients before rounding to f32.
    let r2 = (r * r) as f32;
    let q = ((r - 1.0) / (r + 1.0)) as f32;
    gpu.batch(|batch| {
        for step in 0..steps {
            // Each ordered dispatch reads completed rows and writes a new row.
            // count <= u32::MAX and n >= 3 also bound step and row offsets.
            batch.dispatch(
                &kernel,
                &[
                    Arg::Buf(&history),
                    Arg::Buf(&prev),
                    Arg::U32(n32),
                    Arg::U32(step as u32),
                    Arg::F32(r2),
                    Arg::F32(q),
                ],
                [n, 1, 1],
            )?;
        }
        Ok(())
    })?;
    out.f32(
        "wavefield",
        &[rows, n],
        &gpu.read::<f32>(&history, count),
    )
}

fn main() {
    outcrop_common::run(&[("propagate", propagate)]);
}
