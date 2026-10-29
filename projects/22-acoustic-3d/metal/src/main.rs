//! Periodic 3D second-order acoustic propagation.
mod buffers;
mod input;

use outcrop_common::gpu::Arg;
use outcrop_common::{Gpu, Inputs, Outputs, Result};
use std::path::PathBuf;

// Keep the original float32 multiplication for ordinary inputs. If either
// operand would be subnormal or the ratio would overflow, combine the scales
// in float64 on the host and upload float32 Courant values instead. For positive
// float32 velocities satisfying CFL, dt/spacing cannot overflow float64; if it
// underflows float64, even the largest float32 velocity has a negligible update.
fn prepare_courant(velocity: &mut [f32], dt: f64, spacing: f64) -> Result<f32> {
    if velocity.iter().any(|&v| !v.is_finite() || v <= 0.0) {
        return Err("velocity must be finite and positive".into());
    }
    let ratio64 = dt / spacing;
    let ratio = ratio64 as f32;
    if ratio.is_normal() && velocity.iter().all(|v| v.is_normal()) {
        return Ok(ratio);
    }
    if !ratio64.is_finite() {
        return Err("Courant number exceeds the finite float32 range".into());
    }
    for v in velocity {
        let courant = (f64::from(*v) * ratio64) as f32;
        if !courant.is_finite() {
            return Err("Courant number exceeds the finite float32 range".into());
        }
        *v = courant;
    }
    Ok(1.0)
}

fn propagate(inp: &Inputs, out: &mut Outputs) -> Result<()> {
    // The common dispatcher still owns CLI/scalar/output handling. Use its input
    // directory argument here because Inputs does not expose a checked reader.
    let directory = PathBuf::from(std::env::args_os().nth(2).ok_or("missing input directory")?);
    let mut velocity = input::read_f32(&directory.join("velocity.npy"))?;
    let initial = input::read_f32(&directory.join("initial.npy"))?;
    let previous = input::read_f32(&directory.join("previous.npy"))?;
    let shape = &velocity.shape;
    if shape.len() != 3
        || shape.contains(&0)
        || initial.shape != *shape
        || previous.shape != *shape
    {
        return Err("velocity, initial, and previous must have the same nonempty 3D shape".into());
    }
    let n = shape.iter()
        .try_fold(1usize, |a, &b| a.checked_mul(b))
        .ok_or("grid size overflow")?;
    if n > u32::MAX as usize {
        return Err("grid is too large for Metal indexing".into());
    }
    let spacing = inp.num("spacing")?;
    let dt = inp.num("dt")?;
    let steps = inp.int("steps")?;
    if !spacing.is_finite() || spacing <= 0.0 || !dt.is_finite() || dt <= 0.0 || steps <= 0 {
        return Err("spacing, dt, and steps must be positive".into());
    }
    let ratio = prepare_courant(&mut velocity.data, dt, spacing)?;
    let gpu = Gpu::new()?;
    let library = gpu.compile(include_str!("../kernels/propagate.metal"))?;
    let kernel = gpu.pipeline(&library, "step")?;
    let v = buffers::copy_from(&gpu, &velocity.data)?;
    let buffers = [
        buffers::copy_from(&gpu, &previous.data)?,
        buffers::copy_from(&gpu, &initial.data)?,
        buffers::allocate(&gpu, n)?,
    ];
    let (mut prev, mut curr, mut next) = (0usize, 1usize, 2usize);
    gpu.batch(|batch| {
        for _ in 0..steps {
            batch.dispatch(
                &kernel,
                &[
                    Arg::Buf(&v),
                    Arg::Buf(&buffers[prev]),
                    Arg::Buf(&buffers[curr]),
                    Arg::Buf(&buffers[next]),
                    Arg::U32(shape[2] as u32),
                    Arg::U32(shape[1] as u32),
                    Arg::U32(shape[0] as u32),
                    Arg::F32(ratio),
                ],
                [n, 1, 1],
            )?;
            (prev, curr, next) = (curr, next, prev);
        }
        Ok(())
    })?;
    out.f32("final", shape, &buffers::read(&buffers[curr], n)?)?;
    out.f32("last", shape, &buffers::read(&buffers[prev], n)?)?;
    Ok(())
}

fn main() {
    outcrop_common::run(&[("propagate", propagate)]);
}
