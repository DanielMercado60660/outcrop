//! Newtonian gravity of point masses and homogeneous spheres.
use outcrop_common::gpu::Arg;
use outcrop_common::{Gpu, Inputs, Outputs, Result};

fn gravity(inp: &Inputs, out: &mut Outputs) -> Result<()> {
    let coordinates = inp.f32("coordinates")?;
    let centers = inp.f32("centers")?;
    let masses = inp.f32("masses")?;
    let radii = inp.f32("radii")?;
    if coordinates.shape.last() != Some(&3) {
        return Err("coordinates must have shape (..., 3)".into());
    }
    if centers.shape.len() != 2 || centers.shape[1] != 3 {
        return Err("centers must have shape (S, 3)".into());
    }
    let sources = centers.shape[0];
    if masses.shape != [sources] || radii.shape != [sources] {
        return Err("masses and radii must have shape (S,)".into());
    }
    // Flatten only for dispatch: preserve every leading observation dimension,
    // including no leading dimensions (one position, scalar potential).
    let observations = coordinates.data.len() / 3;
    let n = u32::try_from(observations)?;
    let s = u32::try_from(sources)?;
    let mut potential = vec![0.0f32; observations];
    let mut acceleration = vec![0.0f32; observations * 3];
    // Empty grids and empty source sets have correctly shaped zero outputs and
    // need neither a Metal device nor a zero-sized dispatch.
    if observations != 0 && sources != 0 {
        let gpu = Gpu::new()?;
        let library = gpu.compile(include_str!("../kernels/gravity.metal"))?;
        let kernel = gpu.pipeline(&library, "gravity")?;
        let xyz_buffer = gpu.buffer_from(&coordinates.data);
        let centers_buffer = gpu.buffer_from(&centers.data);
        let masses_buffer = gpu.buffer_from(&masses.data);
        let radii_buffer = gpu.buffer_from(&radii.data);
        let potential_buffer = gpu.buffer_zeroed::<f32>(observations);
        let acceleration_buffer = gpu.buffer_zeroed::<f32>(observations * 3);
        gpu.dispatch(
            &kernel,
            &[
                Arg::Buf(&xyz_buffer),
                Arg::Buf(&centers_buffer),
                Arg::Buf(&masses_buffer),
                Arg::Buf(&radii_buffer),
                Arg::Buf(&potential_buffer),
                Arg::Buf(&acceleration_buffer),
                Arg::U32(n),
                Arg::U32(s),
            ],
            [observations, 1, 1],
        )?;
        potential = gpu.read(&potential_buffer, observations);
        acceleration = gpu.read(&acceleration_buffer, observations * 3);
    }
    out.f32(
        "potential",
        &coordinates.shape[..coordinates.shape.len() - 1],
        &potential,
    )?;
    out.f32("acceleration", &coordinates.shape, &acceleration)?;
    Ok(())
}

fn main() {
    outcrop_common::run(&[("gravity", gravity)]);
}
