use outcrop_common::{gpu::Arg, Gpu, Inputs, Outputs, Result};

mod checked_npy;

fn prism_gravity(inp: &Inputs, out: &mut Outputs) -> Result<()> {
    // Inputs does not expose its directory. The shared CLI has already checked
    // argv; use its input-directory argument with our checked local NPY reader.
    let input_dir = std::env::args_os().nth(2).ok_or("missing input directory")?;
    let input_dir = std::path::Path::new(&input_dir);
    let coordinates = checked_npy::read_f32(input_dir, "coordinates")?;
    let prisms = checked_npy::read_f32(input_dir, "prisms")?;
    let density = checked_npy::read_f32(input_dir, "density")?;
    let field = inp.str("field")?;
    let mode = match field.as_str() {
        "potential" => 0, "g_e" => 1, "g_n" => 2, "g_z" => 3,
        _ => return Err(format!("unknown field: {field}").into()),
    };
    if coordinates.shape.len() != 2 || coordinates.shape[1] != 3 || coordinates.shape[0] == 0 {
        return Err("coordinates must have shape (N, 3), N > 0".into());
    }
    if prisms.shape.len() != 2 || prisms.shape[1] != 6 || prisms.shape[0] == 0 {
        return Err("prisms must have shape (P, 6), P > 0".into());
    }
    let n = coordinates.shape[0];
    let p = prisms.shape[0];
    if density.shape != [p] || n > u32::MAX as usize || p > u32::MAX as usize {
        return Err("invalid density shape or array too large".into());
    }
    let gpu = Gpu::new()?;
    let lib = gpu.compile(include_str!("../kernels/prism.metal"))?;
    let pipeline = gpu.pipeline(&lib, "prism_gravity")?;
    let xyz = gpu.buffer_from(&coordinates.data);
    let bounds = gpu.buffer_from(&prisms.data);
    let rho = gpu.buffer_from(&density.data);
    let values = gpu.buffer_zeroed::<f32>(n);
    gpu.dispatch(&pipeline, &[Arg::Buf(&xyz), Arg::Buf(&bounds), Arg::Buf(&rho),
        Arg::Buf(&values), Arg::U32(n as u32), Arg::U32(p as u32), Arg::U32(mode)], [n, 1, 1])?;
    out.f32(&field, &[n], &gpu.read::<f32>(&values, n))
}

fn main() {
    outcrop_common::run(&[("prism_gravity", prism_gravity)]);
}
