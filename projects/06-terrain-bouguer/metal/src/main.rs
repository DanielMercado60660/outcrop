use outcrop_common::gpu::{Arg, Buffer};
use outcrop_common::{Gpu, Inputs, Outputs, Result};

fn buffer_from<T: Copy>(gpu: &Gpu, data: &[T]) -> Result<Buffer> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| gpu.buffer_from(data)))
        .map_err(|_| "Metal buffer allocation failed".into())
}

fn buffer_zeroed(gpu: &Gpu, n: usize) -> Result<Buffer> {
    n.checked_mul(std::mem::size_of::<f32>()).ok_or("buffer size overflow")?;
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| gpu.buffer_zeroed::<f32>(n)))
        .map_err(|_| "Metal buffer allocation failed".into())
}

fn terrain_bouguer(inp: &Inputs, out: &mut Outputs) -> Result<()> {
    let x = inp.f32("easting")?;
    let y = inp.f32("northing")?;
    let s = inp.f32("surface")?;
    let r = inp.f32("reference")?;
    let d = inp.f32("density")?;
    let c = inp.f32("coordinates")?;
    if x.shape.len() != 1 || y.shape.len() != 1 || x.data.len() < 2 || y.data.len() < 2 {
        return Err("invalid grid axes".into());
    }
    let (nx, ny) = (x.data.len(), y.data.len());
    if s.shape != [ny, nx] || r.shape != [ny, nx] || d.shape != [ny, nx] {
        return Err("grid array shapes do not match axes".into());
    }
    if c.shape.len() != 2 || c.shape[1] != 3 { return Err("coordinates must have shape (n, 3)".into()); }
    let n = c.shape[0];
    if nx > i32::MAX as usize
        || ny > i32::MAX as usize
        || n > (u32::MAX as usize / 3)
        || nx.checked_mul(ny).is_none_or(|v| v > u32::MAX as usize)
    {
        return Err("input too large for Metal indices".into());
    }
    let gpu = Gpu::new()?;
    let lib = gpu.compile(include_str!("../kernels/terrain.metal"))?;
    let k = gpu.pipeline(&lib, "terrain_bouguer")?;
    let bx = buffer_from(&gpu, &x.data)?;
    let by = buffer_from(&gpu, &y.data)?;
    let bs = buffer_from(&gpu, &s.data)?;
    let br = buffer_from(&gpu, &r.data)?;
    let bd = buffer_from(&gpu, &d.data)?;
    let bc = buffer_from(&gpu, &c.data)?;
    let bl = buffer_zeroed(&gpu, n)?;
    let bb = buffer_zeroed(&gpu, n)?;
    let bt = buffer_zeroed(&gpu, n)?;
    let bf = buffer_zeroed(&gpu, n)?;
    if n != 0 {
        gpu.dispatch(&k, &[
            Arg::Buf(&bx), Arg::Buf(&by), Arg::Buf(&bs), Arg::Buf(&br),
            Arg::Buf(&bd), Arg::Buf(&bc), Arg::Buf(&bl), Arg::Buf(&bb),
            Arg::Buf(&bt), Arg::Buf(&bf), Arg::U32(nx as u32), Arg::U32(ny as u32),
            Arg::U32(n as u32),
        ], [n, 1, 1])?;
    }
    out.f32("layer_gz", &[n], &gpu.read::<f32>(&bl, n))?;
    out.f32("slab_gz", &[n], &gpu.read::<f32>(&bb, n))?;
    out.f32("terrain_gz", &[n], &gpu.read::<f32>(&bt, n))?;
    out.f32("correction_gz", &[n], &gpu.read::<f32>(&bf, n))?;
    Ok(())
}

fn main() { outcrop_common::run(&[("terrain_bouguer", terrain_bouguer)]); }
