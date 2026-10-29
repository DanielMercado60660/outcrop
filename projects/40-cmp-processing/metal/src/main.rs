use outcrop_common::{gpu::Arg, Gpu, Inputs, Outputs, Result};
const KERNELS: &str = include_str!("../kernels/cmp.metal");

fn dimensions(g: &outcrop_common::Array<f32>, o: &outcrop_common::Array<f32>) -> Result<(usize, usize)> {
    if g.shape.len() != 2 || o.shape.len() != 1 || o.shape[0] != g.shape[0] { return Err("incompatible shapes".into()); }
    Ok((g.shape[0], g.shape[1]))
}
fn correct_or_stack(inp: &Inputs, out: &mut Outputs, stack: bool) -> Result<()> {
    let g = inp.f32("gather")?;
    let o = inp.f32("offsets")?;
    let (no, nt) = dimensions(&g, &o)?;
    let dt = inp.num("dt")? as f32;
    let velocity = inp.num("velocity")? as f32;
    let n = if stack { nt } else { no * nt };
    let gpu = Gpu::new()?;
    let lib = gpu.compile(KERNELS)?;
    let pipeline = gpu.pipeline(&lib, if stack { "stack_kernel" } else { "nmo_kernel" })?;
    let bg = gpu.buffer_from(&g.data);
    let bo = gpu.buffer_from(&o.data);
    let result = gpu.buffer_zeroed::<f32>(n);
    if n > 0 { gpu.dispatch(&pipeline, &[Arg::Buf(&bg), Arg::Buf(&bo), Arg::Buf(&result), Arg::U32(nt as u32), Arg::U32(no as u32), Arg::F32(dt), Arg::F32(velocity)], [n, 1, 1])?; }
    let values = gpu.read::<f32>(&result, n);
    if stack { out.f32("stack", &[nt], &values) } else { out.f32("corrected", &[no, nt], &values) }
}
fn nmo(inp: &Inputs, out: &mut Outputs) -> Result<()> { correct_or_stack(inp, out, false) }
fn stack(inp: &Inputs, out: &mut Outputs) -> Result<()> { correct_or_stack(inp, out, true) }
fn semblance(inp: &Inputs, out: &mut Outputs) -> Result<()> {
    let g = inp.f32("gather")?;
    let o = inp.f32("offsets")?;
    let v = inp.f32("velocities")?;
    let (no, nt) = dimensions(&g, &o)?;
    if v.shape.len() != 1 { return Err("velocities must be one-dimensional".into()); }
    let nv = v.shape[0];
    let half = inp.int("half_window")?;
    if half < 0 { return Err("half_window must be nonnegative".into()); }
    let dt = inp.num("dt")? as f32;
    let n = nt * nv;
    let gpu = Gpu::new()?;
    let lib = gpu.compile(KERNELS)?;
    let terms = gpu.pipeline(&lib, "semblance_terms")?;
    let window = gpu.pipeline(&lib, "semblance_window")?;
    let bg = gpu.buffer_from(&g.data);
    let bo = gpu.buffer_from(&o.data);
    let bv = gpu.buffer_from(&v.data);
    let bn = gpu.buffer_zeroed::<f32>(n);
    let bd = gpu.buffer_zeroed::<f32>(n);
    let result = gpu.buffer_zeroed::<f32>(n);
    if n > 0 {
        gpu.batch(|batch| {
            batch.dispatch(&terms, &[Arg::Buf(&bg), Arg::Buf(&bo), Arg::Buf(&bv), Arg::Buf(&bn), Arg::Buf(&bd), Arg::U32(nt as u32), Arg::U32(no as u32), Arg::U32(nv as u32), Arg::F32(dt)], [n, 1, 1])?;
            batch.dispatch(&window, &[Arg::Buf(&bn), Arg::Buf(&bd), Arg::Buf(&result), Arg::U32(nt as u32), Arg::U32(no as u32), Arg::U32(nv as u32), Arg::U32((half as u64).min(u32::MAX as u64) as u32)], [nv, 1, 1])
        })?;
    }
    out.f32("semblance", &[nv, nt], &gpu.read::<f32>(&result, n))
}
fn main() { outcrop_common::run(&[("nmo", nmo), ("semblance", semblance), ("stack", stack)]); }
