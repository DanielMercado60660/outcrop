use outcrop_common::{gpu::Arg, Gpu, Inputs, Outputs, Result};
const SOURCE: &str = include_str!("../kernels/rays.metal");
fn layered(inp: &Inputs, out: &mut Outputs, bend: bool) -> Result<()> {
    let interfaces = inp.f32("interfaces")?;
    let velocities = inp.f32("velocities")?;
    let rays = inp.f32(if bend { "offsets" } else { "angles" })?;
    if velocities.data.is_empty() || interfaces.data.len() + 1 != velocities.data.len() { return Err("incompatible layer arrays".into()); }
    let depth = inp.num("depth")? as f32;
    let n = rays.data.len();
    let gpu = Gpu::new()?;
    let lib = gpu.compile(SOURCE)?;
    let kernel = gpu.pipeline(&lib, "layered")?;
    let bi = gpu.buffer_from(&interfaces.data);
    let bv = gpu.buffer_from(&velocities.data);
    let br = gpu.buffer_from(&rays.data);
    let be = gpu.buffer_zeroed::<f32>(2*n);
    let bt = gpu.buffer_zeroed::<f32>(n);
    let ba = gpu.buffer_zeroed::<f32>(n);
    gpu.dispatch(&kernel, &[Arg::Buf(&bi), Arg::Buf(&bv), Arg::Buf(&br), Arg::Buf(&be), Arg::Buf(&bt), Arg::Buf(&ba), Arg::U32(velocities.data.len() as u32), Arg::F32(depth), Arg::U32(n as u32), Arg::U32(bend as u32)], [n,1,1])?;
    out.f32("endpoints", &[n,2], &gpu.read::<f32>(&be,2*n))?;
    out.f32("times", &[n], &gpu.read::<f32>(&bt,n))?;
    if bend { out.f32("angles", &[n], &gpu.read::<f32>(&ba,n))?; }
    Ok(())
}
fn gradient(inp: &Inputs, out: &mut Outputs, bend: bool) -> Result<()> {
    let rays = inp.f32(if bend { "offsets" } else { "angles" })?;
    let n = rays.data.len();
    let gpu = Gpu::new()?;
    let lib = gpu.compile(SOURCE)?;
    let kernel = gpu.pipeline(&lib, "gradient")?;
    let br = gpu.buffer_from(&rays.data);
    let be = gpu.buffer_zeroed::<f32>(2*n);
    let bt = gpu.buffer_zeroed::<f32>(n);
    let ba = gpu.buffer_zeroed::<f32>(n);
    gpu.dispatch(&kernel, &[Arg::Buf(&br), Arg::Buf(&be), Arg::Buf(&bt), Arg::Buf(&ba), Arg::F32(inp.num("v0")? as f32), Arg::F32(inp.num("gradient")? as f32), Arg::F32(inp.num("depth")? as f32), Arg::U32(n as u32), Arg::U32(bend as u32)], [n,1,1])?;
    out.f32("endpoints", &[n,2], &gpu.read::<f32>(&be,2*n))?;
    out.f32("times", &[n], &gpu.read::<f32>(&bt,n))?;
    if bend { out.f32("angles", &[n], &gpu.read::<f32>(&ba,n))?; }
    Ok(())
}
fn shoot_layered(i: &Inputs, o: &mut Outputs) -> Result<()> { layered(i,o,false) }
fn bend_layered(i: &Inputs, o: &mut Outputs) -> Result<()> { layered(i,o,true) }
fn shoot_gradient(i: &Inputs, o: &mut Outputs) -> Result<()> { gradient(i,o,false) }
fn bend_gradient(i: &Inputs, o: &mut Outputs) -> Result<()> { gradient(i,o,true) }
fn main() { outcrop_common::run(&[("shoot_layered",shoot_layered),("bend_layered",bend_layered),("shoot_gradient",shoot_gradient),("bend_gradient",bend_gradient)]); }
