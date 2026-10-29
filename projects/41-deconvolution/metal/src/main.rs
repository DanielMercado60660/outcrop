use outcrop_common::{gpu::Arg, Gpu, Inputs, Outputs, Result};

const SHADER: &str = include_str!("../kernels/deconvolution.metal");

fn positive(inp: &Inputs, name: &str) -> Result<usize> {
    let v = inp.int(name)?;
    if v < 1 || v > u32::MAX as i64 { return Err(format!("invalid {name}").into()); }
    Ok(v as usize)
}

fn execute(inp: &Inputs, out: &mut Outputs, pred: bool) -> Result<()> {
    let trace = inp.f32("trace")?;
    if trace.shape.len() != 1 || trace.data.len() > u32::MAX as usize { return Err("invalid trace".into()); }
    let n = trace.data.len();
    let p = positive(inp, if pred { "order" } else { "inverse_length" })?;
    let lag = if pred { positive(inp, "lag")? } else { 0 };
    let load = inp.num("stabilization")?;
    if !load.is_finite() || load < 0.0 || load > f32::MAX as f64 { return Err("invalid stabilization".into()); }
    let wave = if pred { None } else { Some(inp.f32("wavelet")?) };
    if let Some(ref w) = wave {
        if w.shape.len() != 1 || w.data.is_empty() || w.data.len() > u32::MAX as usize { return Err("invalid wavelet".into()); }
    }
    let count = p.checked_add(lag).filter(|&v| v <= u32::MAX as usize).ok_or("correlation size overflow")?;
    let matrix_len = p.checked_mul(p).filter(|&v| v <= u32::MAX as usize).ok_or("matrix size overflow")?;
    let gpu = Gpu::new()?;
    let lib = gpu.compile(SHADER)?;
    let corr = gpu.pipeline(&lib, "autocorrelation")?;
    let system = gpu.pipeline(&lib, "system")?;
    let solve = gpu.pipeline(&lib, "solve")?;
    let apply = gpu.pipeline(&lib, "apply_filter")?;
    let x = gpu.buffer_from(&trace.data);
    let source = wave.as_ref().map_or(&trace.data, |w| &w.data);
    let source_buf = gpu.buffer_from(source);
    let r = gpu.buffer_zeroed::<f32>(count);
    let a = gpu.buffer_zeroed::<f32>(matrix_len);
    let b = gpu.buffer_zeroed::<f32>(p);
    let status = gpu.buffer_zeroed::<u32>(1);
    let y = gpu.buffer_zeroed::<f32>(n);
    gpu.dispatch(&corr, &[Arg::Buf(&source_buf), Arg::Buf(&r), Arg::U32(source.len() as u32), Arg::U32(count as u32)], [count, 1, 1])?;
    gpu.dispatch(&system, &[Arg::Buf(&r), Arg::Buf(&source_buf), Arg::Buf(&a), Arg::Buf(&b), Arg::U32(p as u32), Arg::U32(lag as u32), Arg::F32(load as f32), Arg::U32(pred as u32)], [p, 1, 1])?;
    gpu.dispatch(&solve, &[Arg::Buf(&a), Arg::Buf(&b), Arg::U32(p as u32), Arg::Buf(&status)], [1, 1, 1])?;
    if gpu.read::<u32>(&status, 1)[0] != 0 { return Err("singular deconvolution system".into()); }
    gpu.dispatch(&apply, &[Arg::Buf(&x), Arg::Buf(&b), Arg::Buf(&y), Arg::U32(n as u32), Arg::U32(p as u32), Arg::U32(lag as u32), Arg::U32(pred as u32)], [n, 1, 1])?;
    out.f32(if pred { "coefficients" } else { "inverse" }, &[p], &gpu.read::<f32>(&b, p))?;
    out.f32(if pred { "residual" } else { "recovered" }, &[n], &gpu.read::<f32>(&y, n))?;
    Ok(())
}
fn spiking(inp: &Inputs, out: &mut Outputs) -> Result<()> { execute(inp, out, false) }
fn predictive(inp: &Inputs, out: &mut Outputs) -> Result<()> { execute(inp, out, true) }
fn main() { outcrop_common::run(&[("spiking_deconvolution", spiking), ("predictive_deconvolution", predictive)]); }
