//! Valid, locally demeaned correlation for each template/station pair.
mod buffer;
mod input;

use outcrop_common::gpu::Arg;
use outcrop_common::{Gpu, Inputs, Outputs, Result};
use std::path::PathBuf;

fn template_match(inp: &Inputs, out: &mut Outputs) -> Result<()> {
    // Inputs does not expose its directory. run has already validated the CLI;
    // read the same directory locally to avoid the frozen NPY parser's unchecked
    // header slices and shape products.
    let input_dir = PathBuf::from(std::env::args_os().nth(2).ok_or("missing input directory")?);
    let data = input::read_f32(&input_dir.join("data.npy"))?;
    let templates = input::read_f32(&input_dir.join("templates.npy"))?;
    let threshold = inp.num("threshold")?;
    if data.shape.len() != 2 || templates.shape.len() != 3 {
        return Err("data must be (S,N) and templates must be (T,S,M)".into());
    }
    let (s, n) = (data.shape[0], data.shape[1]);
    let (t, ts, m) = (templates.shape[0], templates.shape[1], templates.shape[2]);
    if s != ts || m < 2 || m > n || !(-1.0..=1.0).contains(&threshold) {
        return Err("incompatible shapes or invalid threshold".into());
    }
    if data
        .data
        .iter()
        .chain(&templates.data)
        .any(|v| !v.is_finite())
    {
        return Err("data and templates must contain only finite float32 values".into());
    }
    let windows = n - m + 1;
    let count = t
        .checked_mul(s)
        .and_then(|v| v.checked_mul(windows))
        .ok_or("output too large")?;
    let shape = [t, s, windows];
    if count == 0 {
        out.f32("correlation", &shape, &[])?;
        out.f32("detections", &shape, &[])?;
        return Ok(());
    }
    let [s32, n32, m32, w32, count32] = [s, n, m, windows, count].map(u32::try_from);
    let (s32, n32, m32, w32, count32) = (s32?, n32?, m32?, w32?, count32?);
    // The shader uses uint offsets for both input buffers as well as the output.
    u32::try_from(data.data.len())?;
    u32::try_from(templates.data.len())?;
    let gpu = Gpu::new()?;
    let library = gpu.compile(include_str!("../kernels/template_match.metal"))?;
    let kernel = gpu.pipeline(&library, "template_match")?;
    let data_buf = buffer::from_slice(&gpu, &data.data)?;
    drop(data);
    let template_buf = buffer::from_slice(&gpu, &templates.data)?;
    drop(templates);
    // Every dispatched output is written by the shader, including degeneracies.
    let result_buf = buffer::allocate(&gpu, count)?;
    gpu.dispatch(
        &kernel,
        &[
            Arg::Buf(&data_buf),
            Arg::Buf(&template_buf),
            Arg::Buf(&result_buf),
            Arg::U32(s32),
            Arg::U32(n32),
            Arg::U32(m32),
            Arg::U32(w32),
            Arg::U32(count32),
        ],
        [count, 1, 1],
    )?;
    drop(data_buf);
    drop(template_buf);
    let correlation = buffer::read(&result_buf, count)?;
    drop(result_buf);
    // Compare the returned coefficient against the original Python float, keeping
    // the inclusive threshold and reporting every qualifying window.
    let mut detections = Vec::new();
    detections.try_reserve_exact(count)?;
    detections.extend(
        correlation
            .iter()
            .map(|&x| if (x as f64) >= threshold { 1.0 } else { 0.0 }),
    );
    out.f32("correlation", &shape, &correlation)?;
    out.f32("detections", &shape, &detections)?;
    Ok(())
}

fn main() {
    outcrop_common::run(&[("template_match", template_match)]);
}
