mod fft;
mod input;

use outcrop_common::{gpu::Arg, Gpu, Inputs, Outputs, Result};
use std::f64::consts::PI;

fn signed_bin(i: usize, n: usize) -> f64 {
    if i < n.div_ceil(2) { i as f64 } else { i as f64 - n as f64 }
}

fn calculate(inp: &Inputs, out: &mut Outputs, radon: bool) -> Result<()> {
    let gather = input::f32("gather")?;
    if gather.shape.len() != 2 || gather.shape[0] == 0 || gather.shape[1] == 0 {
        return Err("gather must be a nonempty 2-D array".into());
    }
    let (nx, nt) = (gather.shape[0], gather.shape[1]);
    let count = nx.checked_mul(nt).filter(|&n| n <= u32::MAX as usize)
        .ok_or("gather dimensions exceed Metal limits")?;
    let dt = inp.num("dt")?;
    if !dt.is_finite() || dt <= 0.0 { return Err("dt must be positive and finite".into()); }

    // Keep grid construction and long Radon phase reduction in host float64,
    // just as in the previously passing implementation.
    let (phase, mask) = if radon {
        let offsets = input::f32("offsets")?;
        if offsets.shape != [nx] { return Err("offsets must have shape (nx,)".into()); }
        let slow = inp.num("slowness")?;
        if !slow.is_finite() || offsets.data.iter().any(|v| !v.is_finite()) {
            return Err("slowness and offsets must be finite".into());
        }
        let phases: Vec<[f32; 2]> = offsets.data.iter().flat_map(|&offset| (0..nt).map(move |f| {
            let freq = signed_bin(f, nt) / (nt as f64 * dt);
            let angle = -2.0 * PI * freq * slow * (offset as f64).powi(2);
            [angle.cos() as f32, angle.sin() as f32]
        })).collect();
        (Some(phases), None)
    } else {
        let dx = inp.num("dx")?;
        let cutoff = inp.num("velocity_cutoff")?;
        if !dx.is_finite() || !cutoff.is_finite() || dx <= 0.0 || cutoff <= 0.0 {
            return Err("dx and velocity_cutoff must be positive and finite".into());
        }
        // Space FFTs use (frequency, wavenumber) layout after transposition.
        let mask: Vec<f32> = (0..nt).flat_map(|f| (0..nx).map(move |k| {
            let freq = signed_bin(f, nt).abs() / (nt as f64 * dt);
            let wave = signed_bin(k, nx).abs() / (nx as f64 * dx);
            if freq <= cutoff * wave { 0.0 } else { 1.0 }
        })).collect();
        (None, Some(mask))
    };

    let gpu = Gpu::new()?;
    let lib = gpu.compile(include_str!("../kernels/filters.metal"))?;
    let kernels = fft::Kernels::new(&gpu, &lib)?;
    let time = fft::Plan::new(&gpu, &kernels, nt, nx)?;
    let input = gpu.buffer_from(&gather.data);
    let a = gpu.buffer_zeroed::<[f32; 2]>(count);
    let b = gpu.buffer_zeroed::<[f32; 2]>(count);
    let result = gpu.buffer_zeroed::<f32>(count);
    let to_complex = gpu.pipeline(&lib, "real_to_complex")?;
    let to_real = gpu.pipeline(&lib, "complex_to_real")?;
    let (x, t) = (nx as u32, nt as u32);
    gpu.batch(|batch| {
        batch.dispatch(&to_complex, &[Arg::Buf(&input), Arg::Buf(&a)], [count, 1, 1])?;
        time.encode(batch, &kernels, &a, &b, false)
    })?;

    if let Some(phases) = phase {
        let phase = gpu.buffer_from(&phases);
        let coefficients = gpu.buffer_zeroed::<[f32; 2]>(nt);
        let dot = gpu.pipeline(&lib, "radon_coefficients")?;
        let project = gpu.pipeline(&lib, "project_radon")?;
        gpu.batch(|batch| {
            batch.dispatch(&dot, &[Arg::Buf(&b), Arg::Buf(&phase), Arg::Buf(&coefficients),
                Arg::U32(x), Arg::U32(t)], [nt, 1, 1])?;
            batch.dispatch(&project, &[Arg::Buf(&b), Arg::Buf(&phase), Arg::Buf(&coefficients),
                Arg::U32(t)], [nt, nx, 1])
        })?;
    } else if let Some(mask) = mask {
        let mask = gpu.buffer_from(&mask);
        let c = gpu.buffer_zeroed::<[f32; 2]>(count);
        let space = fft::Plan::new(&gpu, &kernels, nx, nt)?;
        let transpose = gpu.pipeline(&lib, "transpose")?;
        let apply_mask = gpu.pipeline(&lib, "apply_mask")?;
        gpu.batch(|batch| {
            batch.dispatch(&transpose, &[Arg::Buf(&b), Arg::Buf(&a), Arg::U32(t), Arg::U32(x)], [nt, nx, 1])?;
            space.encode(batch, &kernels, &a, &c, false)?;
            batch.dispatch(&apply_mask, &[Arg::Buf(&c), Arg::Buf(&mask)], [count, 1, 1])?;
            space.encode(batch, &kernels, &c, &a, true)?;
            batch.dispatch(&transpose, &[Arg::Buf(&a), Arg::Buf(&b), Arg::U32(x), Arg::U32(t)], [nx, nt, 1])
        })?;
    }
    gpu.batch(|batch| {
        time.encode(batch, &kernels, &b, &a, true)?;
        batch.dispatch(&to_real, &[Arg::Buf(&a), Arg::Buf(&result)], [count, 1, 1])
    })?;
    out.f32("filtered", &[nx, nt], &gpu.read::<f32>(&result, count))
}

fn fk_filter(inp: &Inputs, out: &mut Outputs) -> Result<()> { calculate(inp, out, false) }
fn radon_demultiple(inp: &Inputs, out: &mut Outputs) -> Result<()> { calculate(inp, out, true) }

fn main() {
    outcrop_common::run(&[("fk_filter", fk_filter), ("radon_demultiple", radon_demultiple)]);
}
