use outcrop_common::{gpu::Arg, Gpu, Inputs, Outputs, Result};

const SECONDS_PER_MYR: f64 = 1_000_000.0 * 365.25 * 86400.0;
const SOURCE: &str = include_str!("../kernels/geotherm.metal");

// Surface flux is coefficient / sqrt(age in Myr).
fn parameters(inp: &Inputs) -> Result<(f32, f32, f32, f32)> {
    let surface = inp.num("surface_temp_c")?;
    let delta = inp.num("mantle_temp_c")? - surface;
    let diffusivity = inp.num("diffusivity_m2_s")?;
    let conductivity = inp.num("conductivity_w_m_k")?;
    let length = (diffusivity * SECONDS_PER_MYR).sqrt();
    let coefficient = conductivity * delta / (std::f64::consts::PI.sqrt() * length);
    Ok((surface as f32, delta as f32, length as f32, coefficient as f32))
}

fn geotherm(inp: &Inputs, out: &mut Outputs) -> Result<()> {
    let depth = inp.f32("depth_m")?;
    let age = inp.f32("age_myr")?;
    let (surface, delta, length, coefficient) = parameters(inp)?;
    let (n, d) = (age.data.len(), depth.data.len());
    let gpu = Gpu::new()?;
    let lib = gpu.compile(SOURCE)?;
    let kernel = gpu.pipeline(&lib, "geotherm")?;
    let bd = gpu.buffer_from(&depth.data);
    let ba = gpu.buffer_from(&age.data);
    let bt = gpu.buffer_zeroed::<f32>(n * d);
    let bq = gpu.buffer_zeroed::<f32>(n * d);
    gpu.dispatch(&kernel, &[Arg::Buf(&bd), Arg::Buf(&ba), Arg::Buf(&bt), Arg::Buf(&bq),
        Arg::U32(n as u32), Arg::U32(d as u32), Arg::F32(surface), Arg::F32(delta),
        Arg::F32(length), Arg::F32(coefficient)], [n * d, 1, 1])?;
    out.f32("temperature_c", &[n, d], &gpu.read::<f32>(&bt, n * d))?;
    out.f32("heat_flow_w_m2", &[n, d], &gpu.read::<f32>(&bq, n * d))
}

fn invert_age(inp: &Inputs, out: &mut Outputs) -> Result<()> {
    let flux = inp.f32("surface_heat_flow_w_m2")?;
    let (_, _, _, coefficient) = parameters(inp)?;
    let n = flux.data.len();
    let gpu = Gpu::new()?;
    let lib = gpu.compile(SOURCE)?;
    let kernel = gpu.pipeline(&lib, "invert_age")?;
    let bq = gpu.buffer_from(&flux.data);
    let ba = gpu.buffer_zeroed::<f32>(n);
    gpu.dispatch(&kernel, &[Arg::Buf(&bq), Arg::Buf(&ba), Arg::U32(n as u32),
        Arg::F32(coefficient)], [n, 1, 1])?;
    out.f32("age_myr", &[n], &gpu.read::<f32>(&ba, n))
}

fn heat_flow_misfit_and_grad(inp: &Inputs, out: &mut Outputs) -> Result<()> {
    let age = inp.f32("age_myr")?;
    let observed = inp.f32("observed_heat_flow_w_m2")?;
    let (_, _, _, coefficient) = parameters(inp)?;
    let scale = inp.num("flux_scale_w_m2")? as f32;
    let n = age.data.len();
    if n == 0 || observed.data.len() != n { return Err("age and observed flux must have equal nonzero length".into()); }
    let n_u32 = u32::try_from(n).map_err(|_| "too many observations for Metal indexing")?;
    let gpu = Gpu::new()?;
    let lib = gpu.compile(SOURCE)?;
    let kernel = gpu.pipeline(&lib, "misfit")?;
    let reduce = gpu.pipeline(&lib, "reduce_misfit")?;
    let ba = gpu.buffer_from(&age.data);
    let bo = gpu.buffer_from(&observed.data);
    let bg = gpu.buffer_zeroed::<f32>(n);
    let mut sums = gpu.buffer_zeroed::<f32>(n);
    let mut scratch = gpu.buffer_zeroed::<f32>(n.div_ceil(4));
    // Encode all passes in order with one wait and no intermediate readback.
    // Ping-pong buffers remain alive until the entire batch has completed.
    gpu.batch(|batch| {
        batch.dispatch(&kernel, &[Arg::Buf(&ba), Arg::Buf(&bo), Arg::Buf(&bg),
            Arg::Buf(&sums), Arg::U32(n_u32), Arg::F32(coefficient), Arg::F32(scale)],
            [n, 1, 1])?;
        let mut count = n;
        loop {
            let next_count = count.div_ceil(4);
            batch.dispatch(&reduce, &[Arg::Buf(&sums), Arg::Buf(&scratch),
                Arg::U32(count as u32), Arg::U32(n_u32)], [next_count, 1, 1])?;
            std::mem::swap(&mut sums, &mut scratch);
            if next_count == 1 {
                break;
            }
            count = next_count;
        }
        Ok(())
    })?;
    out.f32("objective", &[], &gpu.read::<f32>(&sums, 1))?;
    out.f32("gradient", &[n], &gpu.read::<f32>(&bg, n))
}

fn main() {
    outcrop_common::run(&[("geotherm", geotherm), ("invert_age", invert_age),
        ("heat_flow_misfit_and_grad", heat_flow_misfit_and_grad)]);
}
