use outcrop_common::gpu::Arg;
use outcrop_common::{Gpu, Inputs, Outputs, Result};

fn travel_time(inp: &Inputs, out: &mut Outputs) -> Result<()> {
    let phi = inp.f32("phi")?;
    let speed = inp.f32("speed")?;
    let spacing = inp.num("spacing")? as f32;
    if !(phi.shape.len() == 2 || phi.shape.len() == 3)
        || phi.shape != speed.shape || phi.shape.contains(&0)
        || !spacing.is_finite() || spacing <= 0.0
        || speed.data.iter().any(|v| !v.is_finite() || *v <= 0.0)
        || phi.data.iter().any(|v| !v.is_finite())
    {
        return Err("invalid grid, speed, or spacing".into());
    }
    let n = phi.data.len();
    if n > u32::MAX as usize { return Err("grid too large".into()); }
    let (n0, n1, n2) = if phi.shape.len() == 2 {
        (phi.shape[0], phi.shape[1], 1)
    } else {
        (phi.shape[0], phi.shape[1], phi.shape[2])
    };
    if speed.data.iter().any(|&v| {
        let step = spacing / v;
        !step.is_finite() || step < f32::MIN_POSITIVE
    }) {
        return Err("grid spacing and speed exceed supported float32 range".into());
    }
    let gpu = Gpu::new()?;
    let lib = gpu.compile(include_str!("../kernels/sweep.metal"))?;
    let init = gpu.pipeline(&lib, "initialize")?;
    let sweep = gpu.pipeline(&lib, "sweep")?;
    // Convert allocation failures in the shared helpers into a CLI error.
    let (p, v, t, fixed) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        (gpu.buffer_from(&phi.data), gpu.buffer_from(&speed.data),
         gpu.buffer_zeroed::<f32>(n), gpu.buffer_zeroed::<u32>(n))
    })).map_err(|_| "Metal buffer allocation failed")?;
    gpu.dispatch(&init, &[Arg::Buf(&p), Arg::Buf(&v), Arg::Buf(&t), Arg::Buf(&fixed),
        Arg::U32(n0 as u32), Arg::U32(n1 as u32), Arg::U32(n2 as u32), Arg::F32(spacing)], [n, 1, 1])?;
    let mut previous: Vec<f32> = gpu.read(&t, n);
    if previous.iter().any(|x| x.is_nan()) {
        return Err("source crossing is below the supported float32 range".into());
    }
    let max_cycles = 8 * (n0 + n1 + n2) + 512;
    for cycle in 0..max_cycles {
        for parity in 0..2 {
            gpu.dispatch(&sweep, &[Arg::Buf(&v), Arg::Buf(&t), Arg::Buf(&fixed),
                Arg::U32(n0 as u32), Arg::U32(n1 as u32), Arg::U32(n2 as u32),
                Arg::F32(spacing), Arg::U32(parity)], [n, 1, 1])?;
        }
        if cycle % 16 == 15 {
            let current: Vec<f32> = gpu.read(&t, n);
            if current == previous {
                if current.iter().any(|x| !x.is_finite()) {
                    return Err("source does not reach the entire grid or travel time exceeds float32 range".into());
                }
                return out.f32("travel", &phi.shape, &current);
            }
            previous = current;
        }
    }
    Err("fast sweeping did not converge".into())
}

fn main() {
    outcrop_common::run(&[("travel_time", travel_time)]);
}
