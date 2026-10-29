use outcrop_common::{gpu::Arg, Gpu, Inputs, Outputs, Result};

fn travel_time(inp: &Inputs, out: &mut Outputs) -> Result<()> {
    let phi = inp.f32("phi")?;
    let speed = inp.f32("speed")?;
    let spacing = inp.nums("spacing")?;
    let order = inp.int("order")?;
    let dims = phi.shape.len();
    if !(dims == 2 || dims == 3) || speed.shape != phi.shape || spacing.len() != dims
        || !(order == 1 || order == 2) || phi.shape.contains(&0)
        || spacing.iter().any(|&x| !x.is_finite() || x <= 0.0)
        || speed.data.iter().any(|&x| !x.is_finite() || x <= 0.0) {
        return Err("invalid grid, spacing, speed or order".into());
    }
    let n = phi.data.len();
    if n > i32::MAX as usize { return Err("grid too large".into()); }
    let mut shape = [1u32; 3];
    let mut stride = [1u32; 3];
    let mut h = [1f32; 3];
    for a in 0..dims {
        shape[a] = phi.shape[a] as u32;
        stride[a] = phi.shape[a + 1..].iter().product::<usize>() as u32;
        h[a] = spacing[a] as f32;
    }
    let params = [n as u32, dims as u32, order as u32, shape[0], shape[1], shape[2], stride[0], stride[1], stride[2]];
    let gpu = Gpu::new()?;
    let lib = gpu.compile(include_str!("../kernels/marching.metal"))?;
    let init = gpu.pipeline(&lib, "initialize")?;
    let march = gpu.pipeline(&lib, "march")?;
    let bp = gpu.buffer_from(&params);
    let bh = gpu.buffer_from(&h);
    let bphi = gpu.buffer_from(&phi.data);
    let bspeed = gpu.buffer_from(&speed.data);
    let bt = gpu.buffer_zeroed::<f32>(n);
    let bfrozen = gpu.buffer_zeroed::<u32>(n);
    let bheap = gpu.buffer_zeroed::<u32>(n);
    let bpos = gpu.buffer_zeroed::<i32>(n);
    let bstate = gpu.buffer_zeroed::<u32>(3);
    gpu.dispatch(&init, &[Arg::Buf(&bp), Arg::Buf(&bh), Arg::Buf(&bphi), Arg::Buf(&bspeed), Arg::Buf(&bt), Arg::Buf(&bfrozen), Arg::Buf(&bpos)], [n, 1, 1])?;
    for _ in 0..(2 * n.div_ceil(4096) + 1) {
        gpu.dispatch(&march, &[Arg::Buf(&bp), Arg::Buf(&bh), Arg::Buf(&bphi), Arg::Buf(&bspeed), Arg::Buf(&bt), Arg::Buf(&bfrozen), Arg::Buf(&bheap), Arg::Buf(&bpos), Arg::Buf(&bstate)], [1, 1, 1])?;
        let state: Vec<u32> = gpu.read(&bstate, 3);
        if state[1] != 0 && state[0] == 0 { break; }
    }
    let result: Vec<f32> = gpu.read(&bt, n);
    out.f32("time", &phi.shape, &result)
}
fn main() { outcrop_common::run(&[("travel_time", travel_time)]); }
