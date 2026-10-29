//! Homogeneous arrival-time inversion. Rust handles the array protocol; all
//! travel times, grid candidates, gradients and centered QR solves run in Metal.
use outcrop_common::gpu::Arg;
use outcrop_common::{Array, Gpu, Inputs, Outputs, Result};

struct Data {
    stations: Array<f32>,
    arrivals: Array<f32>,
    sigma: Array<f32>,
    n: usize,
    b: usize,
    velocity: f32,
}

fn data(inp: &Inputs) -> Result<Data> {
    let stations = inp.f32("stations")?;
    let arrivals = inp.f32("arrivals")?;
    let sigma = inp.f32("sigma")?;
    if stations.shape.len() != 2 || stations.shape[1] != 3 || arrivals.shape.len() != 2 {
        return Err("invalid stations or arrivals shape".into());
    }
    let n = stations.shape[0];
    let b = arrivals.shape[0];
    if n == 0 || arrivals.shape[1] != n || sigma.shape != [n] {
        return Err("inconsistent station dimensions".into());
    }
    let velocity = inp.num("velocity")? as f32;
    if !velocity.is_finite() || velocity <= 0.0 {
        return Err("velocity must be finite and positive".into());
    }
    if sigma.data.iter().any(|s| !s.is_finite() || *s <= 0.0) {
        return Err("sigma must be finite and positive".into());
    }
    // Every shader offset is a uint: e*n+i for picks, 3*i+k for stations,
    // 4*e+k for models, and 16*e+4*k+l for the full covariance.
    let picks = b.checked_mul(n).ok_or("input dimensions overflow")?;
    let coordinates = n.checked_mul(3).ok_or("station dimensions overflow")?;
    let covariance = b.checked_mul(16).ok_or("covariance dimensions overflow")?;
    if [picks, coordinates, covariance]
        .iter()
        .any(|&len| len > u32::MAX as usize)
    {
        return Err("input dimensions exceed Metal index range".into());
    }
    Ok(Data {
        stations,
        arrivals,
        sigma,
        n,
        b,
        velocity,
    })
}

fn misfit_and_grad(inp: &Inputs, out: &mut Outputs) -> Result<()> {
    let d = data(inp)?;
    let models = inp.f32("models")?;
    if models.shape != [d.b, 4] {
        return Err("invalid models shape".into());
    }
    let gpu = Gpu::new()?;
    let lib = gpu.compile(include_str!("../kernels/location.metal"))?;
    let kernel = gpu.pipeline(&lib, "misfit")?;
    let s = gpu.buffer_from(&d.stations.data);
    let a = gpu.buffer_from(&d.arrivals.data);
    let sig = gpu.buffer_from(&d.sigma.data);
    let m = gpu.buffer_from(&models.data);
    let pred = gpu.buffer_zeroed::<f32>(d.b * d.n);
    let obj = gpu.buffer_zeroed::<f32>(d.b);
    let grad = gpu.buffer_zeroed::<f32>(d.b * 4);
    gpu.dispatch(
        &kernel,
        &[
            Arg::Buf(&s),
            Arg::Buf(&a),
            Arg::Buf(&sig),
            Arg::Buf(&m),
            Arg::Buf(&pred),
            Arg::Buf(&obj),
            Arg::Buf(&grad),
            Arg::U32(d.n as u32),
            Arg::U32(d.b as u32),
            Arg::F32(d.velocity),
        ],
        [d.b, 1, 1],
    )?;
    out.f32("predicted", &[d.b, d.n], &gpu.read(&pred, d.b * d.n))?;
    out.f32("objective", &[d.b], &gpu.read(&obj, d.b))?;
    out.f32("gradient", &[d.b, 4], &gpu.read(&grad, d.b * 4))
}

fn grid_search(inp: &Inputs, out: &mut Outputs) -> Result<()> {
    let d = data(inp)?;
    let x = inp.f32("x")?;
    let y = inp.f32("y")?;
    let z = inp.f32("z")?;
    if x.shape.len() != 1
        || y.shape.len() != 1
        || z.shape.len() != 1
        || x.data.is_empty()
        || y.data.is_empty()
        || z.data.is_empty()
    {
        return Err("invalid grid axes".into());
    }
    let (nx, ny, nz) = (x.data.len(), y.data.len(), z.data.len());
    let nodes = nx
        .checked_mul(ny)
        .and_then(|len| len.checked_mul(nz))
        .filter(|&len| len <= u32::MAX as usize)
        .ok_or("grid dimensions exceed Metal index range")?;
    let total = d.b
        .checked_mul(nodes)
        .filter(|&len| len <= u32::MAX as usize)
        .ok_or("batched grid exceeds Metal index range")?;
    let gpu = Gpu::new()?;
    let lib = gpu.compile(include_str!("../kernels/location.metal"))?;
    let candidates = gpu.pipeline(&lib, "grid_nodes")?;
    let reduce = gpu.pipeline(&lib, "grid_reduce")?;
    let s = gpu.buffer_from(&d.stations.data);
    let a = gpu.buffer_from(&d.arrivals.data);
    let sig = gpu.buffer_from(&d.sigma.data);
    let bx = gpu.buffer_from(&x.data);
    let by = gpu.buffer_from(&y.data);
    let bz = gpu.buffer_from(&z.data);
    let losses = gpu.buffer_zeroed::<f32>(total);
    let times = gpu.buffer_zeroed::<f32>(total);
    let models = gpu.buffer_zeroed::<f32>(d.b * 4);
    let obj = gpu.buffer_zeroed::<f32>(d.b);
    gpu.dispatch(
        &candidates,
        &[
            Arg::Buf(&s),
            Arg::Buf(&a),
            Arg::Buf(&sig),
            Arg::Buf(&bx),
            Arg::Buf(&by),
            Arg::Buf(&bz),
            Arg::Buf(&losses),
            Arg::Buf(&times),
            Arg::U32(d.n as u32),
            Arg::U32(d.b as u32),
            Arg::U32(nx as u32),
            Arg::U32(ny as u32),
            Arg::U32(nz as u32),
            Arg::F32(d.velocity),
        ],
        [total, 1, 1],
    )?;
    gpu.dispatch(
        &reduce,
        &[
            Arg::Buf(&losses),
            Arg::Buf(&times),
            Arg::Buf(&bx),
            Arg::Buf(&by),
            Arg::Buf(&bz),
            Arg::Buf(&models),
            Arg::Buf(&obj),
            Arg::U32(d.b as u32),
            Arg::U32(nx as u32),
            Arg::U32(ny as u32),
            Arg::U32(nz as u32),
        ],
        [d.b, 1, 1],
    )?;
    out.f32("models", &[d.b, 4], &gpu.read(&models, d.b * 4))?;
    out.f32("objective", &[d.b], &gpu.read(&obj, d.b))
}

fn geiger(inp: &Inputs, out: &mut Outputs) -> Result<()> {
    let d = data(inp)?;
    let initial = inp.f32("initial")?;
    if initial.shape != [d.b, 4] {
        return Err("invalid initial shape".into());
    }
    let gpu = Gpu::new()?;
    let lib = gpu.compile(include_str!("../kernels/location.metal"))?;
    let kernel = gpu.pipeline(&lib, "locate")?;
    let s = gpu.buffer_from(&d.stations.data);
    let a = gpu.buffer_from(&d.arrivals.data);
    let sig = gpu.buffer_from(&d.sigma.data);
    let init = gpu.buffer_from(&initial.data);
    let models = gpu.buffer_zeroed::<f32>(d.b * 4);
    let obj = gpu.buffer_zeroed::<f32>(d.b);
    let cov = gpu.buffer_zeroed::<f32>(d.b * 16);
    let ellipse = gpu.buffer_zeroed::<f32>(d.b * 4);
    let status = gpu.buffer_zeroed::<u32>(d.b);
    gpu.dispatch(
        &kernel,
        &[
            Arg::Buf(&s),
            Arg::Buf(&a),
            Arg::Buf(&sig),
            Arg::Buf(&init),
            Arg::Buf(&models),
            Arg::Buf(&obj),
            Arg::Buf(&cov),
            Arg::Buf(&ellipse),
            Arg::U32(d.n as u32),
            Arg::U32(d.b as u32),
            Arg::F32(d.velocity),
            Arg::Buf(&status),
        ],
        [d.b, 1, 1],
    )?;
    // Match LocateStatus in location.metal. Check all events before writing
    // any outputs, so an unresolved solve cannot escape as NaN covariance.
    for (event, code) in gpu.read::<u32>(&status, d.b).into_iter().enumerate() {
        let reason = match code {
            0 => continue,
            1 => "invalid or coincident source/station geometry",
            2 => "centered Jacobian is rank deficient at float32 precision",
            3 => "non-finite or unrepresentable solve/covariance",
            _ => "unknown Metal solver status",
        };
        return Err(format!("geiger event {event}: {reason}").into());
    }
    out.f32("models", &[d.b, 4], &gpu.read(&models, d.b * 4))?;
    out.f32("objective", &[d.b], &gpu.read(&obj, d.b))?;
    out.f32("covariance", &[d.b, 4, 4], &gpu.read(&cov, d.b * 16))?;
    out.f32("ellipse", &[d.b, 2, 2], &gpu.read(&ellipse, d.b * 4))
}

fn main() {
    outcrop_common::run(&[
        ("misfit_and_grad", misfit_and_grad),
        ("grid_search", grid_search),
        ("geiger", geiger),
    ]);
}
