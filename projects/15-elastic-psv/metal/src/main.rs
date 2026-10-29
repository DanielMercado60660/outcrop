//! Locked second-order staggered P-SV scheme and explicit discrete adjoint.
use outcrop_common::gpu::{Arg, Batch, Buffer, Pipeline};
use outcrop_common::{Array, Gpu, Inputs, Outputs, Result};
use std::collections::BTreeMap;

#[derive(Clone, Copy)]
struct Params {
    ny: u32,
    nx: u32,
    cells: u32,
    grid: u32,
    shots: u32,
    sources: u32,
    receivers: u32,
    nt: u32,
    source_groups: u32,
    receiver_groups: u32,
    model_ny: u32,
    model_nx: u32,
    top: u32,
    left: u32,
    history: u32,
    dy: f32,
    dx: f32,
    dt: f32,
}

impl Params {
    // Explicit serialization keeps the Rust/MSL ABI independent of padding.
    fn bytes(self) -> Vec<u8> {
        let mut b = Vec::with_capacity(72);
        for x in [
            self.ny, self.nx, self.cells, self.grid, self.shots, self.sources,
            self.receivers, self.nt, self.source_groups, self.receiver_groups,
            self.model_ny, self.model_nx, self.top, self.left, self.history,
        ] {
            b.extend_from_slice(&x.to_le_bytes());
        }
        for x in [self.dy, self.dx, self.dt] {
            b.extend_from_slice(&x.to_le_bytes());
        }
        b
    }
}

struct Groups {
    // Triples: global field index, first entry, one past last entry.
    ranges: Vec<u32>,
    entries: Vec<u32>,
}

impl Groups {
    fn new(locations: &[u32]) -> Self {
        let mut map = BTreeMap::<u32, Vec<u32>>::new();
        for (i, &cell) in locations.iter().enumerate() {
            map.entry(cell).or_default().push(i as u32);
        }
        let mut ranges = Vec::with_capacity(3 * map.len());
        let mut entries = Vec::with_capacity(locations.len());
        for (cell, ids) in map {
            let begin = entries.len() as u32;
            entries.extend(ids);
            ranges.extend([cell, begin, entries.len() as u32]);
        }
        Self { ranges, entries }
    }
}

struct Problem {
    p: Params,
    model: Vec<f32>,
    source_y: Array<f32>,
    source_x: Array<f32>,
    receiver_ids: Vec<u32>,
    sg: Groups,
    rg: Groups,
    profiles: Vec<f32>,
}

fn checked_u32(x: usize) -> Result<u32> {
    u32::try_from(x).map_err(|_| "array too large for Metal grid indices".into())
}

fn shape(a: &[usize], expected: &[usize], name: &str) -> Result<()> {
    if a != expected {
        return Err(format!("{name}: shape {a:?}, expected {expected:?}").into());
    }
    Ok(())
}

/// Fixed profile design in f64; all propagation is f32. Layout is float2(a,b):
/// y integer, y half, x integer, x half.
fn axis_profiles(
    n: usize, lo: usize, hi: usize, h: f64, dt: f64, freq: f64, speed: f64,
) -> Vec<f32> {
    let padded = n + lo + hi;
    let mut result = Vec::with_capacity(4 * padded);
    for offset in [0.0, 0.5] {
        for k in 0..padded {
            let u = k as f64 + offset;
            let sl = if lo == 0 { 0.0 } else {
                ((lo as f64 - u) / lo as f64).clamp(0.0, 1.0)
            };
            let sh = if hi == 0 { 0.0 } else {
                ((u - (lo + n - 1) as f64) / hi as f64).clamp(0.0, 1.0)
            };
            let s = sl.max(sh);
            if s == 0.0 {
                result.extend([0.0, 0.0]);
            } else {
                let c = -1.5 * speed * 0.001f64.ln() / h;
                let sigma = c * (if lo == 0 { 0.0 } else { sl * sl / lo as f64 }
                    + if hi == 0 { 0.0 } else { sh * sh / hi as f64 });
                let alpha = std::f64::consts::PI * freq * (1.0 - s);
                let z = -(sigma + alpha) * dt;
                result.extend([z.exp() as f32, (sigma / (sigma + alpha) * z.exp_m1()) as f32]);
            }
        }
    }
    result
}

impl Problem {
    fn load(inp: &Inputs, history: bool) -> Result<Self> {
        let vp = inp.f32("vp")?;
        let vs = inp.f32("vs")?;
        let rho = inp.f32("rho")?;
        if vp.shape.len() != 2 || vp.shape.iter().any(|&n| n < 2) {
            return Err("material arrays must have shape (ny>=2,nx>=2)".into());
        }
        shape(&vs.shape, &vp.shape, "vs")?;
        shape(&rho.shape, &vp.shape, "rho")?;
        let (my, mx) = (vp.shape[0], vp.shape[1]);
        let sy = inp.f32("source_y")?;
        let sx = inp.f32("source_x")?;
        if sy.shape.len() != 3 || sy.shape[0] == 0 || sy.shape[2] == 0 {
            return Err("source arrays must have shape (nshot,nsource,nt), nshot and nt positive".into());
        }
        shape(&sx.shape, &sy.shape, "source_x")?;
        let (shots, ns, nt) = (sy.shape[0], sy.shape[1], sy.shape[2]);
        let sl = inp.i64("source_locations")?;
        let rl = inp.i64("receiver_locations")?;
        shape(&sl.shape, &[shots, ns, 2], "source_locations")?;
        if rl.shape.len() != 3 || rl.shape[0] != shots || rl.shape[2] != 2 {
            return Err("receiver_locations must have shape (nshot,nreceiver,2)".into());
        }
        let nr = rl.shape[1];
        let widths = inp.nums("pml_width")?;
        let spacing = inp.nums("spacing")?;
        if widths.len() != 4 || widths.iter().any(|&w| w < 0.0 || w.fract() != 0.0) {
            return Err("pml_width must contain four nonnegative integers".into());
        }
        if spacing.len() != 2 || spacing.iter().any(|&h| !h.is_finite() || h <= 0.0) {
            return Err("spacing must contain two positive numbers".into());
        }
        // Bound widths before the float-to-usize cast, which otherwise
        // saturates on very large values. Metal grid indices are uint.
        if widths.iter().any(|&w| w > u32::MAX as f64) {
            return Err("pml_width exceeds Metal grid index range".into());
        }
        let w: Vec<usize> = widths.iter().map(|&v| v as usize).collect();
        let ny = my.checked_add(w[0]).and_then(|n| n.checked_add(w[1]))
            .ok_or("padded y dimension overflow")?;
        let nx = mx.checked_add(w[2]).and_then(|n| n.checked_add(w[3]))
            .ok_or("padded x dimension overflow")?;
        let cells = ny.checked_mul(nx).ok_or("grid size overflow")?;
        let grid = cells.checked_mul(shots).ok_or("shot grid size overflow")?;
        // Plane offsets in the shaders are uint. History offsets are ulong.
        checked_u32(grid.checked_mul(26).ok_or("state size overflow")?)?;
        checked_u32(shots * ns * nt * 2)?;
        checked_u32(shots * nr * nt * 3)?;
        let locations = |a: &Array<i64>, count: usize| -> Result<Vec<u32>> {
            let mut ids = Vec::with_capacity(shots * count);
            for shot in 0..shots {
                for q in 0..count {
                    let i = a.data[2 * (shot * count + q)];
                    let j = a.data[2 * (shot * count + q) + 1];
                    if i < 0 || j < 0 || i as usize >= my - 1 || j as usize >= mx - 1 {
                        return Err("locations must lie inside [0,ny-1) x [0,nx-1)".into());
                    }
                    ids.push(checked_u32(shot * cells + (i as usize + w[0]) * nx + j as usize + w[2])?);
                }
            }
            Ok(ids)
        };
        let source_ids = locations(&sl, ns)?;
        let receiver_ids = locations(&rl, nr)?;
        let sg = Groups::new(&source_ids);
        let rg = Groups::new(&receiver_ids);
        let dt = inp.num("dt")?;
        let freq = inp.num("pml_freq")?;
        let speed = inp.num("max_vel")?;
        if !dt.is_finite() || dt <= 0.0 || !freq.is_finite() || freq < 0.0
            || !speed.is_finite() || speed <= 0.0 {
            return Err("invalid dt, pml_freq or max_vel".into());
        }
        let mut profiles = axis_profiles(my, w[0], w[1], spacing[0], dt, freq, speed);
        profiles.extend(axis_profiles(mx, w[2], w[3], spacing[1], dt, freq, speed));
        let mut model = vp.data;
        model.extend(vs.data);
        model.extend(rho.data);
        let p = Params {
            ny: checked_u32(ny)?, nx: checked_u32(nx)?, cells: checked_u32(cells)?,
            grid: checked_u32(grid)?, shots: checked_u32(shots)?, sources: checked_u32(ns)?,
            receivers: checked_u32(nr)?, nt: checked_u32(nt)?,
            source_groups: checked_u32(sg.ranges.len() / 3)?,
            receiver_groups: checked_u32(rg.ranges.len() / 3)?,
            model_ny: checked_u32(my)?, model_nx: checked_u32(mx)?,
            top: checked_u32(w[0])?, left: checked_u32(w[2])?, history: u32::from(history),
            dy: spacing[0] as f32, dx: spacing[1] as f32, dt: dt as f32,
        };
        Ok(Self { p, model, source_y: sy, source_x: sx, receiver_ids, sg, rg, profiles })
    }

    fn output_shape(&self) -> [usize; 3] {
        [self.p.shots as usize, self.p.receivers as usize, self.p.nt as usize]
    }
}

struct Kernels {
    material: Pipeline,
    record: Pipeline,
    velocity: Pipeline,
    inject: Pipeline,
    stress: Pipeline,
    reverse_stress: Pipeline,
    gather_velocity: Pipeline,
    reverse_source: Pipeline,
    reverse_velocity: Pipeline,
    gather_stress: Pipeline,
    receiver_adjoint: Pipeline,
    material_adjoint: Pipeline,
    unpad: Pipeline,
}

struct Solver {
    gpu: Gpu,
    k: Kernels,
    p: Params,
    bytes: Vec<u8>,
    model: Buffer,
    coefficients: Buffer,
    profiles: Buffer,
    state: Buffer,
    history: Buffer,
    traces: Buffer,
    sy: Buffer,
    sx: Buffer,
    source_groups: Buffer,
    source_entries: Buffer,
    receiver_ids: Buffer,
    receiver_groups: Buffer,
    receiver_entries: Buffer,
}

impl Solver {
    fn new(problem: &Problem) -> Result<Self> {
        let gpu = Gpu::new()?;
        let lib = gpu.compile(include_str!("../kernels/elastic.metal"))?;
        let k = Kernels {
            material: gpu.pipeline(&lib, "material")?,
            record: gpu.pipeline(&lib, "record")?,
            velocity: gpu.pipeline(&lib, "velocity")?,
            inject: gpu.pipeline(&lib, "inject")?,
            stress: gpu.pipeline(&lib, "stress")?,
            reverse_stress: gpu.pipeline(&lib, "reverse_stress")?,
            gather_velocity: gpu.pipeline(&lib, "gather_velocity")?,
            reverse_source: gpu.pipeline(&lib, "reverse_source")?,
            reverse_velocity: gpu.pipeline(&lib, "reverse_velocity")?,
            gather_stress: gpu.pipeline(&lib, "gather_stress")?,
            receiver_adjoint: gpu.pipeline(&lib, "receiver_adjoint")?,
            material_adjoint: gpu.pipeline(&lib, "material_adjoint")?,
            unpad: gpu.pipeline(&lib, "unpad")?,
        };
        let p = problem.p;
        let g = p.grid as usize;
        let hist_len = if p.history != 0 {
            g.checked_mul(5).and_then(|v| v.checked_mul(p.nt as usize)).ok_or("history size overflow")?
        } else { 1 };
        let model = gpu.buffer_from(&problem.model);
        // Forward values use two float32 planes (rounded value + residual).
        // The adjoint consumes the rounded coefficient/history planes as usual.
        let coefficients = gpu.buffer_zeroed::<f32>(10 * p.cells as usize);
        let profiles = gpu.buffer_from(&problem.profiles);
        let state = gpu.buffer_zeroed::<f32>(26 * g);
        let history = gpu.buffer_zeroed::<f32>(hist_len);
        let traces = gpu.buffer_zeroed::<f32>(3 * p.shots as usize * p.receivers as usize * p.nt as usize);
        let sy = gpu.buffer_from(&problem.source_y.data);
        let sx = gpu.buffer_from(&problem.source_x.data);
        let source_groups = gpu.buffer_from(&problem.sg.ranges);
        let source_entries = gpu.buffer_from(&problem.sg.entries);
        let receiver_ids = gpu.buffer_from(&problem.receiver_ids);
        let receiver_groups = gpu.buffer_from(&problem.rg.ranges);
        let receiver_entries = gpu.buffer_from(&problem.rg.entries);
        Ok(Self {
            gpu, k, p, bytes: p.bytes(), model, coefficients, profiles, state, history, traces,
            sy, sx, source_groups, source_entries, receiver_ids, receiver_groups, receiver_entries,
        })
    }

    fn forward_step(&self, b: &mut Batch, time: u32) -> Result<()> {
        let p = &self.p;
        b.dispatch(&self.k.record, &[
            Arg::Buf(&self.state), Arg::Buf(&self.receiver_ids), Arg::Buf(&self.traces),
            Arg::Bytes(&self.bytes), Arg::U32(time),
        ], [(p.shots * p.receivers) as usize, 1, 1])?;
        b.dispatch(&self.k.velocity, &[
            Arg::Buf(&self.state), Arg::Buf(&self.coefficients), Arg::Buf(&self.profiles),
            Arg::Buf(&self.history), Arg::Bytes(&self.bytes), Arg::U32(time),
        ], [p.grid as usize, 1, 1])?;
        if p.source_groups != 0 {
            b.dispatch(&self.k.inject, &[
                Arg::Buf(&self.state), Arg::Buf(&self.coefficients), Arg::Buf(&self.sy),
                Arg::Buf(&self.sx), Arg::Buf(&self.source_groups), Arg::Buf(&self.source_entries),
                Arg::Bytes(&self.bytes), Arg::U32(time),
            ], [p.source_groups as usize, 1, 1])?;
        }
        b.dispatch(&self.k.stress, &[
            Arg::Buf(&self.state), Arg::Buf(&self.coefficients), Arg::Buf(&self.profiles),
            Arg::Buf(&self.history), Arg::Bytes(&self.bytes), Arg::U32(time),
        ], [p.grid as usize, 1, 1])
    }

    fn forward(&self) -> Result<Vec<f32>> {
        self.gpu.dispatch(&self.k.material, &[
            Arg::Buf(&self.model), Arg::Buf(&self.coefficients), Arg::Bytes(&self.bytes),
        ], [self.p.cells as usize, 1, 1])?;
        // Bounded command buffers avoid thousands of CPU/GPU synchronizations.
        for start in (0..self.p.nt).step_by(32) {
            self.gpu.batch(|b| {
                for time in start..(start + 32).min(self.p.nt) {
                    self.forward_step(b, time)?;
                }
                Ok(())
            })?;
        }
        let count = 3 * self.p.shots as usize * self.p.receivers as usize * self.p.nt as usize;
        Ok(self.gpu.read(&self.traces, count))
    }

    fn adjoint(&self, residual: &[f32]) -> Result<(Vec<f32>, Vec<f32>)> {
        let p = self.p;
        let grid = p.grid as usize;
        let adj = self.gpu.buffer_zeroed::<f32>(13 * grid);
        let derivative_adj = self.gpu.buffer_zeroed::<f32>(4 * grid);
        // Per-shot coefficients avoid floating atomics, including between shots.
        let coeff_adj = self.gpu.buffer_zeroed::<f32>(5 * grid);
        let nsamples = p.shots as usize * p.sources as usize * p.nt as usize;
        let source_adj = self.gpu.buffer_zeroed::<f32>(2 * nsamples);
        let residual = self.gpu.buffer_from(residual);
        for end in (1..=p.nt).rev().step_by(32) {
            let begin = end.saturating_sub(32);
            self.gpu.batch(|b| {
                for time in (begin..end).rev() {
                    b.dispatch(&self.k.reverse_stress, &[
                        Arg::Buf(&adj), Arg::Buf(&self.coefficients), Arg::Buf(&self.profiles),
                        Arg::Buf(&self.history), Arg::Buf(&coeff_adj), Arg::Buf(&derivative_adj),
                        Arg::Bytes(&self.bytes), Arg::U32(time),
                    ], [grid, 1, 1])?;
                    b.dispatch(&self.k.gather_velocity, &[
                        Arg::Buf(&adj), Arg::Buf(&derivative_adj), Arg::Bytes(&self.bytes),
                    ], [grid, 1, 1])?;
                    if p.source_groups != 0 {
                        b.dispatch(&self.k.reverse_source, &[
                            Arg::Buf(&adj), Arg::Buf(&self.coefficients), Arg::Buf(&self.sy),
                            Arg::Buf(&self.sx), Arg::Buf(&self.source_groups), Arg::Buf(&self.source_entries),
                            Arg::Buf(&coeff_adj), Arg::Buf(&source_adj), Arg::Bytes(&self.bytes), Arg::U32(time),
                        ], [p.source_groups as usize, 1, 1])?;
                    }
                    b.dispatch(&self.k.reverse_velocity, &[
                        Arg::Buf(&adj), Arg::Buf(&self.coefficients), Arg::Buf(&self.profiles),
                        Arg::Buf(&self.history), Arg::Buf(&coeff_adj), Arg::Buf(&derivative_adj),
                        Arg::Bytes(&self.bytes), Arg::U32(time),
                    ], [grid, 1, 1])?;
                    b.dispatch(&self.k.gather_stress, &[
                        Arg::Buf(&adj), Arg::Buf(&derivative_adj), Arg::Bytes(&self.bytes),
                    ], [grid, 1, 1])?;
                    if p.receiver_groups != 0 {
                        b.dispatch(&self.k.receiver_adjoint, &[
                            Arg::Buf(&adj), Arg::Buf(&residual), Arg::Buf(&self.receiver_groups),
                            Arg::Buf(&self.receiver_entries), Arg::Bytes(&self.bytes), Arg::U32(time),
                        ], [p.receiver_groups as usize, 1, 1])?;
                    }
                }
                Ok(())
            })?;
        }
        let padded_grad = self.gpu.buffer_zeroed::<f32>(3 * p.cells as usize);
        let model_cells = p.model_ny as usize * p.model_nx as usize;
        let model_grad = self.gpu.buffer_zeroed::<f32>(3 * model_cells);
        self.gpu.batch(|b| {
            b.dispatch(&self.k.material_adjoint, &[
                Arg::Buf(&self.model), Arg::Buf(&self.coefficients), Arg::Buf(&coeff_adj),
                Arg::Buf(&padded_grad), Arg::Bytes(&self.bytes),
            ], [p.cells as usize, 1, 1])?;
            b.dispatch(&self.k.unpad, &[
                Arg::Buf(&padded_grad), Arg::Buf(&model_grad), Arg::Bytes(&self.bytes),
            ], [model_cells, 1, 1])
        })?;
        Ok((self.gpu.read(&model_grad, 3 * model_cells), self.gpu.read(&source_adj, 2 * nsamples)))
    }
}

fn elastic(inp: &Inputs, out: &mut Outputs) -> Result<()> {
    let problem = Problem::load(inp, false)?;
    let solver = Solver::new(&problem)?;
    let traces = solver.forward()?;
    let shape = problem.output_shape();
    let n: usize = shape.iter().product();
    for (i, name) in ["vy", "vx", "p"].iter().enumerate() {
        out.f32(name, &shape, &traces[i * n..(i + 1) * n])?;
    }
    Ok(())
}

fn misfit_and_grad(inp: &Inputs, out: &mut Outputs) -> Result<()> {
    let problem = Problem::load(inp, true)?;
    let oy = inp.f32("observed_y")?;
    let ox = inp.f32("observed_x")?;
    shape(&oy.shape, &problem.output_shape(), "observed_y")?;
    shape(&ox.shape, &problem.output_shape(), "observed_x")?;
    let scale = inp.num("data_scale")?;
    if scale <= 0.0 || !scale.is_finite() || oy.data.is_empty() {
        return Err("positive data_scale and at least one receiver are required".into());
    }
    let solver = Solver::new(&problem)?;
    let traces = solver.forward()?;
    // Only this small scalar reduction runs on the host. It matches the frozen
    // objective definition without introducing a float32 sum into the loss.
    let n = oy.data.len();
    let normalization = 1.0 / (n as f64 * scale * scale);
    let mut sum = 0.0f64;
    let mut residual = Vec::with_capacity(2 * n);
    for (channel, obs) in [&oy.data, &ox.data].iter().enumerate() {
        for i in 0..n {
            let r = traces[channel * n + i] as f64 - obs[i] as f64;
            sum += r * r;
            residual.push((r * normalization) as f32);
        }
    }
    let (model_grad, source_grad) = solver.adjoint(&residual)?;
    out.f32("objective", &[], &[(0.5 * normalization * sum) as f32])?;
    let mshape = [problem.p.model_ny as usize, problem.p.model_nx as usize];
    let m = mshape[0] * mshape[1];
    for (i, name) in ["grad_vp", "grad_vs", "grad_rho"].iter().enumerate() {
        out.f32(name, &mshape, &model_grad[i * m..(i + 1) * m])?;
    }
    let s = problem.source_y.data.len();
    out.f32("grad_source_y", &problem.source_y.shape, &source_grad[..s])?;
    out.f32("grad_source_x", &problem.source_x.shape, &source_grad[s..])?;
    Ok(())
}

fn main() {
    outcrop_common::run(&[("elastic", elastic), ("misfit_and_grad", misfit_and_grad)]);
}
