//! Nodal, stress-free mantle convection. All field arithmetic runs on Metal.
//! The host validates inputs, builds transform tables, and encodes work.

use outcrop_common::gpu::{Arg, Batch, Buffer, Pipeline};
use outcrop_common::{Array, Gpu, Inputs, Outputs, Result};

mod npy;

fn main() {
    outcrop_common::run(&[("stokes", stokes), ("convection", convection)]);
}

fn input(inp: &Inputs) -> Result<(Array<f32>, usize, f32)> {
    // Inputs does not expose its directory. Use the same CLI argument as
    // outcrop_common::run, but avoid the shared parser's unchecked header slices.
    let dir = std::env::args_os().nth(2).ok_or("missing input directory")?;
    let t = npy::read_temperature(&std::path::PathBuf::from(dir).join("temperature.npy"))?;
    if t.shape.len() != 2 || t.shape[0] != t.shape[1] || t.shape[0] < 5 {
        return Err("temperature must have shape (N+1,N+1), N >= 4".into());
    }
    if t.data.iter().any(|v| !v.is_finite()) {
        return Err("temperature must be finite".into());
    }
    let n = t.shape[0] - 1;
    if n > u32::MAX as usize / (n + 1) {
        return Err("grid exceeds Metal's 32-bit indexing".into());
    }
    let ra = inp.num("rayleigh")?;
    if !ra.is_finite() || ra < 0.0 || ra > f32::MAX as f64 {
        return Err("rayleigh must be finite, nonnegative, and representable in float32".into());
    }
    Ok((t, n, ra as f32))
}

struct Solver {
    gpu: Gpu,
    n: usize,
    ra: f32,
    fft: bool,
    first: Pipeline,
    vertical: Pipeline,
    vertical_inverse: Option<Pipeline>,
    last: Pipeline,
    advance: Pipeline,
    velocity: Pipeline,
    diagnostics: Pipeline,
    transform: Buffer,
    inverse_biharmonic: Buffer,
    a: Buffer,
    b: Buffer,
    psi: Buffer,
}

impl Solver {
    fn new(n: usize, ra: f32) -> Result<Self> {
        let gpu = Gpu::new()?;
        // One 2N-point odd-extension FFT per line. Other grid sizes use the
        // same DST-I via a dense sine table, without padding the PDE grid.
        let fft = n.is_power_of_two() && n <= 128;
        let lib = gpu.compile(include_str!("../kernels/convection.metal"))?;
        let first = gpu.pipeline(&lib, if fft { "fft_rhs_x" } else { "dense_rhs_x" })?;
        let vertical = gpu.pipeline(&lib, if fft { "fft_solve_z" } else { "dense_forward_z" })?;
        let vertical_inverse = if fft {
            None
        } else {
            Some(gpu.pipeline(&lib, "dense_inverse_z")?)
        };
        let last = gpu.pipeline(&lib, if fft { "fft_inverse_x" } else { "dense_inverse_x" })?;
        let advance = gpu.pipeline(&lib, "midpoint_stage")?;
        let velocity = gpu.pipeline(&lib, "velocity_field")?;
        let diagnostics = gpu.pipeline(&lib, "observables")?;
        let side = n + 1;
        let size = side * side;
        let nf = n as f64;
        let transform = if fft {
            // exp(-2 pi i k/(2N)), rounded once from host double precision.
            let twiddles: Vec<[f32; 2]> = (0..n)
                .map(|k| {
                    let theta = -std::f64::consts::PI * k as f64 / nf;
                    if k == 0 {
                        [1.0, 0.0]
                    } else if 2 * k == n {
                        [0.0, -1.0]
                    } else {
                        [theta.cos() as f32, theta.sin() as f32]
                    }
                })
                .collect();
            gpu.buffer_from(&twiddles)
        } else {
            let mut sine = vec![0.0f32; size];
            for k in 1..n {
                for i in 1..n {
                    sine[k * side + i] =
                        (std::f64::consts::PI * (k * i) as f64 / nf).sin() as f32;
                }
            }
            gpu.buffer_from(&sine)
        };
        // L's eigenvalues are -4 N^2(sin^2(kx*pi/2N)+sin^2(kz*pi/2N)).
        // Two Poisson solves multiply by 1/lambda^2. Absorb one inverse-DST
        // normalization (2/N) here; the last x transform applies the other.
        let eigenvalues: Vec<f64> = (0..=n)
            .map(|k| -4.0 * nf * nf * (std::f64::consts::PI * k as f64 / (2.0 * nf)).sin().powi(2))
            .collect();
        let mut inverse = vec![0.0f32; size];
        for kx in 1..n {
            for kz in 1..n {
                inverse[kx * side + kz] =
                    ((2.0 / nf) / (eigenvalues[kx] + eigenvalues[kz]).powi(2)) as f32;
            }
        }
        let inverse_biharmonic = gpu.buffer_from(&inverse);
        let a = gpu.buffer_zeroed::<f32>(size);
        let b = gpu.buffer_zeroed::<f32>(size);
        let psi = gpu.buffer_zeroed::<f32>(size);
        Ok(Self {
            gpu, n, ra, fft, first, vertical, vertical_inverse, last, advance,
            velocity, diagnostics, transform, inverse_biharmonic, a, b, psi,
        })
    }

    fn fft_grid(&self) -> [usize; 3] {
        // Batch uses at most 256 threads/group. If a device chooses fewer,
        // extra groups exit uniformly; each useful group strides its work.
        [256 * (self.n - 1), 1, 1]
    }

    fn stokes(&self, batch: &mut Batch, t: &Buffer) -> Result<()> {
        if self.ra == 0.0 {
            // psi was initialized to exact zero and stays zero for this call.
            return Ok(());
        }
        let n = self.n as u32;
        let dense_grid = [(self.n - 1) * (self.n - 1), 1, 1];
        batch.dispatch(
            &self.first,
            &[Arg::Buf(t), Arg::Buf(&self.a), Arg::Buf(&self.transform), Arg::U32(n), Arg::F32(self.ra)],
            if self.fft { self.fft_grid() } else { dense_grid },
        )?;
        batch.dispatch(
            &self.vertical,
            &[Arg::Buf(&self.a), Arg::Buf(&self.b), Arg::Buf(&self.transform), Arg::Buf(&self.inverse_biharmonic), Arg::U32(n)],
            if self.fft { self.fft_grid() } else { dense_grid },
        )?;
        let transformed = if let Some(inverse_z) = &self.vertical_inverse {
            batch.dispatch(
                inverse_z,
                &[Arg::Buf(&self.b), Arg::Buf(&self.a), Arg::Buf(&self.transform), Arg::U32(n)],
                dense_grid,
            )?;
            &self.a
        } else {
            &self.b
        };
        batch.dispatch(
            &self.last,
            &[Arg::Buf(transformed), Arg::Buf(&self.psi), Arg::Buf(&self.transform), Arg::U32(n)],
            if self.fft { self.fft_grid() } else { dense_grid },
        )
    }

    fn stage(&self, batch: &mut Batch, eval: &Buffer, base: &Buffer, result: &Buffer, dt: f32) -> Result<()> {
        self.stokes(batch, eval)?;
        batch.dispatch(
            &self.advance,
            &[Arg::Buf(eval), Arg::Buf(base), Arg::Buf(&self.psi), Arg::Buf(result), Arg::U32(self.n as u32), Arg::F32(dt)],
            [(self.n + 1) * (self.n + 1), 1, 1],
        )
    }

    fn final_velocity(&self, batch: &mut Batch, t: &Buffer, u: &Buffer, w: &Buffer) -> Result<()> {
        self.stokes(batch, t)?;
        batch.dispatch(
            &self.velocity,
            &[Arg::Buf(&self.psi), Arg::Buf(u), Arg::Buf(w), Arg::U32(self.n as u32)],
            [(self.n + 1) * (self.n + 1), 1, 1],
        )
    }
}

fn stokes(inp: &Inputs, out: &mut Outputs) -> Result<()> {
    let (temperature, n, ra) = input(inp)?;
    let s = Solver::new(n, ra)?;
    let t = s.gpu.buffer_from(&temperature.data);
    let size = temperature.data.len();
    let u = s.gpu.buffer_zeroed::<f32>(size);
    let w = s.gpu.buffer_zeroed::<f32>(size);
    s.gpu.batch(|batch| s.final_velocity(batch, &t, &u, &w))?;
    for (name, buf) in [("psi", &s.psi), ("u", &u), ("w", &w)] {
        out.f32(name, &temperature.shape, &s.gpu.read::<f32>(buf, size))?;
    }
    Ok(())
}

fn convection(inp: &Inputs, out: &mut Outputs) -> Result<()> {
    let (temperature, n, ra) = input(inp)?;
    let dt = inp.num("dt")?;
    let steps = inp.int("steps")?;
    if !dt.is_finite() || dt <= 0.0 || !(dt as f32).is_finite() || dt as f32 == 0.0 || steps < 0 {
        return Err("dt must be positive and finite; steps must be nonnegative".into());
    }
    let side = n + 1;
    if temperature.data[..side].iter().any(|&v| v != 1.0)
        || temperature.data[n * side..].iter().any(|&v| v != 0.0)
    {
        return Err("convection requires bottom temperature 1 and top temperature 0".into());
    }
    let s = Solver::new(n, ra)?;
    let size = temperature.data.len();
    let t = s.gpu.buffer_from(&temperature.data);
    let midpoint = s.gpu.buffer_zeroed::<f32>(size);
    let u = s.gpu.buffer_zeroed::<f32>(size);
    let w = s.gpu.buffer_zeroed::<f32>(size);
    // Bound command-buffer size while avoiding a host/GPU wait at each stage.
    // No steps are omitted, even near a steady state.
    let mut done = 0i64;
    while done < steps {
        let count = (steps - done).min(64);
        s.gpu.batch(|batch| {
            for _ in 0..count {
                s.stage(batch, &t, &t, &midpoint, 0.5 * dt as f32)?;
                s.stage(batch, &midpoint, &t, &t, dt as f32)?;
            }
            Ok(())
        })?;
        done += count;
    }
    let diagnostics = s.gpu.buffer_zeroed::<f32>(3);
    s.gpu.batch(|batch| {
        s.final_velocity(batch, &t, &u, &w)?;
        batch.dispatch(
            &s.diagnostics,
            &[Arg::Buf(&t), Arg::Buf(&u), Arg::Buf(&w), Arg::Buf(&diagnostics), Arg::U32(n as u32)],
            [256, 1, 1],
        )
    })?;
    for (name, buf) in [("temperature", &t), ("u", &u), ("w", &w)] {
        out.f32(name, &temperature.shape, &s.gpu.read::<f32>(buf, size))?;
    }
    let d = s.gpu.read::<f32>(&diagnostics, 3);
    for (name, value) in ["nu_top", "nu_bottom", "vrms"].iter().zip(d) {
        // Outputs::scalar writes f64; this API explicitly requires f32 scalars.
        out.f32(name, &[], &[value])?;
    }
    Ok(())
}
