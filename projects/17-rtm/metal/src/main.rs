//! Fourth-order acoustic Born modelling and its Euclidean transpose.
//! Propagation, the hand-written adjoint, and reductions run on Metal.
mod buffers;
use buffers::{from_slice as buffer_from, zeroed as buffer_zeroed};
use outcrop_common::gpu::Arg;
use outcrop_common::{Array, Gpu, Inputs, Outputs, Result};

fn main() {
    outcrop_common::run(&[("rtm", rtm), ("born_misfit_and_grad", born_misfit_and_grad)]);
}

fn require(ok: bool, message: &str) -> Result<()> {
    if ok { Ok(()) } else { Err(message.into()) }
}

fn product(dims: &[usize]) -> Result<usize> {
    let n = dims.iter().try_fold(1usize, |a, &b| a.checked_mul(b))
        .ok_or("array size overflow")?;
    require(n <= u32::MAX as usize, "array exceeds Metal index range")?;
    Ok(n)
}

fn shape<T>(a: &Array<T>, expected: &[usize], name: &str) -> Result<()> {
    require(a.shape == expected, &format!("{name}: invalid shape"))
}

// Nodal CFS-PML from projects 13/14. Include the two exterior FD guards
// when differentiating b, even though their wavefields stay zero. Fixed
// setup coefficients are calculated in f64 then rounded once; all wave
// propagation, adjoint arithmetic, and reductions are float32.
fn profile(n: usize, width: usize, h: f64, dt: f64, freq: f64, vmax: f64) -> Vec<[f32; 4]> {
    let mut result = vec![[0.0f32; 4]; n];
    if width == 0 { return result; }
    let mut a = vec![0.0f64; n];
    let mut b = vec![0.0f64; n];
    let left = (width + 2) as f64;
    let right = (n - 1 - 2 - width) as f64;
    let sigma0 = 3.0 * vmax * 1000.0f64.ln() / (2.0 * width as f64 * h);
    for i in 0..n {
        let fraction = ((left - i as f64) / width as f64).clamp(0.0, 1.0)
            .max(((i as f64 - right) / width as f64).clamp(0.0, 1.0));
        if fraction > 0.0 {
            let sigma = sigma0 * fraction * fraction;
            let rate = sigma + std::f64::consts::PI * freq * (1.0 - fraction);
            a[i] = (-rate * dt).exp();
            b[i] = (sigma / rate) * (-rate * dt).exp_m1();
        }
    }
    for i in 0..n {
        result[i][0] = a[i] as f32;
        result[i][1] = b[i] as f32;
        if i >= 2 && i < n - 2 {
            result[i][2] = (((2.0 / 3.0) * (b[i + 1] - b[i - 1])
                - (1.0 / 12.0) * (b[i + 2] - b[i - 2])) / h) as f32;
        }
    }
    result
}

// Full padded-grid indices including the shot stride. Point order is
// preserved; coincident sources are handled without floating atomics.
fn locations(a: &Array<i64>, nz: usize, nx: usize, plane: usize, nxp: usize, pad: usize) -> Result<Vec<u32>> {
    a.data.as_chunks::<2>().0.iter().enumerate().map(|(j, q)| {
        require(q[0] >= 0 && q[0] < nz as i64 && q[1] >= 0 && q[1] < nx as i64,
                "source/receiver location outside the physical model")?;
        Ok(((j / a.shape[1]) * plane + (q[0] as usize + pad) * nxp
            + q[1] as usize + pad) as u32)
    }).collect()
}

fn point_lists(indices: &[u32], cells: usize) -> (Vec<u32>, Vec<u32>) {
    let mut offsets = vec![0u32; cells + 1];
    for &i in indices { offsets[i as usize + 1] += 1; }
    for i in 1..offsets.len() { offsets[i] += offsets[i - 1]; }
    let mut cursor = offsets[..cells].to_vec();
    let mut ids = vec![0u32; indices.len()];
    for (j, &i) in indices.iter().enumerate() {
        ids[cursor[i as usize] as usize] = j as u32;
        cursor[i as usize] += 1;
    }
    (offsets, ids)
}

fn calculate(inp: &Inputs, out: &mut Outputs, misfit: bool) -> Result<()> {
    let velocity = inp.f32("velocity")?;
    let source = inp.f32("source_amplitudes")?;
    let sl = inp.i64("source_locations")?;
    let rl = inp.i64("receiver_locations")?;
    let data = inp.f32("data")?;
    require(velocity.shape.len() == 2, "velocity must have shape (nz,nx)")?;
    require(source.shape.len() == 3, "source_amplitudes must have shape (shots,sources,nt)")?;
    require(rl.shape.len() == 3, "receiver_locations must have shape (shots,receivers,2)")?;
    let (nz, nx) = (velocity.shape[0], velocity.shape[1]);
    let (shots, ns, nt) = (source.shape[0], source.shape[1], source.shape[2]);
    let nr = rl.shape[1];
    require(nz > 0 && nx > 0, "model dimensions must be positive")?;
    shape(&sl, &[shots, ns, 2], "source_locations")?;
    shape(&rl, &[shots, nr, 2], "receiver_locations")?;
    shape(&data, &[shots, nr, nt], "data")?;
    require(velocity.data.iter().all(|&v| v.is_finite() && v > 0.0), "invalid velocity")?;
    require(source.data.iter().chain(&data.data).all(|v| v.is_finite()), "nonfinite source or data")?;
    let scatter = if misfit {
        let a = inp.f32("scatter")?;
        shape(&a, &[nz, nx], "scatter")?;
        require(a.data.iter().all(|v| v.is_finite()), "nonfinite scatter")?;
        a.data
    } else { vec![0.0; nz * nx] };
    let spacing = inp.nums("grid_spacing")?;
    require(spacing.len() == 2 && spacing.iter().all(|h| h.is_finite() && *h > 0.0),
            "grid_spacing must be [dz,dx], both positive")?;
    let (dz, dx) = (spacing[0], spacing[1]);
    let dt = inp.num("dt")?;
    let freq = inp.num("pml_freq")?;
    require(dt.is_finite() && dt > 0.0 && freq.is_finite() && freq > 0.0,
            "dt and pml_freq must be finite and positive")?;
    let width = inp.int("pml_width")?;
    require(width >= 0, "pml_width must be nonnegative")?;
    let width = usize::try_from(width)?;
    let pad = width.checked_add(2).ok_or("PML size overflow")?;
    let twice_pad = pad.checked_mul(2).ok_or("PML size overflow")?;
    let nzp = nz.checked_add(twice_pad).ok_or("grid size overflow")?;
    let nxp = nx.checked_add(twice_pad).ok_or("grid size overflow")?;
    let plane = product(&[nzp, nxp])?;
    let cells = product(&[shots, plane])?;
    let traces = product(&[shots, nr])?;
    let samples = product(&[traces, nt])?;
    let history_len = product(&[nt, cells])?;
    product(&[8, cells])?;
    product(&[shots, ns, nt])?;

    // Empty acquisitions retain the exact-zero linear-map convention.
    if nt == 0 || shots == 0 || nr == 0 {
        let zero = vec![0.0f32; nz * nx];
        if misfit {
            out.f32("predicted", &[shots, nr, nt], &vec![0.0; samples])?;
            out.f32("loss", &[], &[0.0f32])?;
            return out.f32("gradient", &[nz, nx], &zero);
        }
        return out.f32("image", &[nz, nx], &zero);
    }

    // Params in rtm.metal: ten uints followed by five floats (60 bytes).
    // Time is passed separately so these bytes stay fixed per survey.
    let mut params = Vec::with_capacity(60);
    for v in [nz, nx, nzp, nxp, shots, nt, ns, nr, pad, usize::from(misfit)] {
        params.extend_from_slice(&(v as u32).to_ne_bytes());
    }
    for v in [1.0 / dz, 1.0 / dx, 1.0 / (dz * dz), 1.0 / (dx * dx), dt * dt] {
        params.extend_from_slice(&(v as f32).to_ne_bytes());
    }
    let vmax = velocity.data.iter().copied().fold(0.0f32, f32::max) as f64;
    let si = locations(&sl, nz, nx, plane, nxp, pad)?;
    let ri = locations(&rl, nz, nx, plane, nxp, pad)?;
    let (so, sid) = point_lists(&si, cells);
    let (ro, rid) = point_lists(&ri, cells);

    let gpu = Gpu::new()?;
    let lib = gpu.compile(include_str!("../kernels/rtm.metal"))?;
    let setup = gpu.pipeline(&lib, "extend_model")?;
    let record = gpu.pipeline(&lib, "record_receivers")?;
    let forward = gpu.pipeline(&lib, "forward_step")?;
    let adj_local = gpu.pipeline(&lib, "adjoint_local")?;
    let adj_gather = gpu.pipeline(&lib, "adjoint_gather")?;
    let fold = gpu.pipeline(&lib, "fold_gradient")?;
    let bv = buffer_from(&gpu, &velocity.data)?;
    let bscatter = buffer_from(&gpu, &scatter)?;
    let coeff = buffer_zeroed::<[f32; 2]>(&gpu, plane)?;
    let padded_scatter = buffer_zeroed::<f32>(&gpu, plane)?;
    let pz = buffer_from(&gpu, &profile(nzp, width, dz, dt, freq, vmax))?;
    let px = buffer_from(&gpu, &profile(nxp, width, dx, dt, freq, vmax))?;
    let bs = buffer_from(&gpu, &source.data)?;
    let bso = buffer_from(&gpu, &so)?;
    let bsid = buffer_from(&gpu, &sid)?;
    let bri = buffer_from(&gpu, &ri)?;
    let bro = buffer_from(&gpu, &ro)?;
    let brid = buffer_from(&gpu, &rid)?;
    let bd = buffer_from(&gpu, &data.data)?;
    let predicted = buffer_zeroed::<f32>(&gpu, samples)?;
    let history = buffer_zeroed::<f32>(&gpu, history_len)?;
    let u: Vec<_> = (0..3).map(|_| buffer_zeroed::<f32>(&gpu, cells)).collect::<Result<_>>()?;
    let s: Vec<_> = (0..3).map(|_| buffer_zeroed::<f32>(&gpu, cells)).collect::<Result<_>>()?;
    let memory: Vec<_> = (0..2).map(|_| buffer_zeroed::<f32>(&gpu, 8 * cells)).collect::<Result<_>>()?;
    gpu.dispatch(&setup, &[Arg::Bytes(&params), Arg::Buf(&bv), Arg::Buf(&bscatter),
        Arg::Buf(&coeff), Arg::Buf(&padded_scatter)], [plane, 1, 1])?;

    for start in (0..nt).step_by(32) {
        gpu.batch(|batch| {
            for k in start..(start + 32).min(nt) {
                if misfit {
                    batch.dispatch(&record, &[Arg::Bytes(&params), Arg::Buf(&s[k % 3]),
                        Arg::Buf(&bri), Arg::Buf(&predicted), Arg::U32(k as u32)], [traces, 1, 1])?;
                }
                // Sample k reads s^k; the final forcing cannot affect any
                // returned receiver sample, so there is no final update.
                if k + 1 < nt {
                    batch.dispatch(&forward, &[Arg::Bytes(&params), Arg::Buf(&coeff),
                        Arg::Buf(&padded_scatter), Arg::Buf(&pz), Arg::Buf(&px), Arg::Buf(&bs),
                        Arg::Buf(&bso), Arg::Buf(&bsid), Arg::Buf(&u[k % 3]),
                        Arg::Buf(&u[(k + 2) % 3]), Arg::Buf(&u[(k + 1) % 3]),
                        Arg::Buf(&s[k % 3]), Arg::Buf(&s[(k + 2) % 3]), Arg::Buf(&s[(k + 1) % 3]),
                        Arg::Buf(&memory[k % 2]), Arg::Buf(&memory[1 - k % 2]),
                        Arg::Buf(&history), Arg::U32(k as u32)], [cells, 1, 1])?;
                }
            }
            Ok(())
        })?;
    }

    let lambda: Vec<_> = (0..3).map(|_| buffer_zeroed::<f32>(&gpu, cells)).collect::<Result<_>>()?;
    let adj_memory: Vec<_> = (0..2).map(|_| buffer_zeroed::<f32>(&gpu, 4 * cells)).collect::<Result<_>>()?;
    let temporary = buffer_zeroed::<f32>(&gpu, 6 * cells)?;
    let extended_gradient = buffer_zeroed::<f32>(&gpu, cells)?;
    let gradient = buffer_zeroed::<f32>(&gpu, nz * nx)?;
    for start in (0..nt).step_by(32) {
        gpu.batch(|batch| {
            for j in start..(start + 32).min(nt) {
                let k = (nt - 1 - j) as u32;
                batch.dispatch(&adj_local, &[Arg::Bytes(&params), Arg::Buf(&coeff),
                    Arg::Buf(&pz), Arg::Buf(&px), Arg::Buf(&lambda[j % 3]),
                    Arg::Buf(&adj_memory[j % 2]), Arg::Buf(&temporary)], [cells, 1, 1])?;
                batch.dispatch(&adj_gather, &[Arg::Bytes(&params), Arg::Buf(&coeff),
                    Arg::Buf(&pz), Arg::Buf(&px), Arg::Buf(&lambda[j % 3]),
                    Arg::Buf(&lambda[(j + 2) % 3]), Arg::Buf(&lambda[(j + 1) % 3]),
                    Arg::Buf(&adj_memory[j % 2]), Arg::Buf(&adj_memory[1 - j % 2]),
                    Arg::Buf(&temporary), Arg::Buf(&history), Arg::Buf(&extended_gradient),
                    Arg::Buf(&predicted), Arg::Buf(&bd), Arg::Buf(&bro), Arg::Buf(&brid),
                    Arg::U32(k)], [cells, 1, 1])?;
            }
            Ok(())
        })?;
    }
    gpu.dispatch(&fold, &[Arg::Bytes(&params), Arg::Buf(&extended_gradient), Arg::Buf(&gradient)],
                 [nz * nx, 1, 1])?;
    if misfit {
        let loss_blocks = gpu.pipeline(&lib, "loss_blocks")?;
        let loss_finish = gpu.pipeline(&lib, "loss_finish")?;
        let blocks = samples.div_ceil(128);
        let partial = buffer_zeroed::<f32>(&gpu, blocks)?;
        let loss = buffer_zeroed::<f32>(&gpu, 1)?;
        gpu.batch(|batch| {
            batch.dispatch(&loss_blocks, &[Arg::Bytes(&params), Arg::Buf(&predicted),
                Arg::Buf(&bd), Arg::Buf(&partial)], [blocks, 1, 1])?;
            batch.dispatch(&loss_finish, &[Arg::Buf(&partial), Arg::Buf(&loss),
                Arg::U32(blocks as u32)], [1, 1, 1])
        })?;
        out.f32("predicted", &[shots, nr, nt], &gpu.read::<f32>(&predicted, samples))?;
        out.f32("loss", &[], &gpu.read::<f32>(&loss, 1))?;
        out.f32("gradient", &[nz, nx], &gpu.read::<f32>(&gradient, nz * nx))
    } else {
        out.f32("image", &[nz, nx], &gpu.read::<f32>(&gradient, nz * nx))
    }
}

fn rtm(inp: &Inputs, out: &mut Outputs) -> Result<()> { calculate(inp, out, false) }
fn born_misfit_and_grad(inp: &Inputs, out: &mut Outputs) -> Result<()> { calculate(inp, out, true) }
