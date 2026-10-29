//! Acoustic least squares with a hand-written discrete C-PML adjoint.
mod allocation;

use allocation::Buffers;
use outcrop_common::gpu::Arg;
use outcrop_common::{Array, Gpu, Inputs, Outputs, Result};

fn main() {
    outcrop_common::run(&[("misfit_and_grad", misfit_and_grad)]);
}

fn require(condition: bool, message: &str) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(message.into())
    }
}

fn product(dims: &[usize]) -> Result<usize> {
    let n = dims.iter().try_fold(1usize, |a, &b| a.checked_mul(b));
    let n = n.ok_or("array size overflow")?;
    require(n <= u32::MAX as usize, "array exceeds Metal index range")?;
    Ok(n)
}

fn shape<T>(a: &Array<T>, expected: &[usize], name: &str) -> Result<()> {
    require(a.shape == expected, &format!("{name}: invalid shape"))
}

fn positive(inp: &Inputs, name: &str) -> Result<f32> {
    let v = inp.num(name)? as f32;
    require(v.is_finite() && v > 0.0, &format!("{name} must be finite and positive"))?;
    Ok(v)
}

/// Fixed coefficients, calculated once in float32, independently of velocity.
/// The guard profiles are retained when taking the centered derivative of b.
fn profile(n: usize, w: usize, h: f32, dt: f32, freq: f32, max_vel: f32) -> Vec<[f32; 4]> {
    let mut p = vec![[0.0f32; 4]; n];
    if w == 0 {
        return p;
    }
    let left = (w + 2) as f32;
    let right = (n - 1 - 2 - w) as f32;
    let sigma0 = -3.0 * max_vel * 0.001f32.ln() / (2.0 * w as f32 * h);
    for (i, value) in p.iter_mut().enumerate() {
        let tl = ((left - i as f32) / w as f32).clamp(0.0, 1.0);
        let tr = ((i as f32 - right) / w as f32).clamp(0.0, 1.0);
        let t = tl.max(tr);
        let sigma = sigma0 * (tl * tl + tr * tr);
        let alpha = std::f32::consts::PI * freq * (1.0 - t);
        let rate = sigma + alpha;
        let e = -rate * dt;
        value[0] = if t > 0.0 { e.exp() } else { 0.0 };
        value[1] = if rate > 0.0 { sigma / rate * e.exp_m1() } else { 0.0 };
    }
    for i in 2..n - 2 {
        p[i][2] = ((2.0 / 3.0) * (p[i + 1][1] - p[i - 1][1])
            - (1.0 / 12.0) * (p[i + 2][1] - p[i - 2][1]))
            / h;
    }
    p
}

/// Point indices include the shot stride. Repeated points remain repeated.
fn locations(
    a: &Array<i64>,
    nz: usize,
    nx: usize,
    nzp: usize,
    nxp: usize,
    pad: usize,
) -> Result<Vec<u32>> {
    let per_shot = a.shape[1];
    a.data
        .as_chunks::<2>()
        .0
        .iter()
        .enumerate()
        .map(|(j, loc)| {
            require(
                loc[0] >= 0 && loc[0] < nz as i64 && loc[1] >= 0 && loc[1] < nx as i64,
                "source/receiver outside original model",
            )?;
            Ok(((j / per_shot) * nzp * nxp
                + (loc[0] as usize + pad) * nxp
                + loc[1] as usize
                + pad) as u32)
        })
        .collect()
}

/// Grid-to-point lists let each GPU thread sum its own sources/residuals.
/// This handles coincident points without scatter races or floating atomics.
fn point_lists(indices: &[u32], cells: usize) -> (Vec<u32>, Vec<u32>) {
    let mut offsets = vec![0u32; cells + 1];
    for &i in indices {
        offsets[i as usize + 1] += 1;
    }
    for i in 1..offsets.len() {
        offsets[i] += offsets[i - 1];
    }
    let mut cursor = offsets[..cells].to_vec();
    let mut ids = vec![0u32; indices.len()];
    for (j, &i) in indices.iter().enumerate() {
        ids[cursor[i as usize] as usize] = j as u32;
        cursor[i as usize] += 1;
    }
    (offsets, ids)
}

fn misfit_and_grad(inp: &Inputs, out: &mut Outputs) -> Result<()> {
    let velocity = inp.f32("velocity")?;
    let source = inp.f32("source_amplitudes")?;
    let source_locations = inp.i64("source_locations")?;
    let receiver_locations = inp.i64("receiver_locations")?;
    let observed = inp.f32("observed")?;
    require(velocity.shape.len() == 2, "velocity must have shape [nz,nx]")?;
    require(source.shape.len() == 3, "source_amplitudes must have shape [shots,sources,nt]")?;
    require(observed.shape.len() == 3, "observed must have shape [shots,receivers,nt]")?;
    let (nz, nx) = (velocity.shape[0], velocity.shape[1]);
    let (shots, sources, nt) = (source.shape[0], source.shape[1], source.shape[2]);
    let receivers = observed.shape[1];
    require(nz > 0 && nx > 0 && shots > 0 && receivers > 0 && nt > 0, "empty model or acquisition")?;
    shape(&source_locations, &[shots, sources, 2], "source_locations")?;
    shape(&receiver_locations, &[shots, receivers, 2], "receiver_locations")?;
    shape(&observed, &[shots, receivers, nt], "observed")?;
    require(velocity.data.iter().all(|v| v.is_finite() && *v > 0.0), "invalid velocity")?;
    require(source.data.iter().chain(&observed.data).all(|v| v.is_finite()), "nonfinite acquisition")?;
    require(inp.int("accuracy")? == 4, "only accuracy=4 is supported")?;
    let width = inp.int("pml_width")?;
    require(width >= 0, "pml_width must be nonnegative")?;
    let w = usize::try_from(width)?;
    let pad = w.checked_add(2).ok_or("PML size overflow")?;
    let twice_pad = pad.checked_mul(2).ok_or("PML size overflow")?;
    let nzp = nz.checked_add(twice_pad).ok_or("grid size overflow")?;
    let nxp = nx.checked_add(twice_pad).ok_or("grid size overflow")?;
    let plane = product(&[nzp, nxp])?;
    let cells = product(&[shots, plane])?;
    let samples = product(&[shots, receivers, nt])?;
    let traces = product(&[shots, receivers])?;
    let history_len = product(&[nt, cells])?;
    product(&[shots, sources, nt])?;
    let h = positive(inp, "grid_spacing")?;
    let dt = positive(inp, "dt")?;
    let max_vel = positive(inp, "max_vel")?;
    let freq = inp.num("pml_freq")? as f32;
    require(freq.is_finite() && freq >= 0.0, "invalid pml_freq")?;

    // Exact byte layout of Params in fwi.metal: eight uints, four floats.
    let mut params = Vec::with_capacity(48);
    for value in [nz, nx, nzp, nxp, shots, nt, receivers, pad] {
        params.extend_from_slice(&(value as u32).to_ne_bytes());
    }
    for value in [1.0 / h, 1.0 / (h * h), dt * dt, 1.0 / samples as f32] {
        params.extend_from_slice(&value.to_ne_bytes());
    }

    let source_indices = locations(&source_locations, nz, nx, nzp, nxp, pad)?;
    let receiver_indices = locations(&receiver_locations, nz, nx, nzp, nxp, pad)?;
    let (source_offsets, source_ids) = point_lists(&source_indices, cells);
    let (receiver_offsets, receiver_ids) = point_lists(&receiver_indices, cells);

    let gpu = Gpu::new()?;
    let lib = gpu.compile(include_str!("../kernels/fwi.metal"))?;
    let setup = gpu.pipeline(&lib, "extend_velocity")?;
    let record = gpu.pipeline(&lib, "record_receivers")?;
    let forward = gpu.pipeline(&lib, "forward_step")?;
    let loss_blocks = gpu.pipeline(&lib, "loss_blocks")?;
    let loss_finish = gpu.pipeline(&lib, "loss_finish")?;
    let adj_local = gpu.pipeline(&lib, "adjoint_local")?;
    let adj_gather = gpu.pipeline(&lib, "adjoint_gather")?;
    let fold = gpu.pipeline(&lib, "fold_gradient")?;

    let buffers = Buffers::new(&gpu);
    let bv = buffers.copied(&velocity.data, "velocity")?;
    let coefficients = buffers.zeroed::<[f32; 2]>(plane, "coefficients")?;
    let pz = buffers.copied(&profile(nzp, w, h, dt, freq, max_vel), "z profile")?;
    let px = buffers.copied(&profile(nxp, w, h, dt, freq, max_vel), "x profile")?;
    let bs = buffers.copied(&source.data, "source amplitudes")?;
    let bso = buffers.copied(&source_offsets, "source offsets")?;
    let bsi = buffers.copied(&source_ids, "source ids")?;
    let bri = buffers.copied(&receiver_indices, "receiver indices")?;
    let bro = buffers.copied(&receiver_offsets, "receiver offsets")?;
    let brids = buffers.copied(&receiver_ids, "receiver ids")?;
    let obs = buffers.copied(&observed.data, "observed")?;
    let predicted = buffers.zeroed::<f32>(samples, "predicted")?;
    let history = buffers.zeroed::<f32>(history_len, "history")?;
    let u = [buffers.zeroed::<f32>(cells, "u[0]")?, buffers.zeroed::<f32>(cells, "u[1]")?];
    let delta = buffers.zeroed::<f32>(cells, "delta")?;
    let memory = [
        buffers.zeroed::<[f32; 4]>(cells, "memory[0]")?,
        buffers.zeroed::<[f32; 4]>(cells, "memory[1]")?,
    ];

    gpu.dispatch(&setup, &[Arg::Bytes(&params), Arg::Buf(&bv), Arg::Buf(&coefficients)], [plane, 1, 1])?;
    // Chunk command buffers to bound command lifetime on larger acquisitions.
    for start in (0..nt).step_by(32) {
        gpu.batch(|b| {
            for k in start..(start + 32).min(nt) {
                let old = k % 2;
                let new = 1 - old;
                b.dispatch(&record, &[
                    Arg::Bytes(&params), Arg::Buf(&u[old]), Arg::Buf(&bri),
                    Arg::Buf(&predicted), Arg::U32(k as u32),
                ], [traces, 1, 1])?;
                b.dispatch(&forward, &[
                    Arg::Bytes(&params), Arg::Buf(&coefficients), Arg::Buf(&pz), Arg::Buf(&px),
                    Arg::Buf(&bs), Arg::Buf(&bso), Arg::Buf(&bsi),
                    Arg::Buf(&u[old]), Arg::Buf(&delta), Arg::Buf(&memory[old]),
                    Arg::Buf(&u[new]), Arg::Buf(&memory[new]), Arg::Buf(&history), Arg::U32(k as u32),
                ], [cells, 1, 1])?;
            }
            Ok(())
        })?;
    }
    drop(u);
    drop(delta);
    drop(memory);

    let blocks = samples.div_ceil(128);
    let partial_loss = buffers.zeroed::<f32>(blocks, "partial loss")?;
    let objective = buffers.zeroed::<f32>(1, "objective")?;
    gpu.batch(|b| {
        b.dispatch(&loss_blocks, &[
            Arg::Bytes(&params), Arg::Buf(&predicted), Arg::Buf(&obs), Arg::Buf(&partial_loss),
        ], [blocks, 1, 1])?;
        b.dispatch(&loss_finish, &[
            Arg::Bytes(&params), Arg::Buf(&partial_loss), Arg::Buf(&objective), Arg::U32(blocks as u32),
        ], [1, 1, 1])
    })?;

    let lambda_u = buffers.zeroed::<f32>(cells, "lambda u")?;
    let lambda_delta = buffers.zeroed::<f32>(cells, "lambda delta")?;
    let lambda_memory = [
        buffers.zeroed::<[f32; 4]>(cells, "lambda memory[0]")?,
        buffers.zeroed::<[f32; 4]>(cells, "lambda memory[1]")?,
    ];
    let temporary = buffers.zeroed::<[f32; 4]>(cells, "adjoint temporary")?;
    let extended_gradient = buffers.zeroed::<f32>(cells, "extended gradient")?;
    for start in (0..nt).step_by(32) {
        gpu.batch(|b| {
            for reverse in start..(start + 32).min(nt) {
                let k = nt - 1 - reverse;
                let old = reverse % 2;
                let new = 1 - old;
                b.dispatch(&adj_local, &[
                    Arg::Bytes(&params), Arg::Buf(&coefficients), Arg::Buf(&pz), Arg::Buf(&px),
                    Arg::Buf(&lambda_u), Arg::Buf(&lambda_delta), Arg::Buf(&lambda_memory[old]),
                    Arg::Buf(&history), Arg::Buf(&temporary), Arg::Buf(&extended_gradient), Arg::U32(k as u32),
                ], [cells, 1, 1])?;
                b.dispatch(&adj_gather, &[
                    Arg::Bytes(&params), Arg::Buf(&coefficients), Arg::Buf(&pz), Arg::Buf(&px),
                    Arg::Buf(&temporary), Arg::Buf(&lambda_memory[old]), Arg::Buf(&lambda_memory[new]),
                    Arg::Buf(&lambda_u), Arg::Buf(&lambda_delta), Arg::Buf(&bro), Arg::Buf(&brids),
                    Arg::Buf(&predicted), Arg::Buf(&obs), Arg::U32(k as u32),
                ], [cells, 1, 1])?;
            }
            Ok(())
        })?;
    }

    let gradient = buffers.zeroed::<f32>(nz * nx, "gradient")?;
    gpu.dispatch(&fold, &[
        Arg::Bytes(&params), Arg::Buf(&extended_gradient), Arg::Buf(&gradient),
    ], [nz * nx, 1, 1])?;
    out.f32("predicted", &[shots, receivers, nt], &gpu.read::<f32>(&predicted, samples))?;
    out.f32("objective", &[], &gpu.read::<f32>(&objective, 1))?;
    out.f32("gradient", &[nz, nx], &gpu.read::<f32>(&gradient, nz * nx))?;
    Ok(())
}
