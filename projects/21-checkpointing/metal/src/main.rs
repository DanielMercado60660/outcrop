//! Acoustic C-PML with a discrete hand adjoint and optimal recursive replay.
mod planner;
mod device;

use outcrop_common::gpu::{Arg, Buffer, Pipeline};
use outcrop_common::{Array, Inputs, Outputs, Result};
use device::Device as Gpu;
use planner::{Event, Plan};

fn main() {
    device::autorelease_pool(|| outcrop_common::run(&[("misfit_and_grad", misfit_and_grad)]));
}

fn require(ok: bool, message: &str) -> Result<()> {
    if ok { Ok(()) } else { Err(message.into()) }
}

fn shape<T>(a: &Array<T>, expected: &[usize], name: &str) -> Result<()> {
    require(a.shape == expected, &format!("invalid {name} shape"))
}

fn product(dims: &[usize]) -> Result<usize> {
    let n = dims.iter().try_fold(1usize, |a, &b| a.checked_mul(b)).ok_or("size overflow")?;
    require(n <= u32::MAX as usize, "array exceeds Metal indexing range")?;
    Ok(n)
}

/// Logical retained bytes are an int64 audit, not a Metal buffer index.
/// Multiply the occupancy first: no retained states means zero bytes even
/// if a hypothetical single restart would exceed the reporting range.
fn checkpoint_payload(peak: usize, shots: usize, nz: usize, nx: usize) -> Result<i64> {
    if peak == 0 { return Ok(0); }
    [peak, 6, shots, nz, nx, 4].iter().try_fold(1i64, |bytes, &factor| {
        bytes.checked_mul(i64::try_from(factor)?)
            .ok_or_else(|| "checkpoint byte count exceeds int64".into())
    })
}

fn positive(x: f64) -> bool { x.is_finite() && x > 0.0 }

/// Fixed C-PML design, shared with project 13. Profiles include the constant
/// zero stencil guards; D(b) uses the same centered first derivative as D(u).
#[allow(clippy::too_many_arguments)]
fn profile(n: usize, low: usize, high: usize, r: usize, h: f64,
           dt: f64, freq: f64, vmax: f64) -> Vec<[f32; 4]> {
    let mut p = vec![[0.0f32; 4]; n];
    for (i, value) in p.iter_mut().enumerate() {
        let ql = if low == 0 { 0.0 } else {
            ((r as f64 + low as f64 - i as f64) / low as f64).clamp(0.0, 1.0)
        };
        let qh = if high == 0 { 0.0 } else {
            ((i as f64 - (n - 1 - r - high) as f64) / high as f64).clamp(0.0, 1.0)
        };
        let q = ql.max(qh);
        if q > 0.0 {
            let sigma = 3.0 * vmax * 1000.0f64.ln() / (2.0 * h)
                * (if low == 0 { 0.0 } else { ql * ql / low as f64 }
                  + if high == 0 { 0.0 } else { qh * qh / high as f64 });
            let rate = sigma + std::f64::consts::PI * freq * (1.0 - q);
            value[0] = (-rate * dt).exp() as f32;
            value[1] = if rate == 0.0 { 0.0 } else {
                (sigma / rate * (-rate * dt).exp_m1()) as f32
            };
        }
    }
    for i in r..n - r {
        p[i][2] = if r == 1 {
            0.5 * (p[i + 1][1] - p[i - 1][1]) / h as f32
        } else {
            ((2.0 / 3.0) * (p[i + 1][1] - p[i - 1][1])
             - (1.0 / 12.0) * (p[i + 2][1] - p[i - 2][1])) / h as f32
        };
    }
    p
}

#[allow(clippy::too_many_arguments)]
fn locations(a: &Array<i64>, nz: usize, nx: usize, nzp: usize, nxp: usize,
             zoff: usize, xoff: usize) -> Result<Vec<u32>> {
    let per_shot = a.shape[1];
    a.data.as_chunks::<2>().0.iter().enumerate().map(|(j, loc)| {
        require(loc[0] >= 0 && loc[0] < nz as i64 && loc[1] >= 0 && loc[1] < nx as i64,
                "source/receiver outside model")?;
        Ok(((j / per_shot) * nzp * nxp + (loc[0] as usize + zoff) * nxp
            + loc[1] as usize + xoff) as u32)
    }).collect()
}

/// One cell owns each source/residual sum: no scatter or float atomics.
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

/// A complete restart: u[t], u[t-1], psi_z, psi_x, zeta_z, zeta_x.
/// Buffers stay on the GPU, including saved states. Guards are always zero.
struct State { u: Buffer, prev: Buffer, memory: Buffer }
impl State {
    fn new(gpu: &Gpu, cells: usize) -> Self {
        Self { u: gpu.buffer_zeroed::<f32>(cells), prev: gpu.buffer_zeroed::<f32>(cells),
               memory: gpu.buffer_zeroed::<[f32; 4]>(cells) }
    }
}

struct Kernels {
    forward: Pipeline, record: Pipeline, copy: Pipeline, clear: Pipeline,
    adj_local: Pipeline, adj_gather: Pipeline,
}

struct Execution {
    gpu: Gpu, kernels: Kernels, params: Vec<u8>, cells: usize, traces: usize,
    coefficients: Buffer, pz: Buffer, px: Buffer, source: Buffer,
    source_offsets: Buffer, source_ids: Buffer, receiver_indices: Buffer,
    receiver_offsets: Buffer, receiver_ids: Buffer, observed: Buffer, predicted: Buffer,
    current: State, next: State, slots: Vec<Option<State>>,
    lambda_u: Buffer, lambda_prev: Buffer, lambda_memory: [Buffer; 2],
    temporary: Buffer, extended_gradient: Buffer, reverse_count: usize,
}

impl Execution {
    fn copy_state(&self, from: &State, to: &State) -> Result<()> {
        self.gpu.dispatch(&self.kernels.copy, &[
            Arg::Bytes(&self.params), Arg::Buf(&from.u), Arg::Buf(&from.prev),
            Arg::Buf(&from.memory), Arg::Buf(&to.u), Arg::Buf(&to.prev), Arg::Buf(&to.memory),
        ], [self.cells, 1, 1])
    }

    /// A segment is one synchronous command buffer, at most 32 steps long.
    /// Replays never record data and cannot retain a command graph/history.
    fn advance(&mut self, start: usize, end: usize, record: bool) -> Result<()> {
        for begin in (start..end).step_by(32) {
            self.gpu.batch(|batch| {
                for t in begin..(begin + 32).min(end) {
                    if record {
                        batch.dispatch(&self.kernels.record, &[
                            Arg::Bytes(&self.params), Arg::Buf(&self.current.u),
                            Arg::Buf(&self.receiver_indices), Arg::Buf(&self.predicted), Arg::U32(t as u32),
                        ], [self.traces.max(1), 1, 1])?;
                    }
                    batch.dispatch(&self.kernels.forward, &[
                        Arg::Bytes(&self.params), Arg::Buf(&self.coefficients),
                        Arg::Buf(&self.pz), Arg::Buf(&self.px), Arg::Buf(&self.source),
                        Arg::Buf(&self.source_offsets), Arg::Buf(&self.source_ids),
                        Arg::Buf(&self.current.u), Arg::Buf(&self.current.prev), Arg::Buf(&self.current.memory),
                        Arg::Buf(&self.next.u), Arg::Buf(&self.next.prev), Arg::Buf(&self.next.memory),
                        Arg::U32(t as u32),
                    ], [self.cells, 1, 1])?;
                    std::mem::swap(&mut self.current, &mut self.next);
                }
                Ok(())
            })?;
        }
        Ok(())
    }

    fn reverse(&mut self, time: usize) -> Result<()> {
        let old = self.reverse_count % 2;
        let new = 1 - old;
        self.gpu.batch(|batch| {
            batch.dispatch(&self.kernels.adj_local, &[
                Arg::Bytes(&self.params), Arg::Buf(&self.coefficients), Arg::Buf(&self.pz), Arg::Buf(&self.px),
                Arg::Buf(&self.lambda_u), Arg::Buf(&self.lambda_memory[old]),
                Arg::Buf(&self.current.u), Arg::Buf(&self.current.memory), Arg::Buf(&self.source),
                Arg::Buf(&self.source_offsets), Arg::Buf(&self.source_ids),
                Arg::Buf(&self.temporary), Arg::Buf(&self.extended_gradient), Arg::U32(time as u32),
            ], [self.cells, 1, 1])?;
            batch.dispatch(&self.kernels.adj_gather, &[
                Arg::Bytes(&self.params), Arg::Buf(&self.coefficients), Arg::Buf(&self.pz), Arg::Buf(&self.px),
                Arg::Buf(&self.temporary), Arg::Buf(&self.lambda_memory[old]), Arg::Buf(&self.lambda_memory[new]),
                Arg::Buf(&self.lambda_u), Arg::Buf(&self.lambda_prev),
                Arg::Buf(&self.receiver_offsets), Arg::Buf(&self.receiver_ids),
                Arg::Buf(&self.predicted), Arg::Buf(&self.observed), Arg::U32(time as u32),
            ], [self.cells, 1, 1])
        })?;
        self.reverse_count += 1;
        Ok(())
    }

    fn execute(&mut self, plan: &Plan, nt: usize) -> Result<(Vec<i64>, i64, usize)> {
        let (mut current_time, mut next_reverse, mut work, mut live, mut peak) = (0, nt, 0i64, 0, 0);
        let mut initial_complete = false;
        let mut log = Vec::with_capacity(plan.events.len() * 3);
        for &event in &plan.events {
            let (opcode, time, slot) = match event {
                Event::Advance(t) => {
                    require(t > current_time && t <= nt, "invalid advance")?;
                    self.advance(current_time, t, !initial_complete)?;
                    work += (t - current_time) as i64;
                    current_time = t;
                    initial_complete |= t == nt;
                    (0, t, -1)
                }
                Event::Save(t, slot) => {
                    require(t == current_time && t > 0 && t < nt && self.slots[slot].is_none(), "invalid save")?;
                    let saved = State::new(&self.gpu, self.cells);
                    self.copy_state(&self.current, &saved)?;
                    self.slots[slot] = Some(saved);
                    live += 1;
                    peak = peak.max(live);
                    (1, t, slot as i64)
                }
                Event::Restore(t, slot) => {
                    require(initial_complete, "restore before original forward complete")?;
                    if let Some(slot) = slot {
                        let saved = self.slots[slot].as_ref().ok_or("restore from empty slot")?;
                        self.copy_state(saved, &self.current)?;
                    } else {
                        require(t == 0, "only state zero is implicit")?;
                        self.gpu.dispatch(&self.kernels.clear, &[
                            Arg::Bytes(&self.params), Arg::Buf(&self.current.u),
                            Arg::Buf(&self.current.prev), Arg::Buf(&self.current.memory),
                        ], [self.cells, 1, 1])?;
                    }
                    current_time = t;
                    (2, t, slot.map_or(-1, |s| s as i64))
                }
                Event::Drop(t, slot) => {
                    require(self.slots[slot].take().is_some(), "drop from empty slot")?;
                    live -= 1;
                    (3, t, slot as i64)
                }
                Event::Reverse(t) => {
                    require(initial_complete && next_reverse > 0 && t == next_reverse - 1 && t == current_time,
                            "invalid reverse order/state")?;
                    self.reverse(t)?;
                    next_reverse -= 1;
                    (4, t, -1)
                }
            };
            // Append only AFTER executing the operation; this is an execution audit.
            log.extend_from_slice(&[opcode, time as i64, slot]);
        }
        require(next_reverse == 0 && live == 0 && work == plan.forward_steps,
                "incomplete execution or work/planner mismatch")?;
        Ok((log, work, peak))
    }
}

fn misfit_and_grad(inp: &Inputs, out: &mut Outputs) -> Result<()> {
    let velocity = inp.f32("velocity")?;
    let source = inp.f32("source_amplitudes")?;
    let sl = inp.i64("source_locations")?;
    let rl = inp.i64("receiver_locations")?;
    let observed = inp.f32("observed")?;
    require(velocity.shape.len() == 2 && source.shape.len() == 3 && observed.shape.len() == 3,
            "invalid input ranks")?;
    let (nz, nx) = (velocity.shape[0], velocity.shape[1]);
    let (shots, ns, nt) = (source.shape[0], source.shape[1], source.shape[2]);
    let nr = observed.shape[1];
    require(nz > 0 && nx > 0 && shots > 0 && nt > 0, "empty model or time axis")?;
    shape(&sl, &[shots, ns, 2], "source_locations")?;
    shape(&rl, &[shots, nr, 2], "receiver_locations")?;
    shape(&observed, &[shots, nr, nt], "observed")?;
    require(velocity.data.iter().all(|&v| positive(v as f64))
            && source.data.iter().chain(&observed.data).all(|v| v.is_finite()), "nonfinite/invalid inputs")?;
    let r = match inp.int("accuracy")? { 2 => 1, 4 => 2, _ => return Err("accuracy must be 2 or 4".into()) };
    let w = inp.nums("pml_width")?;
    require(w.len() == 4 && w.iter().all(|x| x.is_finite() && *x >= 0.0 && x.fract() == 0.0
            && *x <= u32::MAX as f64), "invalid PML widths")?;
    let w: Vec<usize> = w.iter().map(|&x| x as usize).collect();
    let h = inp.nums("grid_spacing")?;
    require(h.len() == 2 && h.iter().all(|&x| positive(x)), "invalid grid spacing")?;
    let dt = inp.num("dt")?;
    let freq = inp.num("pml_freq")?;
    let vmax = inp.num("max_vel")?;
    require(positive(dt) && positive(vmax) && freq.is_finite() && freq >= 0.0, "invalid timestep/PML design")?;
    let budget = inp.int("checkpoint_budget")?;
    require(budget >= -1 && (budget < 0 || budget < nt as i64), "invalid checkpoint budget")?;
    let capacity = if budget < 0 { nt - 1 } else { budget as usize };
    let nzp = nz.checked_add(w[0]).and_then(|v| v.checked_add(w[1] + 2 * r)).ok_or("grid overflow")?;
    let nxp = nx.checked_add(w[2]).and_then(|v| v.checked_add(w[3] + 2 * r)).ok_or("grid overflow")?;
    let plane = product(&[nzp, nxp])?;
    let cells = product(&[shots, plane])?;
    let samples = product(&[shots, nr, nt])?;
    let traces = product(&[shots, nr])?;
    product(&[shots, ns, nt])?;
    let (zoff, xoff) = (r + w[0], r + w[2]);
    let source_indices = locations(&sl, nz, nx, nzp, nxp, zoff, xoff)?;
    let receiver_indices = locations(&rl, nz, nx, nzp, nxp, zoff, xoff)?;
    let (so, si) = point_lists(&source_indices, cells);
    let (ro, ri) = point_lists(&receiver_indices, cells);

    // Exact MSL Params layout: ten uints followed by five floats.
    let mut params = Vec::with_capacity(60);
    for value in [nz, nx, nzp, nxp, shots, nt, nr, r, zoff, xoff] {
        params.extend_from_slice(&(value as u32).to_ne_bytes());
    }
    for value in [1.0 / h[0], 1.0 / h[1], 1.0 / (h[0] * h[0]), 1.0 / (h[1] * h[1]), dt * dt] {
        params.extend_from_slice(&(value as f32).to_ne_bytes());
    }
    let plan = Plan::new(nt, capacity)?;
    let gpu = Gpu::new()?;
    let lib = gpu.compile(include_str!("../kernels/checkpoint.metal"))?;
    let setup = gpu.pipeline(&lib, "extend_velocity")?;
    let loss_blocks = gpu.pipeline(&lib, "loss_blocks")?;
    let loss_finish = gpu.pipeline(&lib, "loss_finish")?;
    let fold = gpu.pipeline(&lib, "fold_gradient")?;
    let kernels = Kernels {
        forward: gpu.pipeline(&lib, "forward_step")?, record: gpu.pipeline(&lib, "record_receivers")?,
        copy: gpu.pipeline(&lib, "copy_state")?, clear: gpu.pipeline(&lib, "clear_state")?,
        adj_local: gpu.pipeline(&lib, "adjoint_local")?, adj_gather: gpu.pipeline(&lib, "adjoint_gather")?,
    };
    let bv = gpu.buffer_from(&velocity.data);
    let coefficients = gpu.buffer_zeroed::<[f32; 2]>(plane);
    gpu.dispatch(&setup, &[Arg::Bytes(&params), Arg::Buf(&bv), Arg::Buf(&coefficients)], [plane, 1, 1])?;
    let mut execution = Execution {
        pz: gpu.buffer_from(&profile(nzp, w[0], w[1], r, h[0], dt, freq, vmax)),
        px: gpu.buffer_from(&profile(nxp, w[2], w[3], r, h[1], dt, freq, vmax)),
        source: gpu.buffer_from(&source.data), source_offsets: gpu.buffer_from(&so), source_ids: gpu.buffer_from(&si),
        receiver_indices: gpu.buffer_from(&receiver_indices), receiver_offsets: gpu.buffer_from(&ro),
        receiver_ids: gpu.buffer_from(&ri), observed: gpu.buffer_from(&observed.data),
        predicted: gpu.buffer_zeroed::<f32>(samples), current: State::new(&gpu, cells), next: State::new(&gpu, cells),
        slots: (0..capacity).map(|_| None).collect(),
        lambda_u: gpu.buffer_zeroed::<f32>(cells), lambda_prev: gpu.buffer_zeroed::<f32>(cells),
        lambda_memory: [gpu.buffer_zeroed::<[f32; 4]>(cells), gpu.buffer_zeroed::<[f32; 4]>(cells)],
        temporary: gpu.buffer_zeroed::<[f32; 4]>(cells), extended_gradient: gpu.buffer_zeroed::<f32>(cells),
        gpu, kernels, params, cells, traces, coefficients, reverse_count: 0,
    };
    let (schedule, forward_steps, peak) = execution.execute(&plan, nt)?;
    let checkpoint_bytes = checkpoint_payload(peak, shots, nzp - 2 * r, nxp - 2 * r)?;
    let g = &execution.gpu;
    let blocks = samples.div_ceil(128);
    let partial = g.buffer_zeroed::<f32>(blocks);
    let objective = g.buffer_zeroed::<f32>(1);
    let gradient = g.buffer_zeroed::<f32>(nz * nx);
    g.batch(|b| {
        b.dispatch(&loss_blocks, &[
            Arg::Bytes(&execution.params), Arg::Buf(&execution.predicted), Arg::Buf(&execution.observed), Arg::Buf(&partial),
        ], [blocks.max(1), 1, 1])?;
        b.dispatch(&loss_finish, &[
            Arg::Buf(&partial), Arg::Buf(&objective), Arg::U32(blocks as u32),
        ], [1, 1, 1])?;
        b.dispatch(&fold, &[
            Arg::Bytes(&execution.params), Arg::Buf(&execution.extended_gradient), Arg::Buf(&gradient),
        ], [nz * nx, 1, 1])
    })?;
    out.f32("objective", &[], &g.read::<f32>(&objective, 1))?;
    out.f32("predicted", &[shots, nr, nt], &g.read::<f32>(&execution.predicted, samples))?;
    out.f32("gradient", &[nz, nx], &g.read::<f32>(&gradient, nz * nx))?;
    out.i64("schedule", &[schedule.len() / 3, 3], &schedule)?;
    out.i64("forward_steps", &[], &[forward_steps])?;
    out.i64("peak_checkpoints", &[], &[peak as i64])?;
    out.i64("checkpoint_bytes", &[], &[checkpoint_bytes])?;
    // All checkpoints were actually dropped during execution. Working GPU
    // buffers and pipelines are local to this call and are released here.
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::checkpoint_payload;

    #[test]
    fn logical_payload_uses_int64_instead_of_metal_index_limits() {
        assert_eq!(checkpoint_payload(1, 1, 16_384, 16_384).unwrap(), 6_442_450_944);
        assert_eq!(checkpoint_payload(3, 1, 16_384, 16_384).unwrap(), 19_327_352_832);
        assert_eq!(checkpoint_payload(0, usize::MAX, usize::MAX, usize::MAX).unwrap(), 0);
        assert!(checkpoint_payload(usize::MAX, 1, 1, 1).is_err());
        assert!(checkpoint_payload(1, 1, usize::MAX, usize::MAX).is_err());
    }
}
