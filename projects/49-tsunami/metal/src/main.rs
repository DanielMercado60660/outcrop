//! Periodic, cell-centered shallow water and its discrete initial-state adjoint.

mod allocation;

use allocation::{allocated_bytes, try_buffer, working_set_budget};
use outcrop_common::{
    gpu::{Arg, Buffer, Pipeline},
    Gpu, Inputs, Outputs, Result,
};

struct Problem {
    data: Vec<f32>,
    n: usize,
    steps: usize,
    ratio: f32,
    mode: u32,
    gh: f32,
    reference_depth: f32,
}

impl Problem {
    fn read(inp: &Inputs) -> Result<Self> {
        let initial = inp.f32("initial")?;
        if initial.shape.len() != 2 || initial.shape[1] != 2 || initial.shape[0] == 0 {
            return Err("initial must have shape (n, 2), n > 0".into());
        }
        let n = initial.shape[0];
        if n > u32::MAX as usize || initial.data.len() != 2 * n {
            return Err("initial is too large or has inconsistent shape".into());
        }
        if !initial.data.iter().all(|x| x.is_finite()) {
            return Err("initial must be finite".into());
        }
        let steps = usize::try_from(inp.int("steps")?)
            .map_err(|_| "steps must be a nonnegative addressable integer")?;
        let mode = match inp.str("mode")?.as_str() {
            "linear" => 0,
            "nonlinear" => 1,
            _ => return Err("mode must be 'linear' or 'nonlinear'".into()),
        };
        let dx = inp.num("dx")?;
        let dt = inp.num("dt")?;
        if !dx.is_finite() || dx <= 0.0 || !dt.is_finite() || dt < 0.0 {
            return Err("dx must be positive and dt nonnegative, both finite".into());
        }
        let ratio = (dt / dx) as f32;
        if !ratio.is_finite() {
            return Err("dt/dx must be representable in float32".into());
        }
        // H belongs only to the linear equations; nonlinear depth is the first
        // state column, and the API's scalar depth argument is ignored.
        let gh = if mode == 0 {
            let depth = inp.num("depth")?;
            let value = (9.81 * depth) as f32;
            if !depth.is_finite() || value <= 0.0 || !value.is_finite() {
                return Err("linear depth must be positive and finite".into());
            }
            value
        } else {
            if initial.data.iter().step_by(2).any(|&h| h <= 0.0) {
                return Err("nonlinear water depths must be strictly positive".into());
            }
            0.0
        };
        // A zero timestep is the identity map for both handlers, regardless
        // of the requested step count. Avoid allocating a useless history.
        let steps = if ratio == 0.0 { 0 } else { steps };
        // Nonlinear trajectories store e = h - H_ref. Choose H_ref internally;
        // the API's depth argument is never used. Retain total depths for an
        // identity call so neither handler introduces subtraction/addition
        // rounding when steps == 0. Linear state already stores eta.
        let reference_depth = if mode != 0 && steps != 0 { initial.data[0] } else { 0.0 };
        Ok(Self { data: initial.data, n, steps, ratio, mode, gh, reference_depth })
    }
}

// Separate buffers make every Euler update depend exclusively on the old
// state. Only a nonlinear adjoint needs U^0 through U^steps: the linear
// step Jacobian is independent of the trajectory.
fn forward(gpu: &Gpu, kernel: &Pipeline, shift: &Pipeline, p: &Problem, keep: bool) -> Result<Vec<Buffer>> {
    let count = if p.steps == 0 {
        1
    } else if keep {
        p.steps.checked_add(1).ok_or("forward history count overflows addressable memory")?
    } else {
        2
    };
    let slots_bytes = count.checked_mul(std::mem::size_of::<Buffer>())
        .filter(|&bytes| bytes <= isize::MAX as usize)
        .ok_or("forward history handles exceed addressable memory")?;
    let state_bytes = p.data.len().checked_mul(std::mem::size_of::<f32>())
        .ok_or("state buffer size overflows addressable memory")?;
    count.checked_mul(state_bytes)
        .ok_or("forward history size overflows addressable memory")?;

    let budget = working_set_budget(gpu);
    let initial = try_buffer(gpu, p.data.len())?;
    // Account for Metal's actual allocation granularity, especially when a
    // tiny grid requests many steps. Leave room for the adjoint and reduction
    // buffers too (each is no larger than one state allocation).
    let buffer_count = count.checked_add(if keep { 4 } else { 0 })
        .ok_or("forward history count overflows addressable memory")?;
    let bytes = buffer_count.checked_mul(allocated_bytes(&initial).max(state_bytes))
        .and_then(|bytes| bytes.checked_add(slots_bytes))
        .ok_or("forward history size overflows addressable memory")?;
    if bytes > budget {
        return Err(format!("forward history needs {bytes} bytes, exceeding the Metal working-set budget of {budget} bytes").into());
    }
    let mut states = Vec::new();
    states.try_reserve_exact(count)
        .map_err(|e| format!("could not allocate forward history handles: {e}"))?;
    gpu.write(&initial, &p.data);
    states.push(initial);
    for _ in 1..count {
        states.push(try_buffer(gpu, p.data.len())?);
    }
    if p.steps == 0 {
        return Ok(states);
    }
    gpu.batch(|batch| {
        if p.mode != 0 {
            // Convert just once, before recording the nonlinear trajectory.
            batch.dispatch(
                shift,
                &[
                    Arg::Buf(&states[0]), Arg::U32(p.n as u32),
                    Arg::F32(-p.reference_depth),
                ],
                [p.n, 1, 1],
            )?;
        }
        for t in 0..p.steps {
            let (a, b) = if keep { (t, t + 1) } else { (t % 2, (t + 1) % 2) };
            batch.dispatch(
                kernel,
                &[
                    Arg::Buf(&states[a]), Arg::Buf(&states[b]),
                    Arg::U32(p.n as u32), Arg::F32(p.ratio),
                    Arg::U32(p.mode), Arg::F32(p.gh),
                    Arg::F32(p.reference_depth),
                ],
                [p.n, 1, 1],
            )?;
        }
        Ok(())
    })?;
    Ok(states)
}

fn propagate(inp: &Inputs, out: &mut Outputs) -> Result<()> {
    let p = Problem::read(inp)?;
    if p.steps == 0 {
        return out.f32("state", &[p.n, 2], &p.data);
    }
    let gpu = Gpu::new()?;
    let lib = gpu.compile(include_str!("../kernels/tsunami.metal"))?;
    let kernel = gpu.pipeline(&lib, "advance")?;
    let shift = gpu.pipeline(&lib, "shift_depth")?;
    let states = forward(&gpu, &kernel, &shift, &p, false)?;
    if p.mode != 0 {
        // Round to total depth only at the public output boundary.
        gpu.dispatch(
            &shift,
            &[
                Arg::Buf(&states[p.steps % 2]), Arg::U32(p.n as u32),
                Arg::F32(p.reference_depth),
            ],
            [p.n, 1, 1],
        )?;
    }
    let result: Vec<f32> = gpu.read(&states[p.steps % 2], 2 * p.n);
    out.f32("state", &[p.n, 2], &result)
}

fn objective_and_grad(inp: &Inputs, out: &mut Outputs) -> Result<()> {
    let p = Problem::read(inp)?;
    let weights = inp.f32("weights")?;
    if weights.shape != [p.n, 2] || weights.data.len() != 2 * p.n {
        return Err("weights must match initial shape".into());
    }
    if !weights.data.iter().all(|x| x.is_finite()) {
        return Err("weights must be finite".into());
    }
    let gpu = Gpu::new()?;
    let lib = gpu.compile(include_str!("../kernels/tsunami.metal"))?;
    let kernel = gpu.pipeline(&lib, "advance")?;
    let shift = gpu.pipeline(&lib, "shift_depth")?;
    let adj = gpu.pipeline(&lib, "adjoint")?;
    let obj = gpu.pipeline(&lib, "objective_terms")?;
    let reduce = gpu.pipeline(&lib, "reduce_sum")?;
    let keep_history = p.mode != 0;
    let states = forward(&gpu, &kernel, &shift, &p, keep_history)?;
    let final_index = if keep_history { p.steps } else { p.steps % 2 };
    let gradients = [try_buffer(&gpu, 2 * p.n)?, try_buffer(&gpu, 2 * p.n)?];
    gpu.write(&gradients[0], &weights.data);
    let sums = [try_buffer(&gpu, p.n)?, try_buffer(&gpu, p.n)?];
    let mut sum_index = 0;
    gpu.batch(|batch| {
        batch.dispatch(
            &obj,
            &[
                Arg::Buf(&states[final_index]), Arg::Buf(&gradients[0]),
                Arg::Buf(&sums[0]), Arg::U32(p.n as u32),
                Arg::F32(p.reference_depth),
            ],
            [p.n, 1, 1],
        )?;
        // A balanced float32 sum limits accumulation error and keeps all
        // objective arithmetic on the GPU. Each pass also handles odd n.
        let mut count = p.n;
        while count > 1 {
            let next = count.div_ceil(2);
            batch.dispatch(
                &reduce,
                &[
                    Arg::Buf(&sums[sum_index]), Arg::Buf(&sums[1 - sum_index]),
                    Arg::U32(count as u32),
                ],
                [next, 1, 1],
            )?;
            sum_index = 1 - sum_index;
            count = next;
        }
        // lambda^steps = weights; apply the transpose of each Euler-step
        // Jacobian in reverse time. No finite differences or float atomics.
        // H_ref is a fixed coordinate offset during this differentiation.
        // d(h-H_ref)/dh = 1 and d(e+H_ref)/de = 1, so the initial e-adjoint
        // is also the requested h-adjoint. Although H_ref was selected from
        // the input, the physical map is independent of that choice: all
        // constant reference-pressure terms cancel in the flux divergence.
        for k in 0..p.steps {
            // The linear branch does not use the state values, so either
            // of the forward buffers supplies its unused state argument.
            let state_index = if keep_history { p.steps - 1 - k } else { final_index };
            batch.dispatch(
                &adj,
                &[
                    Arg::Buf(&states[state_index]), Arg::Buf(&gradients[k % 2]),
                    Arg::Buf(&gradients[(k + 1) % 2]), Arg::U32(p.n as u32),
                    Arg::F32(p.ratio), Arg::U32(p.mode), Arg::F32(p.gh),
                    Arg::F32(p.reference_depth),
                ],
                [p.n, 1, 1],
            )?;
        }
        Ok(())
    })?;
    let value: Vec<f32> = gpu.read(&sums[sum_index], 1);
    let gradient: Vec<f32> = gpu.read(&gradients[p.steps % 2], 2 * p.n);
    out.f32("objective", &[], &value)?;
    out.f32("gradient", &[p.n, 2], &gradient)
}

fn main() {
    outcrop_common::run(&[("propagate", propagate), ("objective_and_grad", objective_and_grad)]);
}
