//! Deterministic float32 fast sweeping and the converged discrete adjoint.
//! The host handles validation, buffers, and convergence checks; all numerical
//! updates, receiver sampling, loss evaluation, and differentiation use Metal.

use outcrop_common::{gpu::Arg, Array, Gpu, Inputs, Outputs, Result};

fn node_indices(array: &Array<i64>, nz: usize, nx: usize) -> Result<Vec<u32>> {
    if array.shape.len() != 2 || array.shape[1] != 2 {
        return Err("node indices must have shape (count, 2)".into());
    }
    array
        .data
        .as_chunks::<2>()
        .0
        .iter()
        .map(|ij| {
            let (z, x) = (ij[0], ij[1]);
            if z < 0 || x < 0 || z as usize >= nz || x as usize >= nx {
                Err("node index outside the slowness grid".into())
            } else {
                Ok((z as usize * nx + x as usize) as u32)
            }
        })
        .collect()
}

fn misfit_and_grad(inp: &Inputs, out: &mut Outputs) -> Result<()> {
    let slowness = inp.f32("slowness")?;
    let spacing = inp.nums("spacing")?;
    let sources = inp.i64("sources")?;
    let receivers = inp.i64("receivers")?;
    let observed = inp.f32("observed")?;
    let weights = inp.f32("weights")?;

    if slowness.shape.len() != 2 || slowness.shape.contains(&0) {
        return Err("slowness must have nonempty shape (nz, nx)".into());
    }
    if slowness.data.iter().any(|&s| !s.is_finite() || s <= 0.0) {
        return Err("slowness must be finite and strictly positive".into());
    }
    if spacing.len() != 2 || spacing.iter().any(|&h| !h.is_finite() || h <= 0.0) {
        return Err("spacing must contain positive finite [dz, dx]".into());
    }
    let h: Vec<f32> = spacing.iter().map(|&x| x as f32).collect();
    if h.iter()
        .any(|&x| !x.is_finite() || x <= 0.0 || !(x * x).is_normal())
    {
        return Err("spacing squared must be representable in float32".into());
    }

    let (nz, nx) = (slowness.shape[0], slowness.shape[1]);
    let n = slowness.data.len();
    if n > i32::MAX as usize {
        return Err("grid exceeds Metal signed node-index range".into());
    }
    let source_nodes = node_indices(&sources, nz, nx)?;
    let receiver_nodes = node_indices(&receivers, nz, nx)?;
    let (ns, nr) = (source_nodes.len(), receiver_nodes.len());
    if observed.shape != [ns, nr] || weights.shape != [ns, nr] {
        return Err("observed and weights must have shape (ns, nr)".into());
    }
    if observed.data.iter().any(|v| !v.is_finite())
        || weights.data.iter().any(|&w| !w.is_finite() || w < 0.0)
    {
        return Err("observations must be finite; weights finite and nonnegative".into());
    }
    let total = n.checked_mul(ns).ok_or("shot grid size overflow")?;
    let samples = ns.checked_mul(nr).ok_or("receiver size overflow")?;
    if [ns, nr, total, samples]
        .iter()
        .any(|&v| v > u32::MAX as usize)
    {
        return Err("shot/receiver arrays exceed Metal indexing range".into());
    }

    // p = [nz, nx, ns, nr, nodes_per_shot]. All offsets are C order.
    let params = [nz as u32, nx as u32, ns as u32, nr as u32, n as u32];
    let gpu = Gpu::new()?;
    let library = gpu.compile(include_str!("../kernels/eikonal.metal"))?;
    let initialize = gpu.pipeline(&library, "initialize")?;
    let sweep = gpu.pipeline(&library, "sweep_cycle")?;
    let adjoint = gpu.pipeline(&library, "discrete_adjoint")?;
    let reduce_gradient = gpu.pipeline(&library, "reduce_gradient")?;
    let reduce_loss = gpu.pipeline(&library, "reduce_loss")?;

    let bp = gpu.buffer_from(&params);
    let bh = gpu.buffer_from(&h);
    let bs = gpu.buffer_from(&slowness.data);
    let bsrc = gpu.buffer_from(&source_nodes);
    let brec = gpu.buffer_from(&receiver_nodes);
    let bobs = gpu.buffer_from(&observed.data);
    let bw = gpu.buffer_from(&weights.data);
    let bt = gpu.buffer_zeroed::<f32>(total);
    // (parent, depth) for one-axis updates whose increment rounded away.
    // Depth preserves causality when several adjacent arrivals are equal.
    let brounded = gpu.buffer_zeroed::<[u32; 2]>(total);
    let bchanged = gpu.buffer_zeroed::<u32>(ns);
    let bpred = gpu.buffer_zeroed::<f32>(samples);
    let bloss_terms = gpu.buffer_zeroed::<f32>(samples);
    let brhs = gpu.buffer_zeroed::<f32>(total);
    let bshot_grad = gpu.buffer_zeroed::<f32>(total);
    let border = gpu.buffer_zeroed::<u32>(total);
    let bgrad = gpu.buffer_zeroed::<f32>(n);
    let bloss = gpu.buffer_zeroed::<f32>(1);

    gpu.dispatch(
        &initialize,
        &[
            Arg::Buf(&bp),
            Arg::Buf(&bsrc),
            Arg::Buf(&bt),
            Arg::Buf(&brounded),
        ],
        [total, 1, 1],
    )?;

    // Each thread owns an entire shot, so the four Gauss-Seidel traversals
    // have a fixed order. Stop only at a bitwise fixed point, not an atol.
    // The generous safety bound reports failure rather than returning an
    // unconverged field. Typical small grids need only a few cycles.
    let max_cycles = n.saturating_mul(4).saturating_add(16);
    let mut converged = false;
    for _ in 0..max_cycles {
        gpu.dispatch(
            &sweep,
            &[
                Arg::Buf(&bp),
                Arg::Buf(&bh),
                Arg::Buf(&bs),
                Arg::Buf(&bsrc),
                Arg::Buf(&bt),
                Arg::Buf(&bchanged),
                Arg::Buf(&brounded),
            ],
            [ns, 1, 1],
        )?;
        if gpu.read::<u32>(&bchanged, ns).iter().all(|&c| c == 0) {
            converged = true;
            break;
        }
    }
    if !converged {
        return Err("float32 fast sweeping did not reach a bitwise fixed point".into());
    }

    gpu.dispatch(
        &adjoint,
        &[
            Arg::Buf(&bp),
            Arg::Buf(&bh),
            Arg::Buf(&bs),
            Arg::Buf(&bsrc),
            Arg::Buf(&brec),
            Arg::Buf(&bobs),
            Arg::Buf(&bw),
            Arg::Buf(&bt),
            Arg::Buf(&bpred),
            Arg::Buf(&bloss_terms),
            Arg::Buf(&brhs),
            Arg::Buf(&bshot_grad),
            Arg::Buf(&border),
            Arg::Buf(&brounded),
        ],
        [ns, 1, 1],
    )?;
    gpu.batch(|batch| {
        batch.dispatch(
            &reduce_gradient,
            &[Arg::Buf(&bp), Arg::Buf(&bshot_grad), Arg::Buf(&bgrad)],
            [n, 1, 1],
        )?;
        batch.dispatch(
            &reduce_loss,
            &[Arg::Buf(&bp), Arg::Buf(&bloss_terms), Arg::Buf(&bloss)],
            [1, 1, 1],
        )
    })?;

    let traveltime = gpu.read::<f32>(&bt, total);
    let predicted = gpu.read::<f32>(&bpred, samples);
    let gradient = gpu.read::<f32>(&bgrad, n);
    let objective = gpu.read::<f32>(&bloss, 1);
    if traveltime
        .iter()
        .chain(&predicted)
        .chain(&gradient)
        .chain(&objective)
        .any(|v| !v.is_finite())
    {
        return Err("eikonal solve or adjoint produced a nonfinite output".into());
    }
    out.f32("traveltime", &[ns, nz, nx], &traveltime)?;
    out.f32("predicted", &[ns, nr], &predicted)?;
    // Outputs::scalar writes f64; use the f32 writer with a zero-dimensional shape.
    out.f32("objective", &[], &objective)?;
    out.f32("gradient", &[nz, nx], &gradient)
}

fn main() {
    outcrop_common::run(&[("misfit_and_grad", misfit_and_grad)]);
}
