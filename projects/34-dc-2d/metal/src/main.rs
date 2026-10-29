use outcrop_common::{gpu::Arg, Gpu, Inputs, Outputs, Result};

// Host setup is float32 too. Compensating the coordinate prefix sum avoids
// accumulating padding-width roundoff before reaching the small core cells.
fn centers(h: &[f32], start: f32) -> (Vec<f32>, Vec<f32>) {
    let mut nodes = vec![start];
    let mut cells = Vec::with_capacity(h.len());
    let mut sum = start;
    let mut correction = 0.0;
    for &width in h {
        cells.push(sum + (0.5 * width - correction));
        let increment = width - correction;
        let next = sum + increment;
        correction = (next - sum) - increment;
        sum = next;
        nodes.push(sum);
    }
    (nodes, cells)
}

fn nearest(c: &[f32], x: f32) -> usize {
    (0..c.len())
        .min_by(|&a, &b| (c[a] - x).abs().total_cmp(&(c[b] - x).abs()))
        .unwrap()
}

fn forward(inp: &Inputs, out: &mut Outputs) -> Result<()> {
    let hx = inp.f32("hx")?;
    let hz = inp.f32("hz")?;
    let origin = inp.f32("origin")?;
    let sig = inp.f32("conductivity")?;
    let ab = inp.f32("ab")?;
    let mn = inp.f32("mn")?;
    let cur = inp.f32("currents")?;
    let ky = inp.f32("ky")?;
    let weights = inp.f32("weights")?;
    let nx = hx.data.len();
    let nz = hz.data.len();
    let n = nx * nz;
    let ns = cur.data.len();
    let nk = ky.data.len();
    if nx == 0
        || nz == 0
        || nk == 0
        || hx.shape != [nx]
        || hz.shape != [nz]
        || origin.shape != [2]
        || sig.shape != [nx, nz]
        || ab.shape != [ns, 2]
        || mn.shape != [ns, 2]
        || cur.shape != [ns]
        || ky.shape != [nk]
        || weights.shape != [nk]
    {
        return Err("invalid input shapes".into());
    }
    if hx.data.iter().chain(&hz.data).chain(&sig.data).chain(&ky.data)
        .any(|&v| !v.is_finite() || v <= 0.0)
        || origin.data.iter().chain(&ab.data).chain(&mn.data)
            .chain(&cur.data).chain(&weights.data).any(|v| !v.is_finite())
    {
        return Err("expected finite inputs and positive widths, conductivity, and ky".into());
    }
    if ns == 0 {
        out.f32("voltage", &[0], &[])?;
        out.f32("apparent_resistivity", &[0], &[])?;
        return Ok(());
    }

    let (xn, xc) = centers(&hx.data, origin.data[0]);
    let (zn, zc) = centers(&hz.data, origin.data[1]);
    let source_z = nearest(&zc, 0.0);
    let src: Vec<u32> = ab.data.iter()
        .map(|&v| (nearest(&xc, v) + source_z * nx) as u32)
        .collect();

    let gpu = Gpu::new()?;
    let setup = gpu.compile(include_str!("../kernels/prepare.metal"))?;
    let faces = gpu.pipeline(&setup, "faces")?;
    let diagonal = gpu.pipeline(&setup, "diagonal")?;
    let lib = gpu.compile(include_str!("../kernels/solve.metal"))?;
    let init = gpu.pipeline(&lib, "init")?;
    let initdot = gpu.pipeline(&lib, "initdot")?;
    let mat = gpu.pipeline(&lib, "mat")?;
    let step = gpu.pipeline(&lib, "step")?;
    let update = gpu.pipeline(&lib, "update")?;
    let dot = gpu.pipeline(&lib, "dot")?;
    let direction = gpu.pipeline(&lib, "direction")?;
    let receivers = gpu.pipeline(&lib, "receivers")?;
    let integrate = gpu.pipeline(&lib, "integrate")?;

    // Solver cells are x-fast: c = z*nx+x. Input conductivity is x-first
    // NumPy C order: sigma[x*nz+z]. The assembly kernels perform this mapping.
    let systems = ns * nk;
    let total = n * systems;
    if total > u32::MAX as usize {
        return Err("mesh/source/mode product exceeds Metal index range".into());
    }
    let (nu, nsu, nxu, nzu, nku) = (n as u32, ns as u32, nx as u32, nz as u32, nk as u32);
    let bhx = gpu.buffer_from(&hx.data);
    let bhz = gpu.buffer_from(&hz.data);
    let bsig = gpu.buffer_from(&sig.data);
    let bxn = gpu.buffer_from(&xn);
    let bzn = gpu.buffer_from(&zn);
    let bxc = gpu.buffer_from(&xc);
    let bzc = gpu.buffer_from(&zc);
    let bky = gpu.buffer_from(&ky.data);
    let be = gpu.buffer_zeroed::<f32>(n);
    let bu = gpu.buffer_zeroed::<f32>(n);
    let bd = gpu.buffer_zeroed::<f32>(n * nk);
    gpu.batch(|b| {
        b.dispatch(&faces, &[
            Arg::Buf(&bhx), Arg::Buf(&bhz), Arg::Buf(&bsig),
            Arg::Buf(&be), Arg::Buf(&bu), Arg::U32(nxu), Arg::U32(nzu),
        ], [n, 1, 1])?;
        b.dispatch(&diagonal, &[
            Arg::Buf(&bhx), Arg::Buf(&bhz), Arg::Buf(&bsig),
            Arg::Buf(&bxn), Arg::Buf(&bzn), Arg::Buf(&bxc), Arg::Buf(&bzc),
            Arg::Buf(&bky), Arg::Buf(&be), Arg::Buf(&bu), Arg::Buf(&bd),
            Arg::U32(nxu), Arg::U32(nzu),
        ], [n * nk, 1, 1])
    })?;

    let bx = gpu.buffer_zeroed::<f32>(total);
    let br = gpu.buffer_zeroed::<f32>(total);
    let bp = gpu.buffer_zeroed::<f32>(total);
    let bq = gpu.buffer_zeroed::<f32>(total);
    let st = gpu.buffer_zeroed::<f32>(systems * 5);
    let bs = gpu.buffer_from(&src);
    let bc = gpu.buffer_from(&cur.data);
    gpu.dispatch(&init, &[
        Arg::Buf(&bx), Arg::Buf(&br), Arg::Buf(&bp), Arg::Buf(&bd),
        Arg::Buf(&bs), Arg::Buf(&bc), Arg::U32(nu), Arg::U32(nsu),
    ], [total, 1, 1])?;
    gpu.dispatch(&initdot, &[
        Arg::Buf(&br), Arg::Buf(&bp), Arg::Buf(&st), Arg::U32(nu),
    ], [systems, 1, 1])?;
    for _ in 0..80 {
        gpu.batch(|b| {
            for _ in 0..50 {
                b.dispatch(&mat, &[
                    Arg::Buf(&bp), Arg::Buf(&bq), Arg::Buf(&bd), Arg::Buf(&be),
                    Arg::Buf(&bu), Arg::U32(nxu), Arg::U32(nzu), Arg::U32(nsu),
                ], [total, 1, 1])?;
                b.dispatch(&step, &[
                    Arg::Buf(&bp), Arg::Buf(&bq), Arg::Buf(&st), Arg::U32(nu),
                ], [systems, 1, 1])?;
                b.dispatch(&update, &[
                    Arg::Buf(&bx), Arg::Buf(&br), Arg::Buf(&bp), Arg::Buf(&bq),
                    Arg::Buf(&st), Arg::U32(nu),
                ], [total, 1, 1])?;
                b.dispatch(&dot, &[
                    Arg::Buf(&br), Arg::Buf(&bd), Arg::Buf(&st), Arg::U32(nu), Arg::U32(nsu),
                ], [systems, 1, 1])?;
                b.dispatch(&direction, &[
                    Arg::Buf(&bp), Arg::Buf(&br), Arg::Buf(&bd), Arg::Buf(&st),
                    Arg::U32(nu), Arg::U32(nsu),
                ], [total, 1, 1])?;
            }
            Ok(())
        })?;
        let state: Vec<f32> = gpu.read(&st, systems * 5);
        if (0..systems).any(|s| state[5 * s + 3] < 0.0) {
            return Err("PCG encountered a non-finite value or breakdown".into());
        }
        if (0..systems).all(|s| state[5 * s + 3] > 0.0) {
            break;
        }
    }
    let state: Vec<f32> = gpu.read(&st, systems * 5);
    if (0..systems).any(|s| state[5 * s + 3] != 1.0) {
        return Err("PCG did not converge within 4000 iterations".into());
    }

    // Sample on the GPU, then integrate the supplied weights there. Only the
    // final float32 voltages/resistivities and convergence state reach the CPU.
    let bmn = gpu.buffer_from(&mn.data);
    let bab = gpu.buffer_from(&ab.data);
    let bw = gpu.buffer_from(&weights.data);
    let samples = gpu.buffer_zeroed::<f32>(systems);
    let bv = gpu.buffer_zeroed::<f32>(ns);
    let brho = gpu.buffer_zeroed::<f32>(ns);
    gpu.batch(|b| {
        b.dispatch(&receivers, &[
            Arg::Buf(&bx), Arg::Buf(&bxc), Arg::Buf(&bzc), Arg::Buf(&bmn),
            Arg::Buf(&samples), Arg::U32(nxu), Arg::U32(nzu), Arg::U32(nsu),
        ], [systems, 1, 1])?;
        b.dispatch(&integrate, &[
            Arg::Buf(&samples), Arg::Buf(&bw), Arg::Buf(&bc), Arg::Buf(&bab),
            Arg::Buf(&bmn), Arg::Buf(&bv), Arg::Buf(&brho),
            Arg::U32(nsu), Arg::U32(nku),
        ], [ns, 1, 1])
    })?;
    let volts: Vec<f32> = gpu.read(&bv, ns);
    let rho: Vec<f32> = gpu.read(&brho, ns);
    if volts.iter().chain(&rho).any(|v| !v.is_finite()) {
        return Err("non-finite receiver response (check electrode geometry and currents)".into());
    }
    out.f32("voltage", &[ns], &volts)?;
    out.f32("apparent_resistivity", &[ns], &rho)?;
    Ok(())
}

fn main() {
    outcrop_common::run(&[("dc_forward", forward)]);
}
