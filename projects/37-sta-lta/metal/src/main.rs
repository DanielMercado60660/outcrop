use outcrop_common::gpu::Arg;
use outcrop_common::{Gpu, Inputs, Outputs, Result};

fn classic_sta_lta(inp: &Inputs, out: &mut Outputs) -> Result<()> {
    let trace = inp.f32("trace")?;
    let n = trace.data.len();
    let nsta = usize::try_from(inp.int("nsta")?)?;
    let nlta = usize::try_from(inp.int("nlta")?)?;
    if trace.shape != [n] || nsta == 0 || nsta > nlta || nlta > n {
        return Err("expected a one-dimensional trace and 1 <= nsta <= nlta <= N".into());
    }
    let mut silent = 0usize;
    for &sample in &trace.data {
        if !sample.is_finite() {
            return Err("trace must contain only finite samples".into());
        }
        // Inspect bits so even the smallest subnormal counts as positive energy.
        silent = if sample.to_bits() & 0x7fff_ffff == 0 {
            silent + 1
        } else {
            0
        };
        if silent >= nlta {
            return Err("every full long window must have positive energy".into());
        }
    }

    let n32 = u32::try_from(n)?;
    let sta32 = u32::try_from(nsta)?;
    let lta32 = u32::try_from(nlta)?;
    let leaves = n.checked_next_power_of_two().ok_or("trace is too long")?;
    let nodes = leaves.checked_mul(2).ok_or("trace is too long")?;
    // Tree indices and exclusive range endpoints are uints in the shader.
    u32::try_from(nodes).map_err(|_| "trace is too long for Metal tree indices")?;
    let leaves32 = u32::try_from(leaves)?;

    let gpu = Gpu::new()?;
    let library = gpu.compile(include_str!("../kernels/sta_lta.metal"))?;
    let initialize = gpu.pipeline(&library, "energy_leaves")?;
    let reduce = gpu.pipeline(&library, "energy_reduce")?;
    let kernel = gpu.pipeline(&library, "classic_sta_lta")?;
    let input = gpu.buffer_from(&trace.data);
    // MSL Energy is { float mantissa; int exponent; }, eight bytes per node.
    let tree = gpu.buffer_zeroed::<[u32; 2]>(nodes);
    let output = gpu.buffer_zeroed::<f32>(n);
    gpu.batch(|batch| {
        batch.dispatch(
            &initialize,
            &[
                Arg::Buf(&input),
                Arg::Buf(&tree),
                Arg::U32(n32),
                Arg::U32(leaves32),
            ],
            [leaves, 1, 1],
        )?;
        let mut first = leaves32 / 2;
        while first > 0 {
            batch.dispatch(
                &reduce,
                &[Arg::Buf(&tree), Arg::U32(first)],
                [first as usize, 1, 1],
            )?;
            first /= 2;
        }
        batch.dispatch(
            &kernel,
            &[
                Arg::Buf(&tree),
                Arg::Buf(&output),
                Arg::U32(n32),
                Arg::U32(sta32),
                Arg::U32(lta32),
                Arg::U32(leaves32),
            ],
            [n, 1, 1],
        )
    })?;
    out.f32("characteristic", &[n], &gpu.read::<f32>(&output, n))
}

fn trigger_onset(inp: &Inputs, out: &mut Outputs) -> Result<()> {
    let cf = inp.f32("characteristic")?;
    let n = cf.data.len();
    let on = inp.num("on")?;
    let off = inp.num("off")?;
    if cf.shape != [n] || !(on > off && off > 0.0) {
        return Err("expected a one-dimensional characteristic and on > off > 0".into());
    }
    let mut pairs = Vec::<i64>::new();
    let mut start = None;
    for (i, &value) in cf.data.iter().enumerate() {
        if !value.is_finite() || value < 0.0 {
            return Err("characteristic must contain finite nonnegative samples".into());
        }
        let value = f64::from(value);
        if start.is_none() && value >= on {
            start = Some(i);
        } else if value < off {
            if let Some(begin) = start.take() {
                pairs.push(i64::try_from(begin)?);
                pairs.push(i64::try_from(i - 1)?);
            }
        }
    }
    if let Some(begin) = start {
        pairs.push(i64::try_from(begin)?);
        pairs.push(i64::try_from(n - 1)?);
    }
    out.i64("triggers", &[pairs.len() / 2, 2], &pairs)
}

fn main() {
    outcrop_common::run(&[
        ("classic_sta_lta", classic_sta_lta),
        ("trigger_onset", trigger_onset),
    ]);
}
