use outcrop_common::{gpu::Arg, Gpu, Inputs, Outputs, Result};

fn coordinates_shape(name: &str, shape: &[usize]) -> Result<usize> {
    if shape.len() != 2 || shape[1] != 3 { return Err(format!("{name} must have shape (n, 3)").into()); }
    Ok(shape[0])
}
fn indices(values: &[i64], bound: usize) -> Result<Vec<u32>> {
    values.iter().map(|&x| if x < 0 || x as usize >= bound { Err("window index out of bounds".into()) } else { Ok(x as u32) }).collect()
}
fn offsets(values: &[i64], len: usize) -> Result<Vec<usize>> {
    let mut result = Vec::with_capacity(values.len());
    for &v in values {
        if v < 0 || v as usize > len || result.last().is_some_and(|&last| (v as usize) < last) { return Err("invalid window offsets".into()); }
        result.push(v as usize);
    }
    if result.first() != Some(&0) || result.last() != Some(&len) { return Err("invalid window offsets".into()); }
    Ok(result)
}
fn equivalent_sources(inp: &Inputs, out: &mut Outputs) -> Result<()> {
    let xyz=inp.f32("coordinates")?;
    let points=inp.f32("points")?;
    let evaluation=inp.f32("evaluation")?;
    let data=inp.f32("data")?;
    let weights=inp.f32("weights")?;
    let n=coordinates_shape("coordinates", &xyz.shape)?;
    let m=coordinates_shape("points", &points.shape)?;
    let k=coordinates_shape("evaluation", &evaluation.shape)?;
    if data.shape != [n] || weights.shape != [n] { return Err("data and weights must match coordinates".into()); }
    let damping=inp.num("damping")? as f32;
    if damping < 0.0 { return Err("negative damping".into()); }
    let method=inp.str("method")?;
    let di=indices(&inp.i64("data_indices")?.data, n)?;
    let si=indices(&inp.i64("source_indices")?.data, m)?;
    let doff=offsets(&inp.i64("data_offsets")?.data, di.len())?;
    let soff=offsets(&inp.i64("source_offsets")?.data, si.len())?;
    if doff.len()!=soff.len() { return Err("window count mismatch".into()); }
    let windows: Vec<(Vec<u32>,Vec<u32>)> = match method.as_str() {
        "standard" => vec![((0..n as u32).collect(),(0..m as u32).collect())],
        "boosted" => (0..doff.len()-1).map(|j| (di[doff[j]..doff[j+1]].to_vec(),si[soff[j]..soff[j+1]].to_vec())).collect(),
        _ => return Err("unknown method".into()),
    };
    let gpu=Gpu::new()?;
    let lib=gpu.compile(include_str!("../kernels/equivalent.metal"))?;
    let matrix=gpu.pipeline(&lib,"make_matrix")?;
    let rhs=gpu.pipeline(&lib,"make_rhs")?;
    let solve=gpu.pipeline(&lib,"solve")?;
    let solve_svd=gpu.pipeline(&lib,"solve_svd")?;
    let update=gpu.pipeline(&lib,"update_residual")?;
    let accumulate=gpu.pipeline(&lib,"accumulate")?;
    let predict=gpu.pipeline(&lib,"predict")?;
    let xyz_b=gpu.buffer_from(&xyz.data);
    let points_b=gpu.buffer_from(&points.data);
    let eval_b=gpu.buffer_from(&evaluation.data);
    let weights_b=gpu.buffer_from(&weights.data);
    let residual_b=gpu.buffer_from(&data.data);
    let total_b=gpu.buffer_zeroed::<f32>(m);
    for (di,si) in windows {
        let wn=di.len(); let wm=si.len();
        if wn==0 || wm==0 { continue; }
        let di_b=gpu.buffer_from(&di); let si_b=gpu.buffer_from(&si);
        let a=gpu.buffer_zeroed::<f32>((wn+wm)*wm);
        let b=gpu.buffer_zeroed::<f32>(wn+wm);
        let scales=gpu.buffer_zeroed::<f32>(wm);
        let coef=gpu.buffer_zeroed::<f32>(wm);
        gpu.dispatch(&matrix,&[Arg::Buf(&xyz_b),Arg::Buf(&points_b),Arg::Buf(&weights_b),Arg::Buf(&di_b),Arg::Buf(&si_b),Arg::Buf(&a),Arg::Buf(&scales),Arg::U32(wn as u32),Arg::U32(wm as u32),Arg::F32(damping)],[wm,1,1])?;
        gpu.dispatch(&rhs,&[Arg::Buf(&residual_b),Arg::Buf(&weights_b),Arg::Buf(&di_b),Arg::Buf(&b),Arg::U32(wn as u32),Arg::U32(wm as u32)],[wn+wm,1,1])?;
        if damping == 0.0 {
            let vectors=gpu.buffer_zeroed::<f32>(wm*wm);
            let beta=gpu.buffer_zeroed::<f32>(wm);
            gpu.dispatch(&solve_svd,&[Arg::Buf(&a),Arg::Buf(&b),Arg::Buf(&scales),Arg::Buf(&coef),Arg::Buf(&vectors),Arg::Buf(&beta),Arg::U32(wn as u32),Arg::U32(wm as u32)],[1,1,1])?;
        } else {
            gpu.dispatch(&solve,&[Arg::Buf(&a),Arg::Buf(&b),Arg::Buf(&scales),Arg::Buf(&coef),Arg::U32(wn as u32),Arg::U32(wm as u32)],[1,1,1])?;
        }
        gpu.dispatch(&update,&[Arg::Buf(&xyz_b),Arg::Buf(&points_b),Arg::Buf(&si_b),Arg::Buf(&coef),Arg::Buf(&residual_b),Arg::U32(n as u32),Arg::U32(wm as u32)],[n,1,1])?;
        gpu.dispatch(&accumulate,&[Arg::Buf(&si_b),Arg::Buf(&coef),Arg::Buf(&total_b),Arg::U32(wm as u32)],[1,1,1])?;
    }
    let output=gpu.buffer_zeroed::<f32>(k);
    gpu.dispatch(&predict,&[Arg::Buf(&eval_b),Arg::Buf(&points_b),Arg::Buf(&total_b),Arg::Buf(&output),Arg::U32(k as u32),Arg::U32(m as u32)],[k,1,1])?;
    out.f32("prediction",&[k],&gpu.read::<f32>(&output,k))
}
fn main() { outcrop_common::run(&[("equivalent_sources",equivalent_sources)]); }
