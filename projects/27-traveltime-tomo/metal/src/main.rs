use outcrop_common::{gpu::Arg, Gpu, Inputs, Outputs, Result};
const SHADER: &str = include_str!("../kernels/tomography.metal");

// Metal indices and dimensions are uints, including products used in kernels.
fn checked_product(a: usize, b: usize) -> Result<usize> {
    let product = a.checked_mul(b).ok_or("array dimensions overflow")?;
    if product > u32::MAX as usize { return Err("array dimensions exceed Metal index range".into()); }
    Ok(product)
}

fn matrix_dims(shape: &[usize], t: &[usize], prior: &[usize]) -> Result<(usize, usize)> {
    if shape.len() != 2 { return Err("matrix must have rank 2".into()); }
    let (m, n) = (shape[0], shape[1]);
    if t != [m] { return Err("travel_times must have shape (nrays,)".into()); }
    if prior != [n] { return Err("reference_slowness must have shape (ncells,)".into()); }
    checked_product(m, n)?;
    checked_product(n, n)?;
    if m > u32::MAX as usize || n > u32::MAX as usize { return Err("array dimensions exceed Metal index range".into()); }
    Ok((m, n))
}

fn ray_matrix(inp: &Inputs, out: &mut Outputs) -> Result<()> {
    let x=inp.f32("x_edges")?; let z=inp.f32("z_edges")?;
    let src=inp.f32("sources")?; let rec=inp.f32("receivers")?;
    if x.shape.len() != 1 || x.shape[0] < 2 || z.shape.len() != 1 || z.shape[0] < 2 {
        return Err("edge arrays must have rank 1 and at least two entries".into());
    }
    if src.shape.len() != 2 || src.shape[1] != 2 { return Err("sources must have shape (nrays, 2)".into()); }
    let nr = src.shape[0];
    if rec.shape != [nr, 2] { return Err("receivers must have shape (nrays, 2) matching sources".into()); }
    let (nx,nz)=(x.shape[0]-1,z.shape[0]-1);
    checked_product(nr, 2)?;
    let nc = checked_product(nx, nz)?;
    let count = checked_product(nr, nc)?;
    let gpu=Gpu::new()?; let lib=gpu.compile(SHADER)?; let k=gpu.pipeline(&lib,"ray_matrix")?;
    let bx=gpu.buffer_from(&x.data); let bz=gpu.buffer_from(&z.data); let bs=gpu.buffer_from(&src.data); let br=gpu.buffer_from(&rec.data); let result=gpu.buffer_zeroed::<f32>(count);
    gpu.dispatch(&k,&[Arg::Buf(&bx),Arg::Buf(&bz),Arg::Buf(&bs),Arg::Buf(&br),Arg::Buf(&result),Arg::U32(nx as u32),Arg::U32(nz as u32),Arg::U32(nr as u32)],[count,1,1])?;
    out.f32("matrix",&[nr,nc],&gpu.read(&result,count))
}

fn solve_tomography(inp: &Inputs, out: &mut Outputs) -> Result<()> {
    let a=inp.f32("matrix")?; let t=inp.f32("travel_times")?; let prior=inp.f32("reference_slowness")?;
    let d=inp.num("damping")?; let lambda2=(d*d) as f32; let (m,n)=matrix_dims(&a.shape,&t.shape,&prior.shape)?; let (mu,nu)=(m as u32,n as u32);
    let gpu=Gpu::new()?; let lib=gpu.compile(SHADER)?;
    let normal=gpu.pipeline(&lib,"normal")?; let factor=gpu.pipeline(&lib,"factor")?; let residual=gpu.pipeline(&lib,"residual")?; let rhs=gpu.pipeline(&lib,"rhs")?; let correction=gpu.pipeline(&lib,"correction")?; let predict=gpu.pipeline(&lib,"predict")?;
    let ba=gpu.buffer_from(&a.data); let bt=gpu.buffer_from(&t.data); let bp=gpu.buffer_from(&prior.data); let bs=gpu.buffer_from(&prior.data);
    let bh=gpu.buffer_zeroed::<f32>(n*n); let br=gpu.buffer_zeroed::<f32>(m); let bb=gpu.buffer_zeroed::<f32>(n); let by=gpu.buffer_zeroed::<f32>(m);
    gpu.batch(|batch| {
        batch.dispatch(&normal,&[Arg::Buf(&ba),Arg::Buf(&bh),Arg::U32(mu),Arg::U32(nu),Arg::F32(lambda2)],[n*n,1,1])?;
        batch.dispatch(&factor,&[Arg::Buf(&bh),Arg::U32(nu)],[1,1,1])?;
        for _ in 0..3 {
            batch.dispatch(&residual,&[Arg::Buf(&ba),Arg::Buf(&bt),Arg::Buf(&bs),Arg::Buf(&br),Arg::U32(mu),Arg::U32(nu)],[m,1,1])?;
            batch.dispatch(&rhs,&[Arg::Buf(&ba),Arg::Buf(&br),Arg::Buf(&bs),Arg::Buf(&bp),Arg::Buf(&bb),Arg::U32(mu),Arg::U32(nu),Arg::F32(lambda2)],[n,1,1])?;
            batch.dispatch(&correction,&[Arg::Buf(&bh),Arg::Buf(&bb),Arg::Buf(&bs),Arg::U32(nu)],[1,1,1])?;
        }
        batch.dispatch(&predict,&[Arg::Buf(&ba),Arg::Buf(&bs),Arg::Buf(&by),Arg::U32(mu),Arg::U32(nu)],[m,1,1])
    })?;
    out.f32("slowness",&[n],&gpu.read(&bs,n))?; out.f32("predicted_times",&[m],&gpu.read(&by,m))
}

fn misfit_and_grad(inp: &Inputs, out: &mut Outputs) -> Result<()> {
    let a=inp.f32("matrix")?; let t=inp.f32("travel_times")?; let s=inp.f32("slowness")?; let prior=inp.f32("reference_slowness")?;
    let d=inp.num("damping")?; let lambda2=(d*d) as f32; let (m,n)=matrix_dims(&a.shape,&t.shape,&prior.shape)?;
    if s.shape != [n] { return Err("slowness must have shape (ncells,)".into()); }
    let (mu,nu)=(m as u32,n as u32);
    let gpu=Gpu::new()?; let lib=gpu.compile(SHADER)?; let residual=gpu.pipeline(&lib,"residual")?; let objective=gpu.pipeline(&lib,"objective")?; let gradient=gpu.pipeline(&lib,"gradient")?;
    let ba=gpu.buffer_from(&a.data); let bt=gpu.buffer_from(&t.data); let bs=gpu.buffer_from(&s.data); let bp=gpu.buffer_from(&prior.data);
    let br=gpu.buffer_zeroed::<f32>(m); let bo=gpu.buffer_zeroed::<f32>(1); let bg=gpu.buffer_zeroed::<f32>(n);
    gpu.batch(|batch| {
        batch.dispatch(&residual,&[Arg::Buf(&ba),Arg::Buf(&bt),Arg::Buf(&bs),Arg::Buf(&br),Arg::U32(mu),Arg::U32(nu)],[m,1,1])?;
        batch.dispatch(&objective,&[Arg::Buf(&br),Arg::Buf(&bs),Arg::Buf(&bp),Arg::Buf(&bo),Arg::U32(mu),Arg::U32(nu),Arg::F32(lambda2)],[1,1,1])?;
        batch.dispatch(&gradient,&[Arg::Buf(&ba),Arg::Buf(&br),Arg::Buf(&bs),Arg::Buf(&bp),Arg::Buf(&bg),Arg::U32(mu),Arg::U32(nu),Arg::F32(lambda2)],[n,1,1])
    })?;
    out.f32("objective",&[],&gpu.read(&bo,1))?; out.f32("gradient",&[n],&gpu.read(&bg,n))
}

fn main() { outcrop_common::run(&[("ray_matrix",ray_matrix),("solve_tomography",solve_tomography),("misfit_and_grad",misfit_and_grad)]); }
