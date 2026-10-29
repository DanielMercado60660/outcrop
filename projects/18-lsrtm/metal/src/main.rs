use outcrop_common::{gpu::{Arg, Buffer, Pipeline}, Array, Gpu, Inputs, Outputs, Result};

// Leave room for the dot kernel's final 256-element block and the time loop's
// 32-step batch endpoint. Every shader element offset is a uint.
const MAX_ELEMENTS: usize = u32::MAX as usize - 255;

fn checked_len(name: &str, dims: &[usize]) -> Result<usize> {
    let mut n = 1usize;
    for &dim in dims {
        if dim > MAX_ELEMENTS {
            return Err(format!("{name}: dimension exceeds Metal uint indexing limits").into());
        }
        n = n.checked_mul(dim).ok_or_else(|| format!("{name}: size overflow"))?;
    }
    if n > MAX_ELEMENTS {
        return Err(format!("{name}: size exceeds Metal uint indexing limits").into());
    }
    // All device elements are four bytes; host profiles also contain f64s.
    let bytes = n.checked_mul(std::mem::size_of::<f64>())
        .ok_or_else(|| format!("{name}: allocation size overflow"))?;
    if bytes > isize::MAX as usize {
        return Err(format!("{name}: allocation exceeds host address limits").into());
    }
    Ok(n)
}

fn check_array<T>(name: &str, a: &Array<T>, rank: usize) -> Result<()> {
    if a.shape.len() != rank {
        return Err(format!("{name}: expected rank {rank}, got shape {:?}", a.shape).into());
    }
    if checked_len(name, &a.shape)? != a.data.len() {
        return Err(format!("{name}: shape does not match data length").into());
    }
    Ok(())
}

fn check_shape<T>(name: &str, a: &Array<T>, shape: &[usize]) -> Result<()> {
    check_array(name, a, shape.len())?;
    if a.shape != shape {
        return Err(format!("{name}: expected shape {shape:?}, got {:?}", a.shape).into());
    }
    Ok(())
}

fn check_locations(name: &str, a: &Array<i64>, nz: usize, nx: usize) -> Result<()> {
    for (i, p) in a.data.as_chunks::<2>().0.iter().enumerate() {
        if p[0] < 0 || p[1] < 0 || p[0] >= nz as i64 || p[1] >= nx as i64 {
            return Err(format!("{name}: node {i} ({},{}) is outside physical grid ({nz},{nx})", p[0], p[1]).into());
        }
    }
    Ok(())
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Params {
    nz: u32, nx: u32, mz: u32, mx: u32, shots: u32, nsrc: u32, nrec: u32,
    nt: u32, pad: u32, n: u32, hz: f32, hx: f32, dt2: f32,
}
impl Params {
    fn bytes(&self) -> &[u8] {
        // repr(C), all fields are initialized 32-bit scalars (no padding).
        unsafe { std::slice::from_raw_parts(self as *const Self as *const u8, std::mem::size_of::<Self>()) }
    }
    // Construction validates these products before Params or any buffer exists.
    fn total(self) -> usize { self.n as usize*self.shots as usize }
    fn model_len(self) -> usize { self.nz as usize*self.nx as usize }
    fn data_len(self) -> usize { self.shots as usize*self.nrec as usize*self.nt as usize }
}
struct Operator {
    gpu: Gpu, c: Params, prof: Buffer, v: Buffer, mask: Buffer, src: Buffer,
    sloc: Buffer, rloc: Buffer, drive: Buffer,
    forward: Pipeline, sample: Pipeline, extend: Pipeline, prepare: Pipeline,
    reverse: Pipeline, fold: Pipeline, axpy: Pipeline, scale: Pipeline,
    subtract: Pipeline, dot: Pipeline,
}

fn profile(n: usize, width: usize, h: f64, dt: f64, freq: f64, vmax: f64) -> Vec<[f32;3]> {
    let mut a=vec![0.0;n];let mut b=vec![0.0;n];
    for i in 0..n {
        let ql=((width+2) as f64-i as f64).max(0.0)/(width as f64);
        let qh=(i as f64-(n-1-width-2) as f64).max(0.0)/(width as f64);
        let ql=ql.min(1.0);let qh=qh.min(1.0);let q=ql.max(qh);
        if q>0.0 {
            let sigma=3.0*vmax*1000_f64.ln()/(2.0*width as f64*h)*(ql*ql+qh*qh);
            let alpha=std::f64::consts::PI*freq*(1.0-q);
            a[i]=(-(sigma+alpha)*dt).exp();
            if sigma+alpha!=0.0 {b[i]=sigma/(sigma+alpha)*(-(sigma+alpha)*dt).exp_m1();}
        }
    }
    (0..n).map(|i| {
        let db=if i>=2&&i<n-2 {((2.0/3.0)*(b[i+1]-b[i-1])-(1.0/12.0)*(b[i+2]-b[i-2]))/h} else {0.0};
        [a[i] as f32,b[i] as f32,db as f32]
    }).collect()
}

impl Operator {
    fn new(inp: &Inputs, model: Option<&Array<f32>>, observed: Option<&Array<f32>>) -> Result<Self> {
        let v=inp.f32("v")?;let mask=inp.f32("mask")?;
        let src=inp.f32("source_amplitudes")?;
        let sloc=inp.i64("source_locations")?;let rloc=inp.i64("receiver_locations")?;
        let h=inp.nums("spacing")?;let dt=inp.num("dt")?;
        let width=inp.int("pml_width")?;
        if inp.int("accuracy")?!=4 || width<=0 {return Err("requires accuracy=4 and positive PML width".into());}
        let w=usize::try_from(width).map_err(|_| "pml_width exceeds host size limits")?;
        check_array("v",&v,2)?;
        let nz=v.shape[0];let nx=v.shape[1];
        if nz==0 || nx==0 {return Err("v: grid axes must be nonempty".into());}
        check_shape("mask",&mask,&[nz,nx])?;
        if let Some(m)=model {check_shape("model",m,&[nz,nx])?;}
        check_array("source_amplitudes",&src,3)?;
        check_array("receiver_locations",&rloc,3)?;
        let shots=src.shape[0];let nsrc=src.shape[1];let nt=src.shape[2];
        let nrec=rloc.shape[1];
        check_shape("source_locations",&sloc,&[shots,nsrc,2])?;
        check_shape("receiver_locations",&rloc,&[shots,nrec,2])?;
        if let Some(d)=observed {check_shape("observed",d,&[shots,nrec,nt])?;}
        check_locations("source_locations",&sloc,nz,nx)?;
        check_locations("receiver_locations",&rloc,nz,nx)?;
        if h.len()!=2 || h.iter().any(|h| !h.is_finite() || *h<=0.0 || !(*h as f32).is_finite() || (*h as f32)==0.0) {
            return Err("spacing must contain two finite positive float32 spacings [dz,dx]".into());
        }
        let freq=inp.num("pml_freq")?;
        if !dt.is_finite() || dt<=0.0 || !freq.is_finite() || freq<=0.0 {
            return Err("dt and pml_freq must be finite and positive".into());
        }
        if v.data.iter().any(|v| !v.is_finite() || *v<=0.0) {
            return Err("v must contain finite positive velocities".into());
        }
        let pad=w.checked_add(2).ok_or("padded dimension overflow")?;
        let padding=pad.checked_mul(2).ok_or("padded dimension overflow")?;
        let mz=nz.checked_add(padding).ok_or("padded z dimension overflow")?;
        let mx=nx.checked_add(padding).ok_or("padded x dimension overflow")?;
        // extend_model converts padded coordinates to signed int for clamping.
        if mz>i32::MAX as usize || mx>i32::MAX as usize {
            return Err("padded dimensions exceed Metal int coordinate limits".into());
        }
        let n=checked_len("padded grid",&[mz,mx])?;
        let total=checked_len("shot wavefields",&[n,shots])?;
        checked_len("six-plane wavefields",&[6,total])?;
        let prof_len=checked_len("PML profiles",&[6,n])?;
        let drive_len=checked_len("background cache",&[total,nt])?;
        checked_len("receiver data",&[shots,nrec,nt])?;
        checked_len("source count",&[shots,nsrc])?;
        checked_len("receiver count",&[shots,nrec])?;
        let vmax=v.data.iter().copied().fold(0.0f32,f32::max) as f64;
        if dt.abs()*vmax*(h[0].powi(-2)+h[1].powi(-2)).sqrt()>0.6 {return Err("CFL domain requires R=1".into());}
        let c=Params {nz:u32::try_from(nz)?,nx:u32::try_from(nx)?,mz:u32::try_from(mz)?,mx:u32::try_from(mx)?,
            shots:u32::try_from(shots)?,nsrc:u32::try_from(nsrc)?,nrec:u32::try_from(nrec)?,
            nt:u32::try_from(nt)?,pad:u32::try_from(pad)?,n:u32::try_from(n)?,
            hz:h[0] as f32,hx:h[1] as f32,dt2:(dt*dt) as f32};
        let pz=profile(mz,w,h[0],dt,freq,vmax);
        let px=profile(mx,w,h[1],dt,freq,vmax);
        let mut prof=vec![0.0;prof_len];let mut vp=vec![0.0;n];
        for (z, zp) in pz.iter().enumerate() {for (x, xp) in px.iter().enumerate() {
            let j=z*mx+x;
            for k in 0..3 {prof[k*mz*mx+j]=zp[k];prof[(3+k)*mz*mx+j]=xp[k];}
            let zz=z.saturating_sub(w+2).min(nz-1);let xx=x.saturating_sub(w+2).min(nx-1);
            vp[j]=v.data[zz*nx+xx];
        }}
        // Coordinates are nonnegative and bounded, and padded-grid size fits uint.
        let locations=|data: &[i64]| -> Vec<u32> {data.as_chunks::<2>().0.iter().map(|p|
            ((p[0] as usize+pad)*mx+p[1] as usize+pad) as u32).collect()};
        let gpu=Gpu::new()?;let lib=gpu.compile(include_str!("../kernels/lsrtm.metal"))?;
        let op=Self {
            prof:gpu.buffer_from(&prof),v:gpu.buffer_from(&vp),mask:gpu.buffer_from(&mask.data),
            src:gpu.buffer_from(&src.data),sloc:gpu.buffer_from(&locations(&sloc.data)),
            rloc:gpu.buffer_from(&locations(&rloc.data)),drive:gpu.buffer_zeroed::<f32>(drive_len),
            forward:gpu.pipeline(&lib,"forward")?,sample:gpu.pipeline(&lib,"sample")?,
            extend:gpu.pipeline(&lib,"extend_model")?,prepare:gpu.pipeline(&lib,"reverse_prepare")?,
            reverse:gpu.pipeline(&lib,"reverse_step")?,fold:gpu.pipeline(&lib,"fold")?,
            axpy:gpu.pipeline(&lib,"axpy")?,scale:gpu.pipeline(&lib,"scale")?,
            subtract:gpu.pipeline(&lib,"subtract")?,dot:gpu.pipeline(&lib,"dot")?,gpu,c,
        };
        op.background()?;
        Ok(op)
    }
    fn zeros(&self,n:usize)->Buffer {self.gpu.buffer_zeroed::<f32>(n)}
    fn background(&self)->Result<()> {
        let states=[self.zeros(6*self.c.total()),self.zeros(6*self.c.total())];
        let dv=self.zeros(self.c.n as usize);
        for start in (0..self.c.nt).step_by(32) {
            self.gpu.batch(|b| {
                for t in start..(start+32).min(self.c.nt) {
                    let i=t as usize%2;
                    b.dispatch(&self.forward,&[Arg::Buf(&states[i]),Arg::Buf(&states[1-i]),
                        Arg::Buf(&self.prof),Arg::Buf(&self.v),Arg::Buf(&dv),Arg::Buf(&self.src),
                        Arg::Buf(&self.sloc),Arg::Buf(&self.drive),Arg::Bytes(self.c.bytes()),Arg::U32(t),Arg::U32(1)],
                        [self.c.total(),1,1])?;
                } Ok(())
            })?;
        } Ok(())
    }
    fn apply(&self,m:&Buffer)->Result<Buffer> {
        let dv=self.zeros(self.c.n as usize);let data=self.zeros(self.c.data_len());
        let states=[self.zeros(6*self.c.total()),self.zeros(6*self.c.total())];
        self.gpu.dispatch(&self.extend,&[Arg::Buf(m),Arg::Buf(&self.mask),Arg::Buf(&dv),Arg::Bytes(self.c.bytes())],[self.c.n as usize,1,1])?;
        for start in (0..self.c.nt).step_by(32) {
            self.gpu.batch(|b| {
                for t in start..(start+32).min(self.c.nt) {
                    let i=t as usize%2;
                    b.dispatch(&self.sample,&[Arg::Buf(&states[i]),Arg::Buf(&self.rloc),Arg::Buf(&data),Arg::Bytes(self.c.bytes()),Arg::U32(t)],[(self.c.shots*self.c.nrec) as usize,1,1])?;
                    b.dispatch(&self.forward,&[Arg::Buf(&states[i]),Arg::Buf(&states[1-i]),
                        Arg::Buf(&self.prof),Arg::Buf(&self.v),Arg::Buf(&dv),Arg::Buf(&self.src),
                        Arg::Buf(&self.sloc),Arg::Buf(&self.drive),Arg::Bytes(self.c.bytes()),Arg::U32(t),Arg::U32(0)],
                        [self.c.total(),1,1])?;
                } Ok(())
            })?;
        } Ok(data)
    }
    fn adjoint(&self,data:&Buffer)->Result<Buffer> {
        let states=[self.zeros(6*self.c.total()),self.zeros(6*self.c.total())];
        let tmp=self.zeros(6*self.c.total());let padded=self.zeros(self.c.total());
        for start in (0..self.c.nt).step_by(32) {
            self.gpu.batch(|b| {
                for step in start..(start+32).min(self.c.nt) {
                    let t=self.c.nt-1-step;let i=step as usize%2;
                    b.dispatch(&self.prepare,&[Arg::Buf(&states[i]),Arg::Buf(&self.prof),Arg::Buf(&self.v),Arg::Buf(&tmp),Arg::Bytes(self.c.bytes())],[self.c.total(),1,1])?;
                    b.dispatch(&self.reverse,&[Arg::Buf(&states[i]),Arg::Buf(&states[1-i]),Arg::Buf(&tmp),
                        Arg::Buf(&self.prof),Arg::Buf(&self.v),Arg::Buf(&self.drive),Arg::Buf(data),
                        Arg::Buf(&self.rloc),Arg::Buf(&padded),Arg::Bytes(self.c.bytes()),Arg::U32(t)],[self.c.total(),1,1])?;
                } Ok(())
            })?;
        }
        let grad=self.zeros(self.c.model_len());
        self.gpu.dispatch(&self.fold,&[Arg::Buf(&padded),Arg::Buf(&self.mask),Arg::Buf(&grad),Arg::Bytes(self.c.bytes())],[self.c.model_len(),1,1])?;
        Ok(grad)
    }
    fn dot(&self,a:&Buffer,b:&Buffer,n:usize)->Result<f64> {
        let blocks=n.div_ceil(256);let sums=self.zeros(blocks);
        self.gpu.dispatch(&self.dot,&[Arg::Buf(a),Arg::Buf(b),Arg::Buf(&sums),Arg::U32(n as u32)],[blocks,1,1])?;
        Ok(self.gpu.read::<f32>(&sums,blocks).iter().map(|&v|v as f64).sum())
    }
    fn axpy(&self,y:&Buffer,x:&Buffer,a:f64,n:usize)->Result<()> {
        self.gpu.dispatch(&self.axpy,&[Arg::Buf(y),Arg::Buf(x),Arg::F32(a as f32),Arg::U32(n as u32)],[n,1,1])
    }
    fn scale(&self,y:&Buffer,a:f64,n:usize)->Result<()> {
        self.gpu.dispatch(&self.scale,&[Arg::Buf(y),Arg::F32(a as f32),Arg::U32(n as u32)],[n,1,1])
    }
    fn residual(&self,d:&Buffer,p:&Buffer)->Result<Buffer> {
        let n=self.c.data_len();let r=self.zeros(n);
        self.gpu.dispatch(&self.subtract,&[Arg::Buf(d),Arg::Buf(p),Arg::Buf(&r),Arg::U32(n as u32)],[n,1,1])?;Ok(r)
    }
    fn output(&self,out:&mut Outputs,name:&str,buf:&Buffer,shape:&[usize])->Result<()> {
        out.f32(name,shape,&self.gpu.read::<f32>(buf,shape.iter().product()))
    }
    fn data_shape(&self)->[usize;3] {[self.c.shots as usize,self.c.nrec as usize,self.c.nt as usize]}
    fn model_shape(&self)->[usize;2] {[self.c.nz as usize,self.c.nx as usize]}
}
fn born_model(inp:&Inputs,out:&mut Outputs)->Result<()> {
    let model=inp.f32("model")?;
    let op=Operator::new(inp,Some(&model),None)?;let m=op.gpu.buffer_from(&model.data);
    let pred=op.apply(&m)?;op.output(out,"predicted",&pred,&op.data_shape())
}
fn misfit_and_grad(inp:&Inputs,out:&mut Outputs)->Result<()> {
    let model=inp.f32("model")?;let observed=inp.f32("observed")?;
    let op=Operator::new(inp,Some(&model),Some(&observed))?;let m=op.gpu.buffer_from(&model.data);
    let d=op.gpu.buffer_from(&observed.data);let pred=op.apply(&m)?;
    let r=op.residual(&pred,&d)?;let grad=op.adjoint(&r)?;
    let phi=0.5*op.dot(&r,&r,op.c.data_len())?;
    out.f32("objective",&[],&[phi as f32])?;
    op.output(out,"gradient",&grad,&op.model_shape())?;
    op.output(out,"predicted",&pred,&op.data_shape())
}
fn lsrtm(inp:&Inputs,out:&mut Outputs)->Result<()> {
    let iterations=inp.int("iterations")?;
    if iterations<0 {return Err("iterations must be nonnegative".into());}
    let iterations=usize::try_from(iterations).map_err(|_| "iterations exceeds host size limits")?;
    let history_len=iterations.checked_add(1).ok_or("residual history size overflow")?;
    checked_len("residual history",&[history_len])?;
    let observed=inp.f32("observed")?;
    let op=Operator::new(inp,None,Some(&observed))?;
    let d=op.gpu.buffer_from(&observed.data);
    let nm=op.c.model_len();let nd=op.c.data_len();let m=op.zeros(nm);
    let mut pred=op.zeros(nd);let mut history=vec![0.0f32;history_len];
    let norm=op.dot(&d,&d,nd)?.sqrt();
    let rtm=if norm==0.0 {op.zeros(nm)} else {op.adjoint(&d)?};
    if norm!=0.0 {
        history[0]=1.0;let mut residual=d.clone();let mut basis:Vec<(Buffer,Buffer)>=Vec::new();
        for k in 0..iterations {
            let p=op.adjoint(&residual)?;
            let q=op.apply(&p)?;
            for _ in 0..2 {for (pi,qi) in &basis {
                let beta=op.dot(qi,&q,nd)?;
                op.axpy(&p,pi,-beta,nm)?;op.axpy(&q,qi,-beta,nd)?;
            }}
            let scale=op.dot(&q,&q,nd)?.sqrt();
            if scale==0.0 {return Err("CGLS exact breakdown on nonzero data".into());}
            op.scale(&p,1.0/scale,nm)?;op.scale(&q,1.0/scale,nd)?;
            let alpha=op.dot(&residual,&q,nd)?;
            op.axpy(&m,&p,alpha,nm)?;basis.push((p,q));
            pred=op.apply(&m)?;residual=op.residual(&d,&pred)?;
            history[k+1]=(op.dot(&residual,&residual,nd)?.sqrt()/norm) as f32;
        }
    }
    op.output(out,"image",&m,&op.model_shape())?;op.output(out,"rtm",&rtm,&op.model_shape())?;
    op.output(out,"predicted",&pred,&op.data_shape())?;
    out.f32("residual_history",&[history.len()],&history)
}
fn main() {
    outcrop_common::run(&[("born_model",born_model),("misfit_and_grad",misfit_and_grad),("lsrtm",lsrtm)]);
}
