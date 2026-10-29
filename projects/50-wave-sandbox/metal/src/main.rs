mod metal;
mod input;

use outcrop_common::gpu::{Buffer, Pipeline};
use outcrop_common::{Gpu, Inputs, Outputs, Result};
use metal::{Arg, Command, Renderer};
use input::CheckedInputs;

// MSL uses uint for buffer offsets and int for stencil/linked-list indices.
// Check complete products before any narrowing or allocation. Six state
// planes also bound w/h enough for signed periodic wrap and stencil offsets.
fn elements(name: &str, dims: &[usize]) -> Result<usize> {
    let n = dims.iter().try_fold(1usize, |n, &d| n.checked_mul(d))
        .ok_or_else(|| format!("{name}: element count overflow"))?;
    u32::try_from(n).map_err(|_| format!("{name}: exceeds Metal uint indexing"))?;
    n.checked_mul(std::mem::size_of::<f32>())
        .filter(|&bytes| bytes <= isize::MAX as usize)
        .ok_or_else(|| format!("{name}: buffer size overflow"))?;
    Ok(n)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Params {
    w: u32, h: u32, ny: u32, nx: u32, p: u32, nt: u32, nr: u32,
    periodic: u32, save_history: u32,
    dt2: f32, iy: f32, ix: f32, scale: f32,
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Profile { a: f32, b: f32, c: f32 }

// Profile setup uses the JSON doubles, then stores float32 coefficients once.
// Values outside the evolution region represent the specified clipped guards.
#[allow(clippy::too_many_arguments)]
fn profile(n: usize, p: usize, radius: usize, h: f64, dt: f64, freq: f64,
           vmax: f64, periodic: bool) -> Vec<Profile> {
    if periodic || p == 0 { return vec![Profile::default(); n]; }
    let sigma0 = -3.0*vmax*0.001_f64.ln()/(2.0*p as f64*h);
    let eval = |i: isize| -> (f64, f64) {
        let ql=((p as f64-i as f64)/p as f64).clamp(0.0,1.0);
        let qr=((i as f64-(n-p-1) as f64)/p as f64).clamp(0.0,1.0);
        let q=ql.max(qr);
        let sigma=sigma0*(ql*ql+qr*qr);
        let alpha=std::f64::consts::PI*freq*(1.0-q);
        let a=if q>0.0 { (-(sigma+alpha)*dt).exp() } else { 0.0 };
        let b=if sigma+alpha==0.0 { 0.0 } else {
            sigma/(sigma+alpha)* (-(sigma+alpha)*dt).exp_m1()
        };
        (a,b)
    };
    let coeff: &[f64]=match radius {
        1=>&[0.5], 2=>&[2.0/3.0,-1.0/12.0],
        3=>&[0.75,-3.0/20.0,1.0/60.0],
        _=>&[0.8,-0.2,4.0/105.0,-1.0/280.0],
    };
    (0..n).map(|i| {
        let (a,b)=eval(i as isize);
        let c: f64=coeff.iter().enumerate().map(|(j,v)| {
            let d=(j+1) as isize;
            v*(eval(i as isize+d).1-eval(i as isize-d).1)/h
        }).sum();
        Profile { a:a as f32, b:b as f32, c:c as f32 }
    }).collect()
}

struct Simulation {
    gpu: Gpu, params: Params,
    states: [Buffer;2], velocity: Buffer, py: Buffer, px: Buffer,
    source_map: Buffer, sources: Buffer, locations: Buffer,
    head: Buffer, links: Buffer, receivers: Buffer, history: Buffer,
    advance: Pipeline, gather: Pipeline,
    library: outcrop_common::gpu::Library,
}
impl Simulation {
    fn new(inp: &CheckedInputs<'_>, gradient: bool) -> Result<Self> {
        let v=inp.f32("velocity")?;
        let state=inp.f32("state")?;
        let sources=inp.f32("source_amplitudes")?;
        let sloc=inp.i64("source_locations")?;
        let rloc=inp.i64("receiver_locations")?;
        if v.shape.len()!=2 || sources.shape.len()!=2 { return Err("invalid model/source shape".into()); }
        let ny=v.shape[0]; let nx=v.shape[1];
        if ny==0 || nx==0 { return Err("velocity dimensions must be nonzero".into()); }
        let ns=sources.shape[0]; let nt=sources.shape[1];
        if ns==0 || nt==0 { return Err("source and time counts must be nonzero".into()); }
        if sloc.shape!=[ns,2] { return Err("source_locations must have shape (ns, 2)".into()); }
        if rloc.shape.len()!=2 || rloc.shape[1]!=2 || rloc.shape[0]==0 {
            return Err("receiver_locations must have shape (nr, 2), nr >= 1".into());
        }
        let nr=rloc.shape[0];
        i32::try_from(ns).map_err(|_| "source count exceeds Metal int indexing")?;
        i32::try_from(nr).map_err(|_| "receiver count exceeds Metal int indexing")?;
        let p=usize::try_from(inp.int("pml_width")?).map_err(|_| "pml_width must be nonnegative")?;
        let accuracy=inp.int("accuracy")?;
        if ![2,4,6,8].contains(&accuracy) { return Err("accuracy must be 2,4,6,8".into()); }
        let periodic=match inp.str("boundary")?.as_str() {
            "pml"=>false,"periodic"=>true,_=>return Err("invalid boundary".into()),
        };
        if periodic && p!=0 { return Err("periodic boundary requires pml_width=0".into()); }
        let twice_p=p.checked_mul(2).ok_or("PML dimension overflow")?;
        let w=nx.checked_add(twice_p).ok_or("extended width overflow")?;
        let h=ny.checked_add(twice_p).ok_or("extended height overflow")?;
        let n=elements("extended model",&[h,w])?;
        let state_len=elements("state",&[6,n])?;
        elements("sources",&[ns,nt])?;
        let receiver_len=elements("receivers",&[nr,nt])?;
        let history_len=if gradient { elements("history",&[n,nt])? } else { 1 };
        let rgba_len=elements("rgba",&[ny,nx,4])?;
        let samples_len=if gradient { 1 } else { elements("texture_samples",&[nt,nr,4])? };
        if state.shape!=[6,h,w] { return Err("invalid six-channel state shape".into()); }
        let spacing=inp.nums("grid_spacing")?;
        if spacing.len()!=2 || spacing.iter().any(|h| !h.is_finite() || *h<=0.0) {
            return Err("grid_spacing must contain exactly two finite positive values".into());
        }
        let dt=inp.num("dt")?;
        let params=Params { w:u32::try_from(w)?,h:u32::try_from(h)?,ny:u32::try_from(ny)?,nx:u32::try_from(nx)?,p:u32::try_from(p)?,
            nt:u32::try_from(nt)?,nr:u32::try_from(nr)?,periodic:periodic as u32,
            save_history:gradient as u32,dt2:(dt*dt) as f32,
            iy:(1.0/spacing[0]) as f32,ix:(1.0/spacing[1]) as f32,
            scale:if gradient { 1.0 } else { inp.num("render_scale")? as f32 } };
        let mut extended=vec![0.0f32;n];
        for y in 0..h { for x in 0..w {
            extended[y*w+x]=v.data[y.saturating_sub(p).min(ny-1)*nx+x.saturating_sub(p).min(nx-1)];
        }}
        let cell=|y:i64,x:i64| -> Result<usize> {
            if y<0 || x<0 || y>=ny as i64 || x>=nx as i64 { return Err("location outside model".into()); }
            Ok((y as usize+p)*w+x as usize+p)
        };
        let mut source_map=vec![-1i32;n];
        for (s,l) in sloc.data.as_chunks::<2>().0.iter().enumerate() {
            let k=cell(l[0],l[1])?;
            source_map[k]=i32::try_from(s)?;
        }
        let mut locations=Vec::new();
        let mut head=vec![-1i32;n]; let mut links=vec![-1i32;params.nr as usize];
        for (r,l) in rloc.data.as_chunks::<2>().0.iter().enumerate() {
            let k=cell(l[0],l[1])?;
            locations.push(u32::try_from(k)?); links[r]=head[k]; head[k]=i32::try_from(r)?;
        }
        let freq=inp.num("pml_freq")?; let vmax=inp.num("max_velocity")?;
        let gpu=Gpu::new()?;
        // common's buffer constructors expect successful allocation. Reject
        // unsupported lengths before calling them (including float3 profiles).
        metal::check_buffer_lengths(&gpu,&[
            state_len, sources.data.len(), receiver_len, history_len,
            rgba_len, samples_len, elements("y profile",&[h,3])?,
            elements("x profile",&[w,3])?,
        ])?;
        let msl=format!("#define RADIUS {}\n{}",accuracy/2,include_str!("../kernels/wave.metal"));
        let library=gpu.compile(&msl)?;
        let advance=gpu.pipeline(&library,"advance")?;
        let gather=gpu.pipeline(&library,"gather")?;
        let project=gpu.pipeline(&library,"project")?;
        let states=[gpu.buffer_from(&state.data),gpu.buffer_zeroed::<f32>(state_len)];
        let command=Command::new(&gpu)?;
        command.compute(&project,&[Arg::Buf(&states[0]),Arg::pod(&params)],n)?;
        command.finish()?;
        let py=gpu.buffer_from(&profile(h,p,accuracy as usize/2,spacing[0],dt,freq,vmax,periodic));
        let px=gpu.buffer_from(&profile(w,p,accuracy as usize/2,spacing[1],dt,freq,vmax,periodic));
        Ok(Self { velocity:gpu.buffer_from(&extended),py,px,
            source_map:gpu.buffer_from(&source_map),sources:gpu.buffer_from(&sources.data),
            locations:gpu.buffer_from(&locations),head:gpu.buffer_from(&head),links:gpu.buffer_from(&links),
            receivers:gpu.buffer_zeroed::<f32>(receiver_len),
            history:gpu.buffer_zeroed::<f32>(history_len),
            gpu,params,states,advance,gather,library })
    }
    fn step(&self, command: &Command, t:u32) -> Result<()> {
        let old=&self.states[t as usize%2]; let next=&self.states[(t as usize+1)%2];
        command.compute_pair(
            (&self.gather,&[Arg::Buf(old),Arg::Buf(&self.locations),Arg::Buf(&self.receivers),
                            Arg::pod(&self.params),Arg::pod(&t)],self.params.nr as usize),
            (&self.advance,&[Arg::Buf(old),Arg::Buf(next),Arg::Buf(&self.velocity),
                             Arg::Buf(&self.py),Arg::Buf(&self.px),Arg::Buf(&self.source_map),
                             Arg::Buf(&self.sources),Arg::Buf(&self.history),Arg::pod(&self.params),
                             Arg::pod(&t)],self.n()))
    }
    fn n(&self)->usize { self.params.w as usize*self.params.h as usize }
}

fn run_frames(inp:&Inputs,out:&mut Outputs)->Result<()> {
    let _pool=metal::Pool::new();
    let inp=CheckedInputs::new(inp)?;
    let s=Simulation::new(&inp,false)?;
    let p=&s.params;
    let render=Renderer::new(&s.gpu,&s.library,p.nx as usize,p.ny as usize)?;
    let probe=s.gpu.pipeline(&s.library,"probe")?;
    let read_image=s.gpu.pipeline(&s.library,"read_image")?;
    let samples=s.gpu.buffer_zeroed::<f32>(p.nt as usize*p.nr as usize*4);
    let rgba=s.gpu.buffer_zeroed::<f32>(p.ny as usize*p.nx as usize*4);
    let command=Command::new(&s.gpu)?;
    for t in 0..p.nt {
        s.step(&command,t)?;
        render.draw(&command,&s.states[(t as usize+1)%2],p)?;
        command.compute_texture(&probe,&[Arg::Buf(&s.locations),Arg::Buf(&samples),Arg::pod(p),
                                        Arg::pod(&t)],p.nr as usize,&render)?;
    }
    command.compute_texture(&read_image,&[Arg::Buf(&rgba),Arg::pod(p)],
                            p.ny as usize*p.nx as usize,&render)?;
    command.finish()?;
    out.f32("state",&[6,p.h as usize,p.w as usize],&s.gpu.read::<f32>(&s.states[p.nt as usize%2],6*s.n()))?;
    out.f32("receiver_amplitudes",&[p.nr as usize,p.nt as usize],
            &s.gpu.read::<f32>(&s.receivers,p.nr as usize*p.nt as usize))?;
    out.f32("rgba",&[p.ny as usize,p.nx as usize,4],
            &s.gpu.read::<f32>(&rgba,p.ny as usize*p.nx as usize*4))?;
    out.f32("texture_samples",&[p.nt as usize,p.nr as usize,4],
            &s.gpu.read::<f32>(&samples,p.nt as usize*p.nr as usize*4))
}

fn misfit_and_grad(inp:&Inputs,out:&mut Outputs)->Result<()> {
    let _pool=metal::Pool::new();
    let inp=CheckedInputs::new(inp)?;
    let s=Simulation::new(&inp,true)?;
    let p=&s.params; let n=s.n();
    let observed=inp.f32("observed")?;
    if observed.shape!=[p.nr as usize,p.nt as usize] { return Err("invalid observed shape".into()); }
    let observed=s.gpu.buffer_from(&observed.data);
    let residual=s.gpu.buffer_zeroed::<f32>(p.nr as usize*p.nt as usize);
    let objective=s.gpu.buffer_zeroed::<f32>(1);
    let adjoints=[s.gpu.buffer_zeroed::<f32>(6*n),s.gpu.buffer_zeroed::<f32>(6*n)];
    let terms=s.gpu.buffer_zeroed::<f32>(6*n);
    let extended_gradient=s.gpu.buffer_zeroed::<f32>(n);
    let gradient=s.gpu.buffer_zeroed::<f32>(p.ny as usize*p.nx as usize);
    let j=s.gpu.pipeline(&s.library,"objective")?;
    let at=s.gpu.pipeline(&s.library,"adjoint_terms")?;
    let astep=s.gpu.pipeline(&s.library,"adjoint_step")?;
    let ext=s.gpu.pipeline(&s.library,"extension_transpose")?;
    let command=Command::new(&s.gpu)?;
    for t in 0..p.nt { s.step(&command,t)?; }
    command.compute(&j,&[Arg::Buf(&s.receivers),Arg::Buf(&observed),Arg::Buf(&residual),
                         Arg::Buf(&objective),Arg::pod(p)],1)?;
    for (iteration,t) in (0..p.nt).rev().enumerate() {
        let l=&adjoints[iteration%2]; let prev=&adjoints[(iteration+1)%2];
        command.compute_pair(
            (&at,&[Arg::Buf(l),Arg::Buf(&s.velocity),Arg::Buf(&s.py),Arg::Buf(&s.px),
                    Arg::Buf(&terms),Arg::pod(p)],n),
            (&astep,&[Arg::Buf(l),Arg::Buf(prev),Arg::Buf(&s.velocity),Arg::Buf(&s.py),Arg::Buf(&s.px),
                       Arg::Buf(&terms),Arg::Buf(&s.history),Arg::Buf(&extended_gradient),
                       Arg::Buf(&residual),Arg::Buf(&s.head),Arg::Buf(&s.links),Arg::pod(p),Arg::pod(&t)],n))?;
    }
    command.compute(&ext,&[Arg::Buf(&extended_gradient),Arg::Buf(&gradient),Arg::pod(p)],
                    p.ny as usize*p.nx as usize)?;
    command.finish()?;
    out.f32("objective",&[],&s.gpu.read::<f32>(&objective,1))?;
    out.f32("gradient",&[p.ny as usize,p.nx as usize],
            &s.gpu.read::<f32>(&gradient,p.ny as usize*p.nx as usize))
}

fn main() {
    outcrop_common::run(&[("run_frames",run_frames),("misfit_and_grad",misfit_and_grad)]);
}
