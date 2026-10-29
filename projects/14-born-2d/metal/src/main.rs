use outcrop_common::{gpu::{Arg,Gpu},Inputs,Outputs,Result};
#[repr(C)]
struct P {ny:u32,nx:u32,shots:u32,nt:u32,ns:u32,nr:u32,r:u32,t:u32,periodic:u32,save_history:u32,ih:f32,ih2:f32,dt2:f32}
fn bytes<T>(v:&T)->&[u8]{unsafe{std::slice::from_raw_parts((v as *const T).cast(),std::mem::size_of::<T>())}}
// Every shader uses uint element offsets. Check products before either host
// allocation or GPU dispatch; also cover the byte sizes of host f64 reductions.
fn checked_len(name: &str, factors: &[usize]) -> Result<usize> {
    let n = factors.iter().try_fold(1usize, |n, &factor| n.checked_mul(factor))
        .ok_or_else(|| format!("{name}: element count overflow"))?;
    if n > u32::MAX as usize {
        return Err(format!("{name}: exceeds Metal uint indexing range").into());
    }
    let bytes = n.checked_mul(std::mem::size_of::<f64>())
        .ok_or_else(|| format!("{name}: byte size overflow"))?;
    if bytes > isize::MAX as usize {
        return Err(format!("{name}: exceeds allocation size range").into());
    }
    Ok(n)
}
fn profile(n:usize,w:usize,r:usize,h:f64,dt:f64,freq:f64,vmax:f64)->Vec<f32>{
 let mut a=vec![0f32;n];let mut b=vec![0f32;n];
 if w>0 {for i in 0..n {let f=(((r+w) as f64-i as f64)/w as f64).clamp(0.,1.).max(((i as f64-(n-1-r-w) as f64)/w as f64).clamp(0.,1.));if f>0. {let sig=3.*vmax*1000f64.ln()/(2.*w as f64*h)*f*f;let sum=sig+std::f64::consts::PI*freq*(1.-f);a[i]=(-sum*dt).exp() as f32;b[i]=if sum==0. {0.}else{(sig/sum*(-sum*dt).exp_m1()) as f32};}}}
 let mut db=vec![0f32;n];if w>0 {for i in r..n-r {for j in 1..=r {let c=if r==1 {0.5} else if j==1 {2./3.} else {-1./12.};db[i]+=c*(b[i+j]-b[i-j])/h as f32;}}} [a,b,db].concat()
}
fn calculate(inp:&Inputs,out:&mut Outputs,gradient:bool)->Result<()> {
 let v=inp.f32("v")?;let dv=inp.f32("dv")?;let source=inp.f32("source_amplitudes")?;let sl=inp.i64("source_locations")?;let rl=inp.i64("receiver_locations")?;
 if v.shape.len()!=2||source.shape.len()!=3||rl.shape.len()!=3{return Err("invalid dimensions".into())} let (my,mx)=(v.shape[0],v.shape[1]);let (shots,ns,nt)=(source.shape[0],source.shape[1],source.shape[2]);let nr=rl.shape[1];
 if my==0||mx==0||dv.shape!=v.shape||sl.shape!=[shots,ns,2]||rl.shape!=[shots,nr,2]{return Err("invalid shapes".into())}
 let r=match inp.int("accuracy")?{2=>1,4=>2,_=>return Err("invalid accuracy".into())};let boundary=inp.str("boundary")?;let periodic=boundary=="periodic";if !periodic&&boundary!="pml"{return Err("invalid boundary".into())}
 let w=usize::try_from(inp.int("pml_width")?)
     .map_err(|_| "pml_width must be nonnegative and fit usize")?;
 let off=if periodic {0} else {w.checked_add(r).ok_or("PML padding overflow")?};
 let padding=off.checked_mul(2).ok_or("PML padding overflow")?;
 let ny=my.checked_add(padding).ok_or("padded ny overflow")?;
 let nx=mx.checked_add(padding).ok_or("padded nx overflow")?;
 // at() uses signed coordinates, including addition during periodic wrapping.
 if ny > (i32::MAX as usize)/2 || nx > (i32::MAX as usize)/2 {
     return Err("padded dimensions exceed Metal signed coordinate range".into());
 }
 for (name,n) in [("shots",shots),("sources",ns),("receivers",nr),("nt",nt)] {
     checked_len(name,&[n])?;
 }
 let model_cells=checked_len("model grid",&[my,mx])?;
 let cells=checked_len("padded grid",&[ny,nx])?;
 let total=checked_len("shot wavefields",&[cells,shots])?;
 let memory_len=checked_len("PML memory",&[8,total])?;
 let traces=checked_len("receiver locations",&[shots,nr])?;
 let data_len=checked_len("receiver data",&[traces,nt])?;
 checked_len("source amplitudes",&[shots,ns,nt])?;
 checked_len("source locations",&[shots,ns])?;
 checked_len("y profile",&[3,ny])?;
 checked_len("x profile",&[3,nx])?;
 // Forward modeling has no history allocation or history-size restriction.
 let history_len=if gradient {Some(checked_len("background history",&[nt,total])?)} else {None};
 let adj_memory_len=if gradient {checked_len("adjoint memory",&[4,total])?} else {0};
 let temp_len=if gradient {checked_len("adjoint temporary",&[6,total])?} else {0};
 let h=inp.num("dx")?;let dt=inp.num("dt")?;let freq=inp.num("pml_freq")?;let vmax=inp.num("max_vel")?;
 let mut vp=vec![0.;cells];let mut dp=vec![0.;cells];for y in 0..ny{for x in 0..nx{let j=y.saturating_sub(off).min(my-1)*mx+x.saturating_sub(off).min(mx-1);vp[y*nx+x]=v.data[j];dp[y*nx+x]=dv.data[j];}}
 let locs=|loc:&[i64]|->Result<Vec<u32>>{loc.chunks_exact(2).map(|a|{if a[0]<0||a[1]<0||a[0] as usize>=my||a[1] as usize>=mx{return Err("location out of bounds".into())}Ok(((a[0] as usize+off)*nx+a[1] as usize+off) as u32)}).collect()};let ls=locs(&sl.data)?;let lr=locs(&rl.data)?;
 let gpu=Gpu::new()?;let lib=gpu.compile(include_str!("../kernels/born.metal"))?;let step=gpu.pipeline(&lib,"step")?;let record=gpu.pipeline(&lib,"record")?;
 let bv=gpu.buffer_from(&vp);let bd=gpu.buffer_from(&dp);let by=gpu.buffer_from(&profile(ny,if periodic{0}else{w},r,h,dt,freq,vmax));let bx=gpu.buffer_from(&profile(nx,if periodic{0}else{w},r,h,dt,freq,vmax));let bs=gpu.buffer_from(&source.data);let bls=gpu.buffer_from(&ls);let blr=gpu.buffer_from(&lr);
 let u:Vec<_>=(0..3).map(|_|gpu.buffer_zeroed::<f32>(total)).collect();let s:Vec<_>=(0..3).map(|_|gpu.buffer_zeroed::<f32>(total)).collect();
 for (name,buf) in [("wavefield_0",&u[0]),("wavefield_m1",&u[2]),("wavefield_sc_0",&s[0]),("wavefield_sc_m1",&s[2])] {let a=inp.f32(name)?;if a.shape!=[shots,my,mx]{return Err("invalid wavefield shape".into())}let mut padded=vec![0.;total];for shot in 0..shots{for y in 0..my{for x in 0..mx{padded[shot*cells+(y+off)*nx+x+off]=a.data[(shot*my+y)*mx+x];}}}gpu.write(buf,&padded);}
 let mem:Vec<_>=(0..2).map(|_|gpu.buffer_zeroed::<f32>(memory_len)).collect();let hist=history_len.map(|n|gpu.buffer_zeroed::<f32>(n));let bu=gpu.buffer_zeroed::<f32>(data_len);let bout=gpu.buffer_zeroed::<f32>(data_len);
 let mut p=P{ny:ny as u32,nx:nx as u32,shots:shots as u32,nt:nt as u32,ns:ns as u32,nr:nr as u32,r:r as u32,t:0,periodic:periodic as u32,save_history:gradient as u32,ih:(1./h) as f32,ih2:(1./(h*h)) as f32,dt2:(dt*dt) as f32};
 // With save_history=0 the kernel never accesses this binding, so an existing
 // buffer supplies the required argument without allocating a dummy history.
 let history_binding=hist.as_ref().unwrap_or(&bv);
 for start in (0..nt).step_by(32){gpu.batch(|batch|{for t in start..(start+32).min(nt){p.t=t as u32;let cur=t%3;let prev=(t+2)%3;let next=(t+1)%3;batch.dispatch(&record,&[Arg::Buf(&u[cur]),Arg::Buf(&s[cur]),Arg::Buf(&blr),Arg::Buf(&bu),Arg::Buf(&bout),Arg::Bytes(bytes(&p))],[traces,1,1])?;batch.dispatch(&step,&[Arg::Buf(&u[cur]),Arg::Buf(&u[prev]),Arg::Buf(&u[next]),Arg::Buf(&s[cur]),Arg::Buf(&s[prev]),Arg::Buf(&s[next]),Arg::Buf(&mem[t%2]),Arg::Buf(&mem[1-t%2]),Arg::Buf(&bv),Arg::Buf(&bd),Arg::Buf(&by),Arg::Buf(&bx),Arg::Buf(&bs),Arg::Buf(&bls),Arg::Buf(history_binding),Arg::Bytes(bytes(&p))],[total,1,1])?;}Ok(())})?;}
 let data:Vec<f32>=gpu.read(&bout,data_len);if !gradient {out.f32("background",&[shots,nr,nt],&gpu.read(&bu,data_len))?;return out.f32("scattered",&[shots,nr,nt],&data)}
 let hist=hist.as_ref().ok_or("missing gradient history")?;
 let obs=inp.f32("observed")?;if obs.shape!=[shots,nr,nt]{return Err("invalid observed shape".into())}let objective=data.iter().zip(&obs.data).map(|(&a,&b)|{let d=(a as f64)-(b as f64);0.5*d*d}).sum::<f64>() as f32;let bo=gpu.buffer_from(&obs.data);
 let prep=gpu.pipeline(&lib,"adjprep")?;let adj=gpu.pipeline(&lib,"adjstep")?;let wave:Vec<_>=(0..3).map(|_|gpu.buffer_zeroed::<f32>(total)).collect();let am:Vec<_>=(0..2).map(|_|gpu.buffer_zeroed::<f32>(adj_memory_len)).collect();let tmp=gpu.buffer_zeroed::<f32>(temp_len);let grad=gpu.buffer_zeroed::<f32>(total);
 for start in (0..nt).step_by(32){gpu.batch(|batch|{for k in start..(start+32).min(nt){let t=nt-1-k;p.t=t as u32;batch.dispatch(&prep,&[Arg::Buf(&wave[k%3]),Arg::Buf(&am[k%2]),Arg::Buf(&tmp),Arg::Buf(&bv),Arg::Buf(&by),Arg::Buf(&bx),Arg::Bytes(bytes(&p))],[total,1,1])?;batch.dispatch(&adj,&[Arg::Buf(&wave[k%3]),Arg::Buf(&wave[(k+2)%3]),Arg::Buf(&wave[(k+1)%3]),Arg::Buf(&am[k%2]),Arg::Buf(&am[1-k%2]),Arg::Buf(&tmp),Arg::Buf(&bv),Arg::Buf(&by),Arg::Buf(&bx),Arg::Buf(hist),Arg::Buf(&grad),Arg::Buf(&bout),Arg::Buf(&bo),Arg::Buf(&blr),Arg::Bytes(bytes(&p))],[total,1,1])?;}Ok(())})?;}
 let padded:Vec<f32>=gpu.read(&grad,total);let mut result=vec![0f64;model_cells];for shot in 0..shots{for y in 0..ny{for x in 0..nx{let j=y.saturating_sub(off).min(my-1)*mx+x.saturating_sub(off).min(mx-1);result[j]+=padded[shot*cells+y*nx+x] as f64;}}}let result:Vec<f32>=result.into_iter().map(|x|x as f32).collect();out.f32("objective",&[],&[objective])?;out.f32("gradient",&[my,mx],&result)
}
fn born(i:&Inputs,o:&mut Outputs)->Result<()>{calculate(i,o,false)}
fn misfit(i:&Inputs,o:&mut Outputs)->Result<()>{calculate(i,o,true)}
fn main(){outcrop_common::run(&[("born",born),("misfit_and_grad",misfit)]);}
