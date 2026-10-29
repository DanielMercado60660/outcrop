use outcrop_common::{gpu::{Arg, Gpu}, Inputs, Outputs, Result};
#[repr(C)]
struct Params { nz:u32,nx:u32,shots:u32,nt:u32,ns:u32,nr:u32,radius:u32,time:u32,inv_h:f32,inv_h2:f32,dt2:f32 }
fn bytes<T>(v:&T)->&[u8] { unsafe { std::slice::from_raw_parts((v as *const T).cast(),std::mem::size_of::<T>()) } }
// Every dimension and flattened offset used in MSL must fit its uint indices.
// Check bytes as well, before either Vec or Metal buffer allocation.
fn checked_size(name: &str, factors: &[usize]) -> Result<usize> {
 let n=factors.iter().try_fold(1usize, |n,&factor| n.checked_mul(factor))
  .ok_or_else(||format!("{name}: size overflow"))?;
 if factors.iter().any(|&v| u32::try_from(v).is_err()) || u32::try_from(n).is_err() {
  return Err(format!("{name}: exceeds Metal u32 indexing").into());
 }
 if n.checked_mul(std::mem::size_of::<u32>()).is_none_or(|bytes|bytes>isize::MAX as usize) {
  return Err(format!("{name}: buffer size overflow").into());
 }
 Ok(n)
}
fn padded_size(n:usize,low:usize,high:usize,p:usize)->Result<usize> {
 let halo=p.checked_mul(2).ok_or("halo size overflow")?;
 let n=n.checked_add(low).and_then(|n|n.checked_add(high)).and_then(|n|n.checked_add(halo))
  .ok_or("padded dimension overflow")?;
 checked_size("padded dimension",&[n])
}
fn profile(n:usize,low:usize,high:usize,p:usize,h:f64,dt:f64,freq:f64,vmax:f64,c:&[f32])->Vec<f32> {
 let mut a=vec![0f32;n]; let mut b=vec![0f32;n];
 for i in 0..n {
  let ql=if low==0 {0.0} else {((p as f64+low as f64-i as f64)/low as f64).clamp(0.0,1.0)};
  let qh=if high==0 {0.0} else {((i as f64-(n-1-p-high) as f64)/high as f64).clamp(0.0,1.0)};
  let q=ql.max(qh);
  if q>0.0 {
   let sigma=3.0*vmax*1000f64.ln()/(2.0*h)*(if low==0 {0.0} else {ql*ql/low as f64}+if high==0 {0.0} else {qh*qh/high as f64});
   let sum=sigma+std::f64::consts::PI*freq*(1.0-q);
   a[i]=(-sum*dt).exp() as f32; b[i]=if sum==0.0 {0.0} else {(sigma/sum*(-sum*dt).exp_m1()) as f32};
  }
 }
 let mut db=vec![0f32;n];
 for i in p..n-p { let mut v=0f32; for j in 1..=p {v+=c[j-1]*(b[i+j]-b[i-j]);} db[i]=v/h as f32; }
 [a,b,db].concat()
}
fn scalar(inp:&Inputs,out:&mut Outputs)->Result<()> {
 let vel=inp.f32("velocity")?; let src=inp.f32("source")?;
 let sl=inp.i64("source_locations")?; let rl=inp.i64("receiver_locations")?;
 if vel.shape.len()!=2 || src.shape.len()!=3 || sl.shape.len()!=3 || rl.shape.len()!=3 {return Err("invalid array ranks".into());}
 let (mz,mx)=(vel.shape[0],vel.shape[1]); let (shots,ns,nt)=(src.shape[0],src.shape[1],src.shape[2]); let nr=rl.shape[1];
 if mz==0 || mx==0 || sl.shape!=[shots,ns,2] || rl.shape!=[shots,nr,2] {return Err("invalid shapes".into());}
 let (c,d):(&[f32],&[f32])=match inp.int("order")? {
  2=>(&[0.5],&[1.0]), 4=>(&[2.0/3.0,-1.0/12.0],&[4.0/3.0,-1.0/12.0]),
  6=>(&[3.0/4.0,-3.0/20.0,1.0/60.0],&[3.0/2.0,-3.0/20.0,1.0/90.0]),
  8=>(&[4.0/5.0,-1.0/5.0,4.0/105.0,-1.0/280.0],&[8.0/5.0,-1.0/5.0,8.0/315.0,-1.0/560.0]),
  _=>return Err("unsupported order".into()) };
 let p=c.len(); let widths=inp.nums("pml_width")?;
 if widths.len()!=4 || widths.iter().any(|v| !v.is_finite() || *v<0.0 || *v>u32::MAX as f64 || v.fract()!=0.0) {return Err("invalid pml_width".into());}
 let w:Vec<usize>=widths.iter().map(|&v|v as usize).collect();
 let h=inp.num("spacing")?; let dt=inp.num("dt")?; let freq=inp.num("pml_freq")?; let vmax=inp.num("max_vel")?;
 let nz=padded_size(mz,w[0],w[1],p)?; let nx=padded_size(mx,w[2],w[3],p)?;
 let cells=checked_size("padded grid",&[nz,nx])?;
 let total=checked_size("wavefields",&[cells,shots])?;
 let source_count=checked_size("source locations",&[shots,ns])?;
 let source_samples=checked_size("source samples",&[shots,ns,nt])?;
 let receiver_count=checked_size("receiver locations",&[shots,nr])?;
 let samples=checked_size("receiver samples",&[shots,nr,nt])?;
 checked_size("z profile",&[3,nz])?; checked_size("x profile",&[3,nx])?;
 if vel.data.len()!=checked_size("velocity",&[mz,mx])? || src.data.len()!=source_samples
  || Some(sl.data.len())!=source_count.checked_mul(2) || Some(rl.data.len())!=receiver_count.checked_mul(2) {
  return Err("array data does not match shape".into());
 }
 let locs=|a:&[i64]|->Result<Vec<u32>> {a.chunks_exact(2).map(|pair| {if pair[0]<0 || pair[1]<0 || pair[0] as usize>=mz || pair[1] as usize>=mx {return Err("location outside model".into());} Ok(((pair[0] as usize+p+w[0])*nx+pair[1] as usize+p+w[2]) as u32)}).collect()};
 let sloc=locs(&sl.data)?; let rloc=locs(&rl.data)?;
 // Stable grouping gives one writer per (shot, cell), with coincident
 // sources subtracted in exactly the original source order. No atomics.
 let mut sources:Vec<(u32,u32)>=sloc.iter().enumerate().map(|(s,&cell)| {
  (((s/ns)*cells+cell as usize) as u32,s as u32)
 }).collect();
 sources.sort_by_key(|&(cell,_)|cell);
 let mut injection_cells=Vec::new(); let mut injection_offsets=Vec::new();
 let mut source_indices=Vec::with_capacity(source_count);
 for (cell,s) in sources {
  if injection_cells.last()!=Some(&cell) {
   injection_cells.push(cell); injection_offsets.push(source_indices.len() as u32);
  }
  source_indices.push(s);
 }
 injection_offsets.push(source_indices.len() as u32);
 checked_size("injection offsets",&[injection_offsets.len()])?;
 if samples==0 {return out.f32("receiver_amplitudes",&[shots,nr,nt],&[]);}
 let mut v2=vec![0f32;cells]; for z in 0..nz {for x in 0..nx {let iz=z.saturating_sub(p+w[0]).min(mz-1); let ix=x.saturating_sub(p+w[2]).min(mx-1); let v=vel.data[iz*mx+ix]; v2[z*nx+x]=v*v;}}
 let profz=profile(nz,w[0],w[1],p,h,dt,freq,vmax,c); let profx=profile(nx,w[2],w[3],p,h,dt,freq,vmax,c);
 let mut coeff=vec![0f32;8]; coeff[..p].copy_from_slice(c); coeff[4..4+p].copy_from_slice(d);
 let gpu=Gpu::new()?; let lib=gpu.compile(include_str!("../kernels/scalar.metal"))?; let step=gpu.pipeline(&lib,"step")?; let record=gpu.pipeline(&lib,"record")?; let inject=gpu.pipeline(&lib,"inject")?;
 let bv=gpu.buffer_from(&v2); let bs=gpu.buffer_from(&src.data); let blr=gpu.buffer_from(&rloc); let bpz=gpu.buffer_from(&profz); let bpx=gpu.buffer_from(&profx); let bc=gpu.buffer_from(&coeff);
 let bic=gpu.buffer_from(&injection_cells); let bio=gpu.buffer_from(&injection_offsets); let bsi=gpu.buffer_from(&source_indices);
 let wave:Vec<_>=(0..3).map(|_|gpu.buffer_zeroed::<f32>(total)).collect(); let psz:Vec<_>=(0..2).map(|_|gpu.buffer_zeroed::<f32>(total)).collect(); let psx:Vec<_>=(0..2).map(|_|gpu.buffer_zeroed::<f32>(total)).collect();
 let zz=gpu.buffer_zeroed::<f32>(total); let zx=gpu.buffer_zeroed::<f32>(total); let result=gpu.buffer_zeroed::<f32>(samples);
 let mut params=Params{nz:nz as u32,nx:nx as u32,shots:shots as u32,nt:nt as u32,ns:ns as u32,nr:nr as u32,radius:p as u32,time:0,inv_h:(1.0/h) as f32,inv_h2:(1.0/(h*h)) as f32,dt2:(dt*dt) as f32};
 for start in (0..nt).step_by(32) {gpu.batch(|batch| {for t in start..start.saturating_add(32).min(nt) {
  params.time=t as u32; let cur=t%3; let prev=(t+2)%3; let next=(t+1)%3; let old=t%2; let new=1-old;
  batch.dispatch(&record,&[Arg::Buf(&wave[cur]),Arg::Buf(&blr),Arg::Buf(&result),Arg::Bytes(bytes(&params))],[receiver_count,1,1])?;
  batch.dispatch(&step,&[Arg::Buf(&wave[cur]),Arg::Buf(&wave[prev]),Arg::Buf(&wave[next]),Arg::Buf(&psz[old]),Arg::Buf(&psz[new]),Arg::Buf(&psx[old]),Arg::Buf(&psx[new]),Arg::Buf(&zz),Arg::Buf(&zx),Arg::Buf(&bv),Arg::Buf(&bpz),Arg::Buf(&bpx),Arg::Buf(&bc),Arg::Bytes(bytes(&params))],[total,1,1])?;
  if !injection_cells.is_empty() {
   batch.dispatch(&inject,&[Arg::Buf(&wave[next]),Arg::Buf(&bv),Arg::Buf(&bs),Arg::Buf(&bic),Arg::Buf(&bio),Arg::Buf(&bsi),Arg::Bytes(bytes(&params)),Arg::U32(injection_cells.len() as u32)],[injection_cells.len(),1,1])?;
  }
 } Ok(())})?;}
 out.f32("receiver_amplitudes",&[shots,nr,nt],&gpu.read(&result,samples))
}
fn main(){outcrop_common::run(&[("scalar",scalar)]);}
