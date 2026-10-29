//! Batched complex FFTs: radix two, with Bluestein for every other length.
//! Float64 host work only constructs O(n) phase tables; all transforms run on Metal.
use outcrop_common::{gpu::{Arg, Batch, Buffer, Library, Pipeline}, Gpu, Result};
use std::f64::consts::PI;

pub struct Kernels {
    reorder: Pipeline,
    stage: Pipeline,
    scale: Pipeline,
    prepare: Pipeline,
    product: Pipeline,
    finish: Pipeline,
}

impl Kernels {
    pub fn new(gpu: &Gpu, lib: &Library) -> Result<Self> {
        Ok(Self {
            reorder: gpu.pipeline(lib, "fft_reorder")?,
            stage: gpu.pipeline(lib, "fft_stage")?,
            scale: gpu.pipeline(lib, "fft_scale")?,
            prepare: gpu.pipeline(lib, "chirp_prepare")?,
            product: gpu.pipeline(lib, "convolution_product")?,
            finish: gpu.pipeline(lib, "chirp_finish")?,
        })
    }
}

struct RadixTwo {
    n: usize,
    twiddle: Buffer,
}

impl RadixTwo {
    fn new(gpu: &Gpu, n: usize) -> Self {
        let twiddle: Vec<[f32; 2]> = (0..n / 2).map(|j| {
            let angle = -2.0 * PI * j as f64 / n as f64;
            [angle.cos() as f32, angle.sin() as f32]
        }).collect();
        Self { n, twiddle: gpu.buffer_from(&twiddle) }
    }

    // The input and output must be distinct; stages then operate in-place.
    fn encode(&self, b: &mut Batch, k: &Kernels, input: &Buffer,
              output: &Buffer, rows: usize, inverse: bool) -> Result<()> {
        let n = self.n as u32;
        b.dispatch(&k.reorder, &[Arg::Buf(input), Arg::Buf(output),
            Arg::U32(n), Arg::U32(self.n.trailing_zeros())], [self.n, rows, 1])?;
        let mut half = 1;
        while half < self.n {
            b.dispatch(&k.stage, &[Arg::Buf(output), Arg::Buf(&self.twiddle),
                Arg::U32(n), Arg::U32(half as u32), Arg::U32(inverse as u32)],
                [self.n / 2, rows, 1])?;
            half *= 2;
        }
        if inverse {
            b.dispatch(&k.scale, &[Arg::Buf(output), Arg::F32(1.0 / self.n as f32)],
                [self.n * rows, 1, 1])?;
        }
        Ok(())
    }
}

struct Bluestein {
    chirp: Buffer,
    spectrum: Buffer,
    work_a: Buffer,
    work_b: Buffer,
}

pub struct Plan {
    n: usize,
    rows: usize,
    radix: RadixTwo,
    bluestein: Option<Bluestein>,
}

impl Plan {
    pub fn new(gpu: &Gpu, k: &Kernels, n: usize, rows: usize) -> Result<Self> {
        let m = if n.is_power_of_two() { n } else {
            n.checked_mul(2).and_then(|v| v.checked_sub(1))
                .and_then(usize::checked_next_power_of_two)
                .ok_or("FFT length overflow")?
        };
        if m.checked_mul(rows).filter(|&v| v <= u32::MAX as usize).is_none() {
            return Err("FFT workspace exceeds Metal indexing limits".into());
        }
        let radix = RadixTwo::new(gpu, m);
        let bluestein = if m == n { None } else {
            // Reduce the integer square before evaluating trig; no large
            // float32 arguments, including for long prime-length traces.
            let chirp: Vec<[f32; 2]> = (0..n).map(|j| {
                let square = (j as u64 * j as u64) % (2 * n as u64);
                let angle = -PI * square as f64 / n as f64;
                [angle.cos() as f32, angle.sin() as f32]
            }).collect();
            let mut convolution = vec![[0.0f32; 2]; m];
            for j in 0..n {
                let v = [chirp[j][0], -chirp[j][1]];
                convolution[j] = v;
                if j != 0 { convolution[m - j] = v; }
            }
            let convolution = gpu.buffer_from(&convolution);
            let spectrum = gpu.buffer_zeroed::<[f32; 2]>(m);
            gpu.batch(|b| radix.encode(b, k, &convolution, &spectrum, 1, false))?;
            Some(Bluestein {
                chirp: gpu.buffer_from(&chirp), spectrum,
                work_a: gpu.buffer_zeroed::<[f32; 2]>(m * rows),
                work_b: gpu.buffer_zeroed::<[f32; 2]>(m * rows),
            })
        };
        Ok(Self { n, rows, radix, bluestein })
    }

    pub fn encode(&self, b: &mut Batch, k: &Kernels, input: &Buffer,
                  output: &Buffer, inverse: bool) -> Result<()> {
        let Some(blue) = &self.bluestein else {
            return self.radix.encode(b, k, input, output, self.rows, inverse);
        };
        let n = self.n as u32;
        let m = self.radix.n as u32;
        b.dispatch(&k.prepare, &[Arg::Buf(input), Arg::Buf(&blue.chirp),
            Arg::Buf(&blue.work_a), Arg::U32(n), Arg::U32(m), Arg::U32(inverse as u32)],
            [m as usize, self.rows, 1])?;
        self.radix.encode(b, k, &blue.work_a, &blue.work_b, self.rows, false)?;
        b.dispatch(&k.product, &[Arg::Buf(&blue.work_b), Arg::Buf(&blue.spectrum),
            Arg::U32(m)], [m as usize, self.rows, 1])?;
        self.radix.encode(b, k, &blue.work_b, &blue.work_a, self.rows, true)?;
        b.dispatch(&k.finish, &[Arg::Buf(&blue.work_a), Arg::Buf(&blue.chirp),
            Arg::Buf(output), Arg::U32(n), Arg::U32(m), Arg::U32(inverse as u32)],
            [self.n, self.rows, 1])
    }
}
