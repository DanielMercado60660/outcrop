//! A small Metal compute context on objc2-metal.
//!
//! ```ignore
//! let gpu = Gpu::new()?;
//! let lib = gpu.compile(include_str!("kernels.metal"))?;
//! let k = gpu.pipeline(&lib, "add")?;
//! let a = gpu.buffer_from(&xs);            // shared-storage buffer holding a copy of xs
//! let out = gpu.buffer_zeroed::<f32>(n);
//! gpu.dispatch(&k, &[Arg::Buf(&a), Arg::Buf(&out), Arg::U32(n as u32)], [n, 1, 1])?;
//! let result: Vec<f32> = gpu.read(&out, n);
//! ```
//! Dispatch waits for completion. For many kernel launches per step (time stepping), use
//! [`Gpu::batch`] to encode several dispatches into one command buffer.

use crate::Result;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary, MTLResourceOptions,
    MTLSize,
};
use std::ffi::c_void;
use std::ptr::NonNull;

pub type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
pub type Library = Retained<ProtocolObject<dyn MTLLibrary>>;
pub type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

/// A kernel argument, bound in order to [[buffer(0)]], [[buffer(1)]], ...
pub enum Arg<'a> {
    Buf(&'a Buffer),
    F32(f32),
    U32(u32),
    I32(i32),
    /// Arbitrary plain-old-data bytes (e.g. a #[repr(C)] params struct matching an MSL struct).
    Bytes(&'a [u8]),
}

pub struct Gpu {
    pub device: Retained<ProtocolObject<dyn MTLDevice>>,
    pub queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
}

impl Gpu {
    pub fn new() -> Result<Self> {
        let device = MTLCreateSystemDefaultDevice().ok_or("no Metal device")?;
        let queue = device.newCommandQueue().ok_or("could not create command queue")?;
        Ok(Gpu { device, queue })
    }

    /// Compile MSL source at runtime with fast math OFF (IEEE-conforming math functions and no
    /// reassociation), so float32 kernels can meet the project-wide parity tolerances.
    pub fn compile(&self, source: &str) -> Result<Library> {
        self.compile_with(source, false)
    }

    /// Compile MSL source with fast math explicitly on or off. Parity-tested kernels use `compile`.
    pub fn compile_with(&self, source: &str, fast_math: bool) -> Result<Library> {
        let src = NSString::from_str(source);
        let opts = objc2_metal::MTLCompileOptions::new();
        #[allow(deprecated)]
        opts.setFastMathEnabled(fast_math);
        self.device
            .newLibraryWithSource_options_error(&src, Some(&opts))
            .map_err(|e| format!("MSL compile error: {}", e.localizedDescription()).into())
    }

    pub fn pipeline(&self, lib: &Library, function: &str) -> Result<Pipeline> {
        let f = lib
            .newFunctionWithName(&NSString::from_str(function))
            .ok_or_else(|| format!("kernel {function:?} not found"))?;
        self.device
            .newComputePipelineStateWithFunction_error(&f)
            .map_err(|e| format!("pipeline {function}: {}", e.localizedDescription()).into())
    }

    /// A shared-storage buffer holding a copy of `data`.
    pub fn buffer_from<T: Copy>(&self, data: &[T]) -> Buffer {
        let len = std::mem::size_of_val(data).max(4);
        if data.is_empty() {
            return self.buffer_bytes(len);
        }
        let ptr = NonNull::new(data.as_ptr() as *mut c_void).unwrap();
        unsafe {
            self.device
                .newBufferWithBytes_length_options(ptr, len, MTLResourceOptions::StorageModeShared)
                .expect("buffer allocation failed")
        }
    }

    /// A zero-filled shared-storage buffer for `n` elements of `T`.
    pub fn buffer_zeroed<T>(&self, n: usize) -> Buffer {
        let b = self.buffer_bytes((n * std::mem::size_of::<T>()).max(4));
        unsafe { std::ptr::write_bytes(b.contents().as_ptr() as *mut u8, 0, b.length()) };
        b
    }

    fn buffer_bytes(&self, len: usize) -> Buffer {
        self.device
            .newBufferWithLength_options(len, MTLResourceOptions::StorageModeShared)
            .expect("buffer allocation failed")
    }

    /// Copy the first `n` elements of a buffer back to the host.
    pub fn read<T: Copy>(&self, buf: &Buffer, n: usize) -> Vec<T> {
        assert!(n * std::mem::size_of::<T>() <= buf.length(), "read past end of buffer");
        let p = buf.contents().as_ptr() as *const T;
        unsafe { std::slice::from_raw_parts(p, n).to_vec() }
    }

    /// Overwrite the start of a buffer with `data`.
    pub fn write<T: Copy>(&self, buf: &Buffer, data: &[T]) {
        assert!(std::mem::size_of_val(data) <= buf.length(), "write past end of buffer");
        let p = buf.contents().as_ptr() as *mut T;
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), p, data.len()) };
    }

    /// Run one kernel over a grid of `grid` threads and wait for it.
    pub fn dispatch(&self, k: &Pipeline, args: &[Arg], grid: [usize; 3]) -> Result<()> {
        self.batch(|b| b.dispatch(k, args, grid))
    }

    /// Encode several dispatches into one command buffer, then commit and wait.
    pub fn batch(&self, f: impl FnOnce(&mut Batch) -> Result<()>) -> Result<()> {
        let cb = self.queue.commandBuffer().ok_or("no command buffer")?;
        let enc = cb.computeCommandEncoder().ok_or("no compute encoder")?;
        let mut b = Batch { enc };
        f(&mut b)?;
        b.enc.endEncoding();
        cb.commit();
        cb.waitUntilCompleted();
        if let Some(err) = cb.error() {
            return Err(format!("Metal command buffer error: {}", err.localizedDescription()).into());
        }
        Ok(())
    }
}

/// Dispatches encoded into a single command buffer (executed in order).
pub struct Batch {
    enc: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
}

impl Batch {
    pub fn dispatch(&mut self, k: &Pipeline, args: &[Arg], grid: [usize; 3]) -> Result<()> {
        self.enc.setComputePipelineState(k);
        for (i, a) in args.iter().enumerate() {
            unsafe {
                match a {
                    Arg::Buf(b) => self.enc.setBuffer_offset_atIndex(Some(b), 0, i),
                    Arg::F32(v) => self.set_bytes(std::slice::from_raw_parts(v as *const f32 as *const u8, 4), i),
                    Arg::U32(v) => self.set_bytes(std::slice::from_raw_parts(v as *const u32 as *const u8, 4), i),
                    Arg::I32(v) => self.set_bytes(std::slice::from_raw_parts(v as *const i32 as *const u8, 4), i),
                    Arg::Bytes(bytes) => self.set_bytes(bytes, i),
                }
            }
        }
        let w = k.threadExecutionWidth();
        let maxt = k.maxTotalThreadsPerThreadgroup();
        let tg = match (grid[1] > 1, grid[2] > 1) {
            (false, false) => MTLSize { width: maxt.min(256), height: 1, depth: 1 },
            (true, false) => MTLSize { width: w, height: (maxt / w).min(8).max(1), depth: 1 },
            _ => MTLSize { width: w, height: (maxt / w / 4).clamp(1, 4), depth: 4.min(maxt / w).max(1) },
        };
        let g = MTLSize { width: grid[0].max(1), height: grid[1].max(1), depth: grid[2].max(1) };
        self.enc.dispatchThreads_threadsPerThreadgroup(g, tg);
        Ok(())
    }

    unsafe fn set_bytes(&self, bytes: &[u8], index: usize) {
        let p = NonNull::new(bytes.as_ptr() as *mut c_void).unwrap();
        self.enc.setBytes_length_atIndex(p, bytes.len(), index);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn saxpy_on_gpu() {
        let gpu = Gpu::new().unwrap();
        let lib = gpu
            .compile(
                "#include <metal_stdlib>\nusing namespace metal;\n\
                 kernel void saxpy(device const float* x [[buffer(0)]], device float* y [[buffer(1)]],\n\
                 constant float& a [[buffer(2)]], constant uint& n [[buffer(3)]], uint i [[thread_position_in_grid]]) {\n\
                 if (i < n) y[i] = a * x[i] + y[i]; }",
            )
            .unwrap();
        let k = gpu.pipeline(&lib, "saxpy").unwrap();
        let n = 1000usize;
        let x: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let bx = gpu.buffer_from(&x);
        let by = gpu.buffer_from(&vec![1.0f32; n]);
        gpu.dispatch(&k, &[Arg::Buf(&bx), Arg::Buf(&by), Arg::F32(2.0), Arg::U32(n as u32)], [n, 1, 1]).unwrap();
        let y: Vec<f32> = gpu.read(&by, n);
        assert_eq!(y[0], 1.0);
        assert_eq!(y[999], 1999.0);
    }
}
