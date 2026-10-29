//! Fallible shared Metal buffers, local until outcrop-common exposes this API.
//!
//! The common crate exposes the retained device/buffer types, but not the
//! Objective-C traits or resource options. Use the Objective-C ABI locally so
//! allocation failure can be returned without adding dependencies or catching
//! the panic in the common allocator.
use outcrop_common::gpu::Buffer;
use outcrop_common::{Gpu, Result};
use std::ffi::{c_char, c_void};

#[link(name = "objc")]
extern "C" {
    fn sel_registerName(name: *const c_char) -> *mut c_void;
    fn objc_msgSend();
}

pub struct Buffers<'a> {
    gpu: &'a Gpu,
}

impl<'a> Buffers<'a> {
    pub fn new(gpu: &'a Gpu) -> Self {
        Self { gpu }
    }

    fn allocate<T>(&self, count: usize, name: &str) -> Result<(Buffer, *mut u8, usize)> {
        let bytes = count
            .checked_mul(std::mem::size_of::<T>())
            .ok_or_else(|| format!("{name}: Metal buffer byte size overflow"))?;
        // Match common's nonempty allocation for empty source/point lists.
        let length = bytes.max(4);
        let device = (&*self.gpu.device) as *const _ as *mut c_void;
        // SAFETY: these are the Objective-C signatures for MTLDevice's
        // maxBufferLength and newBufferWithLength:options:, and MTLBuffer's
        // contents. NSUInteger is usize on macOS; all returns are scalars or
        // pointers, so none require a structure-return calling convention.
        unsafe {
            let max_length: unsafe extern "C" fn(*mut c_void, *mut c_void) -> usize =
                std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
            let limit = max_length(device, sel_registerName(c"maxBufferLength".as_ptr()));
            if length > limit {
                return Err(format!(
                    "{name}: Metal buffer requires {length} bytes, device limit is {limit}"
                )
                .into());
            }
            let new_buffer: unsafe extern "C" fn(
                *mut c_void,
                *mut c_void,
                usize,
                usize,
            ) -> *mut c_void = std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
            // Zero options = default CPU cache + shared storage + default
            // hazard tracking, exactly common's StorageModeShared allocation.
            let raw = new_buffer(
                device,
                sel_registerName(c"newBufferWithLength:options:".as_ptr()),
                length,
                0,
            );
            // The new method family returns +1 ownership. Adopt it once;
            // normal Buffer drop releases it on success and all error paths.
            let buffer = Buffer::from_raw(raw.cast()).ok_or_else(|| {
                format!("{name}: Metal buffer allocation failed ({length} bytes)")
            })?;
            let contents: unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut u8 =
                std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
            let data = contents(raw, sel_registerName(c"contents".as_ptr()));
            if data.is_null() {
                return Err(format!("{name}: Metal shared buffer has no CPU mapping").into());
            }
            Ok((buffer, data, length))
        }
    }

    pub fn copied<T: Copy>(&self, data: &[T], name: &str) -> Result<Buffer> {
        let (buffer, destination, _) = self.allocate::<T>(data.len(), name)?;
        // SAFETY: allocation above checked the slice's byte length. The new
        // buffer cannot overlap the input and has no pending GPU commands.
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr().cast::<u8>(),
                destination,
                std::mem::size_of_val(data),
            );
        }
        Ok(buffer)
    }

    pub fn zeroed<T>(&self, count: usize, name: &str) -> Result<Buffer> {
        let (buffer, data, length) = self.allocate::<T>(count, name)?;
        // SAFETY: data maps this entire shared allocation, which has not been
        // submitted to the GPU. No typed Rust values are constructed here.
        unsafe { std::ptr::write_bytes(data, 0, length) };
        Ok(buffer)
    }
}
