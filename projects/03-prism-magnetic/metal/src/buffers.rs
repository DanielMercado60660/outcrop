//! Fallible shared-buffer allocation without adding an Objective-C dependency.
//!
//! `outcrop-common` exposes the owned Buffer type and device, but its allocation
//! helpers panic and it does not re-export the Metal traits. These two runtime
//! messages provide the local fallible equivalent; all other GPU operations
//! continue to use the common API.

use outcrop_common::{gpu::Buffer, Gpu, Result};
use std::ffi::{c_char, c_void};

#[link(name = "objc")]
extern "C" {
    fn sel_registerName(name: *const c_char) -> *const c_void;
    fn objc_msgSend();
}

/// Allocate storage for `count` float32 values. Contents must be written before
/// they are read: input buffers are copied below, and the kernel writes every
/// output element before the synchronized readback.
pub fn allocate(gpu: &Gpu, count: usize, name: &str) -> Result<Buffer> {
    let bytes = count
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| format!("Metal buffer {name:?} size overflows usize"))?
        .max(4);
    let device = std::ptr::from_ref(&*gpu.device).cast::<c_void>();

    // SAFETY: gpu.device is a live MTLDevice. Both selectors are standard
    // MTLDevice methods; their Objective-C ABI uses pointer receivers/selectors,
    // NSUInteger (usize) lengths/options, and an object pointer for the buffer.
    // Each objc_msgSend cast matches the selected method exactly. The selector
    // names are static NUL-terminated strings. Options = 0 is StorageModeShared,
    // CPUCacheModeDefaultCache, and HazardTrackingModeDefault, as in common.
    unsafe {
        let max_length: unsafe extern "C" fn(*const c_void, *const c_void) -> usize =
            std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
        let limit = max_length(device, sel_registerName(c"maxBufferLength".as_ptr()));
        if bytes > limit {
            return Err(format!(
                "Metal buffer {name:?} needs {bytes} bytes, exceeding device limit {limit}"
            )
            .into());
        }
        let new_buffer: unsafe extern "C" fn(
            *const c_void,
            *const c_void,
            usize,
            usize,
        ) -> *mut c_void = std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
        let raw = new_buffer(
            device,
            sel_registerName(c"newBufferWithLength:options:".as_ptr()),
            bytes,
            0,
        );
        // The `new` method returns an owned (+1) MTLBuffer or nil. Adopt that
        // ownership once; Buffer drops it normally, including on later errors.
        Buffer::from_raw(raw.cast()).ok_or_else(|| {
            format!("failed to allocate Metal buffer {name:?} ({bytes} bytes)").into()
        })
    }
}

pub fn from_slice(gpu: &Gpu, values: &[f32], name: &str) -> Result<Buffer> {
    let buffer = allocate(gpu, values.len(), name)?;
    // allocate checked the byte size, so common's write bound is satisfied.
    gpu.write(&buffer, values);
    Ok(buffer)
}
