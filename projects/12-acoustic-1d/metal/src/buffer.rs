//! Fallible shared Metal allocation without adding dependencies or changing common/.
//!
//! outcrop-common exposes the retained device and Buffer, but not the Objective-C
//! traits needed to call the allocator. Keep the two runtime calls local here.

use outcrop_common::gpu::Buffer;
use outcrop_common::{Gpu, Result};
use std::ffi::{c_char, c_void};

#[link(name = "objc")]
extern "C" {
    fn sel_registerName(name: *const c_char) -> *const c_void;
    fn objc_msgSend();
}

/// Allocate shared storage for f32 elements. The caller initializes it before use.
pub fn allocate_f32(gpu: &Gpu, count: usize, name: &str) -> Result<Buffer> {
    let bytes = count
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| format!("{name}: Metal buffer byte count overflow"))?
        .max(4);
    let device = std::ptr::from_ref(&*gpu.device).cast::<c_void>();

    // SAFETY: gpu keeps this MTLDevice alive. maxBufferLength returns NSUInteger
    // (usize on macOS). The selector is a static NUL-terminated string, and the
    // typed message-send signature matches the Metal SDK declaration exactly.
    let limit = unsafe {
        let send: unsafe extern "C" fn(*const c_void, *const c_void) -> usize =
            std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
        send(device, sel_registerName(c"maxBufferLength".as_ptr()))
    };
    if bytes > limit {
        return Err(format!(
            "{name}: requested {bytes} bytes exceeds Metal maxBufferLength ({limit} bytes)"
        )
        .into());
    }

    // SAFETY: this signature matches newBufferWithLength:options:. Resource
    // options 0 selects Shared storage, DefaultCache, and default hazard tracking,
    // exactly as in outcrop-common. The new-method family returns a +1 retained
    // MTLBuffer (or nil); Buffer::from_raw takes ownership without another retain.
    let buffer = unsafe {
        let send: unsafe extern "C" fn(
            *const c_void,
            *const c_void,
            usize,
            usize,
        ) -> *mut c_void = std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
        let raw = send(
            device,
            sel_registerName(c"newBufferWithLength:options:".as_ptr()),
            bytes,
            0,
        );
        Buffer::from_raw(raw.cast())
    };
    buffer.ok_or_else(|| format!("{name}: Metal buffer allocation failed ({bytes} bytes)").into())
}
