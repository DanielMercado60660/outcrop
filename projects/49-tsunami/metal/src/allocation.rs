//! Fallible Metal allocation without adding dependencies or changing common.
//!
//! outcrop-common exposes the device and Buffer type but not the Objective-C traits
//! needed to call their allocation methods. This small bridge uses the system
//! Objective-C runtime; numerical work still uses the shared GPU dispatcher.

use outcrop_common::{gpu::Buffer, Gpu, Result};
use std::ffi::{c_char, c_void, CStr};

#[link(name = "objc")]
extern "C" {
    fn sel_registerName(name: *const c_char) -> *const c_void;
    fn objc_msgSend();
}

// SAFETY requirements: object must implement a no-argument selector returning
// NSUInteger or uint64_t. Both are 64-bit on the supported macOS Metal targets.
unsafe fn size_property(object: *const c_void, selector: &CStr) -> usize {
    let send: unsafe extern "C" fn(*const c_void, *const c_void) -> usize =
        unsafe { std::mem::transmute(objc_msgSend as unsafe extern "C" fn()) };
    unsafe { send(object, sel_registerName(selector.as_ptr())) }
}

pub fn working_set_budget(gpu: &Gpu) -> usize {
    let device = std::ptr::from_ref(&*gpu.device).cast();
    // SAFETY: these MTLDevice properties have the signatures described above;
    // the retained device remains alive throughout both calls.
    unsafe {
        size_property(device, c"recommendedMaxWorkingSetSize")
            .saturating_sub(size_property(device, c"currentAllocatedSize"))
    }
}

pub fn allocated_bytes(buffer: &Buffer) -> usize {
    // SAFETY: MTLBuffer inherits MTLResource's NSUInteger allocatedSize getter.
    unsafe { size_property(Buffer::as_ptr(buffer).cast(), c"allocatedSize") }
}

// The caller must initialize the buffer before reading it. All forward,
// adjoint and reduction kernels write their complete active output range.
pub fn try_buffer(gpu: &Gpu, elements: usize) -> Result<Buffer> {
    let bytes = elements.checked_mul(std::mem::size_of::<f32>())
        .ok_or("state buffer size overflows addressable memory")?.max(4);
    let device = std::ptr::from_ref(&*gpu.device).cast();
    // SAFETY: maxBufferLength is an NSUInteger property on the live MTLDevice.
    if bytes > unsafe { size_property(device, c"maxBufferLength") } {
        return Err("state buffer exceeds the Metal device's maximum buffer length".into());
    }
    // SAFETY: this is the exact ABI of newBufferWithLength:options:. Zero
    // options mean shared storage, default caching and default hazard tracking,
    // matching outcrop-common. A successful `new` returns an owned (+1) MTLBuffer;
    // Buffer::from_raw adopts it, or returns None for allocation failure.
    unsafe {
        let send: unsafe extern "C" fn(*const c_void, *const c_void, usize, usize) -> *mut c_void =
            std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
        let raw = send(device, sel_registerName(c"newBufferWithLength:options:".as_ptr()), bytes, 0);
        Buffer::from_raw(raw.cast()).ok_or_else(|| "Metal could not allocate a state buffer".into())
    }
}
