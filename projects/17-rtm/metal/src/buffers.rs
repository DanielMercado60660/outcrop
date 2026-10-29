//! Fallible shared Metal allocations without additional dependencies.
//!
//! outcrop-common exposes device/buffer handles but neither the Metal traits nor
//! fallible allocation helpers. These three messages use the Objective-C C
//! runtime directly, then adopt the allocation into the common Buffer type.
use outcrop_common::gpu::Buffer;
use outcrop_common::{Gpu, Result};
use std::ffi::{c_char, c_void};

#[link(name = "objc")]
unsafe extern "C" {
    fn sel_registerName(name: *const c_char) -> *const c_void;
    fn objc_msgSend();
}

fn allocate<T>(gpu: &Gpu, n: usize) -> Result<Buffer> {
    let bytes = n.checked_mul(std::mem::size_of::<T>())
        .ok_or("Metal buffer byte size overflow")?.max(4);
    // ProtocolObject is an Objective-C object, not a Rust trait object.
    let device = std::ptr::from_ref(&*gpu.device).cast::<c_void>();
    // SAFETY: gpu owns a live MTLDevice. maxBufferLength takes no arguments
    // and returns NSUInteger (usize) on the supported macOS targets.
    let limit = unsafe {
        let send: unsafe extern "C" fn(*const c_void, *const c_void) -> usize =
            std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
        send(device, sel_registerName(c"maxBufferLength".as_ptr()))
    };
    if bytes > limit {
        return Err(format!("Metal buffer requires {bytes} bytes, exceeding device limit {limit}").into());
    }
    // SAFETY: The selector takes two NSUInteger arguments (length/options)
    // and returns a nullable, +1 retained id<MTLBuffer>. Options=0 selects
    // shared storage/default CPU cache/default hazard tracking, matching
    // outcrop-common. Buffer::from_raw adopts that +1 reference, including nil.
    let b = unsafe {
        let send: unsafe extern "C" fn(*const c_void, *const c_void, usize, usize) -> *mut c_void =
            std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
        let ptr = send(device, sel_registerName(c"newBufferWithLength:options:".as_ptr()), bytes, 0);
        Buffer::from_raw(ptr.cast())
    };
    b.ok_or_else(|| format!("failed to allocate {bytes}-byte Metal buffer").into())
}

pub fn from_slice<T: Copy>(gpu: &Gpu, data: &[T]) -> Result<Buffer> {
    let b = allocate::<T>(gpu, data.len())?;
    gpu.write(&b, data);
    Ok(b)
}

pub fn zeroed<T>(gpu: &Gpu, n: usize) -> Result<Buffer> {
    let b = allocate::<T>(gpu, n)?;
    // SAFETY: b owns a live shared MTLBuffer; contents returns its CPU
    // pointer. Its allocation length is the checked byte count above.
    unsafe {
        let send: unsafe extern "C" fn(*const c_void, *const c_void) -> *mut u8 =
            std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
        let ptr = send(std::ptr::from_ref(&*b).cast(), sel_registerName(c"contents".as_ptr()));
        if ptr.is_null() {
            return Err("shared Metal buffer has no CPU-accessible contents".into());
        }
        std::ptr::write_bytes(ptr, 0, (n * std::mem::size_of::<T>()).max(4));
    }
    Ok(b)
}
