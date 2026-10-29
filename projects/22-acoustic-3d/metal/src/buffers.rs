//! Fallible allocation without the frozen shared helpers' allocation `expect`.
use outcrop_common::gpu::Buffer;
use outcrop_common::{Gpu, Result};
use std::ffi::{c_char, c_void, CStr};

// common exposes the retained device/buffer objects but not their objc2 traits.
// Use the public Objective-C ABI locally, keeping the std + common dependency
// rule. All messages below have pointer/integer arguments and scalar returns,
// so objc_msgSend is the correct entry point on both Apple architectures.
#[link(name = "objc")]
extern "C" {
    fn sel_registerName(name: *const c_char) -> *const c_void;
    fn objc_msgSend();
}

unsafe fn size_property(object: *const c_void, name: &CStr) -> usize {
    let send: unsafe extern "C" fn(*const c_void, *const c_void) -> usize =
        std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
    send(object, sel_registerName(name.as_ptr()))
}

fn contents(buffer: &Buffer) -> Result<*mut f32> {
    // SAFETY: buffer is a live MTLBuffer allocated with shared storage. contents
    // returns its CPU-accessible allocation, valid while buffer is retained.
    unsafe {
        let send: unsafe extern "C" fn(*const c_void, *const c_void) -> *mut f32 =
            std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
        let pointer = send(
            std::ptr::from_ref(&**buffer).cast(),
            sel_registerName(c"contents".as_ptr()),
        );
        if pointer.is_null() {
            return Err("Metal shared buffer has no CPU-accessible storage".into());
        }
        Ok(pointer)
    }
}

pub fn allocate(gpu: &Gpu, count: usize) -> Result<Buffer> {
    let bytes = count.checked_mul(std::mem::size_of::<f32>()).ok_or("buffer size overflow")?;
    let device = std::ptr::from_ref(&*gpu.device).cast::<c_void>();
    // SAFETY: gpu retains a live MTLDevice; maxBufferLength returns NSUInteger.
    let maximum = unsafe { size_property(device, c"maxBufferLength") };
    if bytes == 0 || bytes > maximum {
        return Err(format!("Metal buffer size {bytes} exceeds the device limit").into());
    }
    // StorageModeShared is the default (zero) resource option. No initialization
    // is needed: each time step writes every element of the destination buffer.
    // SAFETY: newBufferWithLength:options: takes NSUInteger and MTLResourceOptions
    // (both usize). The new-family selector returns a +1 retained MTLBuffer or
    // nil, which Buffer::from_raw adopts without an extra retain.
    unsafe {
        let send: unsafe extern "C" fn(*const c_void, *const c_void, usize, usize) -> *mut c_void =
            std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
        let raw = send(device, sel_registerName(c"newBufferWithLength:options:".as_ptr()), bytes, 0);
        Buffer::from_raw(raw.cast())
            .ok_or_else(|| format!("Metal buffer allocation failed ({bytes} bytes)").into())
    }
}

pub fn copy_from(gpu: &Gpu, values: &[f32]) -> Result<Buffer> {
    let buffer = allocate(gpu, values.len())?;
    // SAFETY: the separate shared allocation holds exactly values.len() floats,
    // and no GPU work has been submitted that could access it yet.
    unsafe {
        std::ptr::copy_nonoverlapping(
            values.as_ptr(), contents(&buffer)?, values.len(),
        );
    }
    Ok(buffer)
}

pub fn read(buffer: &Buffer, count: usize) -> Result<Vec<f32>> {
    let bytes = count.checked_mul(std::mem::size_of::<f32>()).ok_or("buffer size overflow")?;
    // SAFETY: buffer is a live MTLBuffer; length returns NSUInteger.
    let length = unsafe { size_property(std::ptr::from_ref(&**buffer).cast(), c"length") };
    if bytes > length {
        return Err("read past end of Metal buffer".into());
    }
    let mut values = Vec::new();
    values.try_reserve_exact(count)?;
    // SAFETY: length was checked above, values has sufficient capacity, and
    // the caller waits for GPU completion before reading. The allocations do
    // not overlap, and set_len exposes only the initialized floats.
    unsafe {
        std::ptr::copy_nonoverlapping(
            contents(buffer)?, values.as_mut_ptr(), count,
        );
        values.set_len(count);
    }
    Ok(values)
}
