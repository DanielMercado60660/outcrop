//! Fallible shared-storage allocation without adding an objc2 dependency.
use outcrop_common::gpu::Buffer;
use outcrop_common::{Gpu, Result};
use std::ffi::{c_char, c_void, CStr};

// common exposes retained objects, but not the Metal protocol traits. Keep this
// small Objective-C ABI bridge local until common provides fallible helpers.
// All messages below have fixed pointer/NSUInteger signatures on macOS.
#[link(name = "objc")]
extern "C" {
    fn sel_registerName(name: *const c_char) -> *const c_void;
    fn objc_msgSend();
}

unsafe fn integer_property(object: *const c_void, name: &CStr) -> usize {
    let send = std::mem::transmute::<
        unsafe extern "C" fn(),
        unsafe extern "C" fn(*const c_void, *const c_void) -> usize,
    >(objc_msgSend);
    send(object, sel_registerName(name.as_ptr()))
}

fn contents(buffer: &Buffer) -> Result<*mut f32> {
    // SAFETY: buffer is a live MTLBuffer; contents returns void* and takes no
    // arguments. These buffers were allocated with MTLStorageModeShared.
    let pointer = unsafe {
        let send = std::mem::transmute::<
            unsafe extern "C" fn(),
            unsafe extern "C" fn(*const c_void, *const c_void) -> *mut f32,
        >(objc_msgSend);
        send(
            Buffer::as_ptr(buffer).cast(),
            sel_registerName(c"contents".as_ptr()),
        )
    };
    if pointer.is_null() {
        return Err("GPU buffer has no CPU-accessible storage".into());
    }
    Ok(pointer)
}

pub fn allocate(gpu: &Gpu, count: usize) -> Result<Buffer> {
    let bytes = count
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or("GPU buffer size overflow")?
        .max(4);
    let device = std::ptr::from_ref(&*gpu.device).cast::<c_void>();
    // SAFETY: device is a live MTLDevice; maxBufferLength returns NSUInteger.
    let maximum = unsafe { integer_property(device, c"maxBufferLength") };
    if bytes > maximum {
        return Err(format!("GPU buffer of {bytes} bytes exceeds device limit").into());
    }
    // SAFETY: this is the exact newBufferWithLength:options: ABI. Zero options
    // select shared storage/default cache/default hazard tracking. A new-family
    // return has +1 ownership, transferred once to common's retained Buffer;
    // from_raw converts nil to None without a panic.
    let buffer = unsafe {
        let send = std::mem::transmute::<
            unsafe extern "C" fn(),
            unsafe extern "C" fn(*const c_void, *const c_void, usize, usize) -> *mut c_void,
        >(objc_msgSend);
        let raw = send(
            device,
            sel_registerName(c"newBufferWithLength:options:".as_ptr()),
            bytes,
            0,
        );
        Buffer::from_raw(raw.cast())
    };
    buffer.ok_or_else(|| format!("GPU buffer allocation failed ({bytes} bytes)").into())
}

pub fn from_slice(gpu: &Gpu, values: &[f32]) -> Result<Buffer> {
    let buffer = allocate(gpu, values.len())?;
    let pointer = contents(&buffer)?;
    // SAFETY: allocate checked the byte count; shared storage is CPU writable,
    // and this fresh buffer cannot overlap the input slice or be in GPU use.
    unsafe {
        std::ptr::copy_nonoverlapping(values.as_ptr(), pointer, values.len());
    }
    Ok(buffer)
}

pub fn read(buffer: &Buffer, count: usize) -> Result<Vec<f32>> {
    let bytes = count
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or("GPU read size overflow")?;
    // SAFETY: buffer is a live MTLBuffer; length returns NSUInteger.
    let length = unsafe { integer_property(Buffer::as_ptr(buffer).cast(), c"length") };
    if bytes > length {
        return Err("GPU read exceeds buffer length".into());
    }
    let pointer = contents(buffer)?;
    let mut result = Vec::new();
    result.try_reserve_exact(count)?;
    // SAFETY: the caller has waited for GPU completion, the checked shared
    // buffer contains count f32 values, and result has capacity for the copy.
    unsafe {
        std::ptr::copy_nonoverlapping(pointer, result.as_mut_ptr(), count);
        result.set_len(count);
    }
    Ok(result)
}
