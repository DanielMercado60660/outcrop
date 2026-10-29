//! Drain Objective-C autoreleases after EACH synchronous submission. Otherwise
//! an autoreleased command buffer can retain buffers from already-dropped
//! checkpoints until process exit. This wrapper needs no additional crate.
use outcrop_common::gpu::{Arg, Batch, Pipeline};
use outcrop_common::{Gpu, Result};
use std::ffi::c_void;
use std::ops::Deref;

#[link(name = "objc")]
unsafe extern "C" {
    fn objc_autoreleasePoolPush() -> *mut c_void;
    fn objc_autoreleasePoolPop(pool: *mut c_void);
}

pub fn autorelease_pool<T>(f: impl FnOnce() -> T) -> T {
    struct Pool(*mut c_void);
    impl Drop for Pool {
        fn drop(&mut self) {
            // SAFETY: The pool was pushed on this thread and is popped in
            // stack order. It outlives the synchronous work in f, even on unwind.
            unsafe { objc_autoreleasePoolPop(self.0); }
        }
    }
    // SAFETY: Objective-C permits nested pools on the current thread.
    let _pool = Pool(unsafe { objc_autoreleasePoolPush() });
    f()
}

pub struct Device(Gpu);
impl Device {
    pub fn new() -> Result<Self> {
        autorelease_pool(|| Gpu::new().map(Self))
    }

    pub fn batch(&self, f: impl FnOnce(&mut Batch) -> Result<()>) -> Result<()> {
        autorelease_pool(|| self.0.batch(f))
    }

    pub fn dispatch(&self, k: &Pipeline, args: &[Arg<'_>], grid: [usize; 3]) -> Result<()> {
        self.batch(|b| b.dispatch(k, args, grid))
    }
}

impl Deref for Device {
    type Target = Gpu;
    fn deref(&self) -> &Gpu { &self.0 }
}
