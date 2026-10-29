//! outcrop-common: shared Rust for Outcrop Metal implementations.
//!
//! - [`npy`]: read/write NumPy `.npy` arrays (the harness contract in CONTRACT.md).
//! - [`cli`]: the `outcrop-NN <fn> <in_dir> <out_dir>` entry point and typed access to inputs.
//! - [`gpu`]: a small Metal context: compile MSL, make pipelines, buffers, dispatch, read back.
//!

pub mod cli;
pub mod gpu;
pub mod npy;

pub use cli::{run, Inputs, Outputs};
pub use gpu::Gpu;
pub use npy::Array;

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
