//! Entry point for `outcrop-NN <fn> <in_dir> <out_dir>` (see CONTRACT.md).
//!
//! ```ignore
//! fn main() {
//!     outcrop_common::run(&[("prism_gravity", prism_gravity)]);
//! }
//! fn prism_gravity(inp: &outcrop_common::Inputs, out: &mut outcrop_common::Outputs) -> outcrop_common::Result<()> {
//!     let xyz = inp.f32("coordinates")?;        // float arrays are converted to f32
//!     let field = inp.str("field")?;            // scalars come from args.json
//!     out.f32("g_z", &[n], &values)?;           // written as <out_dir>/g_z.npy (<f4)
//!     Ok(())
//! }
//! ```

use crate::npy::{self, Array};
use crate::Result;
use serde_json::Value;
use std::path::PathBuf;

pub type Handler = fn(&Inputs, &mut Outputs) -> Result<()>;

/// Parse argv, dispatch to the named handler, print errors to stderr, exit nonzero on failure.
pub fn run(handlers: &[(&str, Handler)]) {
    let args: Vec<String> = std::env::args().collect();
    let names: Vec<&str> = handlers.iter().map(|(n, _)| *n).collect();
    if args.len() != 4 {
        eprintln!("usage: {} <function> <in_dir> <out_dir>\nfunctions: {}", args[0], names.join(", "));
        std::process::exit(2);
    }
    let Some((_, h)) = handlers.iter().find(|(n, _)| *n == args[1]) else {
        eprintln!("unknown function {:?}; available: {}", args[1], names.join(", "));
        std::process::exit(2);
    };
    let result = Inputs::load(PathBuf::from(&args[2])).and_then(|inp| {
        let mut out = Outputs { dir: PathBuf::from(&args[3]) };
        h(&inp, &mut out)
    });
    if let Err(e) = result {
        eprintln!("error in {}: {e}", args[1]);
        std::process::exit(1);
    }
}

/// Inputs for one call: arrays from `<in_dir>/<name>.npy`, scalars from `<in_dir>/args.json`.
pub struct Inputs {
    dir: PathBuf,
    args: serde_json::Map<String, Value>,
}

impl Inputs {
    pub fn load(dir: PathBuf) -> Result<Self> {
        let p = dir.join("args.json");
        let args = if p.exists() {
            match serde_json::from_str::<Value>(&std::fs::read_to_string(&p)?)? {
                Value::Object(m) => m,
                _ => return Err("args.json is not an object".into()),
            }
        } else {
            Default::default()
        };
        Ok(Inputs { dir, args })
    }

    fn npy_path(&self, name: &str) -> Result<PathBuf> {
        let p = self.dir.join(format!("{name}.npy"));
        if p.exists() {
            Ok(p)
        } else {
            Err(format!("missing array input {name:?}").into())
        }
    }

    pub fn has_array(&self, name: &str) -> bool {
        self.dir.join(format!("{name}.npy")).exists()
    }

    /// A float array as f32 (f64 inputs are rounded to f32).
    pub fn f32(&self, name: &str) -> Result<Array<f32>> {
        let p = self.npy_path(name)?;
        match npy::dtype(&p)?.as_str() {
            "<f4" => npy::read(&p),
            "<f8" => {
                let a: Array<f64> = npy::read(&p)?;
                Ok(Array { shape: a.shape, data: a.data.iter().map(|&x| x as f32).collect() })
            }
            d => Err(format!("{name}: expected float array, got {d}").into()),
        }
    }

    /// A float array as f64 (for host-side setup that needs it).
    pub fn f64(&self, name: &str) -> Result<Array<f64>> {
        let p = self.npy_path(name)?;
        match npy::dtype(&p)?.as_str() {
            "<f8" => npy::read(&p),
            "<f4" => {
                let a: Array<f32> = npy::read(&p)?;
                Ok(Array { shape: a.shape, data: a.data.iter().map(|&x| x as f64).collect() })
            }
            d => Err(format!("{name}: expected float array, got {d}").into()),
        }
    }

    /// An integer array as i64 (accepts <i4 and <i8).
    pub fn i64(&self, name: &str) -> Result<Array<i64>> {
        let p = self.npy_path(name)?;
        match npy::dtype(&p)?.as_str() {
            "<i8" => npy::read(&p),
            "<i4" => {
                let a: Array<i32> = npy::read(&p)?;
                Ok(Array { shape: a.shape, data: a.data.iter().map(|&x| x as i64).collect() })
            }
            d => Err(format!("{name}: expected integer array, got {d}").into()),
        }
    }

    pub fn bool(&self, name: &str) -> Result<Array<bool>> {
        npy::read(self.npy_path(name)?)
    }

    fn arg(&self, name: &str) -> Result<&Value> {
        self.args.get(name).ok_or_else(|| format!("missing scalar argument {name:?}").into())
    }
    pub fn has_arg(&self, name: &str) -> bool {
        self.args.get(name).is_some_and(|v| !v.is_null())
    }
    pub fn num(&self, name: &str) -> Result<f64> {
        self.arg(name)?.as_f64().ok_or_else(|| format!("{name} is not a number").into())
    }
    pub fn int(&self, name: &str) -> Result<i64> {
        let v = self.arg(name)?;
        v.as_i64()
            .or_else(|| v.as_f64().filter(|f| f.fract() == 0.0).map(|f| f as i64))
            .ok_or_else(|| format!("{name} is not an integer").into())
    }
    pub fn flag(&self, name: &str) -> Result<bool> {
        self.arg(name)?.as_bool().ok_or_else(|| format!("{name} is not a bool").into())
    }
    pub fn str(&self, name: &str) -> Result<String> {
        Ok(self.arg(name)?.as_str().ok_or_else(|| format!("{name} is not a string"))?.to_string())
    }
    pub fn nums(&self, name: &str) -> Result<Vec<f64>> {
        self.arg(name)?
            .as_array()
            .ok_or_else(|| format!("{name} is not a list"))?
            .iter()
            .map(|v| v.as_f64().ok_or_else(|| format!("{name}: non-numeric entry").into()))
            .collect()
    }
    /// Raw JSON value for anything else (nested lists, optional values).
    pub fn json(&self, name: &str) -> Option<&Value> {
        self.args.get(name)
    }
}

/// Outputs for one call: each becomes `<out_dir>/<name>.npy`.
pub struct Outputs {
    dir: PathBuf,
}

impl Outputs {
    pub fn f32(&mut self, name: &str, shape: &[usize], data: &[f32]) -> Result<()> {
        npy::write(self.dir.join(format!("{name}.npy")), shape, data)
    }
    pub fn f64(&mut self, name: &str, shape: &[usize], data: &[f64]) -> Result<()> {
        npy::write(self.dir.join(format!("{name}.npy")), shape, data)
    }
    pub fn i64(&mut self, name: &str, shape: &[usize], data: &[i64]) -> Result<()> {
        npy::write(self.dir.join(format!("{name}.npy")), shape, data)
    }
    pub fn i32(&mut self, name: &str, shape: &[usize], data: &[i32]) -> Result<()> {
        npy::write(self.dir.join(format!("{name}.npy")), shape, data)
    }
    pub fn bool(&mut self, name: &str, shape: &[usize], data: &[bool]) -> Result<()> {
        npy::write(self.dir.join(format!("{name}.npy")), shape, data)
    }
    /// A scalar float output, written as a 0-d array.
    pub fn scalar(&mut self, name: &str, v: f64) -> Result<()> {
        npy::write(self.dir.join(format!("{name}.npy")), &[], &[v])
    }
}
