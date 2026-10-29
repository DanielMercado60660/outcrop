use outcrop_common::gpu::Arg;
use outcrop_common::{Gpu, Inputs, Outputs, Result};

// The only f64 arithmetic is this O(1) scalar setup. Python scalars can exceed
// the f32 range: D = k = P = 1e308 would all cast to infinity, giving NaNs,
// although alpha = sqrt(2) and the continuous peak = 1/(2 sqrt(2)) are finite.
// Keep that range handling isolated here; every coordinate and the entire
// deflection profile are evaluated in f32 on the GPU.
fn scalar_parameters(rigidity: f64, foundation: f64, load: f64, broken: bool) -> (f32, i32, f32) {
    // alpha = (4D/k)^(1/4). Taking roots before division avoids overflow.
    let alpha = std::f64::consts::SQRT_2 * rigidity.sqrt().sqrt() / foundation.sqrt().sqrt();
    // P/(k alpha) has units of metres. The full force acts on a broken
    // edge, so its peak is four times the continuous plate's peak.
    let factor: f64 = if broken { 2.0 } else { 0.5 };
    let denominator = foundation * alpha;
    let amplitude = if load == 0.0 {
        0.0
    } else if denominator.is_finite() && denominator > 0.0 && (load / denominator * factor).is_finite() {
        (load / denominator * factor) as f32
    } else {
        (load.signum() * (load.abs().ln() - foundation.ln() - alpha.ln() + factor.ln()).exp()) as f32
    };

    // Pass alpha as a normal f32 mantissa times an integer power of two.
    // Casting alpha directly could overflow/underflow for finite Python
    // scalars. Metal's ldexp performs coordinate scaling without that cast.
    let exponent = alpha.log2().floor() as i32;
    let mantissa = (alpha / 2.0f64.powi(exponent)) as f32;
    (mantissa, exponent, amplitude)
}

fn plate_flexure(inp: &Inputs, out: &mut Outputs) -> Result<()> {
    let x = inp.f32("x")?;
    if x.shape.len() != 1 {
        return Err("x must be a one-dimensional array".into());
    }
    let rigidity = inp.num("rigidity")?;
    let foundation = inp.num("foundation")?;
    let load = inp.num("load")?;
    if !load.is_finite() {
        return Err("load must be finite".into());
    }
    if !rigidity.is_finite() || rigidity <= 0.0 || !foundation.is_finite() || foundation <= 0.0 {
        return Err("rigidity and foundation must be positive and finite".into());
    }
    let broken = match inp.str("plate")?.as_str() {
        "continuous" => false,
        "broken" => true,
        _ => return Err("plate must be continuous or broken".into()),
    };
    let n = x.data.len();
    let n32 = u32::try_from(n).map_err(|_| "x is too large for a Metal dispatch")?;
    if n == 0 {
        return out.f32("deflection", &x.shape, &[]);
    }

    // The continuous solution is
    // P/(2 k alpha) exp(-|x|/alpha) (cos(|x|/alpha)+sin(|x|/alpha)).
    // The edge solution is 2P/(k alpha) exp(-x/alpha) cos(x/alpha).
    // For u = x/alpha, the continuous profile's third u-derivative at
    // 0+ is 4, and even symmetry gives a shear jump of 8D*A/alpha^3.
    // The broken profile has second derivative 0 and third derivative 2
    // at the edge, giving D*w'''(0) = 2D*A/alpha^3. With alpha^4 = 4D/k,
    // both amplitudes above give the required positive force P.
    let (alpha_mantissa, alpha_exponent, amplitude) =
        scalar_parameters(rigidity, foundation, load, broken);
    let gpu = Gpu::new()?;
    let library = gpu.compile(include_str!("../kernels/flexure.metal"))?;
    let kernel = gpu.pipeline(&library, "plate_flexure")?;
    let bx = gpu.buffer_from(&x.data);
    let bw = gpu.buffer_zeroed::<f32>(n);
    gpu.dispatch(
        &kernel,
        &[
            Arg::Buf(&bx),
            Arg::Buf(&bw),
            Arg::F32(amplitude),
            Arg::U32(u32::from(broken)),
            Arg::U32(n32),
            Arg::F32(alpha_mantissa),
            Arg::I32(alpha_exponent),
        ],
        [n, 1, 1],
    )?;
    out.f32("deflection", &x.shape, &gpu.read::<f32>(&bw, n))
}

fn main() {
    outcrop_common::run(&[("plate_flexure", plate_flexure)]);
}
