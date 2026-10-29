"""Cooling half-space temperature, heat flow, and age inversion in MLX."""

import math

import mlx.core as mx

_SECONDS_PER_MYR = 365.25 * 86400.0 * 1_000_000.0


def _erf_nonnegative(x):
    """Approximate erf for nonnegative x with float32 MLX operations.

    A Taylor series avoids cancellation close to zero; elsewhere the
    Abramowitz and Stegun 7.1.26 approximation has error below 1.5e-7.
    """
    x2 = x * x
    series = 1.0 / 9360.0 + x2 * (-1.0 / 75600.0)
    series = -1.0 / 1320.0 + x2 * series
    series = 1.0 / 216.0 + x2 * series
    series = -1.0 / 42.0 + x2 * series
    series = 1.0 / 10.0 + x2 * series
    series = -1.0 / 3.0 + x2 * series
    small = (2.0 / math.sqrt(math.pi)) * x * (1.0 + x2 * series)

    t = 1.0 / (1.0 + 0.3275911 * x)
    polynomial = (((((1.061405429 * t - 1.453152027) * t + 1.421413741)
                    * t - 0.284496736) * t + 0.254829592) * t)
    large = 1.0 - polynomial * mx.exp(-x2)
    return mx.where(x < 0.5, small, large)


def geotherm(depth_m, age_myr, surface_temp_c, mantle_temp_c,
             diffusivity_m2_s, conductivity_w_m_k):
    depth = mx.array(depth_m, dtype=mx.float32)[None, :]
    age = mx.array(age_myr, dtype=mx.float32)[:, None]
    # Take roots before multiplying: kappa * t can overflow float32 even
    # when the diffusion length itself is representable.
    diffusion_length = mx.sqrt(age) * math.sqrt(diffusivity_m2_s * _SECONDS_PER_MYR)
    eta = depth / (2.0 * diffusion_length)
    delta_temp = mantle_temp_c - surface_temp_c
    temperature = surface_temp_c + delta_temp * _erf_nonnegative(eta)
    flux = (conductivity_w_m_k * delta_temp / math.sqrt(math.pi)
            / diffusion_length) * mx.exp(-eta * eta)
    return {"temperature_c": temperature, "heat_flow_w_m2": flux}


def invert_age(surface_heat_flow_w_m2, surface_temp_c, mantle_temp_c,
               diffusivity_m2_s, conductivity_w_m_k):
    flux = mx.array(surface_heat_flow_w_m2, dtype=mx.float32)
    # q_surface = coefficient / sqrt(age_myr). Scale before squaring so
    # the dimensional (k * delta_temp / flux)**2 need not be representable.
    coefficient = (conductivity_w_m_k * (mantle_temp_c - surface_temp_c)
                   / math.sqrt(math.pi * diffusivity_m2_s * _SECONDS_PER_MYR))
    root_age = coefficient / flux
    age = root_age * root_age
    return {"age_myr": age}


def heat_flow_misfit_and_grad(age_myr, observed_heat_flow_w_m2,
                              flux_scale_w_m2, surface_temp_c, mantle_temp_c,
                              diffusivity_m2_s, conductivity_w_m_k):
    age = mx.array(age_myr, dtype=mx.float32)
    observed = mx.array(observed_heat_flow_w_m2, dtype=mx.float32)
    coefficient = (conductivity_w_m_k * (mantle_temp_c - surface_temp_c)
                   / math.sqrt(math.pi * diffusivity_m2_s * _SECONDS_PER_MYR))

    def objective(ages):
        predicted = coefficient / mx.sqrt(ages)
        residual = (predicted - observed) / flux_scale_w_m2
        return 0.5 * mx.mean(residual * residual)

    value, gradient = mx.value_and_grad(objective)(age)
    return {"objective": value, "gradient": gradient}
