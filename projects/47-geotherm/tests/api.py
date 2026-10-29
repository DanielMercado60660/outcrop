"""Contract for the cooling-half-space geotherm on both execution paths."""


def geotherm(depth_m, age_myr, surface_temp_c, mantle_temp_c, diffusivity_m2_s, conductivity_w_m_k):
    """Return temperature and upward conductive heat flow for cooling half-spaces.

    depth_m: float32 array (D,), nonnegative depth in metres, positive downward.
    age_myr: float32 array (N,), positive ages in million Julian years, where
    one year is exactly 365.25 * 86400 seconds. The scalar arguments are Python
    floats: surface_temp_c and mantle_temp_c in degrees C (mantle > surface),
    diffusivity_m2_s in m²/s and conductivity_w_m_k in W/(m K), both positive.
    Initially the entire half-space is at mantle temperature; the surface is
    instantaneously fixed at surface temperature for t > 0. Return a dict of
    float32 arrays temperature_c and heat_flow_w_m2, both (N, D), with age on
    axis 0 and depth on axis 1. Heat flow is positive upward: k * dT/dz.
    """
    raise NotImplementedError


def invert_age(surface_heat_flow_w_m2, surface_temp_c, mantle_temp_c, diffusivity_m2_s, conductivity_w_m_k):
    """Recover independent cooling ages from surface heat-flow observations.

    surface_heat_flow_w_m2: positive float32 array (N,) in W/m², positive
    upward. The four Python float material/boundary arguments have the same
    units and constraints as geotherm. For each observation return a dict
    containing age_myr, a float32 (N,) array of positive ages in million
    Julian years. The observations are exact, with known material properties;
    solve the half-space surface-flux inverse independently for each entry.
    """
    raise NotImplementedError


def heat_flow_misfit_and_grad(age_myr, observed_heat_flow_w_m2, flux_scale_w_m2, surface_temp_c, mantle_temp_c, diffusivity_m2_s, conductivity_w_m_k):
    """Return surface heat-flow least-squares objective and its age gradient.

    age_myr and observed_heat_flow_w_m2 are float32 (N,) arrays, respectively
    positive ages in million Julian years and positive upward fluxes in W/m².
    flux_scale_w_m2 is a positive Python float in W/m². Other Python floats
    have the units and constraints given by geotherm. Define objective as
    0.5 * mean(((predicted_surface_flux - observed_flux) / flux_scale)**2).
    Return a dict with objective, a zero-dimensional float32 array (unitless),
    and gradient, a float32 (N,) array of derivatives per million Julian years.
    MLX must use autodiff and Metal a hand-written derivative.
    """
    raise NotImplementedError
