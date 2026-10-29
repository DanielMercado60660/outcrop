"""Public ray tracing contract, identical for both paths."""


def shoot_layered(interfaces, velocities, angles, depth):
    """Shoot direct transmitted rays from (x,z)=(0,0) to a given depth.

    Coordinates are metres, x positive right and z positive down. `interfaces`
    is strictly increasing float32 (L-1,), positive interface depths in metres;
    `velocities` is float32 (L,), layer velocities in m/s from the surface.
    `angles` is float32 (N,), radians from downward vertical, positive toward
    +x. `depth` is a positive Python float in metres, not on an interface.
    Rays are downgoing and subcritical in traversed layers. Return a dict with
    `endpoints` float32 (N,2), columns x,z in metres, and `times` float32
    (N,), one-way travel times in seconds. No reflected or head waves.
    """
    raise NotImplementedError


def bend_layered(interfaces, velocities, offsets, depth):
    """Find the unique downgoing direct transmitted rays to receivers.

    `interfaces`, `velocities`, `depth`, coordinates and units follow
    shoot_layered. `offsets` is float32 (N,) in metres, and receivers are at
    (offsets[i], depth). Return a dict with `endpoints` float32 (N,2) in
    metres, `times` float32 (N,) in seconds, and `angles` float32 (N,) in
    radians from downward vertical, signed toward +x.
    """
    raise NotImplementedError


def shoot_gradient(v0, gradient, angles, depth):
    """Shoot rays in v(z)=v0+gradient*z from (0,0) to `depth`.

    `v0` is a Python float in m/s, `gradient` a Python float in (m/s)/m,
    `angles` float32 (N,) in radians from downward vertical, signed toward +x,
    and `depth` a positive Python float in metres. x is positive right, z
    positive down. Velocity is positive throughout and rays reach the depth
    before turning. Return `endpoints` float32 (N,2), columns x,z in metres,
    and `times` float32 (N,) in seconds. Zero gradient is homogeneous.
    """
    raise NotImplementedError


def bend_gradient(v0, gradient, offsets, depth):
    """Find unique downgoing direct rays in v(z)=v0+gradient*z.

    `v0`, `gradient`, `depth`, coordinates and units follow shoot_gradient.
    `offsets` is float32 (N,) in metres, with receivers at (offset, depth).
    Rays reach receivers before turning. Return `endpoints` float32 (N,2) in
    metres, `times` float32 (N,) in seconds, and `angles` float32 (N,) in
    radians from downward vertical, signed toward +x.
    """
    raise NotImplementedError
