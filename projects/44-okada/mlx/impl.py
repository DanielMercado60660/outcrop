"""Okada displacement in a homogeneous elastic half-space."""
import math

import mlx.core as mx


def _stack(*x):
    return mx.stack(x, axis=-1)


def _point(c, depth, sd, cd, slip, alpha):
    x, y = c[:, 0], c[:, 1]
    d = depth
    p, q = y*cd+d*sd, y*sd-d*cd
    r2 = x*x+y*y+d*d
    r = mx.sqrt(r2)
    r3, r5 = r*r2, r*r2*r2
    rd = r+d
    d12 = 1/(r*rd*rd)
    d32 = d12*(2*r+d)/r2
    d33 = d12*(3*r+d)/(r2*rd)
    f1 = y*(d12-x*x*d33)
    f2 = x*(d12-y*y*d33)
    f3 = x/r3-f2
    f4 = -x*y*d32
    f5 = 1/(r*rd)-x*x*d32
    qr = 3*q/r5
    a = (1-alpha)/alpha
    s = _stack(-x*x*qr-a*f1*sd, -x*y*qr-a*f2*sd, -d*x*qr-a*f4*sd)
    t = _stack(-x*p*qr+a*f3*sd*cd, -y*p*qr+a*f1*sd*cd, -d*p*qr+a*f5*sd*cd)
    n = _stack(x*q*qr-a*f3*sd*sd, y*q*qr-a*f1*sd*sd, d*q*qr-a*f5*sd*sd)
    return (slip[0]*s+slip[1]*t+slip[2]*n)/(2*math.pi)


def _near_zero(x, scale):
    # Eight float32 epsilons of the local length operands. Both sides scale
    # with the length unit; there must be no absolute length floor here.
    return mx.abs(x) <= (8 * 2.0**-23) * scale


def _snap(x, scale):
    # Preserve derivatives through the numerical zero convention. The scale
    # only selects a branch and contributes no derivative.
    return x + mx.stop_gradient(mx.where(_near_zero(x, scale), 0., x)-x)


def _corner(xi, eta, q, kxi, ket, z, sd, cd, alpha, slip):
    # Coordinates have already been snapped using their construction scales.
    a1, a2, a3, a4, a5 = (1-alpha)/2, alpha/2, (1-alpha)/alpha, 1-alpha, alpha
    x2, e2, q2 = xi*xi, eta*eta, q*q
    r2 = x2+e2+q2
    r = mx.sqrt(r2)
    r3 = r*r2
    y, d = eta*cd+q*sd, eta*sd-q*cd
    rx = mx.where(kxi, 1., r+xi)
    re = mx.where(ket, 1., r+eta)
    alx = mx.where(kxi, -mx.log(mx.where(kxi, r-xi, 1.)), mx.log(rx))
    ale = mx.where(ket, -mx.log(mx.where(ket, r-eta, 1.)), mx.log(re))
    x11 = mx.where(kxi, 0., 1/(r*rx))
    y11 = mx.where(ket, 0., 1/(r*re))
    x32 = mx.where(kxi, 0., (r+rx)*x11*x11/r)
    y32 = mx.where(ket, 0., (r+re)*y11*y11/r)
    # The zero-q branch is evaluated in its limiting form.  A constant
    # correction gives Okada's specified zero value at q=0, while retaining
    # the derivative of the principal arctangent away from a corner edge.
    xe = xi*eta
    safe_xe = mx.where(xe == 0, 1., xe)
    qr = q*r
    normal = mx.arctan(xe/mx.where(q == 0, 1., qr))
    limit = -mx.arctan(qr/safe_xe)
    # At xi=0 and q=0 the prescribed angular term and its q derivative
    # are zero; adding an eta*x11 term here corrupts vertical-dip gradients.
    tt = mx.where(q == 0, mx.where(xe == 0, 0., limit), normal)
    xy, qx, qy = xi*y11, q*x11, q*y11
    ua1 = _stack(tt/2+a2*xi*qy, a2*q/r, a1*ale-a2*q*qy)
    ua2 = _stack(a2*q/r, tt/2+a2*eta*qx, a1*alx-a2*q*qx)
    ua3 = _stack(-a1*ale-a2*q*qy, -a1*alx-a2*q*qx, tt/2-a2*(eta*qx+xi*qy))
    rd = r+d
    if cd != 0:
        xx = mx.sqrt(x2+q2)
        den = xi*(r+xx)*cd
        angle = mx.where(xi == 0, 0., mx.arctan((eta*(xx+q*cd)+xx*(r+xx)*sd)/mx.where(xi == 0, 1., den)))
        ai4 = (xi/rd*sd*cd+2*angle)/(cd*cd)
        ai3 = (y*cd/rd-ale+sd*mx.log(rd))/(cd*cd)
    else:
        ai3 = (eta/rd+y*q/(rd*rd)-ale)/2
        ai4 = xi*y/(2*rd*rd)
    ai1 = -xi/rd*cd-ai4*sd
    ai2 = mx.log(rd)+ai3*sd
    ub1 = _stack(-xi*qy-tt-a3*ai1*sd, -q/r+a3*y/rd*sd, q*qy-a3*ai2*sd)
    ub2 = _stack(-q/r+a3*ai3*sd*cd, -eta*qx-tt-a3*xi/rd*sd*cd, q*qx+a3*ai4*sd*cd)
    ub3 = _stack(q*qy-a3*ai3*sd*sd, q*qx+a3*xi/rd*sd*sd, eta*qx+xi*qy-tt-a3*ai4*sd*sd)
    c = d+z
    z32 = sd/r3-(q*cd-z)*y32
    uc1 = _stack(a4*xy*cd-a5*xi*q*z32, a4*(cd/r+2*qy*sd)-a5*c*q/r3, a4*qy*cd-a5*(c*eta/r3-z*y11+x2*z32))
    uc2 = _stack(a4*cd/r-qy*sd-a5*c*q/r3, a4*y*x11-a5*c*eta*q*x32, -d*x11-xy*sd-a5*c*(x11-q2*x32))
    uc3 = _stack(-a4*(sd/r+qy*cd)-a5*(z*y11-q2*z32), 2*a4*xy*sd+d*x11-a5*c*(x11-q2*x32), a4*(y*x11+xy*cd)+a5*q*(c*eta*x32+xi*z32))
    def combine(v1, v2, v3):
        return (slip[0]*v1+slip[1]*v2+slip[2]*v3)/(2*math.pi)
    return combine(ua1, ua2, ua3), combine(ub1, ub2, ub3), combine(uc1, uc2, uc3)


def _rectangle(c, depth, sd, cd, bounds, slip, alpha):
    x, y, z = c[:, 0], c[:, 1], c[:, 2]
    xis = [_snap(x-bounds[i], mx.abs(x)+mx.abs(bounds[i])) for i in range(2)]
    total = mx.zeros_like(c)
    for image in (False, True):
        dep = depth-z if image else depth+z
        dep_scale = abs(depth)+mx.abs(z)
        p_scale = mx.abs(y*cd)+dep_scale*abs(sd)
        q_scale = mx.abs(y*sd)+dep_scale*abs(cd)
        p, q = y*cd+dep*sd, _snap(y*sd-dep*cd, q_scale)
        es = [_snap(p-bounds[i], p_scale+mx.abs(bounds[i])) for i in (2, 3)]
        r12 = mx.sqrt(xis[0]**2+es[1]**2+q*q)
        r21 = mx.sqrt(xis[1]**2+es[0]**2+q*q)
        r22 = mx.sqrt(xis[1]**2+es[1]**2+q*q)
        # Apply the same relative rule to the negative-extension logarithms.
        # These sums cancel at removable singularities (KXI/KET rules).
        kx = [(xis[0]<0)&_near_zero(r+xis[1], r+mx.abs(xis[1])) for r in (r21, r22)]
        ke = [(es[0]<0)&_near_zero(r+es[1], r+mx.abs(es[1])) for r in (r12, r22)]
        for i in range(2):
            for j in range(2):
                a, b, cc = _corner(xis[i], es[j], q, kx[j], ke[i], z, sd, cd, alpha, slip)
                if image:
                    v = a+b+z[:, None]*cc
                    w = a+b-z[:, None]*cc
                    u = _stack(v[:, 0], v[:, 1]*cd-v[:, 2]*sd, w[:, 1]*sd+w[:, 2]*cd)
                else:
                    u = _stack(-a[:, 0], -a[:, 1]*cd+a[:, 2]*sd, -a[:, 1]*sd-a[:, 2]*cd)
                total = total + (1 if i == j else -1)*u
    return total


def okada(coordinates, source, depth, dip_deg, bounds, slip, alpha):
    c = mx.array(coordinates, dtype=mx.float32)
    b = mx.array(bounds, dtype=mx.float32)
    s = mx.array(slip, dtype=mx.float32)
    sd, cd = math.sin(math.radians(dip_deg)), math.cos(math.radians(dip_deg))
    if abs(cd) < 1e-6:
        sd, cd = (1. if sd > 0 else -1.), 0.
    # Keep inverse powers and their autodiff intermediates in float32 range
    # when the length unit makes the entire geometry very small or large.
    # A power of two changes no input significands. Ordinary-scale inputs
    # retain their existing evaluation, including its rounding behavior.
    length_scale = 1.
    if 0 < depth < 2.0**-8 or depth > 2.0**8:
        length_scale = math.ldexp(1., math.frexp(depth)[1]-1)
    scaled_depth = depth/length_scale
    if source == "point":
        if length_scale == 1.:
            fn = lambda x: _point(x, depth, sd, cd, s, alpha)
        else:
            # The point kernel is homogeneous of degree -2 in length.
            potency = (s/length_scale)/length_scale
            fn = lambda x: _point(x/length_scale, scaled_depth, sd, cd, potency, alpha)
    elif source == "finite_rectangle":
        if length_scale == 1.:
            fn = lambda x: _rectangle(x, depth, sd, cd, b, s, alpha)
        else:
            scaled_bounds = b/length_scale
            fn = lambda x: _rectangle(x/length_scale, scaled_depth, sd, cd, scaled_bounds, s, alpha)
    else:
        raise ValueError("source must be 'point' or 'finite_rectangle'")
    u = fn(c)
    derivatives = mx.stack([mx.grad(lambda x, component=i: mx.sum(fn(x)[:, component]))(c)[:, :2] for i in range(3)], axis=1)
    return {"displacement": u, "horizontal_derivatives": derivatives}
