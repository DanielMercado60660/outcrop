"""Independent float64 NumPy transcription of tests/SCHEME.md.

No Deepwave functions, coefficients, saved wavefields, or implementation code
are used. Background forcing is cached once; the adjoint reverses the written
recurrence, including the transpose of edge replication.
"""

import numpy as np


class SchemeBorn:
    def __init__(self, inputs):
        assert int(inputs["accuracy"]) == 4
        self.v = np.asarray(inputs["v"], np.float64)
        self.mask = np.asarray(inputs["mask"], np.float64)
        self.spacing = np.asarray(inputs["spacing"], np.float64)
        self.dt = float(inputs["dt"])
        self.width = int(inputs["pml_width"])
        assert self.width > 0
        self.step_ratio = max(1, int(np.ceil(
            self.dt * self.v.max() * np.linalg.norm(1 / self.spacing) / 0.6)))
        assert self.step_ratio == 1
        self.pad = self.width + 2
        self.velocity = np.pad(self.v, self.pad, mode="edge")
        self.kv2 = self.dt**2 * self.velocity**2
        self.sources = np.asarray(inputs["source_amplitudes"], np.float64)
        self.nshot, _, self.nt = self.sources.shape
        self.shape = (self.nshot, *self.velocity.shape)
        self.source_nodes = np.asarray(inputs["source_locations"]) + self.pad
        self.receiver_nodes = np.asarray(inputs["receiver_locations"]) + self.pad
        self.shots = np.arange(self.nshot)[:, None]
        self.src = (self.shots, self.source_nodes[..., 0], self.source_nodes[..., 1])
        self.rec = (self.shots, self.receiver_nodes[..., 0], self.receiver_nodes[..., 1])
        self.profiles = []
        for axis, (length, spacing) in enumerate(zip(self.velocity.shape, self.spacing)):
            i = np.arange(length)
            fraction = np.maximum(np.clip((self.pad - i) / self.width, 0, 1),
                                  np.clip((i - (length - 1 - self.pad)) / self.width, 0, 1))
            sigma = 3 * self.v.max() * np.log(1000) / (2 * self.width * spacing) * fraction**2
            alpha = np.pi * float(inputs["pml_freq"]) * (1 - fraction)
            a = np.where(fraction > 0, np.exp(-(sigma + alpha) * self.dt), 0)
            b = sigma / (sigma + alpha) * np.expm1(-(sigma + alpha) * self.dt)
            db = np.zeros(length)
            db[2:-2] = ((2 / 3) * (b[3:-1] - b[1:-3])
                          - (1 / 12) * (b[4:] - b[:-4])) / spacing
            shape = (1, length, 1) if axis == 0 else (1, 1, length)
            self.profiles.append(tuple(q.reshape(shape) for q in (a, b, db)))
        # Padded-to-physical map for the exact transpose of edge extension.
        iz = np.clip(np.arange(self.velocity.shape[0]) - self.pad, 0, self.v.shape[0] - 1)
        ix = np.clip(np.arange(self.velocity.shape[1]) - self.pad, 0, self.v.shape[1] - 1)
        self.fold_indices = (iz[:, None] * self.v.shape[1] + ix[None, :]).ravel()
        self.drive = np.empty((self.nt, *self.shape), np.float64)
        self.background_data = np.empty((*self.receiver_nodes.shape[:2], self.nt))
        q, previous, memories = self.zero_state()
        for n in range(self.nt):
            self.background_data[..., n] = q[self.rec]
            h, memories = self.spatial(q, memories)
            np.add.at(h, self.src, -self.sources[..., n])
            self.drive[n] = (2 * self.dt**2 * self.velocity) * h
            following = 2 * q - previous + self.kv2 * h
            previous, q = q, following

    def zero_state(self):
        return (np.zeros(self.shape), np.zeros(self.shape),
                [(np.zeros(self.shape), np.zeros(self.shape)) for _ in range(2)])

    def derivative(self, q, axis, order):
        """P D_order P; D1.T=-D1 and D2.T=D2 on zero-guard fields."""
        out = np.zeros_like(q)
        centre = q[:, 2:-2, 2:-2]
        if axis == 0:
            plus1, minus1 = q[:, 3:-1, 2:-2], q[:, 1:-3, 2:-2]
            plus2, minus2 = q[:, 4:, 2:-2], q[:, :-4, 2:-2]
        else:
            plus1, minus1 = q[:, 2:-2, 3:-1], q[:, 2:-2, 1:-3]
            plus2, minus2 = q[:, 2:-2, 4:], q[:, 2:-2, :-4]
        if order == 1:
            value = ((2 / 3) * (plus1 - minus1) - (1 / 12) * (plus2 - minus2))
        else:
            value = ((4 / 3) * ((plus1 - centre) + (minus1 - centre))
                     - (1 / 12) * ((plus2 - centre) + (minus2 - centre)))
        out[:, 2:-2, 2:-2] = value / self.spacing[axis]**order
        return out

    def spatial(self, q, memories):
        h = np.zeros_like(q)
        following = []
        for axis, ((a, b, db), (psi, zeta)) in enumerate(zip(self.profiles, memories)):
            d = self.derivative(q, axis, 1)
            t = ((1 + b) * self.derivative(q, axis, 2) + db * d
                 + self.derivative(a * psi, axis, 1))
            h += (1 + b) * t + a * zeta
            following.append((b * d + a * psi, b * t + a * zeta))
        return h, following

    def forward(self, model):
        dv = np.pad(self.mask * np.asarray(model, np.float64), self.pad, mode="edge")
        q, previous, memories = self.zero_state()
        data = np.empty((*self.receiver_nodes.shape[:2], self.nt))
        for n in range(self.nt):
            data[..., n] = q[self.rec]
            h, memories = self.spatial(q, memories)
            following = 2 * q - previous + self.kv2 * h + dv * self.drive[n]
            previous, q = q, following
        return data

    def adjoint(self, data):
        """Exact reverse accumulation of forward(), including R.T and E.T."""
        data = np.asarray(data, np.float64)
        lq, lprevious, memories = self.zero_state()
        padded_gradient = np.zeros_like(self.velocity)
        for n in range(self.nt - 1, -1, -1):
            padded_gradient += np.sum(self.drive[n] * lq, axis=0)
            current = 2 * lq + lprevious
            hbar = self.kv2 * lq
            old_memories = []
            for axis, ((a, b, db), (lpsi, lzeta)) in enumerate(zip(self.profiles, memories)):
                tbar = (1 + b) * hbar + b * lzeta
                current += (self.derivative((1 + b) * tbar, axis, 2)
                            - self.derivative(db * tbar + b * lpsi, axis, 1))
                old_memories.append((a * (lpsi - self.derivative(tbar, axis, 1)),
                                     a * (hbar + lzeta)))
            np.add.at(current, self.rec, data[..., n])
            lprevious, lq, memories = -lq, current, old_memories
        folded = np.bincount(self.fold_indices, weights=padded_gradient.ravel(),
                             minlength=self.v.size).reshape(self.v.shape)
        return self.mask * folded


def cgls(op, observed, iterations):
    """Independent NumPy replay of the exact api.py iteration contract."""
    model = np.zeros_like(op.v)
    residual = np.asarray(observed, np.float64).copy()
    norm0 = np.sqrt(np.sum(residual**2))
    basis = []
    history = [1.0]
    first = None
    for iteration in range(iterations):
        p = op.adjoint(residual)
        q = op.forward(p)
        for _ in range(2):
            for pi, qi in basis:
                beta = np.sum(qi * q)
                p -= beta * pi
                q -= beta * qi
        scale = np.sqrt(np.sum(q**2))
        p, q = p / scale, q / scale
        alpha = np.sum(residual * q)
        model += alpha * p
        basis.append((p, q))
        if first is None:
            first = model.copy()
        residual = observed - op.forward(model)
        history.append(np.sqrt(np.sum(residual**2)) / norm0)
        if (iteration + 1) % 10 == 0:
            print(f"  independent scheme CGLS {iteration + 1}/{iterations}", flush=True)
    return model, np.asarray(history), first


def validate(arrays, tolerance, fd_step):
    """Refuse acceptance unless independently recomputed outputs match.

    Checks both golden bundles, every modeled/adjoint/solver output, all FD
    objectives and both complete 40-step histories. Derived width metrics are
    already checked separately by the generator and the integrity tests.
    """
    op = SchemeBorn(arrays)
    errors = {}

    def check(name, value, expected):
        value, expected = np.asarray(value), np.asarray(expected)
        assert value.shape == expected.shape, name
        assert np.all(np.isfinite(value)), name
        norm = np.max(np.abs(expected))
        error = np.max(np.abs(value - expected))
        relative = float(error / norm) if norm else float(error)
        errors[name] = relative
        assert relative <= tolerance, (name, relative, tolerance)

    check("observed", op.forward(arrays["truth"]), arrays["observed"])
    check("rtm", op.adjoint(arrays["observed"]), arrays["rtm"])
    check("zero_objective", 0.5 * np.sum(arrays["observed"]**2), arrays["zero_objective"])
    check("zero_model_prediction", op.forward(arrays["zero_model"]), np.zeros_like(arrays["observed"]))
    if "iterations" in arrays:
        for prefix in ("", "scaled_"):
            data = arrays[f"{prefix}observed"]
            image, history, first = cgls(op, data, int(arrays["iterations"]))
            check(f"{prefix}image", image, arrays[f"{prefix}image"])
            check(f"{prefix}residual_history", history, arrays[f"{prefix}residual_history"])
            check(f"{prefix}predicted", op.forward(image), arrays[f"{prefix}predicted"])
            if prefix:
                check("scaled_rtm", op.adjoint(data), arrays["scaled_rtm"])
            else:
                check("rtm_scaled", first, arrays["rtm_scaled"])
                check("rtm_scaled_predicted", op.forward(first), arrays["rtm_scaled_predicted"])
    else:
        prediction = op.forward(arrays["model"])
        residual = prediction - arrays["observed"]
        check("predicted", prediction, arrays["predicted"])
        check("objective", 0.5 * np.sum(residual**2), arrays["objective"])
        check("gradient", op.adjoint(residual), arrays["gradient"])
        check("adjoint", op.adjoint(arrays["probe_data"]), arrays["adjoint"])
        check("masked_model_prediction", op.forward(arrays["model_outside"]), arrays["predicted"])
        fds = []
        for plus, minus in zip(arrays["fd_models_plus"], arrays["fd_models_minus"], strict=True):
            rplus, rminus = op.forward(plus) - arrays["observed"], op.forward(minus) - arrays["observed"]
            fds.append((0.5 * np.sum(rplus**2) - 0.5 * np.sum(rminus**2)) / (2 * fd_step))
        check("fd_values", fds, arrays["fd_values"])
    return {"step_ratio": op.step_ratio, "relative_max_errors": errors,
            "max_relative_error": max(errors.values())}
