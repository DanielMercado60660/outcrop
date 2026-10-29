"""Generate the fixed CPU float64 Deepwave and analytic Born fixtures."""
import json
import runpy
from pathlib import Path

import deepwave
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "judge" / "golden"
TOL = runpy.run_path(str(ROOT / "tests" / "tolerances.py"))
torch.set_num_threads(1)


def run(case, dv=None):
    def tensor(name, dtype=torch.float64):
        return torch.as_tensor(case[name], dtype=dtype)

    dv = tensor("dv") if dv is None else dv
    return deepwave.scalar_born(
        tensor("v"), float(case["dx"]), float(case["dt"]), dv=dv,
        source_amplitudes=tensor("source_amplitudes"),
        source_locations=tensor("source_locations", torch.long),
        bg_receiver_locations=tensor("receiver_locations", torch.long),
        sc_receiver_locations=tensor("receiver_locations", torch.long),
        wavefield_0=tensor("wavefield_0"), wavefield_m1=tensor("wavefield_m1"),
        wavefield_sc_0=tensor("wavefield_sc_0"),
        wavefield_sc_m1=tensor("wavefield_sc_m1"),
        accuracy=int(case["accuracy"]), boundary=str(case["boundary"]),
        pml_width=int(case["pml_width"]), pml_freq=float(case["pml_freq"]),
        max_vel=float(case["max_vel"]), origin=[0, 0],
    )


def base(ny, nx, shots, nt, dx, dt, accuracy, boundary, pml_width):
    z = np.zeros((shots, ny, nx), np.float32)
    return {"dx": dx, "dt": dt, "accuracy": accuracy, "boundary": boundary,
            "pml_width": pml_width, "pml_freq": 15., "max_vel": 1800.,
            "wavefield_0": z.copy(), "wavefield_m1": z.copy(),
            "wavefield_sc_0": z.copy(), "wavefield_sc_m1": z.copy()}


def save(name, case):
    np.savez_compressed(OUT / (name + ".npz"), **case)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1402)
    y, x = np.mgrid[:48, :52]
    for order in (2, 4):
        c = base(48, 52, 2, 150, 10., .001, order, "pml", 10)
        c["v"] = (1500 + 100 * np.sin(y / 17) * np.cos(x / 19)).astype(np.float32)
        c["dv"] = (45 * np.exp(-((y-26)**2/30 + (x-25)**2/45))
                   - 25 * np.exp(-((y-32)**2/18 + (x-35)**2/25))).astype(np.float32)
        t = np.arange(150) * c["dt"] - .045
        q = (np.pi * 15 * t)**2
        pulse = (1 - 2*q) * np.exp(-q)
        c["source_amplitudes"] = np.stack([pulse, .8*pulse])[:, None, :].astype(np.float32)
        c["source_locations"] = np.array([[[12, 14]], [[14, 36]]], np.int64)
        c["receiver_locations"] = np.array([[[12, 14], [12, 23], [12, 32], [20, 25]],
                                             [[14, 36], [14, 27], [14, 18], [20, 30]]], np.int64)
        result = run(c)
        c["background"] = result.bg_receiver_amplitudes.detach().numpy()
        c["scattered"] = result.sc_receiver_amplitudes.detach().numpy()
        # Independent, fixed target ensures the residual is not trivially zero.
        c["observed"] = (c["scattered"] * .35).astype(np.float32)
        direction = rng.normal(size=c["dv"].shape) * np.exp(-((y-26)**2+(x-26)**2)/180)
        c["direction"] = (direction / np.max(np.abs(direction))).astype(np.float32)
        dv = torch.tensor(c["dv"], dtype=torch.float64, requires_grad=True)
        residual = run(c, dv).sc_receiver_amplitudes - torch.tensor(c["observed"], dtype=torch.float64)
        objective = .5 * (residual**2).sum()
        gradient, = torch.autograd.grad(objective, dv)
        c["objective"] = objective.detach().numpy()
        c["gradient"] = gradient.detach().numpy()
        eps = .5
        def obj(d, case=c):
            r = run(case, torch.tensor(d, dtype=torch.float64)).sc_receiver_amplitudes.detach().numpy() - case["observed"]
            return .5 * np.sum(r*r)
        fd = (obj(c["dv"].astype(np.float64) + eps*c["direction"]) -
              obj(c["dv"].astype(np.float64) - eps*c["direction"])) / (2*eps)
        adj = np.sum(c["gradient"] * c["direction"])
        assert abs(fd-adj) / abs(adj) < TOL["ORACLE_FD"], (fd, adj)
        print(order, "background/scattered max", np.max(np.abs(c["background"])), np.max(np.abs(c["scattered"])), "objective", c["objective"], "gradient fd", fd, adj)
        save(f"pml{order}", c)

    errors = []
    for n in (24, 48, 96):
        # Periodic smooth mode, spatial period 480 m; exact tangent solution.
        dx = 480. / n
        dt = .2 * dx / 1500.
        nt = 2*n + 1  # receiver's last sample is at fixed physical time
        c = base(n, n, 1, nt, dx, dt, 2, "periodic", 0)
        c["v"] = np.full((n, n), 1500., np.float32)
        c["dv"] = np.full((n, n), 30., np.float32)
        yy, xx = np.mgrid[:n, :n]
        mode = np.cos(2*np.pi*yy/n) * np.cos(2*np.pi*xx/n)
        k = np.sqrt(2)*2*np.pi/480
        c["wavefield_0"] = mode[None].astype(np.float32)
        c["wavefield_m1"] = (mode*np.cos(1500*k*dt))[None].astype(np.float32)
        c["wavefield_sc_m1"] = (-30*k*dt*np.sin(1500*k*dt)*mode)[None].astype(np.float32)
        c["source_amplitudes"] = np.zeros((1, 1, nt), np.float32)
        c["source_locations"] = np.array([[[0, 0]]], np.int64)
        c["receiver_locations"] = np.array([[[0, 0], [n//4, 0]]], np.int64)
        result = run(c)
        c["background"] = result.bg_receiver_amplitudes.detach().numpy()
        c["scattered"] = result.sc_receiver_amplitudes.detach().numpy()
        time = (nt-1)*dt
        c["analytic_final"] = np.array(-30*k*time*np.sin(1500*k*time))
        error = abs(c["scattered"][0, 0, -1] - c["analytic_final"])
        errors.append(error)
        print("convergence", n, "analytic", c["analytic_final"], "error", error)
        save(f"mode{n}", c)
    print("error ratios", np.array(errors[:-1])/errors[1:])
    ratios = np.array(errors[:-1])/errors[1:]
    assert (min(ratios) > TOL["CONVERGENCE_RATIO"]
            and max(ratios) < TOL["CONVERGENCE_RATIO_MAX"]
            and errors[-1] < TOL["CONVERGENCE_FINE"])
    meta = {"seed": 1402, "numpy": np.__version__, "torch": torch.__version__,
            "deepwave": deepwave.__version__, "dtype": "float64 reference; float32 inputs",
            "cases": {"pml2,pml4": "48x52, two shots, 150 samples, dx=10 m, dt=1 ms, 10-cell PML, 15 Hz, max velocity 1800 m/s", "mode24,mode48,mode96": "periodic 480 m square; dx=480/n, dt=0.2 dx/1500, nt=2n+1; 1500 m/s, dv=30 m/s"},
            "analytic": "cos(2pi y/L) cos(2pi x/L), tangent -dv*k*t*sin(v*k*t), k=sqrt(2)*2pi/L",
            "tolerances": {k: {"value": TOL[k], "reason": reason} for k, reason in {
                "PARITY": "float32 versus float64 at identical discretization", "CROSS": "independent float32 implementations", "GRADIENT": "project-wide centered directional derivative limit", "ORACLE_FD": "float64 oracle directional derivative", "CONVERGENCE_RATIO": "second-order convergence allowing pre-asymptotic and rounding effects", "CONVERGENCE_RATIO_MAX": "upper bound for observed second-order convergence", "CONVERGENCE_FINE": "fine-grid absolute modal amplitude error bound with float32 margin"}.items()}}
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    main()
