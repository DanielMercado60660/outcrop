"""Plot a synthetic joint gravity and magnetic inversion."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1] / 'common' / 'py'))
os.environ['OUTCROP_PROJECT'] = str(ROOT)
from outcrop_harness import call


def main():
    shape = (8, 8, 2)
    xyz = np.indices(shape).reshape(3, -1).T
    n = len(xyz)
    d = np.zeros((3, n, n), dtype=np.float32)
    indices = np.arange(n).reshape(shape)
    for axis in range(3):
        for cell in np.ndindex(shape):
            if cell[axis] + 1 < shape[axis]:
                neighbor = list(cell)
                neighbor[axis] += 1
                i, j = indices[cell], indices[tuple(neighbor)]
                d[axis, i, i], d[axis, i, j] = -1, 1
    x, y = np.meshgrid(np.linspace(-0.5, 7.5, 11), np.linspace(-0.5, 7.5, 11), indexing='ij')
    rx, ry = x.ravel()[:, None], y.ravel()[:, None]
    r2 = (rx - xyz[:, 0])**2 + (ry - xyz[:, 1])**2
    depth = xyz[:, 2] + 1.3
    ag = (depth / (r2 + depth**2)**1.5).astype(np.float32)
    am = ((2 * depth**2 - r2) / (r2 + depth**2)**2.5).astype(np.float32)
    u = (np.exp(-((xyz[:, 0] - 2.3)**2 / 2.2 + (xyz[:, 1] - 2.5)**2 / 2.0
                  + xyz[:, 2]**2 / 0.8))
         + 0.75 * np.exp(-((xyz[:, 0] - 5.3)**2 / 1.8 + (xyz[:, 1] - 5.1)**2 / 2.5
                            + xyz[:, 2]**2 / 0.8))).astype(np.float32)
    v = (0.8 * u).astype(np.float32)
    ag /= np.max(np.abs(ag @ u))
    am /= np.max(np.abs(am @ v))
    rng = np.random.default_rng(1101)
    dg = (ag @ u + rng.normal(0, 0.01, len(ag))).astype(np.float32)
    dm = (am @ v + rng.normal(0, 0.01, len(am))).astype(np.float32)
    out = call('mlx', 'invert', gravity_matrix=ag, magnetic_matrix=am,
               gravity_data=dg, magnetic_data=dm, derivatives=d, alpha=0.01,
               smooth=0.005, coupling=0.1, sweeps=20)

    fig, axes = plt.subplots(2, 2, figsize=(9, 7.5), constrained_layout=True)
    for row, (truth, recovered, unit, scale) in enumerate([
        (u, out['density'], 'Density contrast (kg/m³)', 500),
        (v, out['susceptibility'], 'Susceptibility contrast (SI)', 0.05),
    ]):
        values = [truth.reshape(shape)[:, :, 0] * scale,
                  recovered.reshape(shape)[:, :, 0] * scale]
        vmax = max(float(np.max(np.abs(a))) for a in values)
        for col, (ax, value) in enumerate(zip(axes[row], values)):
            im = ax.imshow(value.T, origin='lower', extent=[0, 800, 0, 800],
                           cmap='RdBu_r', vmin=-vmax, vmax=vmax, interpolation='nearest')
            ax.set_title(('Synthetic truth' if col == 0 else 'Joint inversion') +
                         (' · density' if row == 0 else ' · susceptibility'))
            ax.set_xlabel('East (m)')
            ax.set_ylabel('North (m)')
            ax.set_aspect('equal')
        fig.colorbar(im, ax=axes[row, :], label=unit, shrink=0.86, pad=0.02)
    fig.suptitle('Joint gravity–magnetic inversion · shallow layer', fontsize=15, weight='bold')
    fig.savefig(ROOT / 'figure.png', dpi=180, facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    main()
