"""Render a synthetic acoustic wave snapshot using the Metal sandbox."""
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent.parent / 'common' / 'py'))
os.environ['OUTCROP_PROJECT'] = str(ROOT)
from outcrop_harness import call


def main():
    ny, nx, nt, p = 400, 500, 260, 16
    dy = dx = 10.0
    dt = .001
    y, x = np.mgrid[:ny, :nx]
    velocity = (1800 + 450 * (y > 220 + 25 * np.sin(x / 100))
                + 250 * np.exp(-(((x - 325) / 70) ** 2 + ((y - 270) / 45) ** 2)))
    t = np.arange(nt) * dt
    a = (np.pi * 18 * (t - .065)) ** 2
    source = ((1 - 2*a) * np.exp(-a)).astype(np.float32)[None, :]
    loc = np.array([[100, 250]], dtype=np.int64)
    receivers = np.column_stack((np.full(160, 100), np.linspace(10, nx - 11, 160, dtype=np.int64)))
    out = call('metal', 'run_frames', velocity=velocity.astype(np.float32),
               state=np.zeros((6, ny + 2*p, nx + 2*p), np.float32),
               source_amplitudes=source, source_locations=loc, receiver_locations=receivers,
               grid_spacing=[dy, dx], dt=dt, accuracy=4, pml_width=p, pml_freq=18.,
               max_velocity=2500., boundary='pml', render_scale=1.)
    field = out['state'][0, p:p + ny, p:p + nx]
    limit = max(float(np.percentile(np.abs(field), 99.7)), 1e-12)
    fig, ax = plt.subplots(figsize=(9, 6.4), constrained_layout=True)
    im = ax.imshow(field, extent=[0, (nx-1)*dx/1000, (ny-1)*dy/1000, 0],
                   cmap='RdBu_r', vmin=-limit, vmax=limit, interpolation='bilinear', aspect='equal')
    ax.contour(x * dx / 1000, y * dy / 1000, velocity, levels=[2000],
               colors='0.2', linewidths=.8, alpha=.65)
    ax.scatter(loc[:, 1]*dx/1000, loc[:, 0]*dy/1000, marker='*', s=90,
               color='#f3bd43', edgecolors='black', linewidths=.6, label='Source', zorder=3)
    ax.set(xlabel='Distance (km)', ylabel='Depth (km)',
           title='Metal acoustic wave propagation · 0.26 s')
    ax.legend(loc='lower right', framealpha=.9)
    fig.colorbar(im, ax=ax, label='Pressure (arbitrary units)', shrink=.86, pad=.02)
    fig.savefig(ROOT / 'figure.png', dpi=180, facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    main()
