"""Standalone unit-rescaling regression; run only on the GPU."""

import importlib.util
from pathlib import Path

import numpy as np


def main():
    impl_path = Path(__file__).resolve().parents[2] / "mlx" / "impl.py"
    spec = importlib.util.spec_from_file_location("okada_mlx_unit_check", impl_path)
    impl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(impl)

    # Powers of two preserve float32 input geometry exactly. The small scale
    # puts every Table-2 rectangle length below the former absolute cutoff.
    factors = (2.0**-24, 2.0**16)
    cases = [
        ("point", 4., 70., [0., 3., 0., 2.], [[2., 3., 0.]]),
        ("finite_rectangle", 4., 70., [0., 3., 0., 2.], [[2., 3., 0.]]),
        ("finite_rectangle", 4., 90., [0., 3., 0., 2.],
         [[0., 0., 0.], [0.125, 0.25, 0.]]),
        ("finite_rectangle", 4., -90., [0., 3., 0., 2.],
         [[0., 0., 0.], [0.125, 0.25, 0.]]),
        ("finite_rectangle", 50., 70., [-80., 120., -30., 25.],
         [[10., 20., -30.]]),
    ]
    count = 0
    for source, depth, dip, bounds, coordinates in cases:
        for slip in np.eye(3, dtype=np.float32):
            args = dict(
                coordinates=np.asarray(coordinates, dtype=np.float32),
                source=source,
                depth=depth,
                dip_deg=dip,
                bounds=np.asarray(bounds, dtype=np.float32),
                slip=slip,
                alpha=2 / 3,
            )
            base = {k: np.asarray(v) for k, v in impl.okada(**args).items()}
            for factor in factors:
                scaled = impl.okada(**{
                    **args,
                    "coordinates": args["coordinates"] * factor,
                    "depth": depth * factor,
                    "bounds": args["bounds"] * factor,
                    "slip": slip * (factor**3 if source == "point" else factor),
                })
                for key, value in scaled.items():
                    actual = np.asarray(value)
                    assert actual.dtype == np.float32
                    assert np.isfinite(actual).all(), (source, dip, factor, key)
                    # Displacement scales as length; horizontal partials are
                    # dimensionless when slip/potency units scale as well.
                    restored = actual / factor if key == "displacement" else actual
                    np.testing.assert_allclose(
                        restored, base[key], rtol=1e-4, atol=1e-6,
                        err_msg=f"{source}, dip={dip}, factor={factor}, {key}",
                    )
                count += 1
    print(f"Passed {count} unit-rescaling comparisons (both output arrays).")


if __name__ == "__main__":
    main()
