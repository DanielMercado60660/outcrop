# Outcrop test contract (frozen)

How tests, judges and implementations talk to each other. Tests are written against this contract and implementations fill it in. It applies to every project; do not change it inside one.

## Folders in a project

```
projects/NN-slug/
  judge/
    make_goldens.py   # oracle: writes judge/golden/*.npz (and golden/meta.json). Only place reference libraries may be imported.
    golden/           # written by running make_goldens.py
  tests/
    api.py            # THE API: stub functions with full docstrings (names, args, shapes, dtypes, units, return values)
    test_*.py         # pytest tests
  mlx/impl.py         # MLX implementation of every function in tests/api.py
  metal/              # Rust crate outcrop-NN: binary that implements the same functions
  bench/, figure.png, RESULTS.md
```

## The API (tests/api.py)

- Every function the tests call is declared in `tests/api.py` as a stub that raises `NotImplementedError`, with a docstring giving each argument and return value: name, shape, dtype, units, and conventions (axis order, grid origin, sign conventions, coordinate system).
- Inputs and outputs are NumPy arrays (float32 for implementations) plus Python scalars, ints, strings, and lists of these. No custom classes.
- Functions return a dict of named NumPy arrays (`{"gz": arr, ...}`), so the Metal path can return the same thing.
- Gradient projects: the API includes a function that returns the objective and its gradient (for example `misfit_and_grad`). MLX computes it with autodiff; Metal with a hand-written adjoint.

## Calling an implementation from a test

Use the harness in `common/py/outcrop_harness.py` (already provided, on `sys.path` via the repo-root `conftest.py`; do not add your own sys.path hacks):

```python
from outcrop_harness import call, PATHS
out = call(path, "prism_gravity", coordinates=xyz, prisms=p, density=rho, field="g_z")
```

- `path` is `"mlx"` or `"metal"`. Tests are parametrized over the paths the project includes, with the path in the test id, so `pytest -k mlx` and `pytest -k metal` select one path. MLX-only projects use `PATHS = ["mlx"]`, Metal-only use `["metal"]` (plus `"mlx"` if MLX exists as a reference).
- `call(path, fn, **kwargs)` returns a dict of NumPy arrays.
  - mlx: imports `projects/NN-slug/mlx/impl.py` and calls `fn(**kwargs)`; mx.arrays in the result are converted to NumPy.
  - metal: writes array kwargs as `.npy` and the rest as `args.json` into a temp dir, runs the release binary `outcrop-NN` from the Cargo workspace (`target/release/outcrop-NN`) as `outcrop-NN <fn> <in_dir> <out_dir>`, and reads every `<name>.npy` in out_dir.
- MLX vs Metal cross-check tests call both and compare.

## Goldens and tolerances

- `make_goldens.py` uses fixed seeds, float64 where the reference supports it, and writes `.npz` files plus `golden/meta.json` (inputs used, library versions, tolerances and the reason for each).
- Tests load goldens with `outcrop_harness.golden(name)` and never import reference libraries or regenerate data.
- Each tolerance lives in one place (`tests/tolerances.py`), with a comment giving the reason. The project-wide defaults: float32 relative error at or below 1e-4 against analytic or float64 references unless the method itself is approximate (any looser tolerance is documented with the reason), and centered finite-difference gradient checks at or below 1e-3.
- Tests must not skip, xfail, or depend on randomness without a fixed seed. The count of collected tests is recorded at lock time.
- Keep goldens small: under 20 MB per project in total. Use small problem sizes for tests; benchmark sizes live in bench/.

## Running

- Python: `.venv/bin/python` (do not use `uv run`; it writes a lockfile).
- Tests: `.venv/bin/python -m pytest projects/NN-slug/tests -k <path> -rs -v`, on a Mac with an Apple silicon GPU.
