# Outcrop: cross-project findings

2026-10-28. Per-project detail is in each `projects/NN-slug/RESULTS.md`.

## Status

All 50 projects are complete: each has a locked judge and test suite, every path the project includes passes those tests, every project was benchmarked at three sizes, and every project has a figure. None is blocked. The paths split as 33 projects with MLX and Metal, 12 MLX-only and 5 Metal-only.

Ten projects (06, 16, 19, 24, 33, 34, 35, 45, 46, 49) were re-run after the main benchmark run, for precision-policy compliance or judge fixes. Their numbers below are from those re-runs. The MLX-vs-Metal cross-checks were run separately; see [CROSS_CHECKS.md](CROSS_CHECKS.md).

Remaining items are listed under [Open issues](#open-issues).

## How to read the numbers

- **Source.** Every number below comes from `projects/*/bench/results.json` or the per-project test runs. Tables were rendered by script from those files, not typed.
- **Wall time.** Median of three calls after one warm-up.
  - Metal times include process launch, shader compilation and file I/O on every call, a floor of roughly 40–60 ms. Metal therefore loses most small-size comparisons by construction; the large size is the fair comparison.
  - MLX times include conversion of the output to NumPy.
- **Speedup is judge ÷ path.** The judge is a correctness oracle (Harmonica, Deepwave in float64 on one CPU thread, SimPEG, ObsPy, scikit-fmm, or a float64 NumPy reference). It is not a tuned competitor, so speedups say "faster than the oracle", not "faster than the state of the art". In 15 projects the harness skipped the judge at the large size (each RESULTS.md quotes the reason); there the speedup is given at medium and labelled.
- **Memory is not comparable across paths.** Each path uses a different method:
  - judge: tracemalloc, which misses native Torch/NumPy allocations;
  - MLX: the allocator peak;
  - Metal: the child-process `ru_maxrss` high-water delta, which is often 0 after warm-up and never counts GPU buffers.

  Each RESULTS.md prints memory with its method. This report draws no memory conclusions beyond that.
- **Stale wording.** Many `results.json` `notes` fields call large-size timings "estimates" or "unverified". They were written before the run. The numbers are measured; the wording is stale.

## Which methods ran fastest on which path

Out of 50 projects, the fastest path at the large size was MLX in 25 and Metal in 25. That count includes the 17 single-path projects, where the only path wins by default. In the 33 projects where both paths ran:

- **Metal fastest (20):** 01, 03, 04, 05, 13, 14, 15, 16, 17, 18, 21, 24, 28, 30, 34, 36, 40, 44, 48, 49. These are time-stepping and stencil propagators (13–18, 21, 36, 48, 49), gradient-through-solver work (28), and big embarrassingly parallel sums (01, 03, 44). 24 is the odd one out: Metal wins only narrowly (1.21 s vs MLX 1.33 s), and both are far slower than scikit-fmm.
- **MLX fastest (13):** 02, 06, 09, 10, 12, 26, 27, 32, 39, 41, 42, 46, 47. These are:
  - vectorised closed forms and FFT pipelines (26, 32, 39, 42, 46, 47);
  - problems where Metal's per-call launch floor dominates (02, 06, 10, 12);
  - three projects where Metal runs a dense solve on a single GPU thread (09, 27, 41).

**Largest speedups over the oracle.** For each project this takes the best path at the reference size: large, or medium where the judge skipped large.

| Project | Speedup | Path | Size |
|---|---|---|---|
| 05 Talwani | 1,448× | MLX | medium; the judge is a slow 96-point area quadrature |
| 26 ray tracing | 720× | MLX | medium |
| 44 Okada | 663× | Metal | large |
| 40 CMP | 177× | Metal | large |
| 33 MT Occam | 144× | MLX | medium; mostly autodiff vs the judge's finite-difference Jacobian |
| 49 tsunami | 112× | Metal | large |
| 01 point-sphere | 61.5× | Metal | large |
| 07 FFT filters | 44.3× | MLX | large |

**Projects where even the fastest path was slower than the judge (12):** 04, 09, 10, 15, 19, 20, 21, 24, 34, 37, 38, 50. The causes:

- **Problems too small to amortise GPU launch:** 04 (at most 256 observations), 10 (864 cells), 21.
- **Small dense fit on MLX's CPU stream:** 09, where the fastest path is MLX at 0.255×. Metal's 47.4 s single-thread solve is a separate problem.
- **Small FWI grids where MLX loses to the single-thread oracle:** 19, now float32 on the GPU, at 0.128×; 20 at 0.345×.
- **Serial algorithms:**
  - 24, fast marching: a host heap with MLX GPU recompute at 0.0133×, and a single-thread GPU heap on Metal at 0.0147×.
  - 37, STA/LTA: a running-sum recursion, 0.141× ObsPy's C loop.
- **Judge skipped at large, so the comparison is only at medium:** 15, 34, 38, 50.

Project 38 (template matching) was expected to be "dramatically fast". At medium it is 0.332× ObsPy, because the kernel is a direct correlation with no FFT. The judge skipped the large size, so no large comparison exists.

**Single-thread GPU solves.** Three Metal paths put a dense factorisation on one GPU thread:

| Project | What runs serially | Metal (large) | Judge (large) | MLX (large) |
|---|---|---|---|---|
| 09 | QR/SVD | 47.4 s | 14.1 ms | 55.2 ms |
| 27 | Cholesky | 1.63 s | judge skipped | 18.7 ms |
| 41 | Levinson-type solve | 9.80 s | 23.6 ms | 5.84 ms |

All three pass their tests. They are correct but are the slowest results in Outcrop, and the clearest Metal follow-up work.

### Fastest path per project (large size)

"Its speedup vs judge" is for the path that was fastest at large. Where the judge skipped large, that path's medium-size speedup is shown, which can differ from the best medium speedup: 05 MLX was 1,448× at medium. Project 16's large size is Marmousi2 at 140×680, 25 m, 2001 steps and 8 shots.

| # | Project | Judge (large) | MLX (large) | Metal (large) | Fastest at large | Its speedup vs judge |
|---|---|---|---|---|---|---|
| 01 | point-sphere-gravity | 4.30 s | 241 ms | 69.9 ms | Metal | 61.5× |
| 02 | prism-gravity | n/a | 69.8 ms | 235 ms | MLX | 11.1× (medium; judge large skipped) |
| 03 | prism-magnetic | n/a | 305 ms | 80.2 ms | Metal | 2.89× (medium; judge large skipped) |
| 04 | tesseroid-gravity | 1.91 ms | 373 ms | 76.4 ms | Metal | 0.0251× |
| 05 | talwani-2d | n/a | 80.4 ms | 50.7 ms | Metal | 36.5× (medium; judge large skipped) |
| 06 | terrain-bouguer | 536 ms | 160 ms | 235 ms | MLX | 3.35× |
| 07 | fft-filters | 3.03 s | 68.4 ms | n/a | MLX (only path) | 44.3× |
| 08 | normal-gravity | 8.35 s | 289 ms | n/a | MLX (only path) | 28.8× |
| 09 | equivalent-sources | 14.1 ms | 55.2 ms | 47.4 s | MLX | 0.255× |
| 10 | density-inversion | 41.8 ms | 54.0 ms | 180 ms | MLX | 0.774× |
| 11 | joint-grav-mag | 180 ms | 66.2 ms | n/a | MLX (only path) | 2.72× |
| 12 | acoustic-1d | 491 ms | 285 ms | 301 ms | MLX | 1.72× |
| 13 | acoustic-2d | 79.9 ms | 488 ms | 60.0 ms | Metal | 1.33× |
| 14 | born-2d | 1.18 s | 538 ms | 94.8 ms | Metal | 12.5× |
| 15 | elastic-psv | n/a | 697 ms | 79.7 ms | Metal | 0.341× (medium; judge large skipped) |
| 16 | fwi-marmousi | 14.0 s | 18.7 s | 1.95 s | Metal | 7.20× |
| 17 | rtm | 509 ms | 553 ms | 112 ms | Metal | 4.55× |
| 18 | lsrtm | 729 ms | 355 ms | 133 ms | Metal | 5.47× |
| 19 | multiscale-fwi | 87.8 ms | 685 ms | n/a | MLX (only path) | 0.128× |
| 20 | nnfwi | 419 ms | 1.21 s | n/a | MLX (only path) | 0.345× |
| 21 | checkpointing | 144 ms | 256 ms | 161 ms | Metal | 0.897× |
| 22 | acoustic-3d | n/a | n/a | 232 ms | Metal (only path) | 2.99× (medium; judge large skipped) |
| 23 | misfit-fwi | 398 ms | 60.0 ms | n/a | MLX (only path) | 6.64× |
| 24 | fast-marching | 17.7 ms | 1.33 s | 1.21 s | Metal | 0.0147× |
| 25 | fast-sweeping | 445 ms | n/a | 112 ms | Metal (only path) | 3.96× |
| 26 | ray-tracing | n/a | 57.9 ms | 405 ms | MLX | 720× (medium; judge large skipped) |
| 27 | traveltime-tomo | n/a | 18.7 ms | 1.63 s | MLX | 1.70× (medium; judge large skipped) |
| 28 | diff-eikonal | n/a | 3.22 s | 166 ms | Metal | 2.38× (medium; judge large skipped) |
| 29 | eikonet | 143 ms | 32.7 ms | n/a | MLX (only path) | 4.37× |
| 30 | eq-location | 1.65 s | 245 ms | 176 ms | Metal | 9.40× |
| 31 | double-difference | n/a | 416 ms | n/a | MLX (only path) | 1.85× (medium; judge large skipped) |
| 32 | mt-1d | 539 ms | 30.1 ms | 50.4 ms | MLX | 17.9× |
| 33 | mt-occam | n/a | 46.1 ms | n/a | MLX (only path) | 144× (medium; judge large skipped) |
| 34 | dc-2d | n/a | 1.31 s | 925 ms | Metal | 0.599× (medium; judge large skipped) |
| 35 | dc-inversion | 100 ms | 28.5 ms | n/a | MLX (only path) | 3.51× |
| 36 | gpr-fdtd | n/a | 493 ms | 150 ms | Metal | 12.3× (medium; judge large skipped) |
| 37 | sta-lta | 52.3 ms | n/a | 371 ms | Metal (only path) | 0.141× |
| 38 | template-matching | n/a | n/a | 72.1 ms | Metal (only path) | 0.332× (medium; judge large skipped) |
| 39 | ambient-noise | 339 ms | 20.5 ms | 299 ms | MLX | 16.5× |
| 40 | cmp-processing | 8.52 s | 50.1 ms | 48.1 ms | Metal | 177× |
| 41 | deconvolution | 23.6 ms | 5.84 ms | 9.80 s | MLX | 4.04× |
| 42 | fk-radon | 768 ms | 32.5 ms | 406 ms | MLX | 23.6× |
| 43 | phase-picker | 2.70 s | 157 ms | n/a | MLX (only path) | 17.2× |
| 44 | okada | 25.0 s | 42.1 ms | 37.7 ms | Metal | 663× |
| 45 | slip-inversion | n/a | 57.6 ms | n/a | MLX (only path) | 2.15× (medium; judge large skipped) |
| 46 | plate-flexure | 474 ms | 36.5 ms | 280 ms | MLX | 13.0× |
| 47 | geotherm | 839 ms | 144 ms | 443 ms | MLX | 5.82× |
| 48 | mantle-convection | 1.04 s | 323 ms | 93.1 ms | Metal | 11.2× |
| 49 | tsunami | 6.14 s | 284 ms | 54.9 ms | Metal | 112× |
| 50 | wave-sandbox | n/a | n/a | 112 ms | Metal (only path) | 0.481× (medium; judge large skipped) |

## Where MLX autodiff beat hand-written Metal adjoints, and where it did not

These six projects have an MLX gradient by autodiff and a Metal gradient (or gradient-based operator) by hand-written adjoint kernels, with a gradient-bearing operation timed in the benchmark:

| Project | Timed operation | MLX autodiff (large) | Metal adjoint (large) | Faster at large | Medium | Medium rel. diff vs judge (MLX / Metal) | Source size, non-whitespace kB (MLX / Metal) |
|---|---|---|---|---|---|---|---|
| 10 density inversion | full inversion (864 cells) | 54.0 ms | 180 ms | **MLX 3.3×** | MLX 10.8× | 1.12e-6 / 4.10e-8 | 4 / 15 |
| 16 FWI | `misfit_and_grad`, Marmousi2 140×680 @25 m, 2001 steps, 8 shots (medium: 70×340 @50 m, 1001 steps) | 18.7 s | 1.95 s | **Metal 9.6×** | Metal 8.9× | 1.22e-5 / 5.27e-5 | 5 / 20 |
| 17 RTM | RTM image (`mx.grad` vs adjoint kernels) | 553 ms | 112 ms | **Metal 4.9×** | Metal 3.6× | 5.05e-6 / 1.99e-6 | 7 / 19 |
| 18 LSRTM | one CGLS update | 355 ms | 133 ms | **Metal 2.7×** | Metal 2.2× | 2.13e-6 / 1.93e-6 | 9 / 18 |
| 21 checkpointing | checkpointed `misfit_and_grad` (budget 7) | 256 ms | 161 ms | **Metal 1.6×** | MLX 1.25× | 6.99e-7 / 5.01e-7 | 9 / 30 |
| 28 diff. eikonal | eikonal `misfit_and_grad` | 3.22 s | 166 ms | **Metal 19.5×** | Metal 14.4× | 2.03e-6 / 2.46e-6 | 7 / 11 |

**Where Metal won.** At the large size, the hand adjoint won every time-stepping and solver gradient: 16, 17, 18, 21 and 28. The widest gap is 28, at 19.5×, because MLX replays the whole iterative sweep inside `value_and_grad`. The second widest is 16 on Marmousi2, at 9.6×. Against the float64 judge, the hand adjoint was the more accurate path in four of six projects; 16 and 28 are the exceptions.

Against the oracle:

- **16 FWI.** Metal beats single-thread float64 Deepwave by 7.20× at large (1.95 s vs 14.0 s) and 2.03× at medium. MLX autodiff runs at 0.752× and 0.228×.
- **17 and 18.** Metal is 4.55× and 5.47×.
- **21.** Metal is 0.897× the full-storage judge.
- MLX autodiff beats the judge only in 18, at 2.05×.

**Where MLX won:**

- **Small problems.** Project 10's inversion has 864 cells, so Metal's per-call launch floor dominates. MLX also wins 21 at medium, and 17 and 21 at the small size.
- **Code size.** The autodiff path is 1.6–4× smaller in every row above, by non-whitespace characters.
- **Accuracy.** Autodiff never lost on accuracy. Both paths pass the same gates: cosine ≥ 0.999 vs Deepwave or adjoint-state, centered finite differences ≤ 1e-3, and 1e-6 full-storage identity in 21.

**Gradients tested but not timed.** In 14, 15, 44, 47 and 49, MLX autodiff and the Metal hand gradient (closed-form derivatives in 44) both pass the finite-difference checks, but only the forward model was benchmarked. Forward Metal was faster at large in 14 (5.7× MLX), 15 (8.7×) and 49 (5.2×). Forward MLX was faster in 47 (3.1×). 44 is overhead-flat at about 40 ms on both paths.

**MLX-only gradient projects.** Projects 11, 19, 20, 23, 29, 33, 35, 43 and 45 rely on autodiff alone. Against the judge, the fastest path ran at:

| Project | Speedup vs judge | Size |
|---|---|---|
| 33 | 144× | medium |
| 43 | 17.2× | large |
| 23 | 6.64× | large |
| 29 | 4.37× | large |
| 35 | 3.51× | large |
| 11 | 2.72× | large |
| 45 | 2.15× | medium |
| 20 | 0.345× | large |
| 19 | 0.128× | large; now float32 on the GPU |

Project 50 is Metal-only, with a hand adjoint passing gradient cosine ≥ 0.999 against Deepwave.

**Bottom line.** For long time-stepping and solver gradients, the hand-written Metal adjoint is 1.6–19.5× faster at the large size. On the one production-scale case measured, Marmousi2 FWI at 140×680 with 2001 steps and 8 shots, it is 9.6× faster than MLX autodiff and 7.2× faster than single-thread Deepwave.

MLX autodiff is the quicker way to a correct gradient: 1.6–4× less code, and it passes every gate the hand adjoint passes. It also wins whenever the problem is small enough that a per-call process launch matters.

## Biggest surprises

1. **Independent review of the tests caught judge defects in 31 of 50 projects before lock.** Projects 18, 24, 43 and 50 needed two or three rounds of fixes, and 35 was reopened after lock. Typical catches:
   - **Infeasible float32 goldens:**
     - 05: an infinite slab on a ±1e7 m polygon.
     - 04: Harmonica's numba kernel cast its inputs to float32, contaminating the goldens by up to 4.4e-3.
     - 11: a 1e-4 solve parity that the judge's own float32 solve fails, since cond ≈ 1.2e4.
     - 35: an FD step of 1e-3 that fails a correct float32 gradient.
   - **Specs that existed only as reference-library internals:**
     - 13 and 16: the C-PML was specified as "match Deepwave".
     - 24: the order-2 FMM rules lived inside compiled scikit-fmm.
     - 50: the scheme was defined by Deepwave internals.
   - **Tests that could not fail a wrong answer:**
     - 21: nothing failed an implementation that never checkpointed.
     - 41: the regularisation was untested.
     - 42: the f-k cutoff was pinned only loosely.
     - 47: the 5% age gate let wrong inversions pass.
     - 46: a sign error in the broken-plate boundary condition.
   - **Scope defects:** 37 and 38 were Metal-only projects whose first test suites required an MLX path.
2. **The Deepwave PML erratum (project 14 SCHEME erratum).** Project 14's written `tests/SCHEME.md` says non-PML nodes use plain D2 and ignore `db`. Deepwave actually applies the full PML expression everywhere, so the `db·D1` and `D1(a·psi)` terms survive in the first accuracy/2 model nodes next to each PML. Review of project 18 found it, when 14's scheme missed 18's goldens by 1.56e-3 and 3.03e-3. On 14's own goldens either form matches (gap ~1.8e-6), so 14's tests could not see it. An erratum was issued: implement the full expression everywhere, as 13's SCHEME.md already does. Downstream projects 16, 17, 18, 19, 21, 23 and 50 depend on that behaviour.
3. **Float32 limits were real, and each was handled by changing the formula, not by adding precision.**
   - 13: the textbook d₀ form of D2 fails float32 at 4.0e-4; the difference form passes.
   - 04: literal tesseroid formulas fail; stable haversine forms pass.
   - 28: the textbook absolute-T eikonal root failed 8 of 15 finite-difference probes; the stable root became part of the contract.
   - 32: tanh overflowed at kh ≈ 31(1+i); guards were added.
   - 37: a float32 running prefix sum misses 1e-4; Metal uses a tree of positive sums stored as mantissa plus exponent.
   - 02, 03, 06: far-field cancellation gets a documented distance-dependent tolerance.
4. **The precision policy (adopted 2026-10-28) grew out of project 35.** The policy: bulk compute is float32 on the GPU; a small step may run in float64 on the CPU only where float32 is shown to lose accuracy, isolated, commented with the reason and visible in the benchmarks. The precipitating bug: `mx.matmul` on the GPU can use reduced-precision (TF32-style) arithmetic even for float32 inputs. That corrupted the solver's residual corrections and its VJP. The fix replaced every matmul with explicit float32 products. Documented, isolated CPU float64 steps that comply with the policy:
   - 08: q0, series coefficients and an equatorial root, computed once on the host.
   - 21: PML profile setup.
   - 22: the Courant factor, only for out-of-range inputs.
   - 42: a phase table reaching about 1500 rad.

   Steps that do not comply, or are not documented, are listed under Open issues.
5. **MLX linear algebra runs on the CPU.** MLX has no GPU QR, Cholesky or triangular solve, so the solve step in 09, 10, 11, 27, 31 and 45 runs in float32 on MLX's CPU stream. After 45's precision-policy fixes, only its `qr` and `solve_triangular` remain there, documented as an exception. This is not float64, but it is not GPU either.
6. **The fast-marching projects were slower on both GPU paths.** Fast marching is serial. On 24 at large, MLX now runs at 0.0133× scikit-fmm (host heap ordering in float32, with recompute on the GPU) and Metal at 0.0147×. The GPU-friendly variant, 25's red-black fast sweeping on Metal, ran 3.96× faster than the judge at 128³.

## Documented deviations (approved, recorded in the locked files)

- **43 phase picker, F1 gates** (approved 2026-10-28). The bounded, sealed MLX GPU reference reached test F1 of P 0.888–0.893 and S 0.749–0.754 on STEAD analyst picks. The gates are therefore the worst seed minus 0.02: P 0.868, S 0.729, micro 0.801. The median-error gates are unchanged.
- **08 normal gravity, Boule u-component.** Boule returns γ_u, the ellipsoidal u-component, not |∇U|. The two differ by 9.7e-3 at 20,000 km. The API documents this, and the test matches Boule.
- **42 f-k/Radon scope.** Radon demultiple is a single known-slowness projection, not a full Radon transform inversion.
- **16 FWI: tests on a synthetic stand-in, benchmark and figure on Marmousi2.**
  - The locked tests and the 0.70 recovery criterion still use the locked synthetic stand-in, as does the small benchmark size (48×64).
  - The medium and large benchmarks and the figure use Marmousi2 (AGL Elastic Marmousi; Martin, Wiley and Marfurt 2006; Copyright 2004 Allied Geophysical Laboratory, University of Houston). The license is CC BY 4.0, with attribution in `data/marmousi2/README.md`.
  - Medium is 70×340 at 50 m; large is 140×680 at 25 m.
  - Medium differences against Deepwave are MLX 1.22e-5 and Metal 5.27e-5. Both are within the 1e-4 parity used by the tests, but this is a benchmark sanity metric, not a locked test.
  - The figure shows 3 Metal FWI iterations with J/J₀ = 0.576, not convergence. No Marmousi2 model-error number is recorded.
  - Classic IFP Marmousi (`data/marmousi`) is unused.
- **06 terrain/Bouguer, judge revised and re-locked.** A benchmark-regime fixture at 700, 1400, 3000 and 8000 m was added, with tolerances unchanged. Metal now uses a 6×6 Gauss–Legendre distant-prism scheme with an exact vertical integral. Medium differences are MLX 2.62e-7 and Metal 2.40e-7.
- **49 tsunami, judge revised and re-locked.** Deep-water fixtures at 1000 and 4000 m were added, with limits of 1e-4 against the reference and 2e-4 between paths. The 4000 m margin is about 1.65×, set by float32 rounding of the returned depth. Both paths now evolve the perturbation from a resting depth, with the hydrostatic term removed from the momentum flux. Medium differences are MLX 8.69e-6 and Metal 6.58e-6.
- **14 SCHEME erratum.** See Surprise 2.
- **18 LSRTM width ruling** (2026-10-28). The official measure is the vertical FWHM of the analytic-signal envelope. Measured ratios were 0.492–0.727 against a 0.8 gate.
- **FD convergence ruling** (2026-10-28). For 12, 13, 14, 15, 22, 36, 48 and 49, the expected order under joint dx and dt refinement is the combined space-time order, 2. Measured orders were 2.01–2.08.
- **Documented precision exception.** 45 keeps `qr` and `solve_triangular` in float32 on MLX's CPU stream, because MLX has no GPU kernels for them.
- **Other reviewed exceptions:**
  - 11: model parity 3e-3 for conditioning reasons, with data parity kept at 1e-4.
  - 24: order-2 FMM need only match scikit-fmm within 5e-2.
  - 25: convergence window 0.65–1.35, because the oracle itself converges at 0.67–0.72.
  - 44: printed-precision Okada gates.
  - 49: dam-break order ≥ 0.45 for the shock.
  - 29: stricter per-source median, plus a p95 guard.

## Open issues

Already closed: the cross-check sweep, 19's CPU-float64 path (now float32 on the GPU), 06 Metal and 49 parity (now 2.4e-7 and 8.7e-6 / 6.6e-6), and 16's missing Marmousi benchmark (now on Marmousi2). 24, 33, 45 and 46 now also comply with the precision policy. See [CROSS_CHECKS.md](CROSS_CHECKS.md) for the cross-check runs, including the follow-up for 06, 17, 34, 46 and 49.

1. **Precision policy: resolved 2026-10-29.** 34 and 35 had prepared geometry, wavenumbers and Robin factors in host float64. Float32 was not shown to lose accuracy there (it changes the Robin coefficients by at most 3.31e-7 and 2.17e-7 relative), so both now run that setup in float32 and all their MLX tests pass. The only float64 left is SciPy's L-BFGS-B driver in 35 (0.134% of inversion time), which is double precision by construction and documented as the exception in `projects/35-dc-inversion/RESULTS.md`.
2. **33 parity of 1.69e-4 is explained.** It is a componentwise difference with a 1e-8 floor, measured against a finite-difference judge gradient, under a 1e-3 gradient gate.
3. **50 wave sandbox:** only the headless core exists. There is no window, no mouse velocity painting and no live FWI toggle, so the planned hand check of the interactive app cannot happen. The headless benchmark ran 360 frames on 1000×1000 in 112 ms, including launch. The ≥ 60 steps/s test passed, but its measured rate is not recorded.
4. **Empty `mlx/` directories** in Metal-only projects 22, 25, 37, 38 and 50. 22's specification promised "MLX as a reference", and 6 MLX tests were counted at lock, but no MLX reference was built.
5. **Results not archived.** The test runs record pass/fail only, so these values were never written to disk:
   - per-path measured errors;
   - 48's per-path Nusselt number and v_rms;
   - 43's F1 and median pick error;
   - 16's Marmousi2 model error;
   - 21's memory-vs-recompute curve, which lives only in pytest's temp directory;
   - the per-distance error curves that review of 02, 03, 04 and 06 asked for.
   - All 50 `bench/results.json` files and 50 figures are on disk (in each project folder); the values listed above are not, and remain open.
6. **Benchmark scope gaps:**
   - 29: evaluation only.
   - 30: grid search only.
   - 33: objective and gradient only; the new optimizer is not timed.
   - 35: forward only.
   - 43: loss+grad only; no training or inference.
   - 04: sizes too small to mean anything (≤256 observations).
   - 44: MLX and Metal are flat at about 40 ms from 512 to 131,072 observations. Launch and graph-build overhead dominate; the work is materialised inside the timer.
7. **Stale benchmark notes.** Many `results.json` `notes` fields still say "estimate" or "unverified". 19's notes still say "MLX uses CPU float64": the string is hard-coded in `run_bench.py`, though the source is now float32 on the GPU.
   - **19 resolved in RESULTS.md (2026-10-28).** "Benchmark provenance" in `projects/19-multiscale-fwi/RESULTS.md` ties the current numbers to the float32 GPU code, quotes the superseded CPU-float64 run, and flags the stale `notes` string. The string itself changes only if `bench/run_bench.py` is edited. The generic "estimate"/"unverified" wording in other projects is untouched.
## Contact sheet

All 50 figures, generated by `tools/make_contact_sheet.py` (Pillow, 5 × 10 grid, each figure letterboxed to 16:9 and labelled with project number and short name).

![contact sheet](contact_sheet.png)
