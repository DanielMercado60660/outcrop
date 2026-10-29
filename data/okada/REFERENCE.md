# Okada reference values (test oracle)

Machine-readable values: `test_values.json`, in this directory. Each value is stored twice: as the printed string and as a float.

## 1. Okada (1985), BSSA 75(4) 1135–1154, Table 2 "Checklist for numerical calculations" (p. 1149). Status: complete

- **What it is:** surface (z = 0) values of displacement (ux, uy, uz) and six horizontal derivatives (∂ux/∂x, ∂ux/∂y, ∂uy/∂x, ∂uy/∂y, ∂uz/∂x, ∂uz/∂y). There are 4 geometries × 3 dislocation types (strike, dip, tensile), so 12 rows and 108 numbers.
- **Geometries (from the table footnotes):**
  - Case 1: point source; x=2, y=3, d=4, δ=70°
  - Case 2: finite source; x=2, y=3, d=4, δ=70°, L=3, W=2
  - Case 3: finite source; x=0, y=0, d=4, δ=90°, L=3, W=2
  - Case 4: finite source; x=0, y=0, d=4, δ=**−90°**, L=3, W=2
- **Medium:** λ = μ (ν = 0.25; DC3D α = 2/3).
- **Units:** lengths are dimensionless. Case 1 is given per unit potency UΔΣ. Cases 2–4 are given per unit dislocation U, so displacements are in U and derivatives in U per length.
- **Precision:** values are printed to 4 significant figures. Every printed `0` is an exact symmetry zero.

### Conventions (paper pp. 1138, 1143)

- **Axes:** right-handed x, y, z. The medium occupies z ≤ 0 and z points up. x runs parallel to strike.
- **Fault position:** fault point (ξ′, η′), with ξ′ ∈ [0, L] and η′ ∈ [0, W], lies at (ξ′, η′ cosδ, −d + η′ sinδ).
- **Origin and d:** the origin corner is (0, 0, −d). η′ runs **up-dip**, so d is the depth of the **deep (bottom) edge** when sinδ > 0. It is not the top-edge depth and not the centroid depth.
- **Point source:** located at (0, 0, −d).
- **Slip components:** U1, U2 and U3 are the motion of the hanging wall relative to the footwall. For 0 < δ < 90°, +U1 is left-lateral, +U2 is reverse and +U3 is opening. The paper notes that U2 becomes normal-sense when sin2δ < 0.
- **Derivatives:** the derivative columns are raw ∂u_i/∂x_j in this frame.

### Sources and agreement

- **Primary:** the scanned paper on Okada's institute site, read page by page from the page image: https://www.bosai.go.jp/e/pdf/Okada_1985_BSSA.pdf
- **Second source, cases 2–4 (81 values):** F. Beauducel, `okada85_checklist.m` (BSD, MATLAB File Exchange #25982): https://github.com/IPGP/deformation-lib/blob/master/okada/okada85_checklist.m. Identical copies are in `feigl/gipht` and `abarbour/deform`.
- **Third source, cases 1–4 (108 values):** torchdeform tests (Apache-2.0): https://github.com/robertpop99/torchdeform/blob/main/tests/sources/test_okada_source.py
- **Result:** the transcription matches both secondary sources exactly when compared by script. There are 0 mismatches.

**Confidence: very high.**

## 2. Okada (1992), BSSA 82(2) 1018–1040, doi:10.1785/BSSA0820021018. Status: there is no checklist table

- **No check values in the paper:** the 1992 paper has **no table of numerical check values**.
  - Its Table 2 holds the closed-form expressions for point-source internal displacement. Tables 2–9 are all formula tables.
  - The "Numerical results" section (p. 1037) contains only figures.
  - Paper checked at https://www.bosai.go.jp/e/pdf/Okada_1992_BSSA.pdf
- **The only check value Okada published for DC3D** is in the NIED *Manual for subroutines DC3D0 and DC3D* (2012), Appendix-2: https://www.bosai.go.jp/e/pdf/DC3Dmanual.pdf
  - Inputs: ALPHA=2./3., X=10, Y=20, Z=−30, DEPTH=50, DIP=70, AL1=−80, AL2=120, AW1=−30, AW2=25, DISL1=200, DISL2=−150, DISL3=100.
  - Answer: **UX, UY, UZ = −37.8981, 63.1789, 14.9607**. The manual gives displacements only, with no derivatives.
  - The same numbers appear in `hydrocoast/DC3D.f90` `test/call.f90` (MIT). That repo copies the manual, so it is not an independent computation.
  - **Confidence:** high that these are the published numbers. Because the run was single precision, use a tolerance of about 1e-4 absolute.
- **DC3D conventions** (manual; https://www.bosai.go.jp/e/dc3d.html):
  - **Frame:** the same frame as in 1985. Z must be ≤ 0; otherwise IRET=2.
  - **Fault position:** the fault origin is (0, 0, −DEPTH). The fault spans [AL1, AL2] along strike and [AW1, AW2] **up-dip** from that origin.
  - **Slip and medium:** DISL1, DISL2 and DISL3 have the same senses as U1, U2 and U3. ALPHA = (λ+μ)/(λ+2μ).
  - **Outputs:** the derivatives are UXX, UYX, UZX (∂/∂x), UXY, UYY, UZY (∂/∂y) and UXZ, UYZ, UZZ (∂/∂z). In UXY the first letter is the displacement component and the second is the derivative direction.
  - **Precision:** the interface is REAL*4.
- **Link between the two tables (derived, not printed by Okada):**
  - The 1985 Table 2 is the z = 0 special case of DC3D and DC3D0.
  - Finite cases: `DC3D(ALPHA=2/3, X=x, Y=y, Z=0, DEPTH=d, DIP=δ, AL1=0, AL2=L, AW1=0, AW2=W, DISL=U)`.
  - Case 1: `DC3D0(ALPHA=2/3, X=2, Y=3, Z=0, DEPTH=4, DIP=70, POT_i=U_iΔΣ=1, POT4=0)`.
  - Each 1985 case in the JSON carries this call as `dc3d_equivalent_call`. The mapping was taken from the geometry definitions and was **not** run, because no Fortran or numpy was available.
- **Values at depth (z < 0):** no published derivative check values exist. torchdeform's `tests/sources/data/dc3d_golden.json` (Apache-2.0, 24 DC3D.f90 runs at depth, map frame with strike rotation) could be used as a third-party fixture. It was not copied here.

## Watch-outs for judge code

1. **The depth convention is the main trap.** Okada's d (and DEPTH with AW1=0, AW2=W) is the depth of the deep edge. Many codes, including okada85.m, take the centroid or top depth plus a centred fault. The two `okada85_checklist.m` copies in circulation differ (`d+sind(dip)*W/2` vs `d−sind(dip)*W/2`) because okada85.m changed its own depth argument. That is not a discrepancy in the paper, so do not borrow either convention.
2. **Do not copy Beauducel's minus signs.** His `varok` negates uEE, uEN, uNE and uNN to undo okada85.m's output convention. The table values are raw ∂u_i/∂x_j.
3. **Cases 3 and 4 exercise the singularity rules (p. 1148).**
   - In both cases q ≈ 0, and x = 0 puts ξ = 0 on a Chinnery corner.
   - Case 4 also has sinδ < 0, which makes R+η = 0 at the ξ=0 corners. Rule (iii) applies there: drop the 1/(R+η) terms and use ln(R+η) → −ln(R−η).
   - A naive implementation gives NaN or Inf here. Tests should use "small", not `== 0`, as Okada (1992, p. 1037) also advises.
   - Case 2 is the clean case with no zeros.
4. **Case 4 is the only negative-dip test.** The paper passes δ = −90° directly. Beauducel emulates it with d=6, δ=+90°, rake 180° for strike-slip (dip-slip and tensile unchanged). Both reproduce the table, so the judge should state which input form it feeds.
5. **Tolerance:** the values carry 4 significant figures, so use a relative tolerance of about 5e-4 to 1e-3 plus a small absolute floor (about 1e-6) for the exact zeros. Suggested rule: |got − want| ≤ 0.5·10^(e−3) + 1e-9, where e is the printed exponent.
6. **Case 1 needs point-source formulas** (DC3D0 / Okada 1985 eqs. for a point source). A finite-fault implementation alone cannot reproduce it.
