"""Locked numerical criteria, all dimensionless relative errors."""

# Project-wide rule: 1e-4 against the reference on the identical mesh.
REFERENCE_REL = 1e-4
# The 1% project criterion allows finite-volume and Fourier quadrature error.
ANALYTIC_REL = 1e-2
# Two independently rounded float32 implementations should agree at 1e-4.
CROSS_PATH_REL = 1e-4
