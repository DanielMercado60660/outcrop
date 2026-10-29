"""Frozen array API for causal spiking and predictive deconvolution."""


def spiking_deconvolution(trace, wavelet, inverse_length, stabilization):
    """Return a causal least-squares wavelet inverse and deconvolved trace.

    trace: float32 (N,), dimensionless sampled seismic trace, time increasing.
    wavelet: float32 (M,), dimensionless causal wavelet, sample zero at index 0.
    inverse_length: int, number L of causal inverse filter coefficients.
    stabilization: float, dimensionless nonnegative diagonal loading lambda.
    Let A be the (M+L-1, L) full-convolution matrix of wavelet, and e0
    the impulse at index 0. Solve (A.T A + lambda I) h = A.T e0.
    Return dict: inverse, float32 (L,), h; recovered, float32 (N,),
    first N samples of full causal convolution trace * h. No normalization,
    padding beyond zero extension, or time shift is applied.
    """
    raise NotImplementedError


def predictive_deconvolution(trace, order, lag, stabilization):
    """Return causal prediction coefficients and prediction-error trace.

    trace: float32 (N,), dimensionless samples in increasing time order.
    order: int P, number of predictor coefficients; lag: int >= 1 samples.
    stabilization: float, dimensionless relative zero-lag diagonal loading.
    Define unnormalized biased autocorrelation r[k] = sum x[t] x[t+k]
    for t=0,...,N-k-1, with x zero outside the trace. Solve the Toeplitz
    system (r[abs(i-j)] + stabilization*r[0]*delta[i,j]) c[j] =
    r[lag+i], i,j=0,...,P-1. Return dict: coefficients, float32 (P,),
    c; residual, float32 (N,), x[t] - sum_j c[j]*x[t-lag-j], with
    out-of-range samples zero. The error filter has leading coefficient 1
    and coefficients -c at delays lag+j.
    """
    raise NotImplementedError
