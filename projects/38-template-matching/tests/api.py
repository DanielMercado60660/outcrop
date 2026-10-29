"""Shared implementation API."""


def template_match(data, templates, threshold):
    """Return valid zero-mean normalized correlation and threshold detections.

    data: float32 array (S,N), finite station traces, arbitrary amplitude units.
    templates: float32 array (T,S,M), finite template traces in the same units;
        each template has a waveform per station, with 2 <= M <= N.
    threshold: dimensionless Python float in [-1,1], inclusive.

    Return a dict containing float32 arrays correlation and detections, each
    (T,S,N-M+1). correlation[t,s,k] is the Pearson coefficient of template
    [t,s,:] and data[s,k:k+M], both locally demeaned. Zero variance in either
    produces zero. k is the zero-based starting sample. There is no padding or
    station stacking. detections is 1 where correlation >= threshold, else 0;
    every qualifying window is reported, with no peak picking.
    """
    raise NotImplementedError
