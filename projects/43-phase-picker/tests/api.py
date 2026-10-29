"""MLX-only implementation contract for project 43.

No implementation or reference imports belong here. Tests train once in process,
then reuse the resulting weights for all inference. No shipped trained weights,
external data, validation/test labels or golden reads inside the implementation.
Disclose architecture, initialization, training seed and final weight hash in
RESULTS.md. The black-box API cannot establish architecture/autodiff provenance.
The test gates were calibrated on MLX GPU reference runs; see tests/tolerances.py
and RESULTS.md.
"""


def train_phase_picker(waveforms, labels, batch_indices, shifts, polarities,
                       rotation_angles, learning_rates, sample_interval_s, seed,
                       label_sigma_samples):
    """Initialize and train one PhaseNet-style 1-D U-Net during the test run.

    Args:
        waveforms: float32 NumPy array (30000, 3000, 3), raw counts, component
            order Z,N,E. Time axis 1 increases from the window start, sampled
            every 0.01 s. No filtering, demeaning or normalization has occurred.
        labels: int32 array (30000, 2), P,S analyst sample indices; -1 means
            absent. These are TRAIN labels, never validation or test labels.
        batch_indices: int32 array (36000, 32), indices into training records.
            Run exactly 36000 optimizer updates, using these 32-record batches
            in row order. Sampling is with replacement, fixed by the judge.
        shifts: int32 array (36000, 32), signed sample translations. Translate
            each selected waveform right by shift (left if negative), filling
            vacated samples with zero, never circular wrapping. Add shift to
            each present label; set labels outside [0,3000) to -1. Absent
            labels remain -1. Apply translation BEFORE normalization.
        polarities: float32 array (36000, 32), each -1 or +1; multiply all three
            components of each translated trace by the given sign.
        rotation_angles: float32 array (36000, 32), radians in [-pi,pi]. After
            translation and polarity, rotate horizontal components with
            N' = cos(angle)*N - sin(angle)*E and
            E' = sin(angle)*N + cos(angle)*E; Z is unchanged. Apply all these
            transformations BEFORE normalization. Labels do not rotate.
        sample_interval_s: Python float 0.01, seconds/sample.
        seed: Python int 43043; reset model initialization and framework RNG
            to this seed each call. Never load trained weights. Training must
            be deterministic on a fixed device/software stack.
        learning_rates: float32 NumPy array (36000,), dimensionless Adam step
            sizes, one per update. Consume the saved values exactly in row
            order: 1000-step linear warmup 1e-6 to .001, then 35000-step
            cosine decay .001 to 1e-5, inclusive endpoints in each segment.
            Use standard bias-corrected Adam (Kingma & Ba), beta1=.9,
            beta2=.999, eps=1e-8 OUTSIDE sqrt, with initially zero moments:
            m_t=.9*m_(t-1)+.1*g_t; v_t=.999*v_(t-1)+.001*g_t**2;
            mhat=m_t/(1-.9**t); vhat=v_t/(1-.999**t);
            w_t=w_(t-1)-learning_rates[t-1]*mhat/(sqrt(vhat)+1e-8).
            t starts at 1 and never resets during the run. MLX 0.32.4 needs
            optim.Adam(..., bias_correction=True) explicitly; its default is
            False. No weight decay, AMSGrad, gradient clipping, gradient
            accumulation, loss scaling, extra updates, early stopping or
            model selection. Update all weights AND biases every step.
        label_sigma_samples: Python float 10.0, dimensionless samples. At
            integer sample t, form P/S targets exp(-.5*((t-pick)/sigma)**2),
            zero for absent labels; divide both by max(1,P+S), then set noise
            to 1-P-S. Train with mean unweighted soft-label CE over batch and
            time, as defined by phase_loss_and_grad (no class weights).

    Required preprocessing for BOTH training and inference: subtract each
    component's temporal mean; divide all components by the trace's joint RMS
    sqrt(mean(x*x)) floored at 1e-8 counts; clip to [-10,10]. Use float32.
    Required architecture: 1-D U-Net, widths [16,32,64,128,192], 647971 trainable
    scalars. Stem Conv1d(3,16,k=7,stride=1,pad=3), then four Conv1d down layers
    between successive widths (k=7,stride=4,pad=3). ReLU after each. Save the
    stem and all down outputs as skips. Decode deepest to shallowest: linearly
    interpolate to the next skip length, concatenate [upsampled,skip] on the
    channel axis, Conv1d(sum of channels,skip width,k=7,stride=1,pad=3), ReLU.
    Interpolation: input coordinate (j+.5)*old_length/new_length-.5, clamped to
    [0,old_length-1], then linear interpolation of adjacent samples (half-pixel,
    align_corners=False). Head Conv1d(16,3,k=1,stride=1,pad=0), then softmax on
    noise,P,S channels at inference; CE uses the head's logits during training.
    All convolutions are cross-correlations, groups=1,dilation=1, with bias and
    symmetric zero padding. No pooling, batch normalization, dropout, residual
    blocks or unlisted layers. Encoder lengths: 3000,750,188,47,12 samples.

    Required initialization, including head: independently draw EVERY weight
    AND bias uniformly in [-1/sqrt(fan_in),+1/sqrt(fan_in)), where fan_in is
    input_channels*kernel_size. This is the PyTorch Conv1d default distribution
    (Kaiming uniform with a=sqrt(5)), including nonzero uniform biases. For the
    tested MLX path, construct layers, then reset mx.random.seed(seed); overwrite
    defaults in order stem, down shallow-to-deep, up deep-to-shallow, head, with
    one float32 mx.random.uniform call for weight then one for bias per layer.
    Weight layout is (out_channels,kernel_size,in_channels), bias (out_channels,).
    Draw no further randomness; consume only the supplied schedule. Use float32
    parameters, moments, activations, targets and gradients; network/Adam work
    runs on the GPU. No CPU float64 training or Neural Engine use.

    The historical 6000-step smaller reference took 62-66 s in an independent
    measurement on an M5 Max MLX GPU, not 30-40 minutes. This strengthened
    candidate must be timed on the GPU before lock; it has no measured
    runtime/accuracy yet. Reference
    calibration has a 1800 s training cap; tests record actual implementation
    runtime but do not impose a hardware-independent performance gate.

    Returns:
        dict with 'steps': scalar shape () int32 NumPy array, value 36000;
        'loss': scalar shape () float32 NumPy array, finite final pre-update
        training CE, dimensionless. Retain only the final model in the imported
        module for predict_phases calls in the same process. Recalling this
        function resets/retrains; inference never updates the model or reads
        data files. Validation/test data are not arguments to this function.
    """
    raise NotImplementedError


def predict_phases(waveforms, sample_interval_s):
    """Return per-sample probabilities from the frozen phase picker.

    Args:
        waveforms: float32 NumPy array (batch, 3000, 3), components Z,N,E.
            Finite raw counts, not demeaned or normalized; preprocess exactly
            as in train_phase_picker. Time increases on
            axis 1; sample zero is the start of each record. Sampling interval
            is constant within a call. Normalize internally as specified above.
            Support arbitrary positive batch
            sizes, including 1 and 32. Do not modify the caller's input.
        sample_interval_s: positive Python float, seconds per sample (0.01 in
            these tests). It is supplied explicitly, never inferred from labels.

    Returns:
        dict with 'probabilities': float32 NumPy array (batch, samples, 3),
        channel order noise,P,S. Finite values in [0,1], summing to one on
        the final axis, aligned to input samples with no time origin shift.
        No picks or labels are supplied to the model. Tests decode each P/S
        channel at the fixed operating point in tolerances.py, without truth
        windows. Inference is deterministic, independent of batch composition,
        and uses the final weights from train_phase_picker for all calls.
    """
    raise NotImplementedError


def phase_loss_and_grad(logits, targets):
    """Mean soft-label categorical cross entropy and derivative w.r.t. logits.

    Args:
        logits: finite float32 NumPy array (batch, samples, 3), dimensionless
            unnormalized scores, channel order noise,P,S.
        targets: float32 NumPy array of the same shape, dimensionless,
            nonnegative and summing to one per sample. No class weights,
            masking, clipping, label smoothing, or extra regularization here.

    Returns:
        dict with 'loss': scalar (shape ()) float32 NumPy array, dimensionless,
        equal to -sum(targets * log_softmax(logits))/(batch*samples), and
        'gradient': float32 NumPy array of logits.shape, derivative of that
        scalar with respect to every logit. Compute a stable log-softmax;
        gradients are through logits, not through argmax/threshold picks.
        MLX must obtain the derivative via autodiff. This is a primitive for
        training a U-Net, not a claim that the whole network has been checked.
    """
    raise NotImplementedError
