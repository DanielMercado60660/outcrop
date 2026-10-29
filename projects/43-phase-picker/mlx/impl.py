"""PhaseNet-style MLX picker trained from the supplied schedule."""

import hashlib
import itertools
import math
from functools import partial

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

from mlx import nn

_model = None


def _normalize(x):
    x = x - mx.mean(x, axis=1, keepdims=True)
    rms = mx.sqrt(mx.mean(mx.square(x), axis=(1, 2), keepdims=True))
    return mx.clip(x / mx.maximum(rms, 1e-8), -10.0, 10.0)


def _interpolate(x, length):
    old = x.shape[1]
    coord = (mx.arange(length, dtype=mx.float32) + 0.5) * (old / length) - 0.5
    coord = mx.clip(coord, 0, old - 1)
    lower = mx.floor(coord).astype(mx.int32)
    upper = mx.minimum(lower + 1, old - 1)
    fraction = (coord - lower.astype(mx.float32))[None, :, None]
    return mx.take(x, lower, axis=1) * (1.0 - fraction) + mx.take(x, upper, axis=1) * fraction


class _Picker(nn.Module):
    def __init__(self, seed):
        super().__init__()
        widths = [16, 32, 64, 128, 192]
        self.stem = nn.Conv1d(3, 16, 7, padding=3)
        self.down = [nn.Conv1d(a, b, 7, stride=4, padding=3)
                     for a, b in itertools.pairwise(widths)]
        self.up = [nn.Conv1d(widths[i] + widths[i - 1], widths[i - 1], 7, padding=3)
                   for i in range(4, 0, -1)]
        self.head = nn.Conv1d(16, 3, 1)
        mx.random.seed(seed)
        for layer in [self.stem, *self.down, *self.up, self.head]:
            bound = 1.0 / math.sqrt(layer.weight.shape[1] * layer.weight.shape[2])
            layer.weight = mx.random.uniform(-bound, bound, shape=layer.weight.shape, dtype=mx.float32)
            layer.bias = mx.random.uniform(-bound, bound, shape=layer.bias.shape, dtype=mx.float32)

    def __call__(self, x):
        x = nn.relu(self.stem(x))
        skips = [x]
        for layer in self.down:
            x = nn.relu(layer(x))
            skips.append(x)
        for layer, skip in zip(self.up, reversed(skips[:-1])):
            x = _interpolate(x, skip.shape[1])
            x = nn.relu(layer(mx.concatenate([x, skip], axis=-1)))
        return self.head(x)


def _loss(logits, targets):
    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    return -mx.mean(mx.sum(targets * log_probs, axis=-1))


def phase_loss_and_grad(logits, targets):
    x = mx.array(logits, dtype=mx.float32)
    y = mx.array(targets, dtype=mx.float32)
    loss, gradient = mx.value_and_grad(_loss)(x, y)
    return {"loss": np.asarray(loss), "gradient": np.asarray(gradient)}


def _augment(x, labels, shift, polarity, angle, sigma):
    length = x.shape[1]
    positions = mx.arange(length, dtype=mx.int32)[None, :]
    source = positions - shift[:, None]
    valid = (source >= 0) & (source < length)
    source = mx.clip(source, 0, length - 1)
    x = mx.take_along_axis(x, source[:, :, None], axis=1)
    x = mx.where(valid[:, :, None], x, 0.0) * polarity[:, None, None]
    c = mx.cos(angle)[:, None]
    s = mx.sin(angle)[:, None]
    z, north, east = x[:, :, 0], x[:, :, 1], x[:, :, 2]
    x = mx.stack([z, c * north - s * east, s * north + c * east], axis=-1)
    x = _normalize(x)

    shifted = labels + shift[:, None]
    present = (labels >= 0) & (shifted >= 0) & (shifted < length)
    delta = (positions[:, :, None] - shifted[:, None, :]).astype(mx.float32) / sigma
    phase = mx.where(present[:, None, :], mx.exp(-0.5 * mx.square(delta)), 0.0)
    phase = phase / mx.maximum(1.0, mx.sum(phase, axis=-1, keepdims=True))
    noise = 1.0 - mx.sum(phase, axis=-1, keepdims=True)
    return x, mx.concatenate([noise, phase], axis=-1)


def train_phase_picker(waveforms, labels, batch_indices, shifts, polarities,
                       rotation_angles, learning_rates, sample_interval_s, seed,
                       label_sigma_samples):
    global _model
    mx.set_default_device(mx.gpu)
    model = _Picker(seed)
    optimizer = optim.Adam(learning_rate=0.001, betas=[0.9, 0.999], eps=1e-8,
                           bias_correction=True)
    optimizer.init(model.trainable_parameters())
    x_all = mx.array(waveforms, dtype=mx.float32)
    y_all = mx.array(labels, dtype=mx.int32)
    indices = mx.array(batch_indices, dtype=mx.int32)
    shift_all = mx.array(shifts, dtype=mx.int32)
    polarity_all = mx.array(polarities, dtype=mx.float32)
    angle_all = mx.array(rotation_angles, dtype=mx.float32)
    rates = mx.array(learning_rates, dtype=mx.float32)
    mx.eval(model.parameters(), optimizer.state, x_all, y_all, indices, shift_all,
            polarity_all, angle_all, rates)

    def objective(net, x, y):
        return _loss(net(x), y)

    value_and_grad = nn.value_and_grad(model, objective)
    state = [model.state, optimizer.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(x, y, shift, polarity, angle, rate):
        x, targets = _augment(x, y, shift, polarity, angle, label_sigma_samples)
        loss, grads = value_and_grad(model, x, targets)
        optimizer.learning_rate = rate
        optimizer.update(model, grads)
        return loss

    for i in range(batch_indices.shape[0]):
        batch = indices[i]
        loss = step(x_all[batch], y_all[batch], shift_all[i], polarity_all[i],
                    angle_all[i], rates[i])
        mx.eval(state, loss)

    _model = model
    digest = hashlib.sha256()
    for name, value in sorted(tree_flatten(model.parameters())):
        digest.update(name.encode("utf-8"))
        digest.update(np.asarray(value).tobytes())
    print(f"phase picker final weights sha256 (sorted names and float32 bytes): {digest.hexdigest()}")
    return {"steps": np.array(batch_indices.shape[0], dtype=np.int32),
            "loss": np.asarray(loss, dtype=np.float32)}


def predict_phases(waveforms, sample_interval_s):
    if _model is None:
        raise RuntimeError("train_phase_picker must be called before predict_phases")
    x = mx.array(waveforms, dtype=mx.float32)
    probabilities = mx.softmax(_model(_normalize(x)), axis=-1)
    return {"probabilities": np.asarray(probabilities, dtype=np.float32)}
