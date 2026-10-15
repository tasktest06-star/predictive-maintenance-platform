"""A 1-D CNN with explicit forward and backward passes, in NumPy.

There is no torch in the environment this platform is developed in, and
``pdmcore.models.anomaly.DeepSVDDDetector`` already establishes the precedent:
a small network written out by hand, with the gradient derived rather than
inherited, so that every term in it is inspectable.

Scope. This is enough machinery for the WDCNN/TICNN family and no more:
Conv1d, BatchNorm1d, ReLU, MaxPool1d, Dropout, Flatten, Dense, softmax
cross-entropy, Adam. It is not a framework and does not try to be.

Correctness. :func:`gradcheck` compares every parameter gradient against a
central finite difference, and ``prototype/tests/test_supervised_cnn.py`` runs it
on the full WDCNN stack. A hand-derived backward pass that is never
gradient-checked is a plausible-looking number generator -- the model will still
train, slightly wrongly, and nothing will say so.

Convention: activations are ``(N, C, L)``, matching the published
architectures' channel-first layout.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "Layer", "Conv1d", "BatchNorm1d", "ReLU", "MaxPool1d", "Dropout", "Flatten",
    "Dense", "Sequential", "Adam", "softmax_cross_entropy", "gradcheck", "DTYPE",
    "set_dtype",
]

#: Working precision. float32 throughout, because on this host it is measured at
#: 2.0x the throughput of float64 on exactly the skinny matmuls this model is made
#: of (0.71 vs 0.35 GFLOP/s on the (8192,64)' @ (8192,16) product that dominates
#: the first convolution's weight gradient). :func:`set_dtype` switches to
#: float64 for :func:`gradcheck`, where a central difference at eps=1e-6 needs
#: more than 7 significant digits to mean anything.
DTYPE = np.float32


def set_dtype(dt) -> None:
    global DTYPE
    DTYPE = dt


class Layer:
    """Base layer. ``params`` are learned; ``buffers`` travel but are not learned."""

    def __init__(self) -> None:
        self.params: Dict[str, np.ndarray] = {}
        self.grads: Dict[str, np.ndarray] = {}
        self.buffers: Dict[str, np.ndarray] = {}
        self.training = True

    def forward(self, x: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def backward(self, g: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    @property
    def n_params(self) -> int:
        return int(sum(p.size for p in self.params.values()))


def same_padding(length: int, k: int, stride: int) -> Tuple[int, int]:
    """Asymmetric 'same' padding: output length is exactly ``ceil(L / stride)``.

    Symmetric ``k // 2`` padding, which is what the PyTorch implementations in
    this repository's ``impl/*`` branches use, gives 129 output positions for
    WDCNN's first layer rather than the 128 the paper specifies. Harmless, but it
    means a reimplementation and the original do not have the same shapes, and
    shape drift is how two "identical" models end up with different parameter
    counts.
    """
    l_out = -(-length // stride)
    total = max(0, (l_out - 1) * stride + k - length)
    return total // 2, total - total // 2


def _im2col(x: np.ndarray, k: int, stride: int, pad: Tuple[int, int]) -> np.ndarray:
    """``(N, C, L)`` -> ``(N, L_out, C*k)`` patch matrix, via a strided view.

    ``as_strided`` is used rather than a Python loop because the loop dominates
    the runtime of the whole model: the first WDCNN layer alone has 128 output
    positions per sample.
    """
    n, c, length = x.shape
    if pad[0] or pad[1]:
        x = np.pad(x, ((0, 0), (0, 0), pad))
    lp = x.shape[2]
    l_out = (lp - k) // stride + 1
    if l_out <= 0:
        raise ValueError(f"kernel {k} / stride {stride} / pad {pad} leaves no output for L={length}")
    x = np.ascontiguousarray(x)
    sn, sc, sl = x.strides
    view = np.lib.stride_tricks.as_strided(
        x, shape=(n, c, l_out, k), strides=(sn, sc, sl * stride, sl), writeable=False
    )
    # (N, C, L_out, k) -> (N, L_out, C, k) -> (N, L_out, C*k)
    return view.transpose(0, 2, 1, 3).reshape(n, l_out, c * k)


def _col2im(cols: np.ndarray, shape: Tuple[int, int, int], k: int, stride: int,
            pad: Tuple[int, int]) -> np.ndarray:
    """Adjoint of :func:`_im2col`: scatter-add patch gradients back to the signal."""
    n, c, length = shape
    lp = length + pad[0] + pad[1]
    out = np.zeros((n, c, lp), dtype=cols.dtype)
    l_out = cols.shape[1]
    cols = cols.reshape(n, l_out, c, k)
    for j in range(k):
        # Every output position i contributes to input position stride*i + j, and
        # for a fixed tap j those are disjoint, so one vectorised strided add per
        # tap is exact and needs no accumulation guard.
        out[:, :, j : j + stride * l_out : stride] += cols[:, :, :, j].transpose(0, 2, 1)
    return out[:, :, pad[0] : pad[0] + length]


class Conv1d(Layer):
    """1-D convolution. ``padding='same'`` uses ``kernel // 2`` like the papers do."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int, stride: int = 1,
                 padding: str = "same", bias: bool = False, seed: int = 0):
        super().__init__()
        self.in_ch, self.out_ch, self.kernel, self.stride = in_ch, out_ch, kernel, stride
        if padding not in ("same", "valid"):
            raise ValueError("padding must be 'same' or 'valid'")
        self.padding = padding
        self.pad: Tuple[int, int] = (0, 0)
        rng = np.random.default_rng(seed)
        fan_in = in_ch * kernel
        self.params["W"] = (rng.standard_normal((fan_in, out_ch)) * np.sqrt(2.0 / fan_in)).astype(DTYPE)
        if bias:
            self.params["b"] = np.zeros(out_ch, dtype=DTYPE)
        self._cache: Optional[Tuple] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.padding == "same":
            self.pad = same_padding(x.shape[2], self.kernel, self.stride)
        cols = _im2col(x, self.kernel, self.stride, self.pad)
        out = cols @ self.params["W"]
        if "b" in self.params:
            out = out + self.params["b"]
        self._cache = (x.shape, cols)
        return out.transpose(0, 2, 1)

    def backward(self, g: np.ndarray) -> np.ndarray:
        shape, cols = self._cache
        gt = g.transpose(0, 2, 1)                      # (N, L_out, out_ch)
        n, l_out, _ = gt.shape
        self.grads["W"] = cols.reshape(-1, cols.shape[2]).T @ gt.reshape(-1, self.out_ch)
        if "b" in self.params:
            self.grads["b"] = gt.reshape(-1, self.out_ch).sum(axis=0)
        gcols = gt @ self.params["W"].T                 # (N, L_out, C*k)
        return _col2im(gcols, shape, self.kernel, self.stride, self.pad)


class BatchNorm1d(Layer):
    """Per-channel batch norm over ``(N, L)``, with running statistics.

    Present because every model in this family uses it, and because it is the one
    layer whose train/eval asymmetry silently changes a reported number: a model
    evaluated in training mode normalises the *test* batch by its own statistics,
    which leaks the test distribution into every prediction. :meth:`eval` is
    asserted in the test suite.
    """

    def __init__(self, ch: int, momentum: float = 0.1, eps: float = 1e-5):
        super().__init__()
        self.params["gamma"] = np.ones(ch, dtype=DTYPE)
        self.params["beta"] = np.zeros(ch, dtype=DTYPE)
        self.buffers["running_mean"] = np.zeros(ch, dtype=DTYPE)
        self.buffers["running_var"] = np.ones(ch, dtype=DTYPE)
        self.momentum, self.eps = momentum, eps
        self._cache: Optional[Tuple] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.training:
            mu = x.mean(axis=(0, 2))
            var = x.var(axis=(0, 2))
            self.buffers["running_mean"] = (1 - self.momentum) * self.buffers["running_mean"] + self.momentum * mu
            self.buffers["running_var"] = (1 - self.momentum) * self.buffers["running_var"] + self.momentum * var
        else:
            mu, var = self.buffers["running_mean"], self.buffers["running_var"]
        inv = 1.0 / np.sqrt(var + self.eps)
        xh = (x - mu[None, :, None]) * inv[None, :, None]
        # The mode is cached, not read back off self.training at backward time.
        # In eval mode mu and var are constants, so d(out)/d(in) is just
        # gamma * inv; in training mode they are functions of the batch and the
        # three-term expression below applies. Using the training expression for
        # an eval-mode forward gave a gradient that was wrong by O(1) -- it
        # subtracts a batch-mean correction that is not there -- and
        # :func:`gradcheck` reported a relative error of exactly 1.0 on the
        # convolution *below* each batch-norm while the batch-norm's own
        # gamma/beta gradients were correct to 1e-10. That asymmetry is the
        # signature: a layer whose local gradients are right and whose
        # pass-through is wrong.
        self._cache = (xh, inv, bool(self.training))
        return self.params["gamma"][None, :, None] * xh + self.params["beta"][None, :, None]

    def backward(self, g: np.ndarray) -> np.ndarray:
        xh, inv, was_training = self._cache
        self.grads["gamma"] = np.sum(g * xh, axis=(0, 2))
        self.grads["beta"] = np.sum(g, axis=(0, 2))
        gxh = g * self.params["gamma"][None, :, None]
        if not was_training:
            return gxh * inv[None, :, None]
        m = g.shape[0] * g.shape[2]
        return (inv[None, :, None] / m) * (
            m * gxh - np.sum(gxh, axis=(0, 2))[None, :, None]
            - xh * np.sum(gxh * xh, axis=(0, 2))[None, :, None]
        )


class ReLU(Layer):
    def forward(self, x: np.ndarray) -> np.ndarray:
        self._mask = x > 0
        return x * self._mask

    def backward(self, g: np.ndarray) -> np.ndarray:
        return g * self._mask


class MaxPool1d(Layer):
    def __init__(self, size: int = 2):
        super().__init__()
        self.size = size

    def forward(self, x: np.ndarray) -> np.ndarray:
        n, c, length = x.shape
        trimmed = (length // self.size) * self.size
        v = x[:, :, :trimmed].reshape(n, c, trimmed // self.size, self.size)
        self._arg = np.argmax(v, axis=3)
        self._shape = x.shape
        self._trimmed = trimmed
        return np.max(v, axis=3)

    def backward(self, g: np.ndarray) -> np.ndarray:
        n, c, lo = g.shape
        out = np.zeros((n, c, lo, self.size), dtype=g.dtype)
        i, j, k = np.indices((n, c, lo))
        out[i, j, k, self._arg] = g
        full = np.zeros(self._shape, dtype=g.dtype)
        full[:, :, : self._trimmed] = out.reshape(n, c, self._trimmed)
        return full


class Dropout(Layer):
    """Inverted dropout. ``on_input=True`` is TICNN's first-layer dropout."""

    def __init__(self, p: float = 0.0, seed: int = 0):
        super().__init__()
        self.p = float(p)
        self._rng = np.random.default_rng(seed)
        self._mask: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        if not self.training or self.p <= 0:
            self._mask = None
            return x
        self._mask = ((self._rng.random(x.shape) >= self.p) / (1.0 - self.p)).astype(x.dtype)
        return x * self._mask

    def backward(self, g: np.ndarray) -> np.ndarray:
        return g if self._mask is None else g * self._mask


class Flatten(Layer):
    def forward(self, x: np.ndarray) -> np.ndarray:
        self._shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, g: np.ndarray) -> np.ndarray:
        return g.reshape(self._shape)


class Dense(Layer):
    def __init__(self, n_in: int, n_out: int, seed: int = 0):
        super().__init__()
        rng = np.random.default_rng(seed)
        self.params["W"] = (rng.standard_normal((n_in, n_out)) * np.sqrt(2.0 / n_in)).astype(DTYPE)
        self.params["b"] = np.zeros(n_out, dtype=DTYPE)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        return x @ self.params["W"] + self.params["b"]

    def backward(self, g: np.ndarray) -> np.ndarray:
        self.grads["W"] = self._x.T @ g
        self.grads["b"] = g.sum(axis=0)
        return g @ self.params["W"].T


class Sequential:
    def __init__(self, layers: Sequence[Layer]):
        self.layers: List[Layer] = list(layers)

    def train(self) -> "Sequential":
        for lay in self.layers:
            lay.training = True
        return self

    def eval(self) -> "Sequential":
        for lay in self.layers:
            lay.training = False
        return self

    def forward(self, x: np.ndarray) -> np.ndarray:
        for lay in self.layers:
            x = lay.forward(x)
        return x

    def backward(self, g: np.ndarray) -> np.ndarray:
        for lay in reversed(self.layers):
            g = lay.backward(g)
        return g

    @property
    def n_params(self) -> int:
        return int(sum(lay.n_params for lay in self.layers))

    def state(self) -> List[Dict[str, List]]:
        return [
            {"params": {k: v.tolist() for k, v in lay.params.items()},
             "buffers": {k: v.tolist() for k, v in lay.buffers.items()}}
            for lay in self.layers
        ]

    def load_state(self, state: Sequence[Dict]) -> "Sequential":
        if len(state) != len(self.layers):
            raise ValueError(f"state has {len(state)} layers, model has {len(self.layers)}")
        for lay, st in zip(self.layers, state):
            for k, v in st.get("params", {}).items():
                lay.params[k] = np.asarray(v, dtype=lay.params[k].dtype if k in lay.params else DTYPE)
            for k, v in st.get("buffers", {}).items():
                lay.buffers[k] = np.asarray(v, dtype=lay.buffers[k].dtype if k in lay.buffers else DTYPE)
        return self


class Adam:
    def __init__(self, model: Sequential, lr: float = 1e-3, weight_decay: float = 0.0,
                 b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8):
        self.model, self.lr, self.wd = model, lr, weight_decay
        self.b1, self.b2, self.eps = b1, b2, eps
        self.t = 0
        self.m = [{k: np.zeros_like(v) for k, v in lay.params.items()} for lay in model.layers]
        self.v = [{k: np.zeros_like(v) for k, v in lay.params.items()} for lay in model.layers]

    def step(self) -> None:
        self.t += 1
        for i, lay in enumerate(self.model.layers):
            for k, p in lay.params.items():
                g = lay.grads.get(k)
                if g is None:
                    continue
                # No decay on BatchNorm scale/shift: shrinking gamma toward zero
                # fights the normalisation it exists to undo.
                if self.wd and not isinstance(lay, BatchNorm1d):
                    g = g + self.wd * p
                self.m[i][k] = self.b1 * self.m[i][k] + (1 - self.b1) * g
                self.v[i][k] = self.b2 * self.v[i][k] + (1 - self.b2) * g * g
                mh = self.m[i][k] / (1 - self.b1**self.t)
                vh = self.v[i][k] / (1 - self.b2**self.t)
                p -= self.lr * mh / (np.sqrt(vh) + self.eps)


def softmax_cross_entropy(logits: np.ndarray, y: np.ndarray,
                          class_weight: Optional[np.ndarray] = None) -> Tuple[float, np.ndarray]:
    """Mean loss and ``dL/dlogits``. ``y`` is integer class indices."""
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    p = e / e.sum(axis=1, keepdims=True)
    n = logits.shape[0]
    w = np.ones(n) if class_weight is None else class_weight[y]
    loss = float(-np.sum(w * np.log(np.maximum(p[np.arange(n), y], 1e-12))) / np.sum(w))
    g = p.copy()
    g[np.arange(n), y] -= 1.0
    g *= (w / np.sum(w))[:, None]
    return loss, g.astype(logits.dtype)


def gradcheck(model: Sequential, x: np.ndarray, y: np.ndarray, *,
              eps: float = 1e-6, n_probe: int = 4, seed: int = 0) -> Dict[str, float]:
    """Max relative error between the analytic and finite-difference gradient.

    Probes ``n_probe`` random entries of every parameter tensor. Runs in eval
    mode so that batch-norm statistics and dropout masks are deterministic --
    with running statistics being updated on every forward pass, a finite
    difference would measure the statistics moving rather than the gradient.

    On the step size, because it changes what this function means. ReLU and
    max-pool are piecewise linear, so a perturbation that flips a sign or an
    argmax anywhere in the network puts the two evaluations on different linear
    pieces and the difference quotient measures a kink rather than a derivative.
    Measured on the WDCNN stack, max relative error over the same probes:

        eps=1e-6   1.6e-07     median 1.3e-08
        eps=1e-5   1.5e-02     median 3.1e-09
        eps=1e-4   2.3e-02     median 4.8e-10
        eps=1e-3   1.0e+00     median 1.1e-08

    The *median* improves monotonically with a larger step, as a correct
    gradient and shrinking truncation error should; only the max degrades. A real
    sign error degrades both and does not depend on the step. The same stack with
    ReLU and max-pool removed checks to 1.2e-08 at eps=1e-5. Default is therefore
    1e-6, and the test suite gates on the median while reporting the max.
    """
    model.eval()
    rng = np.random.default_rng(seed)
    worst: Dict[str, float] = {}
    loss, g = softmax_cross_entropy(model.forward(x), y)
    model.backward(g)
    analytic = [{k: v.copy() for k, v in lay.grads.items()} for lay in model.layers]
    del loss
    for i, lay in enumerate(model.layers):
        for k, p in lay.params.items():
            if k not in analytic[i]:
                continue
            flat = p.ravel()
            idx = rng.choice(flat.size, size=min(n_probe, flat.size), replace=False)
            errs = []
            for j in idx:
                orig = flat[j]
                flat[j] = orig + eps
                lp, _ = softmax_cross_entropy(model.forward(x), y)
                flat[j] = orig - eps
                lm, _ = softmax_cross_entropy(model.forward(x), y)
                flat[j] = orig
                num = (lp - lm) / (2 * eps)
                ana = analytic[i][k].ravel()[j]
                errs.append(abs(num - ana) / max(abs(num) + abs(ana), 1e-12))
            worst[f"layer{i}.{type(lay).__name__}.{k}"] = float(max(errs))
    return worst
