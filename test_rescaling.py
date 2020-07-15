#!/usr/bin/env python3
import pytest
import eagerpy as ep
import numpy as np
import torch
import jax
import tensorflow as tf

from clipping_aware_rescaling import l2_clipping_aware_rescaling
from naive_rescaling import l2_naive_rescaling


def get_effective_norms(x, eta, delta, a, b):
    x, eta, delta = ep.astensors(x, eta, delta)
    xp = ep.clip(x + eta * delta, a, b)
    diff = xp - x
    N = diff.shape[0]
    norms = ep.norms.l2(diff.reshape((N, -1)), axis=-1)
    norms = norms.numpy()
    return norms


def assert_clipping_aware_correct(x, eta, delta, a, b, eps):
    norms = get_effective_norms(x, eta, delta, a, b)
    print(f"clipping-aware rescaling norms:\n{norms}")
    assert np.allclose(norms, eps)


def assert_naive_too_small(x, eta, delta, a, b, eps):
    norms = get_effective_norms(x, eta, delta, a, b)
    print(f"naive rescaling norms:\n{norms}")
    assert not np.allclose(norms, eps)
    assert np.all(np.logical_or(np.isclose(norms, eps), norms < eps))


def generate_numpy_data(n, shape, a, b):
    np.random.seed(22)
    x = np.random.rand(n, *shape) * (b - a) + a
    delta = np.random.randn(n, *shape)
    return x, delta


def to_numpy(x):
    return x


def to_pytorch(x):
    return torch.from_numpy(x)


def to_tensorflow(x):
    return tf.convert_to_tensor(x)


def to_jax(x):
    return jax.device_put(x)


@pytest.mark.parametrize("n", [1, 10])
@pytest.mark.parametrize("shape", [(784,), (28, 28)])
@pytest.mark.parametrize("a,b,eps", [(0, 1, 3.6), (100, 120, 22.0), (-5, 5, 11.0)])
# @pytest.mark.parametrize("convert", [to_numpy, to_pytorch, to_tensorflow, to_jax])
@pytest.mark.parametrize("convert", [to_numpy, to_pytorch, to_jax])
@pytest.mark.parametrize("eagerpy", [False, True])
def test_rescaling(n, shape, a, b, eps, convert, eagerpy):
    x, delta = generate_numpy_data(n, shape, a, b)
    x = convert(x)
    delta = convert(delta)
    if eagerpy:
        x = ep.astensor(x)
        delta = ep.astensor(delta)

    eta = l2_naive_rescaling(delta, eps)
    assert_naive_too_small(x, eta, delta, a, b, eps)

    eta = l2_clipping_aware_rescaling(x, delta, eps, a, b)
    assert_clipping_aware_correct(x, eta, delta, a, b, eps)
