#!/usr/bin/env python3
import numpy as np
import torch

from clipping_aware_rescaling import l2_clipping_aware_rescaling
from naive_rescaling import l2_naive_rescaling


def generate_data(n, shape, a, b):
    np.random.seed(22)
    x = np.random.rand(n, *shape) * (b - a) + a
    delta = np.random.randn(n, *shape)
    return torch.from_numpy(x), torch.from_numpy(delta)


def get_effective_norms(x, eta, delta, a, b):
    diff = torch.clamp(x + eta * delta, a, b) - x
    norms = torch.norm(diff.reshape((diff.shape[0], -1)), dim=-1)
    return norms


def example():
    x, delta = generate_data(4, (28, 28), 0, 1)

    eps = 3.6
    print(f"target norm: {eps}\n")

    eta = l2_naive_rescaling(delta, eps)
    norms = get_effective_norms(x, eta, delta, 0, 1)
    print(f"naive rescaling norms:\n{norms}\n")

    eta = l2_clipping_aware_rescaling(x, delta, eps, 0, 1)
    norms = get_effective_norms(x, eta, delta, 0, 1)
    print(f"clipping-aware rescaling norms:\n{norms}\n")


if __name__ == "__main__":
    example()
