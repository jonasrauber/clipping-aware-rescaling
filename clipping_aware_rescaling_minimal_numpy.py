#!/usr/bin/env python3
# Copyright (c) 2020, Jonas Rauber
#
# Licensed under the BSD 3-Clause License
#
# Last changed:
# * 2020-07-15
# * 2020-01-08
# * 2019-04-18

import numpy as np


def l2_clipping_aware_rescaling_minimal_numpy(x, delta, eps):
    """Calculates eta such that
    norm(clip(x + eta * delta, 0, 1) - x) == eps.

    Args:
        x: A 1-dimensional NumPy array.
        delta: A 1-dimensional NumPy array.
        eps: A non-negative float.

    Returns:
        eta: A non-negative float.
    """
    delta2 = np.square(delta)
    space = np.where(delta >= 0, 1 - x, x)
    f2 = np.square(space) / delta2
    ks = np.argsort(f2)
    f2_sorted = f2[ks]
    m = np.cumsum(delta2[ks[::-1]])[::-1]
    dx = np.ediff1d(f2_sorted, to_begin=f2_sorted[0])
    dy = m * dx
    y = np.cumsum(dy)
    j = np.flatnonzero(y >= eps**2)[0]
    eta2 = f2_sorted[j] - (y[j] - eps**2) / m[j]
    eta = np.sqrt(eta2).item()
    return eta


if __name__ == "__main__":
    # This is a minimal, self-contained NumPy implementation.
    # For the full, generic EagerPy-based implementation with support for
    # batches as well as custom data domain bounds, please see
    # clipping_aware_rescaling.py

    np.random.seed(22)
    x = np.random.rand(784)
    delta = np.random.randn(784)

    eps = 3.6
    print(f"target norm: {eps}")

    eta = l2_clipping_aware_rescaling_minimal_numpy(x, delta, eps)
    print(f"eta: {eta:.3f}")

    diff = np.clip(x + eta * delta, 0, 1) - x
    norm = np.linalg.norm(diff, axis=-1)
    print(f"output norm: {norm}")
