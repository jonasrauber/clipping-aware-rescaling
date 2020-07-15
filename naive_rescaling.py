# Copyright (c) 2020, Jonas Rauber
#
# Licensed under the BSD 3-Clause License
#
# Last changed:
# * 2020-07-15

import eagerpy as ep


def l2_naive_rescaling(delta, eps: float):
    """Calculates eta = eps / norm(delta) for a batch of deltas.

    Assumes delta has a batch dimension and eps is a scalar.

    Args:
        delta: A batch of perturbation directions (PyTorch Tensor, TensorFlow
            Eager Tensor, NumPy Array, JAX Array, or EagerPy Tensor).
        eps: The target norm (non-negative float).

    Returns:
        eta: A batch of scales with the same number of dimensions as x but all
            axis == 1 except for the batch dimension.
    """
    delta, restore_fn = ep.astensor_(delta)
    N = delta.shape[0]
    ndim = delta.ndim
    delta = delta.reshape((N, -1))
    eta = eps / ep.norms.l2(delta, axis=-1)
    eta = eta.reshape((-1,) + (1,) * (ndim - 1))
    return restore_fn(eta)
