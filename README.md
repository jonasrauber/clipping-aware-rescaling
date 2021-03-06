# Fast Differentiable Clipping-Aware Rescaling

## Signature

```python
def l2_clipping_aware_rescaling(x, delta, eps: float, a: float = 0.0, b: float = 1.0):
    """Calculates eta such that norm(clip(x + eta * delta, a, b) - x) == eps.

    Assumes x and delta have a batch dimension and eps, a, b, and p are
    scalars. If the equation cannot be solved because eps is too large, the
    left hand side is maximized.

    Args:
        x: A batch of inputs (PyTorch Tensor, TensorFlow Eager Tensor, NumPy
            Array, JAX Array, or EagerPy Tensor).
        delta: A batch of perturbation directions (same shape and type as x).
        eps: The target norm (non-negative float).
        a: The lower bound of the data domain (float).
        b: The upper bound of the data domain (float).

    Returns:
        eta: A batch of scales with the same number of dimensions as x but all
            axis == 1 except for the batch dimension.
    """
    ...
```

## How to use

```python
from clipping_aware_rescaling import l2_clipping_aware_rescaling
```

## Example

See `./example.py`:

```
target norm: 3.6

naive rescaling norms:
tensor([3.4086, 3.3889, 3.3422, 3.3861], dtype=torch.float64)

clipping-aware rescaling norms:
tensor([3.6000, 3.6000, 3.6000, 3.6000], dtype=torch.float64)
```

## Tech Report

Our tech report *[Fast Differentiable Clipping-Aware Normalization and Rescaling](https://arxiv.org/abs/2007.07677)* explaining the algortihm is available on [arXiv](https://arxiv.org/abs/2007.07677).

## Citation (BibTeX)

```bibtex
@article{rauber2020fast,
  title={Fast Differentiable Clipping-Aware Normalization and Rescaling},
  author={Rauber, Jonas and Bethge, Matthias},
  journal={arXiv preprint arXiv:2007.07677},
  year={2020},
  url={https://github.com/jonasrauber/clipping-aware-rescaling},
}
```

## License

[BSD 3-Clause License](./LICENSE)

## Author

[Jonas Rauber](https://jonasrauber.de)
