"""Fill tensor in-place with integers increasing along a dimension."""

import torch


def arange_(tensor: torch.Tensor, dim: int) -> None:
    """
    Fill tensor in-place with integers increasing along a dimension.

    Modifies the values in the tensor, such that each view obtained by
    indexing with a single index in `dim` consistes of constant values, and
    each slice with one remaining dimension along `dim` will only contain 
    increasing integers.

    Parameters
    ----------
    tensor
        The tensor to be modified in-place.

    dim
        The dimension along which the resulting values will be increasing
    """
    if not -tensor.ndim <= dim < tensor.ndim:
        raise IndexError(f'Dimension {dim} is out of range for tensor with {tensor.ndim} dimensions.')

    dim = dim % tensor.ndim
    shape = [s if d == dim else 1 for d, s in enumerate(tensor.shape)]
    values = torch.arange(tensor.size(dim), device=tensor.device).reshape(shape)
    tensor[:] = values.expand_as(tensor)
