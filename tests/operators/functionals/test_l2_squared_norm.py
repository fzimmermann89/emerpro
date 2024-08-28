"""Tests for L2-Squared-functional."""

import pytest
import torch
import torch.nn as nn
from mrpro.operators.functionals.l2_squared import L2NormSquared


@pytest.mark.parametrize(
    (
        'x',
        'dim',
        'keepdim',
        'shape_forward_x',
    ),
    [
        (
            torch.rand(16, 8, 1, 160, 160),
            (-2, -1),
            True,
            torch.Size([16, 8, 1, 1, 1]),
        ),
        (
            torch.rand(16, 8, 1, 160, 160),
            (-2, -1),
            False,
            torch.Size([16, 8, 1]),
        ),
        (
            torch.rand(16, 8, 1, 160, 160),
            None,
            False,
            torch.Size([]),
        ),
        (
            torch.rand(16, 8, 1, 160, 160),
            None,
            True,
            torch.Size([1, 1, 1, 1, 1]),
        ),
    ],
)
def test_l2_functional_shape(
    x,
    dim,
    keepdim,
    shape_forward_x,
):
    """Test if L2 norm matches expected values."""
    l2_norm = L2NormSquared(weight=1.0)
    torch.testing.assert_close(
        l2_norm.forward(x, dim=dim, keepdim=keepdim)[0].shape, shape_forward_x, rtol=1e-3, atol=1e-3
    )
    l2_norm = L2NormSquared(weight=1.0, divide_by_n=True)
    torch.testing.assert_close(
        l2_norm.forward(x, dim=dim, keepdim=keepdim)[0].shape, shape_forward_x, rtol=1e-3, atol=1e-3
    )


@pytest.mark.parametrize(
    ('x',),
    [
        (torch.rand(10, 10, 10),),
    ],
)
def test_prox_unity_sigma(
    x,
):
    """Test if Prox of l2 norm is the identity if sigma is 0 or close to 0."""
    l2_norm = L2NormSquared(weight=1, divide_by_n=True)
    torch.testing.assert_close(l2_norm.prox(x, sigma=0)[0], x, rtol=1e-3, atol=1e-3)
    l2_norm = L2NormSquared(weight=1, divide_by_n=False)
    torch.testing.assert_close(l2_norm.prox(x, sigma=0)[0], x, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize(
    ('x',),
    [
        (torch.rand(10, 10, 10),),
    ],
)
def test_prox_unity_W(
    x,
):
    """Test if Prox of l2 norm is the identity if W=0."""
    l2_norm = L2NormSquared(weight=0, divide_by_n=True)
    torch.testing.assert_close(l2_norm.prox(x, sigma=1)[0], x, rtol=1e-3, atol=1e-3)
    l2_norm = L2NormSquared(weight=0, divide_by_n=False)
    torch.testing.assert_close(l2_norm.prox(x, sigma=1)[0], x, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize(
    (
        'x',
        'forward_x',
        'prox',
        'prox_complex_conjugate',
    ),
    [
        (
            torch.tensor([1, 1], dtype=torch.complex64),
            torch.tensor((1.0), dtype=torch.float32),
            torch.tensor([1 / 2, 1 / 2], dtype=torch.complex64),
            torch.tensor([1 / 2, 1 / 2], dtype=torch.complex64),
        ),
        (
            torch.tensor([1 + 1j, 1 + 1j], dtype=torch.complex64),
            torch.tensor((2.0), dtype=torch.float32),
            torch.tensor([(1 + 1j) / 2, (1 + 1j) / 2], dtype=torch.complex64),
            torch.tensor([(1 + 1j) / 2, (1 + 1j) / 2], dtype=torch.complex64),
        ),
        (
            torch.tensor([1 + 0j, 1 + 1j], dtype=torch.complex64),
            torch.tensor((3 / 2), dtype=torch.float32),
            torch.tensor([1 / 2, (1 + 1j) / 2], dtype=torch.complex64),
            torch.tensor([1 / 2, (1 + 1j) / 2], dtype=torch.complex64),
        ),
    ],
)
def test_l2_squared_functional(
    x,
    forward_x,
    prox,
    prox_complex_conjugate,
):
    """Test if l2_squared_norm matches expected values."""
    l2_squared_norm = L2NormSquared(weight=1, divide_by_n=False, keepdim=False)

    torch.testing.assert_close(l2_squared_norm.forward(x)[0], forward_x, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(l2_squared_norm.prox(x, sigma=1, divide_by_n=False)[0], prox, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(
        l2_squared_norm.prox_convex_conj(x, sigma=1)[0], prox_complex_conjugate, rtol=1e-3, atol=1e-3
    )


@pytest.mark.parametrize(
    (
        'x',
        'gradients',
    ),
    [
        (
            torch.tensor([1.0, 2.0, -1.0], dtype=torch.float32, requires_grad=True),
            True,
        ),
    ],
)
def test_l2_loss_backward(
    x,
    gradients,
):
    """Test if L2 loss works with backward pass."""
    model = nn.Linear(3, 1)
    out = model(x)
    l2_norm = L2NormSquared(weight=1, target=torch.tensor([0.5, 1.5, -0.5]))
    loss = l2_norm.forward(out, keepdim=False)[0]
    loss.backward()
    [True for x in model.parameters() if x.grad is not None][0] == gradients


@pytest.mark.parametrize(
    ('x',),
    [
        (100 * torch.randn(16, 8, 1, 160, 160),),
    ],
)
def test_l2_moreau(
    x,
):
    """Test if L2 norm matches expected values."""
    divide_by_n = False
    dim = (-2, -1)
    l2_norm = L2NormSquared(weight=1, keepdim=True, dim=dim, divide_by_n=divide_by_n)
    sigma = 0.5
    x_new = l2_norm.prox(x, sigma=sigma)[0] + sigma * (l2_norm.prox_convex_conj(x / sigma, 1.0 / sigma))[0]
    torch.testing.assert_close(x, x_new, rtol=1e-3, atol=1e-3)
