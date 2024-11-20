"""Primal-Dual Hybrid Gradient Algorithm (PDHG)."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from mrpro.algorithms.optimizers.OptimizerStatus import OptimizerStatus
from mrpro.operators import (
    IdentityOp,
    LinearOperator,
    LinearOperatorMatrix,
    ProximableFunctional,
    ProximableFunctionalSeparableSum,
)
from mrpro.operators.functionals import ZeroFunctional


@dataclass
class PDHGStatus(OptimizerStatus):
    """Status of the PDHG algorithm."""

    objective: Callable[[*tuple[torch.Tensor, ...]], torch.Tensor]
    dual_stepsize: float | torch.Tensor
    primal_stepsize: float | torch.Tensor
    relaxation: float | torch.Tensor
    duals: Sequence[torch.Tensor]
    relaxed: Sequence[torch.Tensor]


def _norm_squared(*values: torch.Tensor) -> torch.Tensor:
    """Calculate the squared L2 norm of the stack of tensors."""
    return sum([torch.vdot(value.flatten(), value.flatten()).real for value in values], start=torch.tensor(0.0))


def pdhg(
    f: ProximableFunctionalSeparableSum | ProximableFunctional | None,
    g: ProximableFunctionalSeparableSum | ProximableFunctional | None,
    operator: LinearOperator | LinearOperatorMatrix | None,
    initial_values: Sequence[torch.Tensor],
    max_iterations: int = 32,
    tolerance: float = 0,
    primal_stepsize: float | None = None,
    dual_stepsize: float | None = None,
    relaxation: float = 1.0,
    initial_relaxed: Sequence[torch.Tensor] | None = None,
    initial_duals: Sequence[torch.Tensor] | None = None,
    callback: Callable[[PDHGStatus], None] | None = None,
) -> tuple[torch.Tensor, ...]:
    r"""Primal-Dual Hybrid Gradient Algorithm (PDHG).

    Solves the minimization problem
        :math:`\min_x g(x) + f(A x)`
    with linear operator A and proximable functionals f and g.

    The operator is supplied as a matrix (tuple of tuples) of linear operators,
    f and g are supplied as tuples of proximable functionals interpreted as separable sums.

    Thus, problem solved is
            :math:`\min_x \sum_i,j g_j(x_j) + f_i(A_ij x_j)`.

    If neither primal nor dual step size are not supplied, they are chose as :math:`1/||A||_{op}`.
    If either is supplied, the other is chosen such that
        primal_stepsize*dual_stepsize = :math:`1/||A||_{op}^2`

    Note that the computation of the operator-norm can be computationally expensive and
    that if no stepsizes are provided, the algorithm runs a power iteration to obtain the
    upper bound of the stepsizes.

    For a warm start, the initial relaxed primal and dual variables can be supplied.
    These might be obtained from the status object of a previous run.

    Parameters
    ----------
    f
        tuple of proximable functionals interpreted as a separable sum
    g
        tuple of proximable functionals interpreted as a separable sum
    operator
        matrix of linear operators
    initial_values
        initial guess of the solution
    max_iterations
        maximum number of iterations
    tolerance
        tolerance for relative change of the primal solution; if set to zero, max_iterations of pdhg are run
    dual_stepsize
        dual step size
    primal_stepsize
        primal step size
    relaxation
        relaxation parameter, 1.0 is no relaxation
    initial_relaxed
        relaxed primals, used for warm start
    initial_duals
        dual variables, used for warm start
    callback
        callback function called after each iteration
    """
    if f is None and g is None:
        warnings.warn(
            'Both f and g are None. The objective is constant. Returning x0 as a possible solution', stacklevel=2
        )
        return tuple(initial_values)

    if operator is None:
        # Use identity operator if no operator is supplied
        n_rows = len(f) if isinstance(f, ProximableFunctionalSeparableSum) else 1
        n_columns = len(f) if isinstance(g, ProximableFunctionalSeparableSum) else 1
        if n_rows != n_columns:
            raise ValueError('If operator is None, the number of elements in f and g should be the same')
        operator_matrix = LinearOperatorMatrix.from_diagonal(*((IdentityOp(),) * n_rows))
    else:
        if isinstance(operator, LinearOperator):
            # We always use a matrix of operators for homogeneous handling
            operator_matrix = LinearOperatorMatrix.from_diagonal(operator)
        else:
            operator_matrix = operator
        n_rows, n_columns = operator_matrix.shape

    # We always use a separable sum for homogeneous handling, even if it is just a ZeroFunctional
    if f is None:
        f_sum = ProximableFunctionalSeparableSum(*(ZeroFunctional(),) * n_rows)
    elif isinstance(f, ProximableFunctional):
        f_sum = ProximableFunctionalSeparableSum(f)
    else:
        f_sum = f

    if g is None:
        g_sum = ProximableFunctionalSeparableSum(*(ZeroFunctional(),) * n_columns)
    elif isinstance(g, ProximableFunctional):
        g_sum = ProximableFunctionalSeparableSum(g)
    else:
        g_sum = g

    if len(f_sum) != n_rows:
        raise ValueError('Number of rows in operator does not match number of functionals in f')

    if len(g) != n_columns:
        raise ValueError('Number of columns in operator does not match number of functionals in g')

    if primal_stepsize is None or dual_stepsize is None:
        # choose primal and dual step size such that their product is 1/|operator|**2
        # to ensure convergence
        operator_norm = operator_matrix.operator_norm(*[torch.randn_like(v) for v in initial_values])
        if primal_stepsize is None and dual_stepsize is None:
            primal_stepsize_ = dual_stepsize_ = 1.0 / operator_norm
        elif primal_stepsize is None:
            primal_stepsize_ = 1 / (operator_norm * dual_stepsize)
            dual_stepsize_ = dual_stepsize
        elif dual_stepsize is None:
            dual_stepsize_ = 1 / (operator_norm * primal_stepsize)
            primal_stepsize_ = primal_stepsize
    else:
        primal_stepsize_ = primal_stepsize
        dual_stepsize_ = dual_stepsize

    primals_relaxed = initial_values if initial_relaxed is None else initial_relaxed
    duals = (0 * operator_matrix)(*initial_values) if initial_duals is None else initial_duals

    if len(duals) != n_rows:
        raise ValueError('if dual variable is supplied, it should be a tuple of same length as the tuple of g')

    primals = initial_values
    for i in range(max_iterations):
        duals = tuple(
            dual + dual_stepsize_ * step for dual, step in zip(duals, operator_matrix(*primals_relaxed), strict=False)
        )
        duals = f_sum.prox_convex_conj(*duals, sigma=dual_stepsize_)

        primals_new = tuple(
            primal - primal_stepsize_ * step for primal, step in zip(primals, operator_matrix.H(*duals), strict=False)
        )
        primals_new = g_sum.prox(*primals_new, sigma=primal_stepsize_)
        primals_relaxed = tuple(
            torch.lerp(primal, primal_new, relaxation) for primal, primal_new in zip(primals, primals_new, strict=False)
        )

        # check if the solution is already accurate enough
        if tolerance != 0:
            change_squared = _norm_squared(*[(old - new) for old, new in zip(primals, primals_new, strict=True)])
            primals_new_squared = _norm_squared(*primals_new)
            if change_squared < tolerance**2 * primals_new_squared:
                return tuple(primals_new)

        primals = primals_new

        if callback is not None:
            status = PDHGStatus(
                iteration_number=i,
                dual_stepsize=dual_stepsize_,
                primal_stepsize=primal_stepsize_,
                relaxation=relaxation,
                duals=duals,
                solution=tuple(primals),
                relaxed=primals_relaxed,
                objective=lambda *x: f_sum.forward(*operator_matrix(*x))[0] + g_sum.forward(*x)[0],
            )
            callback(status)

    return tuple(primals)
