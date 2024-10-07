"""Linear Operator Matrix class."""

import operator
from collections.abc import Callable, Sequence
from functools import reduce
from types import EllipsisType
from typing import Self, TypeVar, cast

import torch

from mrpro.operators.LinearOperator import LinearOperator, LinearOperatorSum
from mrpro.operators.Operator import Operator
from mrpro.operators.ZeroOp import ZeroOp

_SingleIdxType = int | slice | EllipsisType | Sequence[int]
_IdxType = _SingleIdxType | tuple[_SingleIdxType, _SingleIdxType]

T = TypeVar('T', bound=Operator)


class LinearOperatorMatrix(Operator):
    r"""Matrix of Linear Operators.

    A matrix of Linear Operators, where each element is a Linear Operator.

    This matrix can be applied to a sequence of tensors, where the number of tensors should match
    the number of columns of the matrix. The output will be a sequence of tensors, where the number
    of tensors will match the number of rows of the matrix.
    The j-th output tensor is calculated as
    :math:`\sum_i \text{operators}[i][j](x[i])` where :math:`\text{operators}[i][j]` is the linear operator
    in the i-th row and j-th column and :math:`x[i]` is the i-th input tensor.

    The matrix can be indexed and sliced like a regular matrix to get submatrices.
    If indexing returns a single element, it is returned as a Linear Operator.

    Basic arithmetic operations are supported with Linear Operators and Tensors.

    """

    _operators: list[list[LinearOperator]]

    def __init__(self, operators: Sequence[Sequence[LinearOperator]]):
        """Initialize Linear Operator Matrix.

        Create a matrix of LinearOperators from a sequence of rows, where each row is a sequence
        of LinearOperators that represent the columns of the matrix.

        Parameters
        ----------
        operators
            A sequence of rows, which are sequences of Linear Operators.
        """
        if not all(isinstance(op, LinearOperator) for row in operators for op in row):
            raise ValueError('All elements should be Linear Operators.')
        if not all(len(row) == len(operators[0]) for row in operators):
            raise ValueError('All rows should have the same length.')
        super().__init__()
        self._operators = cast(  # cast because ModuleList is not recognized as a list
            list[list[LinearOperator]], torch.nn.ModuleList(torch.nn.ModuleList(row) for row in operators)
        )
        self._shape = (len(operators), len(operators[0]) if operators else 0)

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the Operator Matrix (rows, columns)."""
        return self._shape

    def forward(self, *x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Apply the operator to the input.

        Parameters
        ----------
        x
            Input tensors. Requires the same number of tensors as the operator has columns.

        Returns
        -------
            Output tensors. The same number of tensors as the operator has rows.
        """
        if len(x) != self.shape[1]:
            raise ValueError('Input should have the same length as the operator has columns.')
        return tuple(
            reduce(operator.add, (op(xi)[0] for op, xi in zip(row, x, strict=True))) for row in self._operators
        )

    def __getitem__(self, idx: _IdxType) -> Self | LinearOperator:
        """Index the Operator Matrix.

        Parameters
        ----------
        idx
            Index or slice to select rows and columns.

        Returns
        -------
            Subset LinearOperatorMatrix or Linear Operator.
        """
        idxs: tuple[_SingleIdxType, _SingleIdxType] = idx if isinstance(idx, tuple) else (idx, slice(None))
        if len(idxs) > 2:
            raise IndexError('Too many indices for LinearOperatorMatrix')

        def _to_numeric_index(idx: slice | int | Sequence[int] | EllipsisType, length: int) -> Sequence[int]:
            """Convert index to a sequence of integers."""
            if isinstance(idx, slice):
                return range(*idx.indices(length))
            if isinstance(idx, int):
                return (idx,)
            if idx is Ellipsis:
                return range(length)
            if isinstance(idx, Sequence):
                return idx
            else:
                raise IndexError('Invalid index type')

        row_numbers = _to_numeric_index(idxs[0], self._shape[0])
        col_numbers = _to_numeric_index(idxs[1], self._shape[1])

        sliced_operators = [
            [row[col] for col in col_numbers] for i, row in enumerate(self._operators) if i in row_numbers
        ]
        if len(row_numbers) == 1 and len(col_numbers) == 1:
            return sliced_operators[0][0]
        else:
            return self.__class__(sliced_operators)

    def __repr__(self):
        """Representation of the Operator Matrix."""
        return f'LinearOperatorMatrix(shape={self._shape}, operators={self._operators})'

    # Note: The type ignores are needed because we currently cannot do arithmetic operations with non-linear operators.
    def __add__(self, other: Self | LinearOperator | torch.Tensor) -> Self:  # type: ignore[override]
        """Addition."""
        operators: list[list[LinearOperator]] = []
        if isinstance(other, LinearOperatorMatrix):
            if self.shape != other.shape:
                raise ValueError('OperatorMatrix shapes do not match.')
            for self_row, other_row in zip(self._operators, other._operators, strict=False):
                operators.append([s + o for s, o in zip(self_row, other_row, strict=False)])
        elif isinstance(other, LinearOperator | torch.Tensor):
            if not self.shape[0] == self.shape[1]:
                raise NotImplementedError('Cannot add a LinearOperator to a non-square OperatorMatrix.')
            for i, self_row in enumerate(self._operators):
                operators.append([op + other if i == j else op for j, op in enumerate(self_row)])
        else:
            return NotImplemented  # type: ignore[unreachable]
        return self.__class__(operators)

    def __radd__(self, other: Self | LinearOperator | torch.Tensor) -> Self:
        """Right addition."""
        return self.__add__(other)

    def __mul__(self, other: torch.Tensor | Sequence[torch.Tensor | complex] | complex) -> Self:
        """LinearOperatorMatrix*Tensor multiplication."""
        if isinstance(other, torch.Tensor | complex | float | int):
            other_: Sequence[torch.Tensor | complex] = (other,) * self.shape[1]
        elif len(other) != self.shape[1]:
            raise ValueError('Other should have the same length as the operator has columns.')
        else:
            other_ = other
        operators = []
        for row in self._operators:
            operators.append([op * o for op, o in zip(row, other_, strict=True)])
        return self.__class__(operators)

    def __rmul__(self, other: torch.Tensor | Sequence[torch.Tensor] | complex) -> Self:
        """Tensor*LinearOperatorMatrix multiplication."""
        if isinstance(other, torch.Tensor | complex | float | int):
            other_: Sequence[torch.Tensor | complex] = (other,) * self.shape[1]
        elif len(other) != self.shape[0]:
            raise ValueError('Other should have the same length as the operator has rows.')
        else:
            other_ = other
        operators = []
        for row, o in zip(self._operators, other_, strict=True):
            operators.append([cast(LinearOperator, o * op) for op in row])
        return self.__class__(operators)

    def __matmul__(self, other: LinearOperator | Self) -> Self:  # type: ignore[override]
        """Composition of operators."""
        if isinstance(other, LinearOperator):
            return self._binary_operation(other, '__matmul__')
        elif isinstance(other, LinearOperatorMatrix):
            if self.shape[1] != other.shape[0]:
                raise ValueError('OperatorMatrix shapes do not match.')
            new_operators = []
            for row in self._operators:
                new_row = []
                for other_col in zip(*other._operators, strict=True):
                    elements = [s @ o for s, o in zip(row, other_col, strict=True)]
                    new_row.append(LinearOperatorSum(*elements))
                new_operators.append(new_row)
            return self.__class__(new_operators)
        return NotImplemented  # type: ignore[unreachable]

    @property
    def H(self) -> Self:  # noqa N802
        """Adjoints of the operators."""
        return self.__class__([[op.H for op in row] for row in zip(*self._operators, strict=True)])

    def adjoint(self, *x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Apply the adjoint of the operator to the input.

        Parameters
        ----------
        x
            Input tensors. Requires the same number of tensors as the operator has rows.

        Returns
        -------
            Output tensors. The same number of tensors as the operator has columns.
        """
        return self.H(*x)

    @classmethod
    def from_diagonal(cls, *operators: LinearOperator):
        """Create a diagonal LinearOperatorMatrix.

        Create a square LinearOperatorMatrix with the given Linear Operators on the diagonal.

        Parameters
        ----------
        operators
            Sequence of Linear Operators to be placed on the diagonal.
        """
        operator_matrix: list[list[LinearOperator]] = [
            [op if i == j else ZeroOp() for j in range(len(operators))] for i, op in enumerate(operators)
        ]
        return cls(operator_matrix)

    def operator_norm(
        self,
        *initial_value: torch.Tensor,
        dim: Sequence[int] | None = None,
        max_iterations: int = 20,
        relative_tolerance: float = 1e-4,
        absolute_tolerance: float = 1e-5,
        callback: Callable[[torch.Tensor], None] | None = None,
    ) -> torch.Tensor:
        """Upper bound of operator norm of the Matrix.

        Uses the bounds :math:`||[A, B}^T|||<=sqrt(||A||^2 + ||B||^2)` and :math:`||[A, B]|||<=||A||+||B||`
        to estimate the operator norm of the matrix by calling operator_norm on each element of the matrix.

        Parameters
        ----------
        initial_value
            Initial value(s) for the power iteration, length should match the number of columns
            of the operator matrix.
        dim
            Dimensions to calculate the operator norm over. Other dimensions are assumed to be
            batch dimensions. None means all dimensions.
        max_iterations
            Maximum number of iterations used in the power iteration.
        relative_tolerance
            Relative tolerance for convergence.
        absolute_tolerance
            Absolute tolerance for convergence.
        callback
            Callback function to be called with the current estimate of the operator norm.


        Returns
        -------
        Estimated operator norm upper bound.
        """

        def _singlenorm(op: LinearOperator, initial_value: torch.Tensor):
            return op.operator_norm(
                initial_value,
                dim=dim,
                max_iterations=max_iterations,
                relative_tolerance=relative_tolerance,
                absolute_tolerance=absolute_tolerance,
                callback=callback,
            )

        if len(initial_value) != self.shape[1]:
            raise ValueError('Initial value should have the same length as the operator has columns.')
        norms = torch.tensor(
            [[_singlenorm(op, iv) for op, iv in zip(row, initial_value, strict=True)] for row in self._operators]
        )
        norm = norms.sum(dim=1).square().sum(0).sqrt().unsqueeze(-1)
        return norm
