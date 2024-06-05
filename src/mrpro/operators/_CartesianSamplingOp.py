"""Cartesian Sampling Operator."""

# Copyright 2023 Physikalisch-Technische Bundesanstalt
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings

import torch
from einops import rearrange

from mrpro.data._KTrajectory import KTrajectory
from mrpro.data._SpatialDimension import SpatialDimension
from mrpro.data.enums import TrajType
from mrpro.operators._LinearOperator import LinearOperator


class CartesianSamplingOp(LinearOperator):
    """Cartesian Sampling Operator.

    Selects points on a Cartisian grid based on the the k-space trajectory.
    Thus, the adjoint sorts the data into regular Cartesian sampled grid based on the k-space
    trajectory. Non-acquired points are zero-filled.
    """

    def __init__(self, encoding_matrix: SpatialDimension[int], traj: KTrajectory) -> None:
        """Initialize Sampling Operator class.

        Parameters
        ----------
        encoding_matrix
            shape of the encoded k-space.
            Only values for directions in which the trajectory is Cartesian will be used
            in the adjoint to determine the shape after reordering,
            i.e., the operator's domain.
        traj
            the k-space trajectory describing at which frequencies data is sampled.
            Its broadcasted shape will be used to determine the shape after sampling,
            i.e., the operator's range
        """
        super().__init__()
        # the shape of the k data,
        sorted_grid_shape = SpatialDimension.from_xyz(encoding_matrix)

        # Cache as these might take some time to compute
        traj_type_kzyx = traj.type_along_kzyx
        ktraj_tensor = traj.as_tensor()

        # If a dimension is irregular or singleton, we will not perform any reordering
        # in it and the shape of data will remain.
        # only dimensions on a cartesian grid will be reordered.
        if traj_type_kzyx[-1] == TrajType.ONGRID:  # kx
            kx_idx = ktraj_tensor[-1, ...].round().to(dtype=torch.int64) + sorted_grid_shape.x // 2
        else:
            sorted_grid_shape.x = ktraj_tensor.shape[-1]
            kx_idx = rearrange(torch.arange(ktraj_tensor.shape[-1]), 'kx->1 1 1 kx')

        if traj_type_kzyx[-2] == TrajType.ONGRID:  # ky
            ky_idx = ktraj_tensor[-2, ...].round().to(dtype=torch.int64) + sorted_grid_shape.y // 2
        else:
            sorted_grid_shape.y = ktraj_tensor.shape[-2]
            ky_idx = rearrange(torch.arange(ktraj_tensor.shape[-2]), 'ky->1 1 ky 1')

        if traj_type_kzyx[-3] == TrajType.ONGRID:  # kz
            kz_idx = ktraj_tensor[-3, ...].round().to(dtype=torch.int64) + sorted_grid_shape.z // 2
        else:
            sorted_grid_shape.z = ktraj_tensor.shape[-3]
            kz_idx = rearrange(torch.arange(ktraj_tensor.shape[-3]), 'kz->1 kz 1 1')

        # 1D indices into a flattened tensor.
        kidx = kz_idx * sorted_grid_shape.y * sorted_grid_shape.x + ky_idx * sorted_grid_shape.x + kx_idx
        kidx = rearrange(kidx, '... kz ky kx -> ... 1 (kz ky kx)')

        # check that all points are inside the encoding matrix
        self._data_outside_of_encoding_matrix = False
        inside_encoding_matrix = (
            (kx_idx >= 0)
            & (kx_idx < sorted_grid_shape.x)
            & (ky_idx >= 0)
            & (ky_idx < sorted_grid_shape.y)
            & (kz_idx >= 0)
            & (kz_idx < sorted_grid_shape.z)
        )
        if torch.any(inside_encoding_matrix):
            warnings.warn(
                'K-space points lie outside of the encoding_matrix and will be ignored.'
                ' Increase the encoding_matrix to include these points.',
                stacklevel=2,
            )

            inside_encoding_matrix = rearrange(inside_encoding_matrix, '... kz ky kx -> ... 1 (kz ky kx)')
            inside_encoding_matrix_idx = torch.broadcast_to(torch.arange(0, kidx.shape[-1]), kidx.shape)
            inside_encoding_matrix_idx = inside_encoding_matrix_idx[inside_encoding_matrix]
            inside_encoding_matrix_idx = torch.reshape(
                inside_encoding_matrix_idx, (*kidx.shape[:-1], inside_encoding_matrix_idx.shape[-1])
            )

            kidx = torch.take_along_dim(kidx, inside_encoding_matrix_idx, dim=-1)
            self._data_outside_of_encoding_matrix = True

        self.register_buffer('_fft_idx', kidx)
        self.register_buffer('_inside_encoding_matrix_idx', inside_encoding_matrix_idx)

        # we can skip the indexing if the data is already sorted and nothing needs to be excluded
        self._needs_indexing = not torch.all(torch.diff(kidx) == 1) or self._data_outside_of_encoding_matrix

        self._trajectory_shape = traj.broadcasted_shape
        self._sorted_grid_shape = sorted_grid_shape

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Forward operator which selects acquired k-space data from k-space.

        Parameters
        ----------
        x
            k-space, fully sampled (or zerofilled) and sorted in Cartesian dimensions
            with shape given by encoding_matrix

        Returns
        -------
            selected k-space data in acquired shape (as described by the trajectory)
        """
        if self._sorted_grid_shape != SpatialDimension(*x.shape[-3:]):
            raise ValueError('k-space data shape mismatch')

        if not self._needs_indexing:
            return (x,)

        x_kflat = rearrange(x, '... coil k2_enc k1_enc k0_enc -> ... coil (k2_enc k1_enc k0_enc)')
        # take_along_dim does broadcast, so no need for extending here
        x_inside_encoding_matrix = torch.take_along_dim(x_kflat, self._fft_idx, dim=-1)

        if self._data_outside_of_encoding_matrix:
            x_indexed = self._broadcast_and_scatter_along_last_dim(
                x_inside_encoding_matrix,
                self._trajectory_shape[-1] * self._trajectory_shape[-2] * self._trajectory_shape[-3],
                self._inside_encoding_matrix_idx,
            )
        else:
            x_indexed = x_inside_encoding_matrix

        # reshape to (... other coil, k2, k1, k0)
        x_reshaped = x_indexed.reshape(x.shape[:-3] + self._trajectory_shape[-3:])

        return (x_reshaped,)

    def adjoint(self, y: torch.Tensor) -> tuple[torch.Tensor,]:
        """Adjoint operator sorting data into the encoding_space matrix.

        Parameters
        ----------
        y
            k-space data in acquired shape

        Returns
        -------
            k-space data sorted into encoding_space matrix
        """
        if self._trajectory_shape[-3:] != y.shape[-3:]:
            raise ValueError('k-space data shape mismatch')

        if not self._needs_indexing:
            return (y,)

        y_kflat = rearrange(y, '... coil k2 k1 k0 -> ... coil (k2 k1 k0)')

        if self._data_outside_of_encoding_matrix:
            y_kflat = torch.take_along_dim(y_kflat, self._inside_encoding_matrix_idx, dim=-1)

        y_scattered = self._broadcast_and_scatter_along_last_dim(
            y_kflat, self._sorted_grid_shape.z * self._sorted_grid_shape.y * self._sorted_grid_shape.x, self._fft_idx
        )

        # reshape to  ..., other, coil, k2_enc, k1_enc, k0_enc
        y_reshaped = y_scattered.reshape(
            *y.shape[:-3],
            self._sorted_grid_shape.z,
            self._sorted_grid_shape.y,
            self._sorted_grid_shape.x,
        )

        return (y_reshaped,)

    @staticmethod
    def _broadcast_and_scatter_along_last_dim(
        data_to_scatter: torch.Tensor, n_last_dim: int, scatter_index: torch.Tensor
    ) -> torch.Tensor:
        """Broadcast scatter index and scatter into zero tensor.

        Parameters
        ----------
        data_to_scatter
            Data to be scattered at indices scatter_index
        n_last_dim
            Number of data points in last dimension
        scatter_index
            Indices describing where to scatter data

        Returns
        -------
            Data scattered into tensor along scatter_index
        """
        # scatter does not broadcast, so we need to manually broadcast the indices
        broadcast_shape = torch.broadcast_shapes(scatter_index.shape[:-1], data_to_scatter.shape[:-1])
        idx_expanded = torch.broadcast_to(scatter_index, (*broadcast_shape, scatter_index.shape[-1]))

        # although scatter_ is inplace, this will not cause issues with autograd, as self
        # is always constant zero and gradients w.r.t. src work as expected.
        data_scattered = torch.zeros(
            *broadcast_shape,
            n_last_dim,
            dtype=data_to_scatter.dtype,
            device=data_to_scatter.device,
        ).scatter_(dim=-1, index=idx_expanded, src=data_to_scatter)

        return data_scattered
