"""Tests for Fourier operator."""

import pytest
import torch
from mrpro.data import KData, KTrajectory, SpatialDimension
from mrpro.data.traj_calculators import KTrajectoryCartesian
from mrpro.operators import FourierOp

from tests import RandomGenerator
from tests.conftest import COMMON_MR_TRAJECTORIES, create_traj
from tests.helper import dotproduct_adjointness_test


def create_data(im_shape, k_shape, nkx, nky, nkz, sx, sy, sz):
    random_generator = RandomGenerator(seed=0)

    # generate random image
    img = random_generator.complex64_tensor(size=im_shape)
    # create random trajectories
    trajectory = create_traj(k_shape, nkx, nky, nkz, sx, sy, sz)
    return img, trajectory


@COMMON_MR_TRAJECTORIES
def test_fourier_op_fwd_adj_property(im_shape, k_shape, nkx, nky, nkz, sx, sy, sz, s0, s1, s2):
    """Test adjoint property of Fourier operator."""

    # generate random images and k-space trajectories
    img, trajectory = create_data(im_shape, k_shape, nkx, nky, nkz, sx, sy, sz)

    # create operator
    recon_matrix = SpatialDimension(im_shape[-3], im_shape[-2], im_shape[-1])
    encoding_matrix = SpatialDimension(
        int(trajectory.kz.max() - trajectory.kz.min() + 1),
        int(trajectory.ky.max() - trajectory.ky.min() + 1),
        int(trajectory.kx.max() - trajectory.kx.min() + 1),
    )
    fourier_op = FourierOp(recon_matrix=recon_matrix, encoding_matrix=encoding_matrix, traj=trajectory)

    # apply forward operator
    (kdata,) = fourier_op(img)

    # test adjoint property; i.e. <Fu,v> == <u, F^Hv> for all u,v
    random_generator = RandomGenerator(seed=0)
    u = random_generator.complex64_tensor(size=img.shape)
    v = random_generator.complex64_tensor(size=kdata.shape)
    dotproduct_adjointness_test(fourier_op, u, v)


@pytest.mark.parametrize(
    ('im_shape', 'k_shape', 'nkx', 'nky', 'nkz', 'sx', 'sy', 'sz'),
    [
        # Cartesian FFT dimensions are not aligned with corresponding k2, k1, k0 dimensions
        (
            (5, 3, 48, 16, 32),
            (5, 3, 96, 18, 64),
            (5, 1, 18, 64),
            (5, 96, 1, 1),  # Cartesian ky dimension defined along k2 rather than k1
            (5, 1, 18, 64),
            'nuf',
            'uf',
            'nuf',
        ),
    ],
)
def test_fourier_op_not_supported_traj(im_shape, k_shape, nkx, nky, nkz, sx, sy, sz):
    """Test trajectory not supported by Fourier operator."""

    # generate random images and k-space trajectories
    img, trajectory = create_data(im_shape, k_shape, nkx, nky, nkz, sx, sy, sz)

    # create operator
    recon_matrix = SpatialDimension(im_shape[-3], im_shape[-2], im_shape[-1])
    encoding_matrix = SpatialDimension(k_shape[-3], k_shape[-2], k_shape[-1])
    with pytest.raises(NotImplementedError, match='Cartesian FFT dims need to be aligned'):
        FourierOp(recon_matrix=recon_matrix, encoding_matrix=encoding_matrix, traj=trajectory)


def test_fourier_op_cartesian_sorting(ismrmrd_cart):
    """Verify correct sorting of Cartesian k-space data before FFT."""
    kdata = KData.from_file(ismrmrd_cart.filename, KTrajectoryCartesian())
    ff_op = FourierOp.from_kdata(kdata)
    (img,) = ff_op.adjoint(kdata.data)

    # shuffle the kspace points along k0
    permutation_index = torch.randperm(kdata.data.shape[-1])
    kdata_unsorted = KData(
        header=kdata.header,
        data=kdata.data[..., permutation_index],
        traj=KTrajectory.from_tensor(kdata.traj.as_tensor()[..., permutation_index]),
    )
    ff_op_unsorted = FourierOp.from_kdata(kdata_unsorted)
    (img_unsorted,) = ff_op_unsorted.adjoint(kdata_unsorted.data)

    torch.testing.assert_close(img, img_unsorted)
