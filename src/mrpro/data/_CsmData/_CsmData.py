"""Class for coil sensitivity maps (csm)."""

# Copyright 2023 Physikalisch-Technische Bundesanstalt
#
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#       http://www.apache.org/licenses/LICENSE-2.0
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

from __future__ import annotations

import torch

from mrpro.data import IData
from mrpro.data import QData
from mrpro.data import SpatialDimension
from mrpro.utils.filters import spatial_uniform_filter_3d


class CsmData(QData):
    """Coil sensitivity map class."""

    @staticmethod
    def _iterative_walsh_csm(
        coil_images: torch.Tensor, smoothing_width: SpatialDimension[int], niter: int
    ) -> torch.Tensor:
        """Calculate csm using an iterative version of the Walsh method.

        This function is inspired by https://github.com/ismrmrd/ismrmrd-python-tools.

        More information on the method can be found in
        https://doi.org/10.1002/(SICI)1522-2594(200005)43:5<682::AID-MRM10>3.0.CO;2-G

        Parameters
        ----------
        coil_images
            images for each coil element.
        smoothing_width
            width smoothing filter.
        niter
            number of iterations of Walsh method.
        """
        # Compute the pointwise covariance between coils
        coil_cov = torch.einsum('azyx,bzyx->abzyx', coil_images, coil_images.conj())

        # Smooth the covariance along y-x for 2D and z-y-x for 3D data
        coil_cov = spatial_uniform_filter_3d(coil_cov, filter_width=smoothing_width)

        # At each point in the image, find the dominant eigenvector
        # of the signal covariance matrix using the power method
        v = coil_cov.sum(dim=0)
        for _ in range(niter):
            v /= v.norm(dim=0)
            v = torch.einsum('abzyx,bzyx->azyx', coil_cov, v)
        csm_data = v / v.norm(dim=0)

        # Make sure there are no inf or nan-values due to very small values in the covariance matrix
        # nan_to_num does not work for complexfloat, boolean indexing not with vmap.
        csm_data = torch.where(torch.isfinite(csm_data), csm_data, 0.0)
        return csm_data

    @classmethod
    def from_idata_walsh(
        cls,
        idata: IData,
        smoothing_width: SpatialDimension[int] = SpatialDimension(5, 5, 5),
        niter: int = 3,
        chunk_size_otherdim: int | None = None,
    ) -> CsmData:
        """Create csm object from image data using iterative Walsh method.

        Parameters
        ----------
        idata
            IData object containing the images for each coil element.
        smoothing_width
            width of smoothing filter.
        niter
            number of iterations of Walsh method.
        chunk_size_otherdim:
            How many elements of the other dimensions should be processed at once.
            Default is None, which means that all elements are processed at once.
        """
        csm_fun = torch.vmap(
            lambda img: CsmData._iterative_walsh_csm(img, smoothing_width, niter),
            chunk_size=chunk_size_otherdim,
        )
        csm_data = csm_fun(idata.data)

        return cls(header=idata.header, data=csm_data)

    @classmethod
    def from_idata_inati(cls, idata: IData, ks: int, power: int, padding_mode='circular') -> CsmData:
        """Coil sensitivity maps using the Inati method.

        Details of the method can be found in Inati et al. 2004.

        Parameters
        ----------
        data: Images of shape (coil, E1, E0)
        ks: kernel size
        power: number of iterations
        padding_mode: padding mode for the sliding window
        """

        from mrpro.data._CsmData._inati import coil_map_study_2d_Inati

        return coil_map_study_2d_Inati(torch.Tensor(idata.data.squeeze()), ks, power, padding_mode)
