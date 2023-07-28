# ------------------------------------------------
# Copyright 2021 Kanupriya Pande
# Contact kpande@lbl.gov
# ------------------------------------------------

import numpy as np


class Geometry(object):
    """
    Detector and object setup for parallel beam geometry.
    """

    def __init__(
        self,
        n_proj,
        voxel_shape,
        voxel_pixsize,
        detector_shape,
        detector_pixsize,
        cor_shift=None,
        step_size=1.0,
    ):
        """
        :param n_proj: int, number of projections
        :param voxel_shape: int, (3,)
        :param voxel_pixsize: float (3,)
        :param detector_shape: int (2,)
        :param det_pixsize: float (2,)
        """
        self.n_proj = n_proj
        self.vox_shape = voxel_shape
        self.vox_pix = voxel_pixsize
        self.vox_size = self.vox_shape * self.vox_pix
        self.n_vox = np.prod(self.vox_shape)
        self.det_shape = detector_shape
        self.det_pix = detector_pixsize
        self.det_size = self.det_shape * self.det_pix
        self.n_det = np.prod(self.det_shape)
        self.vox_ds = np.array([1, 1, 1])
        if cor_shift is None:
            self.cor_shift = np.zeros((n_proj, 3))
        else:
            if len(cor_shift.shape) == 2:
                assert cor_shift.shape[0] == n_proj
                assert cor_shift.shape[1] == 3
                self.cor_shift = cor_shift
            elif len(cor_shift.shape) == 1:
                assert np.size(cor_shift) == 3
                self.cor_shift = np.tile(cor_shift, n_proj).reshape(n_proj, 3)
            else:
                print("shape or size of cor_shift not valid")

        self.step_size = step_size
        self._voxel_detector_grid()

    def _geo_parameters(self, angles=None, shifts=None):
        """
        :param angles: float (3, n_proj) (tomo, alpha, beta)
        :param shifts: xyz translations
        :return: self.angles and self.shifts
        """
        if angles is None:
            self.angles = np.zeros((3, self.n_proj))
            self.angles[0] = np.linspace(0.0, np.pi, self.n_proj)
        elif len(angles.shape) == 1:
            assert np.size(angles) == self.n_proj
            self.angles = np.zeros((3, self.n_proj))
            self.angles[0] = angles
        else:
            assert angles.shape[1] == self.n_proj
            self.angles = np.zeros((3, self.n_proj))
            self.angles[0] = angles[0]
            self.angles[1] = angles[1]
            if angles.shape[0] == 3:
                self.angles[2] = angles[2]

        if shifts is None:
            self.shifts = np.zeros((3, self.n_proj))
        else:
            assert shifts.shape[0] == 3
            assert shifts.shape[1] == self.n_proj
            self.shifts = shifts

    def _voxel_detector_grid(self):
        # first set up voxel centers and origin
        nx, ny, nz = self.vox_shape
        sx, sy, sz = self.vox_size
        x = np.linspace(-sx / 2, sx / 2, nx, endpoint=False) + 0.5
        y = np.linspace(-sy / 2, sy / 2, ny, endpoint=False) + 0.5
        z = np.linspace(-sz / 2, sz / 2, nz, endpoint=False) + 0.5
        x, y, z = np.meshgrid(x, y, z, indexing="ij")
        self.vox_centers = np.array([x.ravel(), y.ravel(), z.ravel()])
        self.vox_origin = np.array([x.min(), y.min(), z.min()])

        # now set up detector grid
        nx, nz = self.det_shape
        sx, sz = self.det_size
        x = np.linspace(-sx / 2, sx / 2, nx, endpoint=False) + 0.5
        z = np.linspace(-sz / 2, sz / 2, nz, endpoint=False) + 0.5
        xd, zd = np.meshgrid(x, z, indexing="ij")
        y_source = -sy
        y_det = sy

        # source and detector positions are needed for ray based method
        self.source_centers = np.array(
            [xd.ravel(), y_source * np.ones((self.n_det,)), zd.ravel()]
        )
        self.det_centers = np.array(
            [xd.ravel(), y_det * np.ones((self.n_det,)), zd.ravel()]
        )

        # this information is needed for voxel based method
        self.det_orig = np.array([x.min(), y.min(), z.min()])
        sx, sz = float(self.vox_shape[0] / self.det_shape[0]), float(
            self.vox_shape[2] / self.det_shape[1]
        )
        self.factor = np.array([sx, 1.0, sz])
