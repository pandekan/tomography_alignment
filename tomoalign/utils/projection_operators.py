import numpy as np
from scipy import sparse
import sys
from copy import deepcopy
from utilities.ray_voxel_utilities import forward_sparse as ray_forward_sparse
from utilities.ray_voxel_utilities import forward_proj_grad as ray_forward_proj_grad
from utilities.voxel_utilities import forward_sparse as vox_forward_sparse
from utilities.voxel_utilities import forward_proj_grad as vox_forward_proj_grad


class ProjectionMatrix(object):

    def __init__(self, geometry, precision=np.float32):

        self.geometry = geometry
        self.precision = precision
        self.n_proj = None
        self.angles = None
        self.xyz_shift = None
        self.voxel_mask = None

    def projection_matrix(self, alpha=None, beta=None, phi=None, xyz_shift=None, voxel_mask=None):

        if phi is None:
            self.n_proj = self.geometry.n_proj
            phi = np.linspace(0., np.pi, self.n_proj)
        else:
            self.n_proj = np.size(phi)

        if alpha is None:
            alpha = np.zeros_like(phi)

        if beta is None:
            beta = np.zeros_like(phi)
        
        # translational shift
        if xyz_shift is None:
            xyz_shift = np.zeros((self.n_proj, 3))
        
        phi = np.squeeze(phi)
        alpha = np.squeeze(alpha)
        beta = np.squeeze(beta)
        xyz_shift = np.squeeze(xyz_shift)
        if self.n_proj == 1:
            phi = np.array([phi])
            alpha = np.array([alpha])
            beta = np.array([beta])
            xyz_shift = np.array([xyz_shift])

        self.angles = np.array([phi, alpha, beta]).T
        self.xyz_shift = xyz_shift
        self.voxel_mask = voxel_mask

        weights, detector_inds, data_inds = self._forward_ray()

        weights = np.concatenate(weights)
        detector_inds = np.concatenate(detector_inds)
        data_inds = np.concatenate(data_inds)
        
        if self.voxel_mask is not None:
            voxel_mask = self.voxel_mask.ravel().astype(bool)
            mask = voxel_mask[data_inds]
            if np.sum(mask) == 0:
                print('entire object is masked')
                weights *= 0.0
            else:
                data_inds = data_inds[mask]
                #data_inds = _rank_order(data_inds)[0]
                detector_inds = detector_inds[mask]
                weights = weights[mask]

        # note that weights at duplicate (row, col) pair are summed
        proj_mat = sparse.coo_matrix((weights, (detector_inds, data_inds)),
                                     shape=(self.n_proj * self.geometry.n_det, self.geometry.n_vox))

        return sparse.csr_matrix(proj_mat)

    def _forward_voxel(self):

        weights, detector_inds, data_inds = [], [], []
        for iproj in range(self.n_proj):
            # rotate centers for this set of angles and translations
            this_geo = deepcopy(self.geometry)
            this_geo.cor_shift = self.geometry.cor_shift[iproj]
            phi, alpha, beta = self.angles[iproj]
            xyz_shift = self.xyz_shift[iproj]
            dat_inds, inds, wts = vox_forward_sparse(this_geo, alpha, beta, phi, xyz_shift)
            
            weights.append(wts.astype(self.precision, copy=False))
            detector_inds.append((inds + iproj * self.geometry.n_det).astype(np.int32))
            data_inds.append(dat_inds)
            
        return weights, detector_inds, data_inds

    def _forward_ray(self):
    
        weights, detector_inds, data_inds = [], [], []
        for iproj in range(self.n_proj):
            phi, alpha, beta = self.angles[iproj]
            xyz_shift = self.xyz_shift[iproj]
            this_geo = deepcopy(self.geometry)
            this_geo.cor_shift = self.geometry.cor_shift[iproj]
        
            dat_inds, det_inds, wts = ray_forward_sparse(this_geo, alpha, beta, phi, xyz_shift)
        
            weights.append(wts.astype(self.precision, copy=False))
            data_inds.append(dat_inds.astype(np.int32))
            detector_inds.append(det_inds + iproj * self.geometry.n_det)
    
        return weights, detector_inds, data_inds
    
    def projection_gradient(self, rec, alpha, beta, phi, xyz_shift, cor_shift):
        
        this_geo = deepcopy(self.geometry)
        this_geo.cor_shift = cor_shift
        
        proj_img, gradient = ray_forward_proj_grad(this_geo, alpha, beta, phi, xyz_shift, rec)
        
        proj_img = proj_img.astype(self.precision, copy=False)
        gradient = gradient.astype(self.precision, copy=False)
        
        return proj_img.ravel(), gradient.reshape(6, -1) 
