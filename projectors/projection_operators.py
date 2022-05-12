# ------------------------------------------------
# Copyright 2021 Kanupriya Pande
# Contact kpande@lbl.gov
# ------------------------------------------------

import numpy as np
from scipy import sparse
from projectors.ray_tracing import forward_sparse as ray_forward_sparse
from projectors.ray_tracing import forward_proj_grad as ray_forward_proj_grad
from src import forward_projection, back_projection
from src import projection_gradient

try:
    from mpi4py import MPI
    mpi_available = True
except ImportError:
    mpi_available = False
    

class Projection(object):

    def __init__(self, geometry, method='linop', precision=np.float32, comm=None):
        
        self.geometry = geometry
        self.precision = precision
        self.method = method
        self.n_proj = self.geometry.n_proj
        self.n_rays = self.geometry.n_det
        self.comm = comm
        if self.comm is None:
            self.size = 1
            self.my_rank = 0
        else:
            self.size = self.comm.Get_size()
            self.my_rank = self.comm.Get_rank()
            
        self.alpha = None
        self.beta = None
        self.phi = None
        self.xyz_shifts = None

    def setup(self, angles=None, xyz_shifts=None):
    
        if angles is None:
            self.alpha = np.zeros(self.n_proj, )
            self.beta = np.zeros(self.n_proj, )
            self.phi = np.linspace(0.0, np.pi, self.n_proj)
        else:
            assert (angles.shape[0] == self.n_proj)
            self.alpha = angles[:, 0]
            self.beta = angles[:, 1]
            self.phi = angles[:, 2]
    
        if xyz_shifts is None:
            self.xyz_shifts = np.zeros((self.n_proj, 3))
        else:
            self.xyz_shifts = xyz_shifts
            assert (self.xyz_shifts.shape[0] == self.n_proj)
            assert (self.xyz_shifts.shape[1] == 3)
        
        self.alpha = self.alpha.astype(np.float32, copy=False)
        self.beta = self.beta.astype(np.float32, copy=False)
        self.phi = self.phi.astype(np.float32, copy=False)
        self.xyz_shifts = self.xyz_shifts.astype(np.float32, copy=False)
        self.geometry.cor_shift = self.geometry.cor_shift.astype(np.float32, copy=False)
        self.geometry.source_centers = self.geometry.source_centers.astype(np.float32, copy=False)
        self.geometry.det_centers = self.geometry.det_centers.astype(np.float32, copy=False)
        self.geometry.vox_origin = self.geometry.vox_origin.astype(np.float32, copy=False)
        self.geometry.vox_centers = self.geometry.vox_centers.astype(np.float32, copy=False)
        self.geometry.step_size = np.float32(self.geometry.step_size)

        if self.comm is not None:
            if self.method == 'matrix':
                self.pmat = None
                # if using method 'matrix' we parallelize along projections
                split_index = np.array_split(np.arange(self.n_proj), self.size)
                my_index = split_index[self.my_rank]
                self.my_proj_index = my_index
                self.my_n_proj = np.size(my_index)
                self.my_phi, self.my_alpha, self.my_beta = self.phi[my_index], self.alpha[my_index], self.beta[my_index]
                self.my_xyz_shifts = self.xyz_shifts[my_index, :]
                self.my_cor_shift = self.geometry.cor_shift[my_index, :]
                self.my_source_centers = self.geometry.source_centers[:, :]
                self.my_det_centers = self.geometry.det_centers[:, :]
                self.my_n_rays = self.geometry.n_det
                self.counts = self.geometry.n_det * np.array([np.size(split_index[i]) for i in range(self.size)])
                self.displacements = np.insert(np.cumsum(self.counts), 0, 0)[0:-1]
            else:
                # if using method 'linop' we parallelize along rays
                split_index = np.array_split(np.arange(self.n_rays), self.size)
                my_rays = split_index[self.my_rank]
                self.my_rays = my_rays
                self.my_n_proj = self.n_proj
                self.my_phi, self.my_alpha, self.my_beta = self.phi, self.alpha, self.beta
                self.my_xyz_shifts = self.xyz_shifts
                self.my_cor_shift = self.geometry.cor_shift
                self.my_source_centers = self.geometry.source_centers[:, my_rays]
                self.my_det_centers = self.geometry.det_centers[:, my_rays]
                self.my_n_rays = np.size(my_rays)
                self.counts = self.n_proj * np.array([np.size(split_index[i]) for i in range(self.size)])
                self.displacements = np.insert(np.cumsum(self.counts), 0, 0)[0:-1]

                # for back-projection parallelize along voxel centers
                split_centers = np.array_split(np.arange(self.geometry.n_vox), self.size)
                self.my_centers = split_centers[self.my_rank]
                self.my_vox_centers = self.geometry.vox_centers[:, self.my_centers]
                self.my_n_vox = self.my_vox_centers.shape[1]
                self.vox_counts = np.array([np.size(split_centers[i]) for i in range(self.size)])
                self.vox_displacements = np.insert(np.cumsum(self.vox_counts), 0, 0)[0:-1]
        else:
            self.pmat = None
            self.my_n_proj = self.n_proj
            self.my_phi, self.my_alpha, self.my_beta = self.phi, self.alpha, self.beta
            self.my_xyz_shifts = self.xyz_shifts
            self.my_cor_shift = self.geometry.cor_shift
            self.my_source_centers = self.geometry.source_centers
            self.my_det_centers = self.geometry.det_centers
            self.my_n_rays = self.n_rays
            self.my_vox_centers = self.geometry.vox_centers
            self.my_n_vox = self.geometry.n_vox
            
    def forward_project(self, rec):
        
        rec = rec.ravel().astype(self.precision)
        if self.method == 'matrix':
            f_proj = self._forward_matrix(rec)
        else:
            f_proj = self._forward_linop(rec)
    
        return f_proj

    def _forward_matrix(self, rec):
        
        if self.pmat is None:
            self._sparse_projection_matrix()
            
        if self.comm is None:
            f_proj = sparse.csr_matrix.dot(self.pmat, rec)
        else:
            if self.precision == np.float64:
                mpi_precision = MPI.DOUBLE
            else:
                mpi_precision = MPI.FLOAT

            if self.my_n_proj > 0:
                my_proj = sparse.csr_matrix.dot(self.pmat, rec)
            
                # now gather my_proj and dump into f_proj
                f_proj = np.zeros((self.n_proj * self.geometry.n_det, ), dtype=self.precision)
                self.comm.Gatherv(np.ascontiguousarray(my_proj), 
                                  [f_proj, self.counts, self.displacements, mpi_precision], root=0)
            else:
                f_proj = None
            self.comm.Barrier()
            f_proj = self.comm.bcast(f_proj, root=0)
        
        f_proj = np.reshape(f_proj, (self.n_proj, self.geometry.det_shape[0], self.geometry.det_shape[1]))
        
        return f_proj

    def _sparse_projection_matrix(self):
        
        nx, ny, nz = self.geometry.vox_shape
        if self.pmat is None and self.my_n_proj > 0:
            weights, detector_inds, data_inds = [], [], []
            for iproj in range(self.my_n_proj):
                phi, alpha, beta = self.my_phi[iproj], self.my_alpha[iproj], self.my_beta[iproj]
                xyz_shifts = self.my_xyz_shifts[iproj]
                cor_shift = self.my_cor_shift[iproj]
            
                dat_inds, det_inds, wts = ray_forward_sparse(alpha, beta, phi, xyz_shifts, cor_shift,
                                                             self.my_source_centers, self.my_det_centers,
                                                             self.geometry.vox_origin,
                                                             self.geometry.step_size, self.my_n_rays,
                                                             nx, ny, nz)
            
                weights.append(wts.astype(self.precision, copy=False))
                data_inds.append(dat_inds.astype(np.int32))
                detector_inds.append(det_inds + iproj * self.geometry.n_det)
            weights = np.concatenate(weights)
            detector_inds = np.concatenate(detector_inds)
            data_inds = np.concatenate(data_inds)
        
            # create a sparse matrix
            self.pmat = sparse.coo_matrix((weights, (detector_inds, data_inds)),
                                          shape=(self.my_n_proj * self.geometry.n_det, self.geometry.n_vox),
                                          dtype=self.precision)
            self.pmat = sparse.csr_matrix(self.pmat, dtype=self.precision)
            
    def _forward_linop(self, rec):
        nx, ny, nz = self.geometry.vox_shape
        n_vox = self.geometry.n_vox 
        if self.comm is None:
            f_proj = forward_projection.forward_project(self.alpha, self.beta, self.phi, self.xyz_shifts.T,
                                                        self.geometry.cor_shift.T,
                                                        self.geometry.source_centers, self.geometry.det_centers,
                                                        self.geometry.vox_origin, self.geometry.step_size,
                                                        nx, ny, nz, rec, self.n_proj, self.geometry.n_det, n_vox)
        else:
            pp = np.zeros((self.n_proj, self.geometry.n_det), dtype=np.float32)
            f_proj = np.zeros((self.n_proj, self.geometry.n_det), dtype=np.float32)
            if np.size(self.my_alpha) > 0 and self.my_source_centers.shape[1] > 0:
                my_proj = forward_projection.forward_project(self.my_alpha, self.my_beta, self.my_phi,
                                                             self.my_xyz_shifts.T, self.my_cor_shift.T,
                                                             self.my_source_centers, self.my_det_centers,
                                                             self.geometry.vox_origin, self.geometry.step_size,
                                                             nx, ny, nz, rec, self.my_n_proj, self.my_n_rays, n_vox)
            
                #f_proj = np.zeros((self.n_proj * self.geometry.n_det,), dtype=np.float32)
                #self.comm.Gatherv(np.ascontiguousarray(np.ravel(my_proj, order='F')),
                #                  [f_proj, self.counts, self.displacements, MPI.FLOAT], root=0)
                pp[:, self.my_rays] = my_proj
            #else:
                #f_proj = None
                 
            self.comm.Barrier()
            #f_proj = self.comm.bcast(f_proj, root=0)
            self.comm.Allreduce([pp, MPI.FLOAT], [f_proj, MPI.FLOAT], op=MPI.SUM)
        # this will be consistent with the result for method 'matrix'
        f_proj = np.reshape(f_proj, (self.n_proj, self.geometry.det_shape[0], self.geometry.det_shape[1]))
        f_proj = np.transpose(f_proj, (0, 2, 1))
    
        return f_proj

    def back_project(self, projections):
    
        projections = projections.astype(np.float32, copy=False)
        if self.method == 'matrix':
            b_proj = self._back_matrix(projections)
        else:
            b_proj = self._back_linop(projections)
    
        return b_proj

    def _back_matrix(self, projections):
    
        if self.pmat is None:
            self._sparse_projection_matrix()
    
        if self.comm is None:
            b_proj = sparse.csc_matrix.dot(sparse.csr_matrix.transpose(self.pmat), projections.ravel())
        else:
            b_proj = np.zeros((self.geometry.n_vox, ), dtype=self.precision)
            if np.size(self.my_proj_index) > 0:
                my_proj = projections[self.my_proj_index]
                my_proj = my_proj.ravel() #np.ravel(my_proj, order='F')
                my_b_proj = sparse.csc_matrix.dot(sparse.csr_matrix.transpose(self.pmat), my_proj)
            else:
                my_b_proj = np.zeros((self.geometry.n_vox, ), dtype=self.precision)
            self.comm.Allreduce([my_b_proj, MPI.FLOAT], [b_proj, MPI.FLOAT], op=MPI.SUM)
    
        return b_proj

    def _back_linop(self, projections):
        
        if self.comm is None:
            b_proj = back_projection.back_project(-self.alpha, -self.beta, -self.phi, -self.xyz_shifts.T,
                                                  self.geometry.vox_centers, self.geometry.vox_origin,
                                                  np.asfortranarray(projections), self.n_proj, self.geometry.n_vox,
                                                  self.geometry.det_shape[0], self.geometry.det_shape[1])
        else:
            b_proj = np.zeros((self.geometry.n_vox,), dtype=np.float32)
            bb = np.zeros_like(b_proj)
            if self.my_n_vox > 0:
                my_b_proj = back_projection.back_project(-self.my_alpha, -self.my_beta, -self.my_phi, 
                                                         -self.my_xyz_shifts.T,
                                                         self.my_vox_centers, self.geometry.vox_origin,
                                                         np.asfortranarray(projections),
                                                         self.my_n_proj, self.my_n_vox,
                                                         self.geometry.det_shape[0], self.geometry.det_shape[1])
                #b_proj = np.zeros((self.geometry.n_vox,), dtype=np.float32)
                #self.comm.Gatherv(np.ascontiguousarray(my_b_proj),
                #                  [b_proj, self.vox_counts, self.vox_displacements, MPI.FLOAT], root=0)
                bb[self.my_centers] = my_b_proj
            #else:
            #    b_proj = None

            #b_proj = self.comm.bcast(b_proj, root=0)
            self.comm.Allreduce([bb, MPI.FLOAT], [b_proj, MPI.FLOAT], op=MPI.SUM)
        return b_proj


class ProjectionGradient(object):
    
    def __init__(self, geometry, precision=np.float32, comm=None):
        
        self.geometry = geometry
        self.precision = precision
        self.n_rays = self.geometry.n_det
        self.comm = comm
        if self.comm is None:
            self.size = 1
            self.my_rank = 0
        else:
            self.size = self.comm.Get_size()
            self.my_rank = self.comm.Get_rank()

        self.geometry.source_centers = self.geometry.source_centers.astype(np.float32, copy=False)
        self.geometry.det_centers = self.geometry.det_centers.astype(np.float32, copy=False)
        self.geometry.vox_origin = self.geometry.vox_origin.astype(np.float32, copy=False)
        self.geometry.vox_centers = self.geometry.vox_centers.astype(np.float32, copy=False)
        self.geometry.step_size = np.float32(self.geometry.step_size)

        if self.comm is not None:
            # parallelize along rays
            split_index = np.array_split(np.arange(self.n_rays), self.size)
            my_rays = split_index[self.my_rank]
            self.my_rays = my_rays
            self.my_source_centers = self.geometry.source_centers[:, my_rays]
            self.my_det_centers = self.geometry.det_centers[:, my_rays]
            self.my_n_rays = np.size(my_rays)
            self.counts = np.array([np.size(split_index[i]) for i in range(self.size)])
            self.displacements = np.insert(np.cumsum(self.counts), 0, 0)[0:-1]
        else:
            self.my_source_centers = self.geometry.source_centers
            self.my_det_centers = self.geometry.det_centers
            self.my_n_rays = self.n_rays
            self.my_vox_centers = self.geometry.vox_centers
            self.my_n_vox = self.geometry.n_vox
    
    def proj_gradient(self, recon, alpha, beta, phi, xyz_shift, cor_shift):
        
        recon = recon.ravel().astype(self.precision)
        nx, ny, nz = self.geometry.vox_shape
        p_image, p_gradient = projection_gradient.compute_gradient(alpha, beta, phi, xyz_shift, cor_shift,
                                                                   self.my_source_centers, self.my_det_centers,
                                                                   self.geometry.vox_origin, self.geometry.step_size,
                                                                   nx, ny, nz, recon, self.my_n_rays,
                                                                   self.geometry.n_vox)
        if self.comm is None:
            return p_image, p_gradient
        else:
            f_temp = np.zeros((self.geometry.n_det, ), dtype=np.float32)
            fp_temp = np.zeros((6, self.geometry.n_det), dtype=np.float32)
            f_proj = np.zeros((self.geometry.n_det, ), dtype=np.float32)
            fp_proj = np.zeros((6, self.geometry.n_rays, ), dtype=np.float32)
            if self.my_n_rays > 0:
                f_temp[self.my_rays] = p_image
                fp_temp[:, self.my_rays] = p_gradient
            
            self.comm.Allreduce([f_temp, MPI.FLOAT], [f_proj, MPI.FLOAT], op=MPI.SUM)
            self.comm.Allreduce([fp_temp, MPI.FLOAT], [fp_proj, MPI.FLOAT], op=MPI.SUM)
        
        self.comm.Barrier()
        return f_proj, fp_proj

    
def _rank_order(image):
    
    flat_image = image.ravel()
    sort_order = flat_image.argsort().astype(np.uint32)
    flat_image = flat_image[sort_order]
    sort_rank = np.zeros_like(sort_order)
    is_different = flat_image[:-1] != flat_image[1:]
    np.cumsum(is_different, out=sort_rank[1:])
    original_values = np.zeros((sort_rank[-1] + 1,), image.dtype)
    original_values[0] = flat_image[0]
    original_values[1:] = flat_image[1:][is_different]
    int_image = np.zeros_like(sort_order)
    int_image[sort_order] = sort_rank
    
    return int_image.reshape(image.shape), original_values
