# ------------------------------------------------
# Copyright 2021 Kanupriya Pande
# Contact kpande@lbl.gov
# ------------------------------------------------

import numpy as np
from src import forward_projection, back_projection
try:
    from mpi4py import MPI
    mpi_available = True
except ImportError:
    mpi_available = False


def profile(fnc): 
    import cProfile, pstats, io
    
    def inner(*args, **kwargs):
        
        pr = cProfile.Profile()
        pr.enable()
        retval = fnc(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        return retval
    
    return inner


class ForwardProjection(object):
    
    def __init__(self, geometry, method, precision=np.float32, comm=None):
        
        self.geometry = geometry
        self.method = method
        self.precision = precision
        self.n_proj = geometry.n_proj
        self.n_rays = geometry.n_det
        self.comm = comm
        
        if self.comm is not None:
            self.size = self.comm.Get_size()
            self.my_rank = self.comm.Get_rank()
        else:
            self.size = 1
            self.my_rank = 0

        self.phi = None
        self.alpha = None
        self.beta = None
        self.xyz_shifts = None
        
    def orientation_setup(self, angles=None, xyz_shifts=None):
        
        if angles is None:
            self.phi = np.linspace(0.0, np.pi, self.n_proj)
            self.alpha = np.zeros(self.n_proj, )
            self.beta = np.zeros(self.n_proj, )
        else:
            assert (angles.shape[0] == self.n_proj)
            self.phi = angles[:, 0]
            self.alpha = angles[:, 1]
            self.beta = angles[:, 2]
    
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

    def _parallel_setup(self, parallelize_projections=False):
        
        if parallelize_projections:
            # parallel loop runs over projections
            split_index = np.array_split(np.arange(self.n_proj), self.size)
            my_index = split_index[self.my_rank]
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
            split_index = np.array_split(np.arange(self.n_rays), self.size)
            my_rays = split_index[self.my_rank]
            self.my_n_proj = self.n_proj
            self.my_phi, self.my_alpha, self.my_beta = self.phi, self.alpha, self.beta
            self.my_xyz_shifts = self.xyz_shifts
            self.my_cor_shift = self.geometry.cor_shift
            self.my_source_centers = self.geometry.source_centers[:, my_rays]
            self.my_det_centers = self.geometry.det_centers[:, my_rays]
            self.my_n_rays = np.size(my_rays)
            self.counts = self.n_proj * np.array([np.size(split_index[i]) for i in range(self.size)])
            self.displacements = np.insert(np.cumsum(self.counts), 0, 0)[0:-1]
    
        self.my_cor_shift = self.my_cor_shift.astype(np.float32, copy=False)
        self.my_source_centers = self.my_source_centers.astype(np.float32, copy=False)
        self.my_det_centers = self.my_det_centers.astype(np.float32, copy=False)

    def forward_project(self, rec):
    
        rec = rec.astype(np.float32)
        nx, ny, nz = self.geometry.vox_shape
        
        if self.comm is None:
            f_proj = forward_projection.forward_project(self.alpha, self.beta, self.phi, self.xyz_shifts.T,
                                                        self.geometry.cor_shift.T,
                                                        self.geometry.source_centers, self.geometry.det_centers,
                                                        self.geometry.vox_origin, self.geometry.step_size,
                                                        rec, self.n_proj, self.geometry.n_det, nx, ny, nz)
        else:
            if np.size(self.my_alpha) > 0 and self.my_source_centers.shape[1] > 0:
                my_proj = forward_projection.forward_project(self.my_alpha, self.my_beta, self.my_phi,
                                                             self.my_xyz_shifts.T, self.my_cor_shift.T,
                                                             self.my_source_centers, self.my_det_centers,
                                                             self.geometry.vox_origin, self.geometry.step_size,
                                                             rec, self.my_n_proj, self.my_n_rays, nx, ny, nz)
            
                f_proj = np.zeros((self.n_proj * self.geometry.n_det, ), dtype=np.float64)
                self.comm.Gatherv(np.ascontiguousarray(np.ravel(my_proj, order='F')),
                                  [f_proj, self.counts, self.displacements, MPI.DOUBLE], root=0)
            else:
                f_proj = None
            
            self.comm.Barrier()
            f_proj = self.comm.bcast(f_proj, root=0)

        return f_proj

    
class BackProjection(object):
    
    def __init__(self, geometry, method, precision=np.float64, comm=None):
        
        self.geometry = geometry
        self.method = method
        self.precision = precision
        self.n_proj = geometry.n_proj
        self.n_rays = geometry.n_det
        self.comm = comm

        if self.comm is not None:
            self.size = self.comm.Get_size()
            self.my_rank = self.comm.Get_rank()
        else:
            self.size = 1
            self.my_rank = 0

        self.phi = None
        self.alpha = None
        self.beta = None
        self.xyz_shifts = None

    def orientation_setup(self, angles=None, xyz_shifts=None):
    
        if angles is None:
            self.phi = np.linspace(0.0, np.pi, self.n_proj)
            self.alpha = np.zeros(self.n_proj, )
            self.beta = np.zeros(self.n_proj, )
        else:
            assert (angles.shape[0] == self.n_proj)
            self.phi = angles[:, 0]
            self.alpha = angles[:, 1]
            self.beta = angles[:, 2]
    
        if xyz_shifts is None:
            self.xyz_shifts = np.zeros((self.n_proj, 3))
        else:
            self.xyz_shifts = xyz_shifts
            assert (self.xyz_shifts.shape[0] == self.n_proj)
            assert (self.xyz_shifts.shape[1] == 3)

    def _parallel_setup(self):
    
        # parallel loop runs over projections
        split_index = np.array_split(np.arange(self.n_proj), self.size)
        my_index = split_index[self.my_rank]
        self.my_index = my_index
        self.my_n_proj = np.size(my_index)
        self.my_phi, self.my_alpha, self.my_beta = self.phi[my_index], self.alpha[my_index], self.beta[my_index]
        self.my_xyz_shifts = self.xyz_shifts[my_index, :]

    def back_project(self, projections):
    
        if self.comm is None:
            b_proj = back_projection.back_project(-self.alpha, -self.beta, -self.phi, -self.xyz_shifts.T,
                                                  self.geometry.vox_centers, self.geometry.vox_origin,
                                                  np.asfortranarray(projections), self.n_proj, self.geometry.n_vox,
                                                  self.geometry.det_shape[0], self.geometry.det_shape[1])
        else:
            if self.my_n_proj > 0:
                my_b_proj = back_projection.back_project(-self.my_alpha, -self.my_beta, -self.my_phi,
                                                         -self.my_xyz_shifts.T,
                                                         self.geometry.vox_centers, self.geometry.vox_origin,
                                                         np.asfortranarray(projections[self.my_index]), self.my_n_proj,
                                                         self.geometry.n_vox,
                                                         self.geometry.det_shape[0], self.geometry.det_shape[1])
            else:
                my_b_proj = np.zeros((self.geometry.n_vox, ), dtype=self.precision)
            
            b_proj = np.zeros_like(my_b_proj)
            self.comm.Allreduce([my_b_proj, MPI.DOUBLE], [b_proj, MPI.DOUBLE], op=MPI.SUM)
    
        return b_proj
