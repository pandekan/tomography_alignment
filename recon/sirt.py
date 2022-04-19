# ------------------------------------------------
# Copyright 2021 Kanupriya Pande
# Contact kpande@lbl.gov
# ------------------------------------------------

import numpy as np
from scipy import optimize
import time
from projectors import projection_operators


class SIRT(object):
    """
    Class for Iterative reconstruction methods.
    run_sirt: Typical simultaneous iterative reconstruction technique
    run_tikhonov_gd: Gradient descent for solving Tikhonov regularized least-squares problem
    run_lasso_ista: Iterative soft shrinkage method for solving L1 regularized least-squares problem
    """
    
    def __init__(self, geometry, projections, angles, xyz_shifts, comm=None, options={}):
        
        self.geometry = geometry
        self.projections = projections
        self.angles = angles
        self.xyz_shifts = xyz_shifts
        self.comm = comm
        if self.comm is None:
            self.my_rank = 0
        else:
            self.my_rank = self.comm.Get_rank()
        self.n_proj = self.geometry.n_proj
        # how to compute projection and back-projection: 'matrix' or 'linop'
        self.method = options['method'] if 'method' in options else 'matrix'
        self.ground_truth = options['ground_truth'] if 'ground_truth' in options else None
        self.precision = options['precision'] if 'precision' in options else np.float32
        self.rec = options['rec'] if 'rec' in options else None
        if self.rec is None:
            self.rec = np.zeros(self.geometry.vox_shape, dtype=self.precision)
        self.projections = self.projections.astype(self.precision, copy=False)
        self.voxel_mask = options['voxel_mask'] if 'voxel_mask' in options else None
        self.proj_obj = None
        
        self._initialize()
    
    def _initialize(self):
        
        if self.proj_obj is None:
            # create an instance of the projection operator
            self.proj_obj = projection_operators.Projection(self.geometry, method=self.method,
                                                            precision=self.precision, comm=self.comm)
        
        # experimental setup
        self.proj_obj.setup(angles=self.angles, xyz_shifts=self.xyz_shifts)
        
        # compute normalizing matrices
        self.W = self.proj_obj.forward_project(np.asfortranarray(np.ones(self.geometry.vox_shape,
                                                                         dtype=self.precision)))
        self.V = self.proj_obj.back_project(np.ones_like(self.projections))
        
        self.V[self.V == 0.] = np.inf
        self.W[self.W == 0.] = np.inf
        self.V = 1. / self.V
        self.W = 1. / self.W
    
    def run_sirt(self, niter=100, make_plot=False, projections=None, positivity=False):
        
        if projections is not None:
            self.projections = projections
        
        if self.ground_truth is not None:
            self.ground_truth = self.ground_truth.ravel()
            norm_factor = np.linalg.norm(self.ground_truth)
        else:
            norm_factor = np.linalg.norm(self.projections)
        
        stop = 0
        k = 0
        rms_error = np.zeros((niter,))
        convergence = np.zeros((niter,))
        t_start = time.time()
        self.rec = self.rec.ravel()
        while k < niter and not stop:
            res = self.proj_obj.forward_project(self.rec)
            res = self.projections - res
            back_proj = self.proj_obj.back_project(self.W * res)
            
            self.rec += self.V * back_proj
            if positivity:
                self.rec[self.rec < 0.] = 0.
            
            convergence[k] = np.linalg.norm(res)
            if self.ground_truth is None:
                rms_error[k] = convergence[k] / norm_factor
            else:
                rms_error[k] = np.linalg.norm(self.ground_truth - self.rec) / norm_factor
            
            if k > 0 and rms_error[k] > rms_error[k - 1]:
                stop = 1
                if self.my_rank == 0:
                    print('semi-convergence criterion reached: stopping at k %3d with RMSE = %4.5f'
                            % (k, rms_error[k]))
            
            if k > 0 and k % 20 == 0 and self.my_rank==0:
                print('time taken for 20 SIRT iterations = %4.5f' % (time.time() - t_start))
                t_start = time.time()
            
            if make_plot and self.my_rank == 0:
                if k == 0:
                    import matplotlib.pyplot as plt
                    plt.ion()
                    fig, (ax0, ax1, ax2) = plt.subplots(3, 1)
                
                elif k % 20 == 0:
                    ax0.imshow(np.reshape(self.rec, self.geometry.vox_shape)[:, :, self.geometry.vox_shape[2] // 2])
                    ax0.set_title('SIRT iteration %3d' % (k))
                    
                    ax1.cla()
                    ax1.set_title('Root Mean-Squared Error')
                    ax1.semilogy(rms_error[1:k])
                    
                    ax2.cla()
                    ax2.set_title('Convergence')
                    ax2.semilogy(convergence[1:k])
                    
                    plt.show()
                    plt.pause(0.1)
            
            k += 1
        
        return self.rec, rms_error[:k]