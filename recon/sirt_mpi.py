import numpy as np
from scipy import sparse
import time
from mpi4py import MPI
import matplotlib.pyplot as plt
from utilities import projection_operators
import copy


class SIRT(object):
    
    def __init__(self, comm, geometry, projections, angles, xyz_shifts, options={}):
        self.comm = comm
        self.geometry = geometry
        self.projections = projections
        self.angles = angles
        self.xyz_shifts = xyz_shifts
        self.n_proj = angles.shape[0]

        self.ground_truth = options['ground_truth'] if 'ground_truth' in options else None
        self.rec = options['rec'] if 'rec' in options else None
        if self.rec is None:
            self.rec = np.zeros((self.geometry.n_vox,), dtype=self.projections.dtype)
        self.precision = options['precision'] if 'precision' in options else np.float32
        self.voxel_mask = options['voxel_mask'] if 'voxel_mask' in options else None 
        # make sure all arrays have the same precision as the projector
        self.projections = self.projections.astype(self.precision, copy=False)
        self.rec = self.rec.astype(self.precision, copy=False)
        if self.ground_truth is not None:
            self.ground_truth = self.ground_truth.astype(self.precision, copy=False)
        self.f_proj_obj = None
        self.proj_mat = None
        self.rms_error = None
        self._initialize()

    def _initialize(self):
    
        self.size = self.comm.Get_size()
        self.my_rank = self.comm.Get_rank()
        self.my_index = np.array_split(np.arange(self.n_proj), self.size)[self.my_rank]
        self.my_n_proj = np.size(self.my_index)
        self.my_angles = self.angles[self.my_index]
        self.my_xyz_shifts = self.xyz_shifts[self.my_index]
        self.my_geom = copy.deepcopy(self.geometry)
        self.my_geom.n_proj = self.my_n_proj
        if self.my_geom.n_proj == 1:
            self.my_geom.cor_shift = np.array([self.geometry.cor_shift[self.my_index]])
        else:
            self.my_geom.cor_shift = self.geometry.cor_shift[self.my_index]

        if self.precision == np.float32:
            mpi_float = MPI.FLOAT
        else:
            mpi_float = MPI.DOUBLE
            
        self.V = np.zeros((self.geometry.n_vox, ), dtype=self.precision)
        if self.f_proj_obj is None:
            # create an instance of the projection operator
            self.f_proj_obj = projection_operators.ProjectionMatrix(self.my_geom, precision=self.precision)
            self.proj_mat = self.f_proj_obj.projection_matrix(phi=self.my_angles[:, 0], alpha=self.my_angles[:, 1],
                                                              beta=self.my_angles[:, 2],
                                                              xyz_shift=self.my_xyz_shifts,
                                                              voxel_mask=self.voxel_mask)
        self.my_W = sparse.csr_matrix.dot(self.proj_mat, np.ones((self.geometry.n_vox, ), dtype=self.precision))
        self.my_V = sparse.csc_matrix.dot(sparse.csr_matrix.transpose(self.proj_mat),
                                          np.ones((self.my_n_proj * self.geometry.n_det, ), dtype=self.precision))
        
        self.comm.Allreduce([self.my_V, mpi_float], [self.V, mpi_float], op=MPI.SUM)
        self.V[self.V < 1.e-8] = np.inf
        self.my_W[self.my_W < 1.e-8] = np.inf
        self.V = 1./self.V
        self.my_W = 1./self.my_W

    def run_main_iteration(self, niter=100, positivity=False, make_plot=False, debug=False):

        self.rms_error = np.zeros((niter, ))
        
        if self.ground_truth is not None:
            self.ground_truth = self.ground_truth.ravel()
            norm_factor = np.linalg.norm(self.ground_truth)
        else:
            norm_factor = np.linalg.norm(self.projections)
        
        if self.precision == np.float32:
            mpi_float = MPI.FLOAT
        else:
            mpi_float = MPI.DOUBLE
        
        stop = 0
        k = 0
        convergence = np.zeros((niter,))
        while k < niter and not stop:
            t_start = time.time()
            rec = np.zeros_like(self.rec)
                
            res = sparse.csr_matrix.dot(self.proj_mat, self.rec)
            res = self.projections[self.my_index] - res.reshape(self.my_n_proj, -1)
            my_back_proj = sparse.csc_matrix.dot(sparse.csr_matrix.transpose(self.proj_mat), self.my_W*res.ravel())
                
            my_norm = np.linalg.norm(res)
            my_back_proj *= self.V
            self.comm.Barrier()
            self.comm.Allreduce([my_back_proj, mpi_float], [rec, mpi_float], op=MPI.SUM)
            
            self.rec += rec
            if positivity:
                self.rec[self.rec < 0.] = 0.
            # convergence[k] = np.linalg.norm(current_rec - old_rec) /(np.linalg.norm(current_rec)+eps)
            
            convergence[k] = np.sqrt(self.comm.allreduce(my_norm**2, op=MPI.SUM))
            if self.ground_truth is None:
                self.rms_error[k] = convergence[k]/norm_factor
            else:
                self.rms_error[k] = np.linalg.norm(self.ground_truth - self.rec)/norm_factor
            
            if k > 1 and self.rms_error[k] > self.rms_error[k-1]:
                stop = 1
                if self.my_rank == 0:
                    print('semi-convergence criterion reached: stopping at k %3d with RMSE = %4.5f'
                          % (k, self.rms_error[k]))
        
            if make_plot and self.my_rank == 0:
                if k == 0:
                    plt.ion()
                    fig, (ax0, ax1, ax2) = plt.subplots(3, 1)

                elif k % 20 == 0:
                    ax0.imshow(self.rec.reshape(self.geometry.vox_shape)[:, :, self.geometry.vox_shape[2]//2])
                    ax0.set_title('SIRT iteration %3d' % (k))

                    ax1.cla()
                    ax1.set_title('Root Mean-Squared Error')
                    ax1.semilogy(self.rms_error[1:k])

                    ax2.cla()
                    ax2.set_title('Convergence')
                    ax2.semilogy(convergence[1:k])

                    plt.show()
                    plt.pause(0.1)
            k += 1
            if debug:
                print('in rank %d: time taken for one SIRT iteration for %d projections = %4.5f'
                      % (self.my_rank, self.my_n_proj, time.time()-t_start))
    
        return self.rec.reshape(self.geometry.vox_shape), self.rms_error[:k]
    
    def run_regularized_gradient_descent(self, niter=100, reg_param=1.0, positivity=False,
                                         make_plot=False, debug=False):
        
        from scipy import optimize

        if self.ground_truth is not None:
            self.ground_truth = self.ground_truth.ravel()
            norm_factor = np.linalg.norm(self.ground_truth)
        else:
            norm_factor = np.linalg.norm(self.projections)

        if self.precision == np.float32:
            mpi_float = MPI.FLOAT
        else:
            mpi_float = MPI.DOUBLE

        stop = 0
        k = 0
        self.rms_error = np.zeros((niter,))
        convergence = np.zeros((niter,))
        while k < niter and not stop:
            t_start = time.time()
            rec = np.zeros_like(self.rec)
            grad = np.zeros_like(self.rec)
            # gradient = - At(b - Ax_tilde) + reg_parm * x_tilde, where x_tilde is rec after positivity
            
            # compute back-projection: At(b - Ax)
            res = sparse.csr_matrix.dot(self.proj_mat, self.rec)
            res = self.projections[self.my_index] - res.reshape(self.my_n_proj, -1)
            my_back_proj = sparse.csc_matrix.dot(sparse.csr_matrix.transpose(self.proj_mat), res.ravel())
    
            my_norm = np.linalg.norm(res)
            
            self.comm.Barrier()
            self.comm.Allreduce([my_back_proj, mpi_float], [grad, mpi_float], op=MPI.SUM)
            grad = -grad + reg_param * self.rec  # At(Ax-b) + reg_param * Lt L x
            
            # find step length using line search
            line_out = optimize.line_search(my_f, my_fp, self.rec, -grad,
                                            args=(self.comm, self.proj_mat, self.projections[self.my_index], reg_param))
            alpha = line_out[0]
            if alpha is None:
                alpha = 1.e-3
            
            # update
            self.rec -= alpha * grad
            
            # positivity
            if positivity:
                self.rec[self.rec < 0.] = 0.
    
            convergence[k] = np.sqrt(self.comm.allreduce(my_norm ** 2, op=MPI.SUM))
            if self.ground_truth is None:
                self.rms_error[k] = convergence[k] / norm_factor
            else:
                self.rms_error[k] = np.linalg.norm(self.ground_truth - self.rec) / norm_factor
    
            if k > 1 and self.rms_error[k] > self.rms_error[k - 1]:
                stop = 1
                if self.my_rank == 0:
                    print('semi-convergence criterion reached: stopping at k %3d with RMSE = %4.5f'
                          % (k, self.rms_error[k]))
    
            if make_plot and self.my_rank == 0:
                if k == 0:
                    plt.ion()
                    fig, (ax0, ax1, ax2) = plt.subplots(3, 1)
        
                elif k % 10 == 0:
                    ax0.imshow(self.rec.reshape(self.geometry.vox_shape)[:, :, self.geometry.vox_shape[2] // 2])
                    ax0.set_title('SIRT iteration %3d' % (k))
            
                    ax1.cla()
                    ax1.set_title('Root Mean-Squared Error')
                    ax1.semilogy(self.rms_error[1:k])
            
                    ax2.cla()
                    ax2.set_title('Convergence')
                    ax2.semilogy(convergence[1:k])
            
                    plt.show()
                    plt.pause(0.1)
            k += 1
            if debug:
                print('in rank %d: time taken for one SIRT iteration for %d projections = %4.5f'
                      % (self.my_rank, self.my_n_proj, time.time() - t_start))

        return self.rec.reshape(self.geometry.vox_shape), self.rms_error[:k]


def my_f(x, comm, A, b, _lambda):
    
    my_res = sparse.csr_matrix.dot(A, x.ravel()) - b.ravel()
    res = np.zeros_like(b)

    if b.dtype == np.float32:
        comm.Allreduce([my_res, MPI.FLOAT], [res, MPI.FLOAT], op=MPI.SUM)
    else:
        comm.Allreduce([my_res, MPI.DOUBLE], [res, MPI.DOUBLE], op=MPI.SUM)
    
    res = 0.5 * np.linalg.norm(res)**2 + 0.5 * _lambda * np.linalg.norm(x)**2
    
    return res


def my_fp(x, comm, A, b, _lambda):
    
    my_res = sparse.csr_matrix.dot(A, x.ravel()) - b.ravel()
    my_grad = sparse.csc_matrix.dot(sparse.csr_matrix.transpose(A), my_res)
    
    grad = np.zeros_like(x)
    if x.dtype == np.float32:
        comm.Allreduce([my_grad, MPI.FLOAT], [grad, MPI.FLOAT], op=MPI.SUM)
    else:
        comm.Allreduce([my_grad, MPI.DOUBLE], [grad, MPI.DOUBLE], op=MPI.SUM)
    
    grad += _lambda * x
    
    return grad
