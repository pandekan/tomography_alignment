import numpy as np
from scipy import sparse
from mpi4py import MPI
from utilities import projection_operators
from utilities import linear_operators as tomo_linop


class CGLS(object):
    
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
            self.rec = np.zeros((self.geometry.n_vox, ), dtype=self.projections.dtype)
        self.precision = options['precision'] if 'precision' in options else np.float32
    
        # make sure all image arrays have the same precision else MPI will screw us over
        self.projections = self.projections.astype(self.precision, copy=False)
        self.rec = self.rec.astype(self.precision, copy=False)
        if self.ground_truth is not None:
            self.ground_truth = self.ground_truth.astype(self.precision, copy=False)
        self.rms_error = None
        self.f_proj_obj = None
        self.proj_mat = None
        self._initialize()
        
    def _initialize(self):
        
        self.size = self.comm.Get_size()
        self.my_rank = self.comm.Get_rank()
        self.my_index = np.array_split(np.arange(self.n_proj), self.size)[self.my_rank]
        self.my_n_proj = np.size(self.my_index)
        self.my_angles = self.angles[self.my_index]
        self.my_xyz_shifts = self.xyz_shifts[self.my_index]
        
        self._p = np.zeros_like(self.rec)
        if self.f_proj_obj is None:
            # create an instance of the projection operator
            self.f_proj_obj = projection_operators.ProjectionMatrix(self.geometry, self.precision)
            self.proj_mat = self.f_proj_obj.projection_matrix(phi=self.my_angles[:, 0], alpha=self.my_angles[:, 1],
                                                              beta=self.my_angles[:, 2],
                                                              xyz_shift=self.my_xyz_shifts)
        self._r = self.projections[self.my_index] - sparse.csr_matrix.dot(self.proj_mat,
                                                                          self.rec).reshape(self.my_n_proj, -1)
        _p = sparse.csc_matrix.dot(sparse.csr_matrix.transpose(self.proj_mat), self._r.ravel())
        
        self.comm.Barrier()
        self.comm.Allreduce([_p, MPI.FLOAT], [self._p, MPI.FLOAT], op=MPI.SUM)
        self._gamma = np.linalg.norm(self._p)**2
        
    def run_main_iteration(self, niter=100, make_plot=False):
        
        if self.ground_truth is None:
            norm_factor = np.linalg.norm(self.projections)
        else:
            norm_factor = np.linalg.norm(self.ground_truth)
            
        stop = 0
        k = 0
        self.rms_error = np.zeros((niter,))
        conv = np.zeros((niter, ))
        reinit_iter = 0
        while not stop and k < niter:
            r = sparse.csr_matrix.dot(self.proj_mat, self._p).reshape(self.my_n_proj, -1)
            
            my_sum_r2 = np.sum(np.square(r.ravel()))
            my_sum_d2 = np.sum(np.square(self.projections[self.my_index] - r).ravel())
            sum_r2 = self.comm.allreduce(my_sum_r2)
            sum_d2 = self.comm.allreduce(my_sum_d2)
            
            alpha = self._gamma/sum_r2
            self.rec += alpha * self._p
            conv[k] = np.sqrt(sum_d2)
            if k > 0 and conv[k] > conv[k-1]:
                # re-initialize
                print('reinitializing at iteration %d' %k)
                if reinit_iter + 1 == k:
                    # if also re-initialized in the previous iteration, quit
                    print('need to re-initialize at two consecutive iterations: quitting')
                    return self.rec, self.rms_error[:k]
                self.rec -= alpha * self._p
                self._initialize()
                reinit_iter = k
            
            self._r -= alpha * r
            # now back-project
            p = np.zeros_like(self.rec)

            _p = sparse.csc_matrix.dot(sparse.csr_matrix.transpose(self.proj_mat), self._r.ravel())
            self.comm.Barrier()
            self.comm.Allreduce([_p, MPI.FLOAT], [p, MPI.FLOAT], op=MPI.SUM)
            gamma = np.linalg.norm(p)**2
            beta = gamma/self._gamma
            
            # update gamma and p
            self._gamma = gamma
            self._p = p + beta * self._p
            if self.ground_truth is None:
                my_sum_r2 = np.sum(np.square(self._r.ravel()))
                sum_r2 = self.comm.allreduce(my_sum_r2)
                self.rms_error[k] = np.sqrt(sum_r2)/norm_factor
            else:
                self.rms_error[k] = np.linalg.norm(self.rec - self.ground_truth.ravel())/norm_factor
                
            if make_plot and self.my_rank == 0:
                if k == 0:
                    import matplotlib.pyplot as plt
                    plt.ion()
                    fig, (ax0, ax1, ax2) = plt.subplots(3, 1)

                elif k % 20 == 0:
                    ax0.imshow(self.rec.reshape(self.geometry.vox_shape)[:, :, self.geometry.vox_shape[2] // 2])
                    ax0.set_title('SIRT iteration %3d' % (k))
    
                    ax1.cla()
                    ax1.set_title('Root Mean-Squared Error')
                    ax1.semilogy(self.rms_error[1:k])
    
                    ax2.cla()
                    ax2.set_title('Convergence')
                    ax2.semilogy(conv[1:k])
                    plt.pause(0.1)
            k += 1
     
        return self.rec, self.rms_error

