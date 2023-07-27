import numpy as np
import time
from scipy import sparse
from utilities import projection_operators


class SIRT(object):
    
    def __init__(self, geometry, projections, angles, xyz_shifts, options={}):
        
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
        self.f_proj_obj = None
        self.proj_mat = None
        self._initialize()
    
    def _initialize(self):
        
        if self.f_proj_obj is None:
            # create an instance of the projection operator
            self.f_proj_obj = projection_operators.ProjectionMatrix(self.geometry, precision=self.precision)
            self.proj_mat = self.f_proj_obj.projection_matrix(phi=self.angles[:, 0], alpha=self.angles[:, 1],
                                                              beta=self.angles[:, 2], xyz_shift=self.xyz_shifts)
        self.W = sparse.csr_matrix.dot(self.proj_mat, np.ones((self.geometry.n_vox,), dtype=self.precision))
        self.V = sparse.csc_matrix.dot(sparse.csr_matrix.transpose(self.proj_mat),
                                       np.ones((self.n_proj * self.geometry.n_det,), dtype=self.precision))
        
        self.V[self.V == 0.] = np.inf
        self.W[self.W == 0.] = np.inf
        self.V = 1./self.V
        self.W = 1./self.W
    
    def run_main_iteration(self, niter=100, make_plot=False, projections=None, positivity=False, debug=False):
                
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
        while k < niter and not stop:
            res = sparse.csr_matrix.dot(self.proj_mat, self.rec)
            res = self.projections - res.reshape(self.n_proj, -1)
            back_proj = sparse.csc_matrix.dot(sparse.csr_matrix.transpose(self.proj_mat), self.W * res.ravel())
            
            back_proj *= self.V
            self.rec += back_proj

            if positivity:
                self.rec[self.rec < 0.] = 0.
            
            convergence[k] = np.linalg.norm(res)
            if self.ground_truth is None:
                rms_error[k] = convergence[k] / norm_factor
            else:
                rms_error[k] = np.linalg.norm(self.ground_truth - self.rec) / norm_factor
            
            if k > 0 and rms_error[k] > rms_error[k - 1]:
                stop = 1
                print('semi-convergence criterion reached: stopping at k %3d with RMSE = %4.5f'
                      % (k, rms_error[k]))

            if k > 0 and k % 20 == 0 and debug:
                print('time taken for 20 SIRT iterations = %4.5f' % (time.time() - t_start))
                t_start = time.time()
                
            if make_plot:
                if k == 0:
                    import matplotlib.pyplot as plt
                    plt.ion()
                    fig, (ax0, ax1, ax2) = plt.subplots(3, 1)
                
                elif k % 20 == 0:
                    ax0.imshow(self.rec.reshape(self.geometry.vox_shape)[:, :, self.geometry.vox_shape[2] // 2])
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
        
        return self.rec.reshape(self.geometry.vox_shape), rms_error[:k]

    def run_regularized_gradient_descent(self, niter=100, reg_param=1.0, positivity=True,
                                         make_plot=False, debug=False):

        from scipy import optimize

        if self.ground_truth is not None:
            self.ground_truth = self.ground_truth.ravel()
            norm_factor = np.linalg.norm(self.ground_truth)
        else:
            norm_factor = np.linalg.norm(self.projections)

        stop = 0
        k = 0
        rms_error = np.zeros((niter,))
        convergence = np.zeros((niter,))
        while k < niter and not stop:
            t_start = time.time()
            # gradient = - At(b - Ax_tilde) + reg_parm * x_tilde, where x_tilde is rec after positivity
            # compute back-projection: At(b - Ax)
            res = sparse.csr_matrix.dot(self.proj_mat, self.rec)
            res = self.projections - res.reshape(self.n_proj, -1)
            grad = sparse.csc_matrix.dot(sparse.csr_matrix.transpose(self.proj_mat), res.ravel())
    
            grad = -grad + reg_param * self.rec  # At(Ax-b) + reg_param * Lt L x
    
            # find step length using line search
            line_out = optimize.line_search(my_f, my_fp, self.rec, -grad,
                                            args=(self.proj_mat, self.projections, reg_param))
            alpha = line_out[0]
            if alpha is None:
                alpha = 1.e-3
            
            # update
            self.rec -= alpha * grad
    
            # positivity
            if positivity:
                self.rec[self.rec < 0.] = 0.
    
            convergence[k] = np.linalg.norm(res)
            if self.ground_truth is None:
                rms_error[k] = convergence[k] / norm_factor
            else:
                rms_error[k] = np.linalg.norm(self.ground_truth - self.rec) / norm_factor
    
            if k > 1 and rms_error[k] > rms_error[k - 1]:
                stop = 1
                print('semi-convergence criterion reached: stopping at k %3d with RMSE = %4.5f' % (k, rms_error[k]))
    
            if make_plot:
                if k == 0:
                    import matplotlib.pyplot as plt
                    plt.ion()
                    fig, (ax0, ax1, ax2) = plt.subplots(3, 1)
        
                elif k % 20 == 0:
                    ax0.imshow(self.rec.reshape(self.geometry.vox_shape)[:, :, self.geometry.vox_shape[2] // 2])
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

        return self.rec.reshape(self.geometry.vox_shape), rms_error[:k]


def my_f(x, A, b, _lambda):

    res = sparse.csr_matrix.dot(A, x.ravel()) - b.ravel()

    res = 0.5 * np.linalg.norm(res) ** 2 + 0.5 * _lambda * np.linalg.norm(x) ** 2

    return res


def my_fp(x, A, b, _lambda):

    res = sparse.csr_matrix.dot(A, x.ravel()) - b.ravel()
    grad = sparse.csc_matrix.dot(sparse.csr_matrix.transpose(A), res) + _lambda * x

    return grad
