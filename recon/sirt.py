import numpy as np
from scipy import sparse
from scipy import optimize
import time
from projectors import projection_operators


class SIRT(object):
    
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
        self.f_proj_obj = None
        
        self._initialize()
    
    def _initialize(self):
        
        if self.f_proj_obj is None:
            # create an instance of the projection operator
            self.f_proj_obj = projection_operators.ForwardProjection(self.geometry, method=self.method,
                                                                     precision=self.precision,
                                                                     comm=self.comm)
            
        self.f_proj_obj._setup(angles=self.angles, xyz_shifts=self.xyz_shifts)
        
        # compute normalizing matrices
        self.W = self.f_proj_obj.forward_project(np.asfortranarray(np.ones(self.geometry.vox_shape,
                                                                           dtype=self.precision)))
        self.V = self.f_proj_obj.back_project(np.ones_like(self.projections))
        
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
            res = self.f_proj_obj.forward_project(self.rec)
            res = self.projections - res
            back_proj = self.f_proj_obj.back_project(self.W * res)
            
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
    
    def run_tikhonov_gd(self, niter=100, reg_param=1.0, positivity=False, make_plot=False, projections=None):
        """
        Solve x* = argmin_x 0.5*|Ax - b|^2 + 0.5*lambda * |x|^2
        using gradient descent with simple line search
        :param niter: maximum number of iterations
        :param reg_param: regularization parameter
        :param positivity: enforce positivity constraint
        :param make_plot: show progress every 20 iterations
        :return: reconstruction, rms_error
        """
        
        if reg_param <= 1.e-10:
            return self.run_sirt(niter=niter, make_plot=make_plot, positivity=positivity, projections=projections)
        
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
        self.rec = self.rec.ravel()
        while k < niter and not stop:
            # gradient = - At(b - Ax_tilde) + reg_parm * x_tilde, where x_tilde is rec after positivity
            # compute back-projection: At(b - Ax)
            
            res = self.projections - self.f_proj_obj.forward_project(self.rec)
            grad = self.f_proj_obj.back_project(res)  # At (b - Ax)
            
            grad = -grad + reg_param * self.rec  # At(Ax-b) + reg_param * Lt L x
            
            # find step length using line search
            line_out = optimize.line_search(my_tikh_f, my_tikh_fp, self.rec, -grad,
                                            args=(self.f_proj_obj, self.projections, reg_param))
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
                if self.my_rank == 0:
                    print('semi-convergence criterion reached: stopping at k %3d with RMSE = %4.5f' 
                            % (k, rms_error[k]))
            
            if make_plot and self.my_rank == 0:
                if k == 0:
                    import matplotlib.pyplot as plt
                    plt.ion()
                    fig, (ax0, ax1, ax2) = plt.subplots(3, 1)
                
                elif k % 10 == 0:
                    ax0.imshow(np.reshape(self.rec, self.geometry.vox_shape)[:, :, self.geometry.vox_shape[2] // 2])
                    ax0.set_title('Tikhonov iteration %3d' % k)
                    
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
    
    def run_lasso_ista(self, niter=100, reg_param=1.0, alpha0=1.0, beta=0.5, make_plot=False, projections=None):
        """
        Solve the "Lasso" problem x* = argmin 0.5 |Ax-b|^2 + \lambda |x|_1 = g(x) + h(x)
        with proximal gradient descent and soft-thresholding operator.
        x^(k+1) = prox_{lambda*t_k, h}(x_(k) - t\grad g(x^(k))
        where prox_{alpha, h}(x) = argmin_z (1/(2*alpha) |z-x|^2 + h(z))
        For h(z) = \lambda |z|_1, prox_{alpha, h}(x) = sgn(x)max(|x|-alpha*lambda, 0)
        niter: int, number of iterations
        reg_param: float, regularization parameter lambda
        make_plot: logical, show progress
        projections: ndarray, projection data if different from that used in __init__
        alpha0: float, initial step size
        beta: float, factor to decrease the step size by during backtracking
        Returns
        rec: reconstruction
        err: rms_error
        """
        
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
        step_size = np.zeros(niter, )
        self.rec = self.rec.ravel()
        while k < niter and not stop:
            # gradient of fidelity term = - At(b - Ax_tilde)
            # compute back-projection: At(b - Ax)
            res = self.f_proj_obj.forward_project(self.rec)
            res = res - self.projections
            grad = self.f_proj_obj.back_project(res)
            
            # backtracking linesearch for proximal gradient descent
            _, alpha, success = self._backtrack_lasso(alpha0, beta, res, grad, reg_param)
            step_size[k] = alpha
            if not success:
                print('line search failed to converge')
                stop = 1
            else:
                # update
                self.rec = _soft_thresholding(self.rec - alpha * grad, alpha * reg_param)
            
            convergence[k] = np.linalg.norm(res)
            if self.ground_truth is None:
                rms_error[k] = convergence[k] / norm_factor
            else:
                rms_error[k] = np.linalg.norm(self.ground_truth - self.rec) / norm_factor
            
            if k > 1 and rms_error[k] > rms_error[k - 1]:
                stop = 1
                print('semi-convergence criterion reached: stopping at k %3d with RMSE = %4.5f' % (k, rms_error[k]))
            
            if make_plot and self.my_rank == 0:
                if k == 0:
                    import matplotlib.pyplot as plt
                    plt.ion()
                    fig, (ax0, ax1, ax2) = plt.subplots(3, 1)
                
                elif k % 10 == 0:
                    ax0.imshow(self.rec.reshape(self.geometry.vox_shape)[:, :, self.geometry.vox_shape[2] // 2])
                    ax0.set_title('ISTA iteration %3d' % k)
                    
                    ax1.cla()
                    ax1.set_title('Root Mean-Squared Error')
                    ax1.semilogy(rms_error[1:k])
                    
                    ax2.cla()
                    ax2.set_title('Convergence')
                    ax2.semilogy(convergence[1:k])
                    
                    plt.show()
                    plt.pause(0.1)
            k += 1
        
        plt.figure()
        plt.plot(step_size)
        plt.show()
        return self.rec.reshape(self.geometry.vox_shape), rms_error[:k]
    
    def _backtrack_lasso(self, t, beta, g0, dg0, _lambda):
        
        g0 = 0.5 * np.linalg.norm(g0) ** 2
        while t > 1.e-16:
            xp = _soft_thresholding(self.rec - t * dg0, t * _lambda)
            Gt = self.rec - xp
            
            g = self.f_proj_obj.forward_project(xp) - self.projections
            g = 0.5 * np.linalg.norm(g) ** 2
            gp = g0 - np.dot(dg0.T, Gt) + (0.5 / t) * np.linalg.norm(Gt) ** 2
            # print(t, g, gp)
            if g <= gp:
                return xp, t, True
            else:
                t *= beta
        return xp, t, False
    
    def run_lasso_accelerated(self, niter=100, reg_param=1.0, alpha0=1.0, beta=0.5, make_plot=False, projections=None):
        """
        Solve the "Lasso" problem x* = argmin 0.5 |Ax-b|^2 + \lambda |x|_1 = g(x) + h(x)
        with proximal gradient descent and soft-thresholding operator using the accelerated method.
        x^(k+1) = prox_{_lambda*t_k, h} (v^(k) - t\grad g(x^(k))
        v^(k+1) = x^(k) + (k-1)/(k+2) (x^(k) - x^(k-1))
        Notice that compared to regular ISTA, there is an additional vector in the proximal update term
        niter: int, number of iterations
        reg_param: float, regularization parameter lambda
        make_plot: logical, show progress
        projections: ndarray, projection data if different from that used in __init__
        alpha0: float, initial step size
        beta: float, factor to decrease the step size by during backtracking
        Returns
        rec: reconstruction
        err: rms_error
        """
        
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
        
        self.rec = self.rec.ravel()
        x_0 = np.zeros_like(self.rec)
        x_1 = np.zeros_like(self.rec)
        while k < niter and not stop:
            # gradient of fidelity term = - At(b - Ax_tilde)
            # compute back-projection: At(b - Ax)
            res = self.f_proj_obj.forward_project(self.rec) - self.projections
            grad = self.f_proj_obj.back_project(res)
            
            # backtracking linesearch for proximal gradient descent
            _, alpha, success = self._backtrack_lasso(alpha0, beta, res, grad, reg_param)
            
            if not success:
                print('line search failed to converge')
                stop = 1
            else:
                # update
                v = x_1 + (k - 2) / (k + 1) * (x_1 - x_0)
                self.rec = _soft_thresholding(v - alpha * grad, alpha * reg_param)
                x_0 = x_1.copy()
                x_1 = self.rec.copy()
            
            convergence[k] = np.linalg.norm(res)
            if self.ground_truth is None:
                rms_error[k] = convergence[k] / norm_factor
            else:
                rms_error[k] = np.linalg.norm(self.ground_truth - self.rec) / norm_factor
            
            if k > 1 and rms_error[k] > rms_error[k - 1]:
                stop = 1
                if self.my_rank == 0:
                    print('semi-convergence criterion reached: stopping at k %3d with RMSE = %4.5f'
                          % (k, rms_error[k]))
            
            if make_plot and self.my_rank == 0:
                if k == 0:
                    import matplotlib.pyplot as plt
                    plt.ion()
                    fig, (ax0, ax1, ax2) = plt.subplots(3, 1)
                
                elif k % 10 == 0:
                    ax0.imshow(self.rec.reshape(self.geometry.vox_shape)[:, :, self.geometry.vox_shape[2] // 2])
                    ax0.set_title('ISTA iteration %3d' % k)
                    
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


def my_tikh_f(x, f_proj_obj, b, _lambda):
    
    res = f_proj_obj.forward_project(x) - b
    
    res = 0.5 * np.linalg.norm(res) ** 2 + 0.5 * _lambda * np.linalg.norm(x) ** 2
    
    return res


def my_tikh_fp(x, f_proj_obj, b, _lambda):
    
    res = f_proj_obj.forward_project(x) - b
    grad = f_proj_obj.back_project(res) + _lambda * x

    return grad


def _soft_thresholding(x, _lambda):
    xx = np.zeros_like(x)
    ind_p = np.where(x > _lambda)
    ind_n = np.where(x < -_lambda)
    xx[ind_p] = x[ind_p] - _lambda
    xx[ind_n] = x[ind_n] + _lambda
    
    return xx
