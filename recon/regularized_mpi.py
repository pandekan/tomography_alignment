# ------------------------------------------------
# Copyright 2022 Kanupriya Pande
# Contact kpande@lbl.gov
# ------------------------------------------------

import numpy as np
import copy
from scipy import sparse, optimize
from scipy.optimize.linesearch import line_search_armijo
from mpi4py import MPI
from utilities import projection_operators
from utilities import tv_denoise


class RegularizedRecon(object):
    """
    Class for Iterative reconstruction methods with different regularization terms.
    run_tikhonov_gd: Gradient descent for solving Tikhonov regularized least-squares problem
    run_lasso_ista: Iterative soft shrinkage method for solving L1 regularized least-squares problem
    run_fista: Fast iterative shrinkage thresholding algorithm for solving TV regularized least-squares problem
    """
    
    def __init__(self, comm, geometry, projections, angles, xyz_shifts, options={}):
        
        self.comm = comm
        self.geometry = geometry
        self.projections = projections
        self.angles = angles
        self.xyz_shifts = xyz_shifts
        self.n_proj = angles.shape[0]
        
        self.ground_truth = options['ground_truth'] if 'ground_truth' in options else None
        self.precision = options['precision'] if 'precision' in options else np.float32
        self.rec = options['rec'] if 'rec' in options else None
        if self.rec is None:
            self.rec = np.zeros(self.geometry.n_vox, dtype=self.precision)
        self.rec = self.rec.ravel().astype(self.precision, copy=False)
        self.projections = self.projections.astype(self.precision, copy=False)
        self.voxel_mask = options['voxel_mask'] if 'voxel_mask' in options else None
        
        if self.ground_truth is not None:
            # compute rms error between ground truth and current reconstruction
            self.norm_factor = np.linalg.norm(self.ground_truth)
        else:
            # compute rms error between projection data and forward projection of current reconstruction
            self.norm_factor = np.linalg.norm(self.projections)
            
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

        if self.f_proj_obj is None:
            # create an instance of the projection operator
            self.f_proj_obj = projection_operators.ProjectionMatrix(self.my_geom, precision=self.precision)
            self.proj_mat = self.f_proj_obj.projection_matrix(phi=self.my_angles[:, 0], alpha=self.my_angles[:, 1],
                                                              beta=self.my_angles[:, 2], 
                                                              xyz_shift=self.my_xyz_shifts)

        if self.precision == np.float32:
            self.mpi_precision = MPI.FLOAT
        else:
            self.mpi_precision = MPI.DOUBLE

    def run_fista(self, niter=100, make_plot=False, hyper=1.e4, beta_tv=1.0, niter_tv=20):
        """
              Iteratively minimize TV penalized cost function

              F(x) = 0.5 * |Ax - b |^2 + beta_tv * |\nabla(x)|_1 = f(x) + g(x)
              by forward-backward iterations. Here g(x) is a discrete gradient.
              1) u_k = prox_{gamma g}(x_{k-1} - gamma \nabla f(x_{k-1}))
              2) t_k = 1/2 *(1 + sqrt(1 + 4t_{k-1}^2)); t_0 = 1.0
              3) x_k = u_k + (t_{k-1} - 1)/t_k * (u_k - u_{k-1})

              where step 1 is solved iteratively by the dual approach for g(x) = l1 norm of the isotropic gradient.
              http://www.math.tau.ac.il/~teboulle/papers/tlv.pdf
        """

        # parameters for forward-backward iterations
        gamma = 1./hyper
        t_old = 1.0
        
        # initialize metrics
        self.rms_error = np.zeros(niter, )
        total_cost = np.zeros(niter, )
        cost_data_fidelity = np.zeros(niter, )
        
        u_old = self.rec.copy()
        k = 0
        stop = 0
        
        while k < niter and not stop:
            rec = np.zeros_like(u_old)

            res = sparse.csr_matrix.dot(self.proj_mat, self.rec).reshape(self.my_n_proj, -1)
            res = self.projections[self.my_index] - res
            
            my_back_proj = sparse.csc_matrix.dot(sparse.csr_matrix.transpose(self.proj_mat), res.ravel())
            
            self.comm.Barrier()
            self.comm.Allreduce([my_back_proj, self.mpi_precision], [rec, self.mpi_precision], op=MPI.SUM)

            if self.my_rank == 0:
                # Gradient descent on data fidelity term
                x_tmp = self.rec + gamma * rec
            
                # Minimize the proximal operator using FISTA
                u = tv_denoise.denoise_fista(x_tmp.reshape(self.geometry.vox_shape),
                                             weight=gamma * beta_tv, niter=niter_tv)
            
                # update t
                t = 0.5 * (1.0 + np.sqrt(1 + 4 * t_old**2))
            
                # update reconstruction
                u = u.ravel().astype(self.precision, copy=False)
                self.rec = u + (t_old - 1)/t * (u - u_old)
                
                t_old = t
                u_old = u
            
            self.comm.Barrier()
            self.rec = self.comm.bcast(self.rec, root=0)

            # compute error metrics
            my_proj_err = np.linalg.norm(res)
            cost_data_fidelity[k] = 0.5*self.comm.allreduce(my_proj_err**2, op=MPI.SUM)
            tv_value = beta_tv * tv_denoise.tv_norm_3d(self.rec.reshape(self.geometry.vox_shape))
            total_cost[k] = cost_data_fidelity[k] + tv_value
            
            if self.ground_truth is None:
                # rms error between input data and projection from current reconstruction
                self.rms_error[k] = np.sqrt(2*cost_data_fidelity[k])/self.norm_factor
            else:
                self.rms_error[k] = np.linalg.norm(self.ground_truth.ravel() - self.rec)/self.norm_factor
            
            if k > 0 and self.rms_error[k] > self.rms_error[k - 1]:
                stop = 1
                if self.my_rank == 0:
                    print('semi-convergence criterion reached: stopping at k %3d with RMSE = %4.5f'
                          % (k, self.rms_error[k]))
            
            if make_plot and self.my_rank==0:
                if k == 0:
                    import matplotlib.pyplot as plt
                    plt.ion()
                    fig, ((ax0, ax1, ax2), (ax3, ax4, ax5)) = plt.subplots(nrows=2, ncols=3)
                
                elif k % 20 == 0:
                    plt.suptitle('TV iteration %3d' % k)
                    
                    ax0.imshow(self.rec.reshape(self.geometry.vox_shape)[self.geometry.vox_shape[0] // 2])
                    
                    ax1.imshow(self.rec.reshape(self.geometry.vox_shape)[:, self.geometry.vox_shape[1] // 2, :])
                    
                    ax2.imshow(self.rec.reshape(self.geometry.vox_shape)[:, :, self.geometry.vox_shape[2] // 2])
                    
                    ax3.cla()
                    ax3.set_title('Root Mean-Squared Error')
                    ax3.semilogy(self.rms_error[1:k])
                    
                    ax4.cla()
                    ax4.set_title('Energy')
                    ax4.semilogy(total_cost[1:k])
                    
                    ax5.cla()
                    ax5.set_title('Residual')
                    ax5.semilogy(cost_data_fidelity[1:k])
                    
                    plt.show()
                    plt.pause(0.1)
            
            k += 1
        
        #return self.rec.reshape(self.geometry.vox_shape), rms_error[:k]
        return self.rec, self.rms_error[:k]

    def run_tikhonov_gd(self, niter=100, reg_param=1.0, positivity=False, make_plot=False):
        """
            Solve x* = argmin_x 0.5*|Ax - b|^2 + 0.5*lambda * |x|^2
            using gradient descent with simple line search
            :param niter: maximum number of iterations
            :param reg_param: regularization parameter
            :param positivity: enforce positivity constraint
            :param make_plot: show progress every 20 iterations
            :param projections: projection data
            :return: reconstruction, rms_error
        """

        stop = 0
        k = 0
        rms_error = np.zeros(niter, )
        convergence = np.zeros(niter, )
        self.rec = self.rec.ravel()
        while k < niter and not stop:
            # gradient = - At(b - Ax_tilde) + reg_parm * x_tilde, where x_tilde is rec after positivity
            # compute back-projection: At(b - Ax)
            grad = np.zeros_like(self.rec)
            res = 0.0

            my_res = sparse.csr_matrix.dot(self.proj_mat, self.rec).reshape(self.my_n_proj, -1)
            my_res = self.projections[self.my_index] - my_res
            res = self.comm.allreduce(np.linalg.norm(my_res)**2, op=MPI.SUM)
            cost = 0.5*(res + reg_param*np.linalg.norm(self.rec)**2)
            
            my_grad = sparse.csc_matrix.dot(sparse.csr_matrix.transpose(self.proj_mat), my_res.ravel())
            
            self.comm.Barrier()
            self.comm.Allreduce([my_grad, self.mpi_precision], [grad, self.mpi_precision], op=MPI.SUM)

            grad = -grad + reg_param * self.rec  # At(Ax-b) + reg_param * Lt L x
    
            # find step length using Armijo line search
            alpha, _, f_new = line_search_armijo(my_tikh_f, self.rec, -grad, grad, cost, alpha0=1.0,
                                                 args=(self.proj_mat, self.projections[self.my_index], 
                                                     reg_param, self.comm))
            
            #line_out = optimize.line_search(my_tikh_f, my_tikh_fp, self.rec, -grad,
            #                                args=(self.proj_mat, self.projections[self.my_index], reg_param, self.comm),
            #                                amax=1.0, maxiter=20)
            #alpha = line_out[0]
            if alpha is None:
                if self.comm.Get_rank() == 0:
                    print('line search failed at iteration %3d' %k)
                break

            # update
            self.rec -= alpha * grad
    
            # positivity
            if positivity:
                self.rec[self.rec < 0.] = 0.0
    
            convergence[k] = np.sqrt(res)
            if self.ground_truth is None:
                rms_error[k] = convergence[k] / self.norm_factor
            else:
                rms_error[k] = np.linalg.norm(self.ground_truth.ravel() - self.rec) / self.norm_factor
    
            if k > 1 and rms_error[k] > rms_error[k - 1]:
                stop = 1
                if self.my_rank == 0:
                    print('semi-convergence criterion reached: stopping at k %3d with RMSE = %4.5f'
                          % (k, rms_error[k]))
    
            if make_plot and self.my_rank == 0:
                if k == 0:
                    import matplotlib.pyplot as plt
                    plt.ion()
                    fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=2)
        
                elif k % 10 == 0:
                    plt.suptitle('Tikhonov iteration %3d' % k)
                    
                    ax0[0].cla()
                    ax0[0].set_title('YZ slice')
                    ax0[0].imshow(np.reshape(self.rec, self.geometry.vox_shape)[self.geometry.vox_shape[0] // 2])
            
                    ax0[1].cla()
                    ax0[1].set_title('XZ slice')
                    ax0[1].imshow(np.reshape(self.rec, self.geometry.vox_shape)[:, self.geometry.vox_shape[1] // 2])

                    ax1[0].cla()
                    ax1[0].set_title('XY slice')
                    ax1[0].imshow(np.reshape(self.rec, self.geometry.vox_shape)[:, :, self.geometry.vox_shape[2] // 2])
                    
                    ax1[1].cla()
                    ax1[1].set_title('Root Mean-Squared Error')
                    ax1[1].semilogy(rms_error[1:k])
            
                    plt.show()
                    plt.pause(0.1)
            k += 1

        return self.rec, rms_error[:k]

    def run_lasso_ista(self, niter=100, reg_param=1.0, alpha0=1.0, beta=0.5, make_plot=False):
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

        stop = 0
        k = 0
        rms_error = np.zeros(niter, )
        convergence = np.zeros(niter, )
        step_size = np.zeros(niter, )
        self.rec = self.rec.ravel()
        while k < niter and not stop:
            # gradient of fidelity term = - At(b - Ax_tilde)
            # compute back-projection: At(b - Ax)
            res = 0.0
            grad = np.zeros_like(self.rec)

            my_res = sparse.csr_matrix.dot(self.proj_mat, self.rec).reshape(self.my_n_proj, -1)
            my_res = my_res - self.projections[self.my_index]
            res = self.comm.allreduce(np.linalg.norm(my_res)**2, op=MPI.SUM)

            my_grad = sparse.csc_matrix.dot(sparse.csr_matrix.transpose(self.proj_mat), my_res.ravel())

            self.comm.Barrier()
            self.comm.Allreduce([my_grad, self.mpi_precision], [grad, self.mpi_precision], op=MPI.SUM)

            # backtracking linesearch for proximal gradient descent
            _, alpha, success = self._backtrack_lasso(alpha0, beta, 0.5*res, grad, reg_param)
            step_size[k] = alpha
            if not success:
                print('line search failed to converge at iteration %3d' %k)
                break
            
            # update
            self.rec = soft_thresholding(self.rec - alpha * grad, alpha * reg_param)
    
            convergence[k] = np.sqrt(res)
            if self.ground_truth is None:
                rms_error[k] = convergence[k] / self.norm_factor
            else:
                rms_error[k] = np.linalg.norm(self.ground_truth.ravel() - self.rec) / self.norm_factor
    
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

        #g0 = 0.5 * np.linalg.norm(g0) ** 2
        while t > 1.e-16:
            g = 0.0
            xp = soft_thresholding(self.rec - t * dg0, t * _lambda)
            Gt = self.rec - xp

            my_g = sparse.csr_matrix.dot(self.proj_mat, xp).reshape(self.my_n_proj, -1)
            my_g = my_g - self.projections[self.my_index]
            g = 0.5 * self.comm.allreduce(np.linalg.norm(my_g)**2, op=MPI.SUM)
            
            gp = g0 - np.dot(dg0.T, Gt) + (0.5 / t) * np.linalg.norm(Gt) ** 2
            # print(t, g, gp)
            if g <= gp:
                return xp, t, True
            else:
                t *= beta
        return xp, t, False

    def run_lasso_accelerated(self, niter=100, reg_param=1.0, alpha0=1.0, beta=0.5, make_plot=False):
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
        stop = 0
        k = 0
        rms_error = np.zeros(niter,)
        convergence = np.zeros(niter,)

        self.rec = self.rec.ravel()
        if self.my_rank == 0:
            x_0 = np.zeros_like(self.rec)
            x_1 = np.zeros_like(self.rec)
        else:
            x_0 = None
            x_1 = None

        while k < niter and not stop:
            # gradient of fidelity term = - At(b - Ax_tilde)
            # compute back-projection: At(b - Ax)
            res = 0.0
            grad = np.zeros_like(self.rec)

            my_res = sparse.csr_matrix.dot(self.proj_mat, self.rec).reshape(self.my_n_proj, -1)
            my_res = my_res - self.projections[self.my_index]
            res = self.comm.allreduce(np.linalg.norm(my_res)**2, op=MPI.SUM)

            my_grad = sparse.csc_matrix.dot(sparse.csr_matrix.transpose(self.proj_mat), my_res.ravel())
            self.comm.Barrier()
            self.comm.Allreduce([my_grad, self.mpi_precision], [grad, self.mpi_precision], op=MPI.SUM)

            # backtracking linesearch for proximal gradient descent
            _, alpha, success = self._backtrack_lasso(alpha0, beta, 0.5*res, grad, reg_param)

            if not success:
                if self.my_rank == 0:
                    print('line search failed at iteration %d' %k)
                break

            # update
            if self.my_rank == 0:
                v = x_1 + (k - 2) / (k + 1) * (x_1 - x_0)
                self.rec = soft_thresholding(v - alpha * grad, alpha * reg_param)
                x_0 = x_1.copy()
                x_1 = self.rec.copy()
    
            self.comm.Barrier()
            self.rec = self.comm.bcast(self.rec, root=0)

            convergence[k] = np.sqrt(res)
            if self.ground_truth is None:
                rms_error[k] = convergence[k] / self.norm_factor
            else:
                rms_error[k] = np.linalg.norm(self.ground_truth.ravel() - self.rec) / self.norm_factor
    
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


def my_tikh_f(x, proj_mat, my_proj, _lambda, comm):

    res = 0.0
    my_n_proj = my_proj.shape[0]
    my_res = sparse.csr_matrix.dot(proj_mat, x).reshape(my_n_proj, -1) - my_proj
    my_res = np.linalg.norm(my_res)**2
    comm.Barrier()    
    res = comm.allreduce(my_res, op=MPI.SUM)
    res += _lambda*np.linalg.norm(x)**2

    return 0.5*res


def my_tikh_fp(x, proj_mat, my_proj, _lambda, comm):
        
    grad = np.zeros_like(x)    

    my_n_proj = my_proj.shape[0]
    my_res = sparse.csr_matrix.dot(proj_mat, x).reshape(my_n_proj, -1) - my_proj
    my_grad = sparse.csc_matrix.dot(sparse.csr_matrix.transpose(proj_mat), my_res.ravel())

    comm.Barrier()
    comm.Allreduce([my_grad, MPI.FLOAT], [grad, MPI.FLOAT], op=MPI.SUM)
        
    grad += _lambda * x

    return grad


def soft_thresholding(x, _lambda):
    xx = np.zeros_like(x)
    ind_p = np.where(x > _lambda)
    ind_n = np.where(x < -_lambda)
    xx[ind_p] = x[ind_p] - _lambda
    xx[ind_n] = x[ind_n] + _lambda

    return xx
