# ------------------------------------------------
# Copyright 2021 Kanupriya Pande
# Contact kpande@lbl.gov
# ------------------------------------------------

import numpy as np
from projectors import projection_operators


class CGLS(object):
    
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
    
        self.proj_obj._setup(angles=self.angles, xyz_shifts=self.xyz_shifts)
    
        self._r = self.projections - self.proj_obj.forward_project(self.rec)
        self._p = self.proj_obj.back_project(self._r)
        if self.my_rank == 0:
            print(self._r.shape)
            print(self._p.shape)

        self._gamma = np.linalg.norm(self._p)**2
        
    def run_main_iteration(self, make_plot=False, niter=100):
    
        if self.ground_truth is None:
            norm_factor = np.linalg.norm(self.projections)
        else:
            self.ground_truth = self.ground_truth.ravel()
            norm_factor = np.linalg.norm(self.ground_truth)
            
        stop = 0
        k = 0
        conv = np.zeros((niter, ))
        self.rms_error = np.zeros((niter, ))
        reinit_iter = 0
        self.rec = self.rec.ravel()
        while not stop and k < niter:
            r = self.proj_obj.forward_project(self._p)
            
            alpha = self._gamma/np.linalg.norm(r)**2
            self.rec += alpha * self._p
            conv[k] = np.linalg.norm(self.projections - self.proj_obj.forward_project(self.rec))
            
            if k > 0 and conv[k] > conv[k-1]:
                # re-initialize only if we did not re-initialize in the previous iteration
                if self.my_rank == 0:
                    print('reinitializing at iteration %d' %k)
                if reinit_iter + 1 == k:
                    if self.my_rank == 0:
                        print('need to re-initialize at two consecutive iterations: quitting')
                    return self.rec, self.rms_error[:k]
                self.rec -= alpha * self._p
                self._initialize()
                reinit_iter = k
            
            self._r -= alpha * r
            # now back-project
            p = self.proj_obj.back_project(self._r)
            
            gamma = np.linalg.norm(p)**2
            beta = gamma/self._gamma
            
            # update gamma and p
            self._gamma = gamma
            self._p = p + beta * self._p
            if self.ground_truth is None:
                self.rms_error[k] = np.linalg.norm(self._r)/norm_factor
            else:
                self.rms_error[k] = np.linalg.norm(self.rec - self.ground_truth)/norm_factor
                
            if make_plot and self.my_rank == 0:
                if k == 0:
                    import matplotlib.pyplot as plt
                    plt.ion()
                    fig, (ax0, ax1, ax2) = plt.subplots(3, 1)

                elif k % 20 == 0:
                    ax0.imshow(np.reshape(self.rec, self.geometry.vox_shape)[:, :, self.geometry.vox_shape[2] // 2])
                    ax0.set_title('CGLS iteration %3d' % (k))
    
                    ax1.cla()
                    ax1.set_title('Root Mean-Squared Error')
                    ax1.semilogy(self.rms_error[1:k])
    
                    ax2.cla()
                    ax2.set_title('Convergence')
                    ax2.semilogy(conv[1:k])
                    plt.pause(0.1)
            k += 1
     
        return self.rec, self.rms_error[:k]
