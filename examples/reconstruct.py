import numpy as np
from recon import sirt, cgls
from utilities import geometry


def recon_sirt(projections, angles, translations, niter=100, vox_shape=None,
               vox_pixsize=None, det_pixsize=None, cor_shift=None,
               comm=None, ground_truth=None):
    
    n_proj, nz, nx = projections.shape
    det_shape = np.array([nx, nz])
    
    if vox_shape is None:
        vox_shape = np.array([nx, nx, nz])
    
    if vox_pixsize is None:
        vox_pixsize = 1.0
    
    if det_pixsize is None:
        det_pixsize = 1.0
        
    if cor_shift is None:
        cor_shift = np.zeros((n_proj, 3))
        
    # create geometry object
    geom = geometry.Geometry(n_proj, vox_shape, vox_pixsize, det_shape, det_pixsize,
                             cor_shift=cor_shift, step_size=1.0)
    
    # create reconstruction object
    sirt_options = {'method': 'linop', 'precision': np.float32, 'ground_truth': ground_truth}
    sirt_rec = sirt.SIRT(geom, projections, angles, translations, comm, options=sirt_options)
    
    # regular SIRT
    rec, err = sirt_rec.run_sirt(niter=niter, make_plot=True)
    
    # use gradient descent to minimize cost function with Tikhonov regularization
    # rec, err = sirt_rec.run_tikhonov_gd(niter=niter, reg_param=1.0, make_plot=True)
    
    # use iterative soft-thresholding to solve cost function with |x|_1 regularization
    # rec, err = sirt_rec.run_lasso_ista(niter=niter, make_plot=True)
    
    return rec, err
