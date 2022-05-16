import numpy as np
from recon import sirt, cgls, regularized
from utilities import geometry


def iterative_recon(projections, angles, translations, rec_0=None, niter=100, vox_shape=None,
        vox_pixsize=None, det_pixsize=None, cor_shift=None, comm=None, 
        ground_truth=None, positivity=False, alg='SIRT', beta_tv=1.0, hyper=1.e4):
    
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
    if alg=='SIRT':
        sirt_options = {'method': 'linop', 'precision': np.float32, 'rec': rec_0, 'ground_truth': ground_truth}
        sirt_obj = sirt.SIRT(geom, projections, angles, translations, comm, options=sirt_options)
    
        # regular SIRT
        rec, err = sirt_obj.run_sirt(niter=niter, positivity=positivity, make_plot=True)
    elif alg=='TV':
        tv_options = {'method': 'linop', 'precision': np.float32, 'rec': rec_0, 'ground_truth': ground_truth}
        tv_obj = regularized.RegularizedRecon(geom, projections, angles, translations, comm, options=tv_options)
        rec, err = tv_obj.run_fista(niter=niter,beta_tv=beta_tv,  make_plot=True)

    return rec, err
