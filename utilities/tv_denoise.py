import numpy as np


def tv_norm(im):
    """Compute the (isotropic) TV norm of an image
    sqrt(
    """
    grad_x1 = np.diff(im, axis=0)
    grad_x2 = np.diff(im, axis=1)
    return np.sqrt(grad_x1[:, :-1]**2 + grad_x2[:-1, :]**2).sum()


def tv_norm_anisotropic(im):
    """Compute the anisotropic TV norm of an image"""
    grad_x1 = np.diff(im, axis=0)
    grad_x2 = np.diff(im, axis=1)
    return np.abs(grad_x1[:, :-1]).sum() + np.abs(grad_x2[:-1, :]).sum()


def div(grad):
    """ Compute divergence of image gradient
    Courtesy : E. Gouillart - https://github.com/emmanuelle/tomo-tv/
    """
    res = np.zeros(grad.shape[1:], dtype=grad.dtype)
    for d in range(grad.shape[0]):
        this_grad = np.rollaxis(grad[d], d)
        this_res = np.rollaxis(res, d)
        this_res[:-1] += this_grad[:-1]
        this_res[1:-1] -= this_grad[:-2]
        this_res[-1] -= this_grad[-2]
    return res


def gradient(img):
    """
    Compute gradient of an N-d image

    Parameters
    ===========
    img: ndarray
        N-dimensional image

    Returns
    =======
    gradient: ndarray
        Gradient of the image: the i-th component along the first
        axis is the gradient along the i-th axis of the original
        array img
    Courtesy : E. Gouillart - https://github.com/emmanuelle/tomo-tv/
    """
    shape = [img.ndim, ] + list(img.shape)
    gradient = np.zeros(shape, dtype=img.dtype)
    
    slice_all = [slice(None, -1)]
    for d in range(img.ndim):
        gradient[d][tuple(slice_all)] = np.diff(img, axis=d)
        slice_all.insert(d, slice(None))
        
    return gradient


def tv_norm_3d(x):
    
    return np.linalg.norm(gradient(x))


def _projector_on_dual(grad):
    """
    modifies in place the gradient to project it
    on the L2 unit ball
    """
    norm = np.maximum(np.sqrt(np.sum(grad**2, 0)), 1.)
    for grad_comp in grad:
        grad_comp /= norm
    return grad


def dual_gap(im, new, gap, weight):
    """
    dual gap of total variation denoising
    see "Total variation regularization for fMRI-based prediction of behavior",
    by Michel et al. (2011) for a derivation of the dual gap
    """
    im_norm = (im**2).sum()
    gx, gy = np.zeros_like(new), np.zeros_like(new)
    gx[:-1] = np.diff(new, axis=0)
    gy[:, :-1] = np.diff(new, axis=1)
    if im.ndim == 3:
        gz = np.zeros_like(new)
        gz[..., :-1] = np.diff(new, axis=2)
        tv_new = 2 * weight * np.sqrt(gx**2 + gy**2 + gz**2).sum()
    else:
        tv_new = 2 * weight * np.sqrt(gx**2 + gy**2).sum()
    dual_gap = (gap**2).sum() + tv_new - im_norm + (new**2).sum()
    return 0.5 / im_norm * dual_gap


def denoise_fista(im, weight=50, niter=200, eps=1.e-5, check_gap_frequency=3):

    """
    Perform total-variation denoising on N-d images

    Find the argmin `res` of
        1/2 * ||im - res||^2 + weight * TV(res),

    where TV is the isotropic l1 norm of the gradient.
    This function implements the FISTA (Fast Iterative Shrinkage
    Thresholding Algorithm) algorithm of Beck et Teboulle, adapted to
    total variation denoising in "Fast gradient-based algorithms for
    constrained total variation image denoising and deblurring problems"
    (2009).
    
    Parameters
    ----------
    im: ndarray of floats (2-d or 3-d)
        input data to be denoised. `im` can be of any numeric type,
        but it is cast into an ndarray of floats for the computation
        of the denoised image.

    weight: float, optional
        denoising weight. The greater ``weight``, the more denoising (at
        the expense of fidelity to ``input``)

    eps: float, optional
        precision required. The distance to the exact solution is computed
        by the dual gap of the optimization problem and rescaled by the l2
        norm of the image (for contrast invariance).

    n_iter_max: int, optional
        maximal number of iterations used for the optimization.

    Returns
    -------
    out: ndarray
        denoised image

    """
    #if not im.dtype.kind == 'f':
    #    im = im.astype(np.float)
    
    shape = [im.ndim, ] + list(im.shape)
    if shape[0] == 3:
        factor = 12.0
    else:
        factor = 8.0
    
    grad_im = np.zeros(shape, dtype=im.dtype)
    grad_aux = np.zeros(shape, dtype=im.dtype)
    t = 1.
    i = 0
    new = im.copy()
    while i < niter:
        error = weight * div(grad_aux) - im
        grad_tmp = gradient(error)
        grad_tmp *= 1/(factor * weight)
        grad_aux += grad_tmp
        grad_tmp = _projector_on_dual(grad_aux)
        t_new = 0.5 * (1 + np.sqrt(1 + 4 * t**2))
        t_factor = (t - 1) / t_new
        grad_aux = (1 + t_factor) * grad_tmp - t_factor * grad_im
        grad_im = grad_tmp
        t = t_new
        if (i % check_gap_frequency) == 0:
            gap = weight * div(grad_im)
            new = im - gap
            dgap = dual_gap(im, new, gap, weight)
            if dgap < eps:
                break
        i += 1
    return new
