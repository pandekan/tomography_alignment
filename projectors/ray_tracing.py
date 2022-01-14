# ------------------------------------------------
# Copyright 2021 Kanupriya Pande
# Contact kpande@lbl.gov
# ------------------------------------------------

import numpy as np
from utilities.rotations import rot_x, rot_y, rot_z, der_rot_x, der_rot_y, der_rot_z
from src import ray_wt_grad


def transform_points(x, alpha, beta, phi, t):
    
    rot_pa = np.dot(rot_z(phi), rot_x(alpha))
    xp = np.dot(rot_y(beta), x) + t[:, np.newaxis]
    xp = np.dot(rot_pa, xp)
    
    return xp


def derivative_ray_points(source_points, ray_vector, alpha, beta, phi, xyz_shift):
    """
    Rx --> R_phi * R_alpha * (R_beta x + t)
    p_j = Rs + j*r_hat where r_hat = (Rd-Rs)/|d-s|
    :param source_points: (ub-transformed) source points for this ray
    :param ray_vector: (un-transformed) ray-vector = detector_points - source_points
    :param angles: rotation angles
    :param xyz_shift: translations
    :return: ndarray with derivatives (6, 3)
    """
    R_p = rot_z(phi)
    R_a = rot_x(alpha)
    R_b = rot_y(beta)
    dR_p = der_rot_z(phi)
    dR_a = der_rot_x(alpha)
    dR_b = der_rot_y(beta)
    R_pa = np.dot(R_p, R_a)
    R_ab = np.dot(R_a, R_b)

    der = np.zeros((9, 3, source_points.shape[1]))
    # we have 9 components here since we separate the derivative wrt angles into a part that does not depend on
    # step/ray_length and that which does
    # example derivative wrt phi = dR_p R_a (R_b s + xyz_shift) + (step/ray_length) * dR_p R_a R_b (d-s)
    rpa_dot_eye = np.dot(R_pa, np.eye(3))
    der[0] = rpa_dot_eye[:, 0][:, np.newaxis]  # d/dt_x
    der[1] = rpa_dot_eye[:, 1][:, np.newaxis]  # d/dt_y
    der[2] = rpa_dot_eye[:, 2][:, np.newaxis]  # d/dt_z

    Rb_st = np.dot(R_b, source_points) + xyz_shift[:, np.newaxis]
    der[3] = np.dot(dR_p, np.dot(R_a, Rb_st))
    der[4] = np.dot(R_p, np.dot(dR_a, Rb_st))
    der[5] = np.dot(R_pa, np.dot(dR_b, source_points))
    der[6] = np.dot(dR_p, np.dot(R_ab, ray_vector))[:, np.newaxis]
    der[7] = np.dot(R_p, np.dot(dR_a, np.dot(R_b, ray_vector)))[:, np.newaxis]
    der[8] = np.dot(R_pa, np.dot(dR_b, ray_vector))[:, np.newaxis]
    return der


def forward_sparse(alpha, beta, phi, xyz_shift, cor_shift, source_centers, det_centers, vox_origin,
                   step_size, n_rays, nx, ny, nz):
    """
    Compute detector and data indices and trilinear interpolation weights for projection
    at angle phi with jitter angles alpha and beta, and jitter translations xyz. The indices
    and weights are used to make a sparse matrix for forward projection.
    alpha: rotation about X-axis
    beta: rotation about Y-axis
    phi: tomographic rotation about Z-axis
    xyz_shift: Translational jitter
    cor_shift: shift of center of rotation
    source_centers, det_centers: unpertubed centers of source and detector
    vox_origin: origin of 3D volume
    step_size: step size to compute points along ray
    n_rays: total numbers of rays = source_centers.shape[1]
    nx, ny, nz: shape of 3D volume
    :return: dat_ind, det_ind, wts
    Create sparse matrix by sparse.coo_matrix((wts, (det_inds, dat_inds)), shape=(geo.n_det, geo.n_vox))
    Using sparse matrix create projection by sparse.csr_matrix.dot(pmat, 3d_obj.ravel())
    This function calls fortran function trilinear_ray_sparse for fast computation.
    """
    
    # center-of-rotation shift
    source_centers[0, :] += cor_shift[0]
    det_centers[0, :] += cor_shift[0]
    p0 = transform_points(source_centers, alpha, beta, phi, xyz_shift) - vox_origin[:, np.newaxis]
    p1 = transform_points(det_centers, alpha, beta, phi, xyz_shift) - vox_origin[:, np.newaxis]
    
    # now ray trace
    r = p1 - p0
    r_length = np.linalg.norm(r, axis=0)
    r_hat = r / r_length
    n = int(r_length[0] / step_size)  # how many points on the ray
    r_points = np.zeros((3, n_rays, n))
    r_points[:, :, :] = p0[:, :, np.newaxis]
    step = np.zeros((n_rays, n))
    for j in range(n):
        r_points[:, :, j] += j * step_size * r_hat
        step[:, j] = j * step_size / r_length[0]
    
    floor_points = np.array([np.floor(r_points[dim]) for dim in range(3)]).astype(np.int32)
    
    w_ceil = r_points - floor_points.astype(np.float32)
    w_floor = 1. - w_ceil
    
    # fortran routine computes weights in single precision
    dat_inds, det_inds, wts, n_inds = ray_wt_grad.trilinear_ray_sparse(np.asfortranarray(floor_points.astype(np.int32)),
                                                                       np.asfortranarray(w_floor.astype(np.float32)),
                                                                       nx, ny, nz, n_rays, n)

    dat_inds = dat_inds[:n_inds]
    det_inds = det_inds[:n_inds]
    wts = wts[:n_inds]
    
    return dat_inds, det_inds, wts


def forward_proj_grad(geometry, alpha, beta, phi, xyz_shift, rec):
    """
    Same as forward_sparse, except this function directly computes the forward projection
    and the gradient of this projection wrt parameters (alpha, beta, phi, xyz).
    :param geometry: 
    :param alpha: 
    :param beta: 
    :param phi: 
    :param xyz_shift: 
    :param rec: 
    :return: 
    """
    N = geometry.vox_shape
    n_rays = geometry.n_det
    vox_ds = geometry.vox_ds
   
    geometry.source_centers[0, :] += geometry.cor_shift[0]
    geometry.det_centers[0, :] += geometry.cor_shift[0]
    
    p0 = transform_points(geometry.source_centers, alpha, beta, phi, xyz_shift) - geometry.vox_origin[:, np.newaxis]
    p1 = transform_points(geometry.det_centers, alpha, beta, phi, xyz_shift) - geometry.vox_origin[:, np.newaxis]
    
    # now ray trace
    step_size = geometry.step_size
    r = p1 - p0
    r_length = np.linalg.norm(r, axis=0)
    r_hat = r / r_length
    n = int(r_length[0] / step_size)  # how many points on the ray
    r_points = np.zeros((3, n_rays, n))
    r_points[:, :, :] = p0[:, :, np.newaxis]
    step = np.zeros((n_rays, n))
    for j in range(n):
        r_points[:, :, j] += j * step_size * r_hat
        step[:, j] = j * step_size / r_length[0]
    
    floor_points = np.array([np.floor(r_points[dim]) for dim in range(3)]).astype(np.int32)
    
    w_ceil = r_points - floor_points.astype(np.float64)
    w_floor = 1. - w_ceil
    
    untransformed_ray = geometry.det_centers - geometry.source_centers
    der = derivative_ray_points(geometry.source_centers, untransformed_ray[:, 0], alpha, beta, phi, xyz_shift)

    # fortran routine computes det_img and grad_det_img in single precision
    # input 3d rec also passed in in single precision
    # integers are int32
    det_img, grad_det_img = ray_wt_grad.trilinear_ray_interp(np.asfortranarray(floor_points),
                                                             np.asfortranarray(w_floor),
                                                             N[0], N[1], N[2], n_rays, n,
                                                             np.asfortranarray(rec.ravel().astype(np.float32)),
                                                             np.asfortranarray(step.astype(np.float32)),
                                                             np.asfortranarray(der.astype(np.float32)))
    return det_img, grad_det_img


def ray_tracing_trilinear(vox_shape, p0, p1, vox_ds, step_size, precision=np.float32, return_der=False):
    """
    March along parallel rays and compute trilinear interpolation weights at each point on ray.
    Since rays are parallel, r_hat and r_length will be same for all.
    :param vox_shape: shape of object pierced by the rays
    :param p0: source ray-points
    :param p1: detector points
    :param vox_ds: downsampling of object wrt original voxel size
    :param step_size: step size to increment points
    :param precision: precision for interpolation weights
    :param return_der: if weight derivatives should be returned (needed for parameter optimization)
    :return wts: trilinear interpolations weights
    :return det_inds: mapped detector indices
    :return dat_inds: corners of voxel indices corresponding within which sampled point lies
    """
    
    N = vox_shape
    n_rays = p0.shape[1]
    r = p1 - p0
    r_length = np.linalg.norm(r, axis=0)
    r_hat = r/r_length
    n = int(r_length[0]/step_size)  # how many points on the ray
    r_points = np.zeros((3, n_rays, n))

    r_points[:, :, :] = p0[:, :, np.newaxis]
    for j in range(n):
        r_points[:, :, j] += j * step_size * r_hat
    
    floor_points = np.array([np.floor(r_points[dim]) for dim in range(3)]).astype(np.int32)
    ceil_points = floor_points + 1
    
    w_ceil = r_points - floor_points.astype(np.float32)
    w_floor = 1. - w_ceil
    
    in_0 = np.logical_and(floor_points[0] >= 0, ceil_points[0] < vox_ds[0] * N[0])
    in_1 = np.logical_and(floor_points[1] >= 0, ceil_points[1] < vox_ds[1] * N[1])
    in_2 = np.logical_and(floor_points[2] >= 0, ceil_points[2] < vox_ds[2] * N[2])
    in_volume = in_0 * in_1 * in_2
    
    ray_ind = np.where(in_volume)[0]
    pixel_ind = np.where(in_volume)[1]
    floor_points = floor_points[:, ray_ind, pixel_ind]
    ceil_points = ceil_points[:, ray_ind, pixel_ind]
    norm_factor = np.linalg.norm(r_hat[:, ray_ind], axis=0) ** (1 / 3.)
    w_floor = w_floor[:, ray_ind, pixel_ind] * norm_factor
    w_ceil = w_ceil[:, ray_ind, pixel_ind] * norm_factor
    
    det_inds = np.hstack((ray_ind, ray_ind, ray_ind, ray_ind, ray_ind, ray_ind, ray_ind, ray_ind))

    wts = np.hstack((w_floor[0] * w_floor[1] * w_floor[2],
                     w_floor[0] * w_floor[1] * w_ceil[2],
                     w_floor[0] * w_ceil[1] * w_floor[2],
                     w_floor[0] * w_ceil[1] * w_ceil[2],
                     w_ceil[0] * w_floor[1] * w_floor[2],
                     w_ceil[0] * w_floor[1] * w_ceil[2],
                     w_ceil[0] * w_ceil[1] * w_floor[2],
                     w_ceil[0] * w_ceil[1] * w_ceil[2]))

    # now divide by step to get data indices
    floor_points = np.array([floor_points[dim] / vox_ds[dim] for dim in range(3)]).astype(np.int32)
    ceil_points = np.array([ceil_points[dim] / vox_ds[dim] for dim in range(3)]).astype(np.int32)

    inds = np.hstack(((floor_points[0] * N[1] + floor_points[1]) * N[2] + floor_points[2],
                      (floor_points[0] * N[1] + floor_points[1]) * N[2] + ceil_points[2],
                      (floor_points[0] * N[1] + ceil_points[1]) * N[2] + floor_points[2],
                      (floor_points[0] * N[1] + ceil_points[1]) * N[2] + ceil_points[2],
                      (ceil_points[0] * N[1] + floor_points[1]) * N[2] + floor_points[2],
                      (ceil_points[0] * N[1] + floor_points[1]) * N[2] + ceil_points[2],
                      (ceil_points[0] * N[1] + ceil_points[1]) * N[2] + floor_points[2],
                      (ceil_points[0] * N[1] + ceil_points[1]) * N[2] + ceil_points[2]))
    
    der_wts = None
    if return_der:
        der_wts = np.hstack((np.array([-w_floor[1] * w_floor[2], -w_floor[0] * w_floor[2], -w_floor[0] * w_floor[1]]),
                             np.array([-w_floor[1] * w_ceil[2],  -w_floor[0] * w_ceil[2], w_floor[0] * w_floor[1]]),
                             np.array([-w_ceil[1] * w_floor[2], w_floor[0] * w_floor[2], -w_floor[0] * w_ceil[1]]),
                             np.array([-w_ceil[1] * w_ceil[2],  w_floor[0] * w_ceil[2], w_floor[0] * w_ceil[1]]),
                             np.array([w_floor[1] * w_floor[2], -w_ceil[0] * w_floor[2], -w_ceil[0] * w_floor[1]]),
                             np.array([w_floor[1] * w_ceil[2],  -w_ceil[0] * w_ceil[2], w_ceil[0] * w_floor[1]]),
                             np.array([w_ceil[1] * w_floor[2], w_ceil[0] * w_floor[2], -w_ceil[0] * w_ceil[1]]),
                             np.array([w_ceil[1] * w_ceil[2],  w_ceil[0] * w_ceil[2], w_ceil[0] * w_ceil[1]])))
        der_wts = der_wts.astype(precision, copy=False)
    
    return wts.astype(precision, copy=False), det_inds, inds, der_wts


def ray_weights_der(p0, p1, geometry, angles, xyz_shift, rec, return_der=True):
    
    N = geometry.vox_shape
    n_rays = geometry.n_det
    vox_ds = geometry.vox_ds
    phi, alpha, beta = angles
    
    # now ray trace
    step_size = geometry.step_size
    r = p1 - p0
    r_length = np.linalg.norm(r, axis=0)
    r_hat = r / r_length
    n = int(r_length[0] / step_size)  # how many points on the ray
    r_points = np.zeros((3, n_rays, n))
    r_points[:, :, :] = p0[:, :, np.newaxis]
    step = np.zeros((n_rays, n))
    for j in range(n):
        r_points[:, :, j] += j * step_size * r_hat
        step[:, j] = j*step_size/r_length[0]
        
    floor_points = np.array([np.floor(r_points[dim]) for dim in range(3)]).astype(np.int32)
    ceil_points = floor_points + 1
    
    w_ceil = r_points - floor_points.astype(np.float32)
    w_floor = 1. - w_ceil
    
    in_0 = np.logical_and(floor_points[0] >= 0, ceil_points[0] < vox_ds[0] * N[0])
    in_1 = np.logical_and(floor_points[1] >= 0, ceil_points[1] < vox_ds[1] * N[1])
    in_2 = np.logical_and(floor_points[2] >= 0, ceil_points[2] < vox_ds[2] * N[2])
    in_volume = in_0 * in_1 * in_2
    
    ray_ind = np.where(in_volume)[0]
    pixel_ind = np.where(in_volume)[1]
    step = step[in_volume]
    floor_points = floor_points[:, ray_ind, pixel_ind]
    ceil_points = ceil_points[:, ray_ind, pixel_ind]
    w_floor = w_floor[:, ray_ind, pixel_ind]
    w_ceil = w_ceil[:, ray_ind, pixel_ind]

    wts = np.hstack((w_floor[0] * w_floor[1] * w_floor[2],
                     w_floor[0] * w_floor[1] * w_ceil[2],
                     w_floor[0] * w_ceil[1] * w_floor[2],
                     w_floor[0] * w_ceil[1] * w_ceil[2],
                     w_ceil[0] * w_floor[1] * w_floor[2],
                     w_ceil[0] * w_floor[1] * w_ceil[2],
                     w_ceil[0] * w_ceil[1] * w_floor[2],
                     w_ceil[0] * w_ceil[1] * w_ceil[2]))
    
    inds = np.hstack(((floor_points[0] * N[1] + floor_points[1]) * N[2] + floor_points[2],
                     (floor_points[0] * N[1] + floor_points[1]) * N[2] + ceil_points[2],
                     (floor_points[0] * N[1] + ceil_points[1]) * N[2] + floor_points[2],
                     (floor_points[0] * N[1] + ceil_points[1]) * N[2] + ceil_points[2],
                     (ceil_points[0] * N[1] + floor_points[1]) * N[2] + floor_points[2],
                     (ceil_points[0] * N[1] + floor_points[1]) * N[2] + ceil_points[2],
                     (ceil_points[0] * N[1] + ceil_points[1]) * N[2] + floor_points[2],
                     (ceil_points[0] * N[1] + ceil_points[1]) * N[2] + ceil_points[2]))
    
    der_wts = np.hstack((np.array([-w_floor[1] * w_floor[2], -w_floor[0] * w_floor[2], -w_floor[0] * w_floor[1]]),
                         np.array([-w_floor[1] * w_ceil[2], -w_floor[0] * w_ceil[2], w_floor[0] * w_floor[1]]),
                         np.array([-w_ceil[1] * w_floor[2], w_floor[0] * w_floor[2], -w_floor[0] * w_ceil[1]]),
                         np.array([-w_ceil[1] * w_ceil[2], w_floor[0] * w_ceil[2], w_floor[0] * w_ceil[1]]),
                         np.array([w_floor[1] * w_floor[2], -w_ceil[0] * w_floor[2], -w_ceil[0] * w_floor[1]]),
                         np.array([w_floor[1] * w_ceil[2], -w_ceil[0] * w_ceil[2], w_ceil[0] * w_floor[1]]),
                         np.array([w_ceil[1] * w_floor[2], w_ceil[0] * w_floor[2], -w_ceil[0] * w_ceil[1]]),
                         np.array([w_ceil[1] * w_ceil[2], w_ceil[0] * w_ceil[2], w_ceil[0] * w_ceil[1]])))
    
    inds = inds.reshape(8, -1)
    wts = wts.reshape(8, -1)
    der_wts = der_wts.reshape(3, 8, -1)
    gradient = np.zeros((6, n_rays))
    proj = np.zeros((n_rays, ))
    untransformed_ray = geometry.det_centers - geometry.source_centers
    der = derivative_ray_points(geometry.source_centers, untransformed_ray[:, 0], alpha, beta, phi, xyz_shift)
    for i, ri in enumerate(ray_ind):
        # get derivatives of points wrt angles and translations
        g = np.zeros((6, 3))
        g[:3, :] = der[:3, :, ri]
        g[3, :] = der[3, :, ri] + step[i]*der[6, :, ri]
        g[4, :] = der[4, :, ri] + step[i]*der[7, :, ri]
        g[5, :] = der[5, :, ri] + step[i]*der[8, :, ri]
        g0 = np.sum(der_wts[0, :, i]*rec.ravel()[inds[:, i]])
        g1 = np.sum(der_wts[1, :, i]*rec.ravel()[inds[:, i]])
        g2 = np.sum(der_wts[2, :, i]*rec.ravel()[inds[:, i]])
        gradient[:, ri] += (g0*g[:, 0] + g1*g[:, 1] + g2*g[:, 2])
        proj[ri] += np.sum(rec.ravel()[inds[:, i]]*wts[:, i])
    
    return proj, gradient
