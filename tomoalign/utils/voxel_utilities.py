import numpy as np

from tomoalign import vox_wt_grad
from tomoalign.utils.rotations import (
    der_rot_x,
    der_rot_y,
    der_rot_z,
    rot_x,
    rot_y,
    rot_z,
)


def rigid_transformation(x, alpha, beta, phi, xyz):
    # compute rigid body transformation given alignment parameters
    # taking into account center-of-rotation shift
    # Ax --> R_b (R_a R_t(x) + s).
    # Center-of-rotation shift c affects derivatives wrt alpha, beta and theta only.

    R_b = rot_y(beta)
    R_a = rot_x(alpha)
    R_t = rot_z(phi)

    rtx = np.dot(R_t, x)
    ratx = np.dot(R_a, rtx)
    ax = np.dot(R_b, ratx + xyz[:, np.newaxis])

    return ax


def derivative_rigid(x, a, b, t, s):
    # compute derivatives of rigid body transformation wrt alignment parameters
    # taking into account center-of-rotation shift
    # Ax --> R_b (R_a(R_t(x + c) - c) + t).
    # Center-of-rotation shift c affects derivatives wrt alpha, beta and theta only.

    R_b = rot_y(b)
    R_a = rot_x(a)
    R_t = rot_z(t)
    dR_b = der_rot_y(b)
    dR_a = der_rot_x(a)
    dR_t = der_rot_z(t)

    rtx = np.dot(R_t, x)
    ratx = np.dot(R_a, rtx)
    rba = np.dot(R_b, R_a)
    # order of derivatives in matrix sx, sy, sz, t, a, b, cx
    der = np.zeros((6, x.shape[0], x.shape[1]))
    der[0] = np.dot(R_b, np.array([1.0, 0.0, 0.0]))[:, np.newaxis]  # = R_b[:, 0]
    der[1] = np.dot(R_b, np.array([0.0, 1.0, 0.0]))[:, np.newaxis]  # = R_b[:, 1]
    der[2] = np.dot(R_b, np.array([0.0, 0.0, 1.0]))[:, np.newaxis]  # = R_b[:, 2]
    der[3] = np.dot(rba, np.dot(dR_t, x))
    der[4] = np.dot(R_b, np.dot(dR_a, rtx))
    der[5] = np.dot(dR_b, ratx + s[:, np.newaxis])

    return der


def forward_sparse(geometry, alpha, beta, phi, xyz_shift):
    """
    Compute bilinear interpolation weights for rotated voxel centers projected
    along the y-axis onto planar detector. This function prodces detector and
    data indices, and interpolation weights that are used to make a sparse
    forward projection matrix.
    """

    det_shape = geometry.det_shape
    rot_centers = rigid_transformation(
        geometry.vox_centers, alpha, beta, phi, xyz_shift
    )
    orig = geometry.vox_origin - geometry.cor_shift
    dx = geometry.vox_ds

    floor_x = np.floor((rot_centers[0] - orig[0]) / dx[0]).astype(np.int32)
    floor_z = np.floor((rot_centers[2] - orig[2]) / dx[2]).astype(np.int32)
    alpha_x = ((rot_centers[0] - orig[0] - floor_x * dx[0]) / dx[0]).astype(np.float32)
    alpha_z = ((rot_centers[2] - orig[2] - floor_z * dx[2]) / dx[2]).astype(np.float32)

    dat_inds, det_inds, wts, n_inds = vox_wt_grad.bilinear_sparse(
        geometry.n_vox,
        np.asfortranarray(floor_x),
        np.asfortranarray(floor_z),
        np.asfortranarray(alpha_x),
        np.asfortranarray(alpha_z),
        det_shape[0],
        det_shape[1],
    )
    dat_inds = dat_inds[:n_inds]
    det_inds = det_inds[:n_inds]
    wts = wts[:n_inds]

    return dat_inds, det_inds, wts


def forward_proj_grad(geometry, alpha, beta, phi, xyz_shift, rec):
    """
    Same as forward_sparse, except this function directly computes the projection
    and gradients of the projection wrt to parameters alpha, beta, phi, xyz.
    """

    det_shape = geometry.det_shape
    rot_centers = rigid_transformation(
        geometry.vox_centers, alpha, beta, phi, xyz_shift
    )
    orig = geometry.vox_origin - geometry.cor_shift
    dx = geometry.vox_ds

    floor_x = np.floor((rot_centers[0] - orig[0]) / dx[0]).astype(np.int32)
    floor_z = np.floor((rot_centers[2] - orig[2]) / dx[2]).astype(np.int32)
    alpha_x = ((rot_centers[0] - orig[0] - floor_x * dx[0]) / dx[0]).astype(np.float32)
    alpha_z = ((rot_centers[2] - orig[2] - floor_z * dx[2]) / dx[2]).astype(np.float32)

    der = derivative_rigid(geometry.vox_centers, alpha, beta, phi, xyz_shift)
    der = der.astype(np.float32)

    det_img, gradient = vox_wt_grad.bilinear_vox_interp(
        geometry.n_vox,
        np.asfortranarray(floor_x),
        np.asfortranarray(floor_z),
        np.asfortranarray(alpha_x),
        np.asfortranarray(alpha_z),
        np.asfortranarray(rec.ravel()),
        det_shape[0],
        det_shape[1],
        np.asfortranarray(der),
    )
    return det_img.ravel(), gradient.reshape(6, -1)
