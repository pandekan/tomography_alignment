import copy
import sys

import h5py
import numpy as np
from scipy import optimize

from tomoalign.recon import sirt
from tomoalign.utils import alignment_functions, geometry, projection_operators
from tomoalign.utils.alignment_functions import cost_xzab, gradient_xzab

input_fname = sys.argv[1]
f = h5py.File(input_fname, "r")
proj = f["data/projections"][()]
alpha = f["data/alpha"][()]
beta = f["data/beta"][()]
xyz = f["data/xyz"][()]
phi = f["data/phi"][()]
ground_truth = f["data/phantom"][()]
nx, ny, nz = ground_truth.shape
n_proj = proj.shape[0]

# geometry
geom = geometry.Geometry(
    n_proj,
    np.array([nx, ny, nz]),
    np.ones(
        3,
    ),
    np.array([nx, nz]),
    np.ones(
        2,
    ),
)

proj_obj = projection_operators.ProjectionMatrix(geom, precision=np.float32)

niter_align = 35
niter_rec = 1000
alpha_rec = np.zeros((niter_align, n_proj))
beta_rec = np.zeros((niter_align, n_proj))
xyz_rec = np.zeros((niter_align, n_proj, 3))
residual = np.zeros((niter_align, n_proj))
residual[0] = 1.0e6
rec = np.zeros_like(ground_truth)

for it in range(1, niter_align):
    sirt_obj = sirt.SIRT(
        geom,
        proj.reshape(n_proj, -1),
        np.array([phi, alpha_rec[it - 1], beta_rec[it - 1]]).T,
        xyz_rec[it - 1],
        options={"ground_truth": ground_truth, "rec": rec.ravel()},
    )
    rec, err = sirt_obj.run_main_iteration(
        niter=niter_rec, positivity=True, make_plot=True
    )
    for i in range(n_proj):
        print("==================== %2d ====================" % i)
        this_geo = copy.deepcopy(geom)
        this_geo.cor_shift = geom.cor_shift[i]
        align_obj = alignment_functions.AlignmentUtilities(proj[i], proj_obj, this_geo)
        params = np.array(
            [xyz_rec[it, i, 0], xyz_rec[it, i, 2], alpha_rec[it, i], beta_rec[it, i]]
        )
        res = optimize.minimize(
            cost_xzab,
            params,
            method="L-BFGS-B",
            jac=gradient_xzab,
            args=(
                align_obj,
                rec,
                np.array([phi[i], 0.0, 0.0]),
                np.zeros(
                    3,
                ),
            ),
            bounds=((-3.0, 3.0), (-3.0, 3.0), (-0.02, 0.02), (-0.02, 0.02)),
            options={"disp": False},
        )
        xyz_rec[it, i, 0], xyz_rec[it, i, 2] = res.x[:2]
        alpha_rec[it, i], beta_rec[it, i] = res.x[2:]
        residual[it, i] = res.fun
        print(
            "%4.5f, %4.5f, %4.5f, %4.5f"
            % (xyz[i, 0], xyz[i, 2], np.rad2deg(alpha[i]), np.rad2deg(beta[i]))
        )
        print(
            "%4.5f, %4.5f, %4.5f, %4.5f, %4.5f"
            % (
                xyz_rec[it - 1, i, 0],
                xyz_rec[it - 1, i, 2],
                np.rad2deg(alpha_rec[it - 1, i]),
                np.rad2deg(beta_rec[it - 1, i]),
                residual[it - 1, i],
            )
        )
        print(
            "%4.5f, %4.5f, %4.5f, %4.5f, %4.5f"
            % (res.x[0], res.x[1], np.rad2deg(res.x[2]), np.rad2deg(res.x[3]), res.fun)
        )
