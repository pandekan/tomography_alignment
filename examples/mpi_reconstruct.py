import numpy as np
import h5py
from mpi4py import MPI
from scipy import sparse
import copy
import sys
from utilities import geometry, projection_operators, generate_phantom
from recon import regularized_mpi

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# shepp-logan phantom
ground_truth = generate_phantom.shepp3d(64)
nx, ny, nz = ground_truth.shape

#ground_truth = generate_phantom.shepp3d(nx)
n_proj = 90
phi = np.linspace(0.0, np.pi, n_proj)
alpha = np.zeros(n_proj, )
beta = np.zeros(n_proj, )
xyz = np.zeros((n_proj, 3))

# geometry
geom = geometry.Geometry(n_proj, np.array([nx, ny, nz]), np.ones(3, ),
                         np.array([nx, nz]), np.ones(2, ))

# generate data -- parallelize over projections
my_proj = np.zeros((n_proj, geom.n_det), dtype=np.float32)
proj = np.zeros_like(my_proj)

proj_obj = projection_operators.ProjectionMatrix(geom, precision=np.float32)
my_proj_index = np.array_split(np.arange(n_proj), size)[rank]
my_n_proj = np.size(my_proj_index)
pmat = proj_obj.projection_matrix(alpha=alpha[my_proj_index], beta=beta[my_proj_index], 
        phi=phi[my_proj_index], xyz_shift=xyz[my_proj_index])
my_proj[my_proj_index] = sparse.csr_matrix.dot(pmat, ground_truth.ravel()).reshape(my_n_proj, -1)

comm.Barrier()
comm.Allreduce([my_proj, MPI.FLOAT], [proj, MPI.FLOAT], op=MPI.SUM)

# reconstruct
niter_rec = 500
rec = None
angles = np.array([phi, alpha, beta]).T
rec_obj = regularized_mpi.RegularizedRecon(comm, geom, proj.reshape(n_proj, -1), angles, xyz,
            options={'ground_truth':ground_truth, 'rec':rec})

penalty = 'TV'

if penalty == 'Tikh':
    # Tikhonov penalized
    reg_tv = 0.1
    rec, err = rec_obj.run_tikhonov_gd(niter=niter_rec, reg_param=reg_tv, positivity=True, make_plot=True)
elif penalty == 'Lasso':
    # |x|_1 penalised with iterative soft-thresholding
    reg_ista = 1.0
    beta_ista_ls = 0.8
    rec, err = rec_obj.run_lasso_accelerated(niter=niter_rec, reg_param=reg_ista, beta=beta_ista_ls, make_plot=True)
elif penalty == 'TV':
    # TV penalised with FISTA
    hyper = 1.e4
    beta_tv = 0.1
    rec, err = rec_obj.run_fista(niter=niter_rec, hyper=hyper, beta_tv=beta_tv, make_plot=True)
else:
    if rank == 0:
        print('%s penalty not implemented' %(penalty))

if rank == 0:
    np.save('recon.npy', rec)
