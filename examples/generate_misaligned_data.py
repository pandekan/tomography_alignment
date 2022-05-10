import numpy as np
from utilities import generate_phantom, geometry
from projectors import projection_operators

nx = ny = nz = 64
n_proj = 90

# shepp-logan phantom of size nx X nx X nx
shepp = generate_phantom.shepp3d(nx)

# geometry
geom = geometry.Geometry(n_proj, np.array([nx, ny, nz]), np.ones(3, ),
                         np.array([nx, nz]), np.ones(2, ))

phi = np.linspace(0.0, np.pi, n_proj)
alpha = np.deg2rad(np.random.randint(-100, 100, n_proj)/100)
beta = np.deg2rad(np.random.randint(-100, 100, n_proj)/100)
xyz = np.zeros((n_proj, 3))
# motion along the direction of X-rays does not affect the projection,
# add jitter only along the X- and Z-axes
xyz[:, 0] = np.random.randint(-200, 200, n_proj)/100
xyz[:, 2] = np.random.randint(-200, 200, n_proj)/100

# instantiate projection object with above geometry, method='matrix' and no MPI
proj_obj = projection_operators.Projection(geom, method='matrix', precision=np.float32, comm=None)

# simulate projection images with misalignment
proj_obj.setup(np.array([alpha, beta, phi]).T, xyz_shifts=xyz)
proj = proj_obj.forward_project(shepp)
