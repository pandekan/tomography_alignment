# tomography_alignment
Rigid body alignment for x-ray tomography data

# compile fortran code as follows

cd src
gfortran -c ray_wt_grad.f90 vox_wt_grad.f90

f2py3 -c ray_wt_grad.f90 -m ray_wt_grad

f2py3 -c vox_wt_grad.f90 -m vox_wt_grad
