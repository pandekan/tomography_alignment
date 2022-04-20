# tomography_alignment
Rigid body alignment for x-ray tomography data

# compile fortran code as follows
gfortran -c rotations_module.f90 external_forward_projection.f90 external_back_projection.f90
f2py3 -c rotations_module.f90 external_forward_projection.f90 forward_projection.f90 -m forward_projection
f2py3 -c rotations_module.f90 external_forward_projection.f90 projection_gradient.f90 -m projection_gradient
f2py3 -c rotations_module.f90 external_back_projection.f90 back_projection.f90 -m back_projection
