# tomography_alignment
Rigid body alignment for x-ray tomography data

# compile fortran code as follows
gfortran -c mod1.f90 mod2.f90 .....

f2py3 -c mod1.f90 mod2.f90 .... subroutine.f90 -m subroutine
