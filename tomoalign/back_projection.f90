subroutine back_project(alpha, beta, phi, xyz, voxel_centers, origin, det_image, n_proj, n_vox, n_det_x, n_det_z, atx)
    
    implicit none

    integer(kind=4),                                   intent(in)  :: n_proj
    real(kind=4), dimension(n_proj),                   intent(in)  :: alpha, beta, phi
    real(kind=4), dimension(n_proj, n_det_x, n_det_z), intent(in)  :: det_image
    real(kind=4), dimension(3, n_proj),                intent(in)  :: xyz
    real(kind=4), dimension(3, n_vox),                 intent(in)  :: voxel_centers
    real(kind=4), dimension(3),                        intent(in)  :: origin
    integer(kind=4),                                   intent(in)  :: n_vox, n_det_x, n_det_z
    real(kind=4), dimension(n_vox),                    intent(out) :: atx
    
    !local variables
    integer(kind=4) :: np
    real(kind=4), dimension(3, n_vox) :: rot_voxel_centers
    real(kind=4), dimension(n_vox)    :: vox_image
    
    ! external subroutines from external_back_projection.f90
    external :: voxel_rigid_transformation, voxel_back_bilinear
    
    !write(*, *) n_proj, n_vox, n_det_z, n_det_x
    atx = 0._4
    
    do np = 1, n_proj
        ! apply rigid body transformation to voxel_centers
        call voxel_rigid_transformation(voxel_centers, alpha(np), beta(np), phi(np), xyz(:, np),&
                n_vox, rot_voxel_centers)
        ! compute contribution of this projection to the reconstruction using bilinear interpolation
        call voxel_back_bilinear(rot_voxel_centers, n_vox, n_det_x, n_det_z, origin, det_image(np, :, :), vox_image)
        atx = atx + vox_image
    end do

end subroutine back_project
