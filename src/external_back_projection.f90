subroutine voxel_rigid_transformation(x, a, b, p, t, npoints, xp)
    
    use rotations_module

    implicit none
    
    integer(kind=4),                     intent(in)  :: npoints
    real(kind=4), dimension(3, npoints), intent(in)  :: x
    real(kind=4),                        intent(in)  :: a, b, p
    real(kind=4), dimension(3),          intent(in)  :: t
    real(kind=4), dimension(3, npoints), intent(out) :: xp
    
    ! local variables
    integer(kind=4) :: i
    real(kind=4), dimension(3, 3) :: ra, rb, rp
    
    call rot_x(a, ra)
    call rot_y(b, rb)
    call rot_z(p, rp)
    xp = matmul(rp, x)
    xp = matmul(ra, xp)
    do i = 1, 3
        xp(i, :) = xp(i, :) + t(i)
    end do
    xp = matmul(rb, xp)
    
end subroutine voxel_rigid_transformation


subroutine voxel_back_bilinear(rot_voxel_centers, n_vox, n_det_x, n_det_z, origin, det_image, vox_image)
    
    implicit none
    
    integer(kind=4),                            intent(in) :: n_vox, n_det_x, n_det_z
    real(kind=4), dimension(3, n_vox),          intent(in) :: rot_voxel_centers
    real(kind=4), dimension(3),                 intent(in) :: origin
    real(kind=4), dimension(n_det_x, n_det_z),  intent(in) :: det_image
    real(kind=4), dimension(n_vox),             intent(out):: vox_image
    
    ! local variables
    integer(kind=4)                    :: i, fx, fz
    integer(kind=4), dimension(n_vox)  :: floor_x, floor_z
    real(kind=4), dimension(n_vox)     :: alpha_x, alpha_z
    
    floor_x = floor(rot_voxel_centers(1, :) - origin(1))
    floor_z = floor(rot_voxel_centers(3, :) - origin(3))
    alpha_x = rot_voxel_centers(1, :) - origin(1) - real(floor_x)
    alpha_z = rot_voxel_centers(3, :) - origin(3) - real(floor_z)
    
    vox_image = 0._4
    do i = 1, n_vox
        fx = floor_x(i)+1
        fz = floor_z(i)+1
        if (fx >= 1 .and. fx <= n_det_x .and. fz >= 1 .and. fz <= n_det_z) then
            vox_image(i) = vox_image(i) + det_image(fz, fx) * (1._4-alpha_x(i)) * (1._4-alpha_z(i))
        end if
        if (fx+1 >= 1 .and. fx+1 <= n_det_x .and. fz >= 1 .and. fz <= n_det_z) then
            vox_image(i) = vox_image(i) + det_image(fz, fx+1) * (alpha_x(i)) * (1._4-alpha_z(i))
        end if
        if (fx >= 1 .and. fx <= n_det_x .and. fz+1 >= 1 .and. fz+1 <= n_det_z) then
            vox_image(i) = vox_image(i) + det_image(fz+1, fx) * (1._4-alpha_x(i)) * alpha_z(i)
        end if
        if (fx+1 >= 1 .and. fx+1 <= n_det_x .and. fz+1 >= 1 .and. fz+1 <= n_det_z) then
            vox_image(i) = vox_image(i) + det_image(fz+1, fx+1)  * alpha_x(i) * alpha_z(i)
        end if
    end do
    
end subroutine voxel_back_bilinear
