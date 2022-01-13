subroutine forward_project(alpha, beta, phi, xyz, cor_shift, source_points, detector_points, origin, &
        step_size, nx, ny, nz, recon, n_proj, n_rays, n_vox, ax)
    
    implicit none
    
    integer(kind=4),                         intent(in)  :: n_proj, n_rays, n_vox
    real(kind=4), dimension(n_proj),         intent(in)  :: alpha, beta, phi
    !real(kind=4), dimension(nx, ny, nz),     intent(in)  :: recon
    real(kind=4), dimension(n_vox),          intent(in)  :: recon
    real(kind=4), dimension(3, n_proj),      intent(in)  :: xyz, cor_shift
    real(kind=4), dimension(3, n_rays),      intent(in)  :: source_points, detector_points
    real(kind=4), dimension(3),              intent(in)  :: origin
    real(kind=4),                            intent(in)  :: step_size
    integer(kind=4),                         intent(in)  :: nx, ny, nz
    real(kind=4), dimension(n_proj, n_rays), intent(out) :: ax
    
    !local variables
    integer(kind=4) :: np, n_on_ray, i, j
    real(kind=4), dimension(3, n_rays) :: p0, p1, r
    real(kind=4), dimension(n_rays) :: a_temp
    real(kind=4), dimension(3) :: r_hat
    real(kind=4) :: r_length
    real(kind=4), allocatable, dimension(:, :, :) :: points_on_ray
    
    ! external subroutines from external_forward_projection
    external :: rigid_transformation, ray_forward_trilinear
    
    ax(:, :) = 0._4
    
    do np = 1, n_proj
        ! apply rigid body transformation to source and detector points
        call rigid_transformation(source_points, alpha(np), beta(np), phi(np), xyz(:, np), n_rays, p0)
        call rigid_transformation(detector_points, alpha(np), beta(np), phi(np), xyz(:, np), n_rays, p1)
        
        do i = 1, 3
            p0(i, :) = p0(i, :) - origin(i)
            p1(i, :) = p1(i, :) - origin(i)
        end do
        
        ! rays connecting transformed source and detector points
        r = p1 - p0
        r_length = sqrt(r(1, 1)**2 + r(2, 1)**2 + r(3, 1)**2)
        r_hat = r(:, 1)/r_length
        n_on_ray = nint(r_length/step_size)
        
        allocate( points_on_ray(3, n_rays, n_on_ray) )
        points_on_ray = 0._4
        
        ! march along the rays and get points at regular intervals
        
        do i = 1, 3
            do j = 1, n_on_ray
                points_on_ray(i, :, j) = p0(i, :) + (j-1)*step_size*r_hat(i)
            end do
        end do
        
        ! compute forward projection for all rays for this projection np
        
        !call ray_forward_trilinear(points_on_ray(:, :, 1:n_on_ray), n_rays, n_on_ray, reshape(recon, (/nx*ny*nz/)), &
        !        nx, ny, nz, a_temp)
        
        call ray_forward_trilinear(points_on_ray(:, :, 1:n_on_ray), n_rays, n_on_ray, recon, nx, ny, nz, a_temp)
        ax(np, :) = a_temp(:)
        
        deallocate( points_on_ray )
    end do
    
end subroutine forward_project
