subroutine compute_gradient(alpha, beta, phi, xyz, cor_shift, source_points, detector_points, origin, &
        step_size, nx, ny, nz, recon, n_rays, n_vox, ax, dax)
    
    implicit none
    
    integer(kind=4),                         intent(in)  :: n_rays, nx, ny, nz, n_vox
    real(kind=4),                            intent(in)  :: alpha, beta, phi
    real(kind=4), dimension(n_vox),          intent(in)  :: recon
    real(kind=4), dimension(3),              intent(in)  :: xyz, cor_shift
    real(kind=4), dimension(3, n_rays),      intent(in)  :: source_points, detector_points
    real(kind=4), dimension(3),              intent(in)  :: origin
    real(kind=4),                            intent(in)  :: step_size
    real(kind=4), dimension(n_rays),         intent(out) :: ax
    real(kind=4), dimension(6, n_rays),      intent(out) :: dax
    
    !local variables
    integer(kind=4) :: np, n_on_ray, i, j
    real(kind=4), dimension(3, n_rays) :: p0, p1, r, s, d
    real(kind=4), dimension(3) :: r_hat, x0
    real(kind=4) :: r_length
    real(kind=4), dimension(n_rays) :: a_temp
    real(kind=4), dimension(6, n_rays) :: d_temp
    real(kind=4), dimension(6, 3, n_rays) :: der
    real(kind=4), dimension(3, 3) :: append_der
    real(kind=4), allocatable, dimension(:, :, :) :: points_on_ray
    real(kind=4), allocatable, dimension(:, :)    :: step
    
    ! external subroutines from external_forward_projection
    external :: rigid_transformation, rigid_transformation_der, ray_forward_der_trilinear
    
    ax(:) = 0._4
    dax(:, :) = 0._4
    
    s = source_points
    d = detector_points
    s(1, :) = s(1, :) + cor_shift(1)
    d(1, :) = d(1, :) + cor_shift(1)
    ! apply rigid body transformation to source and detector points
    call rigid_transformation(s, alpha, beta, phi, xyz(:), n_rays, p0)
    call rigid_transformation(d, alpha, beta, phi, xyz(:), n_rays, p1)
        
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
    allocate( step(n_rays, n_on_ray))
    points_on_ray = 0._4
    step = 0._4
        
    ! march along the rays and get points at regular intervals
    do i = 1, 3
        do j = 1, n_on_ray
            points_on_ray(i, :, j) = p0(i, :) + (j-1)*step_size*r_hat(i)
        end do
    end do
    
    do j = 1, n_on_ray
        step(:, j) = (j-1)*step_size/r_length
    end do
    
    ! compute gradient of source points wrt to angles and translations
    x0 = d(:, 1) - s(:, 1) ! untransformed ray vector
    call rigid_transformation_der(s, x0, alpha, beta, phi, xyz, n_rays, der, append_der)
    
    ! compute forward projection and its derivative wrt angles and translations
    call ray_forward_der_trilinear(points_on_ray, n_rays, n_on_ray, nx, ny, nz, recon, step, der, append_der, ax, dax)
    
    deallocate( points_on_ray )
    deallocate( step )

end subroutine compute_gradient
