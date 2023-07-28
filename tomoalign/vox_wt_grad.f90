subroutine bilinear_vox_interp(n_vox, floor_x, floor_z, alpha_x, alpha_z, rec, ndim_x, ndim_z, der_points, &
                               det_img, grad_det_img)
    
    implicit none
    
    integer(kind=4),                                intent(in)  :: n_vox, ndim_x, ndim_z
    integer(kind=4), dimension(:),                  intent(in)  :: floor_x, floor_z
    real(kind=4), dimension(:),                     intent(in)  :: alpha_x, alpha_z, rec
    real(kind=4), dimension(:,:,:),                 intent(in)  :: der_points
    real(kind=4), dimension(ndim_z, ndim_x),        intent(out) :: det_img
    real(kind=4), dimension(6, ndim_z, ndim_x),     intent(out) :: grad_det_img
    
    integer(kind=4) :: i, fx, fz
    real(kind=4), dimension(6)      :: g0, g2
    real(kind=4), dimension(6, 3)   :: g
    
    ! initialize
    det_img(:, :) = 0._4
    grad_det_img(:, :, :) = 0._4
    
    do i = 1, n_vox
        fx = floor_x(i)+1
        fz = floor_z(i)+1
        g = der_points(:, :, i)
        if (fx >= 1 .and. fx <= ndim_x .and. fz >= 1 .and. fz <= ndim_z) then
            det_img(fz, fx) = det_img(fz, fx) + rec(i) * (1._4-alpha_x(i)) * (1._4-alpha_z(i))
            g0 = g(:, 1) * (1._4 - alpha_z(i)) * rec(i)
            g2 = g(:, 3) * (1._4 - alpha_x(i)) * rec(i)
            grad_det_img(:, fz, fx) = grad_det_img(:, fz, fx) + (g0 + g2)
        end if
        if (fx+1 >= 1 .and. fx+1 <= ndim_x .and. fz >= 1 .and. fz <= ndim_z) then
            det_img(fz, fx+1) = det_img(fz, fx+1) + rec(i) * (alpha_x(i)) * (1._4-alpha_z(i))
            g0 = g(:, 1) * (-1._4 * (1._4 - alpha_z(i))) * rec(i)
            g2 = g(:, 3) * alpha_x(i) * rec(i)
            grad_det_img(:, fz, fx+1) = grad_det_img(:, fz, fx+1) + (g0 + g2)
        end if
        if (fx >= 1 .and. fx <= ndim_x .and. fz+1 >= 1 .and. fz+1 <= ndim_z) then
            det_img(fz+1, fx) = det_img(fz+1, fx) + rec(i) * (1._4-alpha_x(i)) * alpha_z(i)
            g0 = g(:, 1) * alpha_z(i) * rec(i)
            g2 = g(:, 3) * (-1._4 * (1._4 - alpha_x(i))) * rec(i)
            grad_det_img(:, fz+1, fx) = grad_det_img(:, fz+1, fx) + (g0 + g2)
        end if
        if (fx+1 >= 1 .and. fx+1 <= ndim_x .and. fz+1 >= 1 .and. fz+1 <= ndim_z) then
            det_img(fz+1, fx+1) = det_img(fz+1, fx+1) + rec(i) * alpha_x(i) * alpha_z(i)
            g0 = g(:, 1) * (-1._4 * alpha_z(i)) * rec(i)
            g2 = g(:, 3) * (-1._4 * alpha_x(i)) * rec(i)
            grad_det_img(:, fz+1, fx+1) = grad_det_img(:, fz+1, fx+1) + (g0 + g2)
            
        end if
        
    end do
    
    return
    
end subroutine bilinear_vox_interp


subroutine bilinear_sparse(n_vox, floor_x, floor_z, alpha_x, alpha_z, ndim_x, ndim_z, dat_inds, det_inds, wts, n_inds)
    
    implicit none
    
    integer(kind=4),                                intent(in)  :: n_vox, ndim_x, ndim_z
    integer(kind=4), dimension(:),                  intent(in)  :: floor_x, floor_z
    real(kind=4), dimension(:),                     intent(in)  :: alpha_x, alpha_z
    integer(kind=4), dimension(4*n_vox),            intent(out) :: dat_inds, det_inds
    real(kind=4), dimension(4*n_vox),               intent(out) :: wts
    integer(kind=4),                                intent(out) :: n_inds
    
    integer(kind=4) :: i, fx, fz
    
    ! initialize -- set indices to -999 and wts to -999.0
    n_inds = 0
    det_inds(:) = -999
    dat_inds(:) = -999
    wts(:) = -999._4
    
    do i = 1, n_vox
        fx = floor_x(i)+1
        fz = floor_z(i)+1
        if (fx >= 1 .and. fx <= ndim_x .and. fz >= 1 .and. fz <= ndim_z) then
            n_inds = n_inds + 1
            dat_inds(n_inds) = i-1 ! starting from index 0 for python
            det_inds(n_inds) = (fx-1) + ndim_x*(fz-1)
            wts(n_inds) = (1._4-alpha_x(i)) * (1._4-alpha_z(i))
        end if
        
        if (fx+1 >= 1 .and. fx+1 <= ndim_x .and. fz >= 1 .and. fz <= ndim_z) then
            n_inds = n_inds + 1
            dat_inds(n_inds) = i-1
            det_inds(n_inds) = fx + ndim_x*(fz-1)
            wts(n_inds) =  alpha_x(i) * (1._4-alpha_z(i))
        end if
        
        if (fx >= 1 .and. fx <= ndim_x .and. fz+1 >= 1 .and. fz+1 <= ndim_z) then
            n_inds = n_inds + 1
            dat_inds(n_inds) = i-1
            det_inds(n_inds) = (fx-1) + ndim_x*fz
            wts(n_inds) =  (1._4-alpha_x(i)) * alpha_z(i)
        end if
        
        if (fx+1 >= 1 .and. fx+1 <= ndim_x .and. fz+1 >= 1 .and. fz+1 <= ndim_z) then
            n_inds = n_inds + 1
            dat_inds(n_inds) = i-1
            det_inds(n_inds) = fx + ndim_x*fz
            wts(n_inds) = alpha_x(i) * alpha_z(i)
        end if
        
    end do
    
    return
    
end subroutine bilinear_sparse