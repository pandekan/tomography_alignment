subroutine trilinear_ray_sparse(floor_points, w_floor, nx, ny, nz, n_rays, n_points, dat_inds, det_inds, wts, n_inds)
    
    implicit none
    
    integer(kind=4), dimension(:, :, :),           intent(in)  :: floor_points
    real(kind=8),    dimension(:, :, :),           intent(in)  :: w_floor
    integer(kind=4),                               intent(in)  :: nx, ny, nz, n_rays, n_points
    integer(kind=4), dimension(8*n_rays*n_points), intent(out) :: dat_inds, det_inds
    real(kind=8),    dimension(8*n_rays*n_points), intent(out) :: wts
    integer(kind=4),                               intent(out) :: n_inds
    integer(kind=4) :: r, p
    integer(kind=4) :: fx, fy, fz, cx, cy, cz
    real(kind=8)    :: wt_fx, wt_fy, wt_fz, wt_cx, wt_cy, wt_cz
    
    dat_inds(:) = -999
    det_inds(:) = -999
    wts(:) = -999._8
    n_inds = 0
    
    do r = 1, n_rays
        do p = 1, n_points
            fx = floor_points(1, r, p) + 1
            fy = floor_points(2, r, p) + 1
            fz = floor_points(3, r, p) + 1
            cx = fx + 1
            cy = fy + 1
            cz = fz + 1
            wt_fx = w_floor(1, r, p)
            wt_fy = w_floor(2, r, p)
            wt_fz = w_floor(3, r, p)
            wt_cx = 1._8 - wt_fx
            wt_cy = 1._8 - wt_fy
            wt_cz = 1._8 - wt_fz
            
            if (fx >= 1 .and. fx <= nx .and. fy >= 1 .and. fy <= ny .and. fz >=1 .and. fz <= nz) then
                n_inds = n_inds + 1
                det_inds(n_inds) = r-1
                dat_inds(n_inds) = (fx-1)*ny*nz + (fy-1)*nz + fz-1
                wts(n_inds) = wt_fx * wt_fy * wt_fz
            end if
            
            if (fx >= 1 .and. fx <= nx .and. fy >= 1 .and. fy <= ny .and. cz >= 1 .and. cz <= nz) then
                n_inds = n_inds + 1
                det_inds(n_inds) = r-1
                dat_inds(n_inds) = (fx-1)*ny*nz + (fy-1)*nz + cz-1
                wts(n_inds) = wt_fx * wt_fy * wt_cz
            end if
            
            if (fx >= 1 .and. fx <= nx .and. cy >= 1 .and. cy <= ny .and. fz >= 1 .and. fz <= nz) then
                n_inds = n_inds + 1
                det_inds(n_inds) = r-1
                dat_inds(n_inds) = (fx-1)*ny*nz + (cy-1)*nz + fz-1
                wts(n_inds) = wt_fx * wt_cy * wt_fz
            end if
            
            if (fx >= 1 .and. fx <= nx .and. cy >= 1 .and. cy <= ny .and. cz >= 1 .and. cz <= nz) then
                n_inds = n_inds + 1
                det_inds(n_inds) = r-1
                dat_inds(n_inds) = (fx-1)*ny*nz + (cy-1)*nz + cz-1
                wts(n_inds) = wt_fx * wt_cy * wt_cz
            end if
            
            if (cx >= 1 .and. cx <= nx .and. fy >= 1 .and. fy <= ny .and. fz >=1 .and. fz <= nz) then
                n_inds = n_inds + 1
                det_inds(n_inds) = r-1
                dat_inds(n_inds) = (cx-1)*ny*nz + (fy-1)*nz + fz-1
                wts(n_inds) = wt_cx * wt_fy * wt_fz
            end if
            
            if (cx >= 1 .and. cx <= nx .and. fy >= 1 .and. fy <= ny .and. cz >= 1 .and. cz <= nz) then
                n_inds = n_inds + 1
                det_inds(n_inds) = r-1
                dat_inds(n_inds) = (cx-1)*ny*nz + (fy-1)*nz + cz-1
                wts(n_inds) = wt_cx * wt_fy * wt_cz
            end if
            
            if (cx >= 1 .and. cx <= nx .and. cy >= 1 .and. cy <= ny .and. fz >= 1 .and. fz <= nz) then
                n_inds = n_inds + 1
                det_inds(n_inds) = r-1
                dat_inds(n_inds) = (cx-1)*ny*nz + (cy-1)*nz + fz-1
                wts(n_inds) = wt_cx * wt_cy * wt_fz
            end if
            
            if (cx >= 1 .and. cx <= nx .and. cy >= 1 .and. cy <= ny .and. cz >= 1 .and. cz <= nz) then
                n_inds = n_inds + 1
                det_inds(n_inds) = r-1
                dat_inds(n_inds) = (cx-1)*ny*nz + (cy-1)*nz + cz-1
                wts(n_inds) = wt_cx * wt_cy * wt_cz
            end if
        end do
    end do
end subroutine trilinear_ray_sparse


subroutine trilinear_ray_interp(floor_points, w_floor, nx, ny, nz, n_rays, n_points, recon, &
                                step, der, det_img, grad_det_img)
    
    implicit none
    
    integer(kind=4), dimension(:, :, :),           intent(in)  :: floor_points
    real(kind=8),    dimension(:, :, :),           intent(in)  :: w_floor
    integer(kind=4),                               intent(in)  :: nx, ny, nz, n_rays, n_points
    real(kind=8),    dimension(:),                 intent(in)  :: recon
    real(kind=8),    dimension(:, :),              intent(in)  :: step
    real(kind=8),    dimension(:, :, :),           intent(in)  :: der
    real(kind=8),    dimension(n_rays),            intent(out) :: det_img
    real(kind=8),    dimension(6, n_rays),         intent(out) :: grad_det_img
    
    integer(kind=4) :: r, p
    integer(kind=4) :: fx, fy, fz, cx, cy, cz
    integer(kind=4) :: dat_ind
    real(kind=8)    :: wt, wt_fx, wt_fy, wt_fz, wt_cx, wt_cy, wt_cz
    
    real(kind=8), dimension(6, 3) :: g
    real(kind=8), dimension(6)    :: g1, g2, g3
    real(kind=8), dimension(9, 3) :: g_temp
    
    det_img(:) = 0._8
    grad_det_img(:, :) = 0._8
    
    do r = 1, n_rays
        do p = 1, n_points
            fx = floor_points(1, r, p) + 1
            fy = floor_points(2, r, p) + 1
            fz = floor_points(3, r, p) + 1
            cx = fx + 1
            cy = fy + 1
            cz = fz + 1
            wt_fx = w_floor(1, r, p)
            wt_fy = w_floor(2, r, p)
            wt_fz = w_floor(3, r, p)
            wt_cx = 1._8 - wt_fx
            wt_cy = 1._8 - wt_fy
            wt_cz = 1._8 - wt_fz
            
            g_temp = der(:, :, r)
            g(:, :) = 0._8
            g = g_temp(:6, :)
            g(4, :) = g(4, :) + step(r, p) * g_temp(7, :)
            g(5, :) = g(5, :) + step(r, p) * g_temp(8, :)
            g(6, :) = g(6, :) + step(r, p) * g_temp(9, :)
            if (fx >= 1 .and. fx <= nx .and. fy >= 1 .and. fy <= ny .and. fz >=1 .and. fz <= nz) then
                dat_ind = (fx-1)*ny*nz + (fy-1)*nz + fz-1 + 1
                wt = wt_fx * wt_fy * wt_fz
                det_img(r) = det_img(r) + recon(dat_ind)*wt
                g1(:) = -wt_fy * wt_fz * recon(dat_ind)*g(:, 1)
                g2(:) = -wt_fx * wt_fz * recon(dat_ind)*g(:, 2)
                g3(:) = -wt_fx * wt_fy * recon(dat_ind)*g(:, 3)
                grad_det_img(:, r) = grad_det_img(:, r) + (g1 + g2 + g3)
            end if
            
            if (fx >= 1 .and. fx <= nx .and. fy >= 1 .and. fy <= ny .and. cz >= 1 .and. cz <= nz) then
                dat_ind = (fx-1)*ny*nz + (fy-1)*nz + cz-1 + 1
                wt = wt_fx * wt_fy * wt_cz
                det_img(r) = det_img(r) + recon(dat_ind)*wt
                g1(:) = -wt_fy * wt_cz * recon(dat_ind)*g(:, 1)
                g2(:) = -wt_fx * wt_cz * recon(dat_ind)*g(:, 2)
                g3(:) = wt_fx * wt_fy * recon(dat_ind)*g(:, 3)
                grad_det_img(:, r) = grad_det_img(:, r) + (g1 + g2 + g3)
            end if
            
            if (fx >= 1 .and. fx <= nx .and. cy >= 1 .and. cy <= ny .and. fz >= 1 .and. fz <= nz) then
                dat_ind = (fx-1)*ny*nz + (cy-1)*nz + fz-1 + 1
                wt = wt_fx * wt_cy * wt_fz
                det_img(r) = det_img(r) + recon(dat_ind)*wt
                g1(:) = -wt_cy * wt_fz * recon(dat_ind)*g(:, 1)
                g2(:) = wt_fx * wt_fz * recon(dat_ind)*g(:, 2)
                g3(:) = -wt_fx * wt_cy * recon(dat_ind)*g(:, 3)
                grad_det_img(:, r) = grad_det_img(:, r) + (g1 + g2 + g3)
            end if
            
            if (fx >= 1 .and. fx <= nx .and. cy >= 1 .and. cy <= ny .and. cz >= 1 .and. cz <= nz) then
                dat_ind = (fx-1)*ny*nz + (cy-1)*nz + cz-1 + 1
                wt = wt_fx * wt_cy * wt_cz
                det_img(r) = det_img(r) + recon(dat_ind)*wt
                g1(:) = -wt_cy * wt_cz * recon(dat_ind)*g(:, 1)
                g2(:) = wt_fx * wt_cz * recon(dat_ind)*g(:, 2)
                g3(:) = wt_fx * wt_cy * recon(dat_ind)*g(:, 3)
                grad_det_img(:, r) = grad_det_img(:, r) + (g1 + g2 + g3)
            end if
            
            if (cx >= 1 .and. cx <= nx .and. fy >= 1 .and. fy <= ny .and. fz >=1 .and. fz <= nz) then
                dat_ind = (cx-1)*ny*nz + (fy-1)*nz + fz-1 + 1
                wt = wt_cx * wt_fy * wt_fz
                det_img(r) = det_img(r) + recon(dat_ind) * wt
                g1(:) = wt_fy * wt_fz * recon(dat_ind) * g(:, 1)
                g2(:) = - wt_cx * wt_fz * recon(dat_ind)*g(:, 2)
                g3(:) = -wt_cx * wt_fy * recon(dat_ind)*g(:, 3)
                grad_det_img(:, r) = grad_det_img(:, r) + (g1 + g2 + g3)
            end if
            
            if (cx >= 1 .and. cx <= nx .and. fy >= 1 .and. fy <= ny .and. cz >= 1 .and. cz <= nz) then
                dat_ind = (cx-1)*ny*nz + (fy-1)*nz + cz-1 + 1
                wt = wt_cx * wt_fy * wt_cz
                det_img(r) = det_img(r) + recon(dat_ind) * wt
                g1(:) = wt_fy * wt_cz * recon(dat_ind) * g(:, 1)
                g2(:) = -wt_cx * wt_cz * recon(dat_ind) * g(:, 2)
                g3(:) = wt_cx * wt_fy * recon(dat_ind) * g(:, 3)
                grad_det_img(:, r) = grad_det_img(:, r) + (g1 + g2 + g3)
            end if
            
            if (cx >= 1 .and. cx <= nx .and. cy >= 1 .and. cy <= ny .and. fz >= 1 .and. fz <= nz) then
                dat_ind = (cx-1)*ny*nz + (cy-1)*nz + fz-1 + 1
                wt = wt_cx * wt_cy * wt_fz
                det_img(r) = det_img(r) + recon(dat_ind)*wt
                g1(:) = wt_cy * wt_fz * recon(dat_ind) * g(:, 1)
                g2(:) = wt_cx * wt_fz * recon(dat_ind) * g(:, 2)
                g3(:) = -wt_cx * wt_cy * recon(dat_ind) * g(:, 3)
                grad_det_img(:, r) = grad_det_img(:, r) + (g1 + g2 + g3)
            end if
            
            if (cx >= 1 .and. cx <= nx .and. cy >= 1 .and. cy <= ny .and. cz >= 1 .and. cz <= nz) then
                dat_ind = (cx-1)*ny*nz + (cy-1)*nz + cz-1 + 1
                wt = wt_cx * wt_cy * wt_cz
                det_img(r) = det_img(r) + recon(dat_ind)*wt
                g1(:) = wt_cy * wt_cz * recon(dat_ind) * g(:, 1)
                g2(:) = wt_cx * wt_cz * recon(dat_ind) * g(:, 2)
                g3(:) = wt_cx * wt_cy * recon(dat_ind) * g(:, 3)
                grad_det_img(:, r) = grad_det_img(:, r) + (g1 + g2 + g3)
            end if
        end do
    end do
end subroutine trilinear_ray_interp
