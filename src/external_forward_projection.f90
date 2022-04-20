subroutine rigid_transformation(x, a, b, p, t, npoints, xp)
    
    use rotations_module
    
    implicit none
    integer(kind=4),                     intent(in)  :: npoints
    real(kind=4), dimension(3, npoints), intent(in)  :: x ! (3, np)
    real(kind=4),                        intent(in)  :: a, b, p
    real(kind=4), dimension(3),          intent(in)  :: t
    real(kind=4), dimension(3, npoints), intent(out) :: xp
    
    real(kind=4), dimension(3, 3) :: rx, ry, rz, rzx
    real(kind=4), dimension(3, npoints) :: tp
    integer(kind=4) :: i
    
    call rot_z(p, rz)
    call rot_x(a, rx)
    call rot_y(b, ry)
    
    rzx = matmul(rz, rx)
    do i = 1, 3
        tp(i, :) = t(i)
    end do
    
    xp = matmul(ry, x) + tp
    xp = matmul(rzx, xp)
    
end subroutine rigid_transformation

subroutine rigid_transformation_der(x, x0, a, b, p, t, npoints, der, append_der)
    
    use rotations_module
    implicit none
    integer(kind=4),                        intent(in)  :: npoints
    real(kind=4), dimension(3, npoints),    intent(in)  :: x
    real(kind=4), dimension(3),             intent(in)  :: x0
    real(kind=4),                           intent(in)  :: a, b, p
    real(kind=4), dimension(3),             intent(in)  :: t
    real(kind=4), dimension(6, 3, npoints), intent(out) :: der
    real(kind=4), dimension(3, 3),          intent(out) :: append_der
    real(kind=4), dimension(3, 3) :: rx, ry, rz, drx, dry, drz, r_zx, r_xy
    real(kind=4), dimension(3, npoints) :: tp, ry_st
    integer(kind=4) :: i
    
    call rot_z(p, rz)
    call rot_x(a, rx)
    call rot_y(b, ry)
    call der_rot_z(p, drz)
    call der_rot_x(a, drx)
    call der_rot_y(b, dry)
    
    r_zx = matmul(rz, rx)
    r_xy = matmul(rx, ry)
    
    ! derivatives wrt to (tx, ty, tz)
    do i = 1, 3
        tp(i, :) = t(i)
        der(1, i, :) = r_zx(i, 1)
        der(2, i, :) = r_zx(i, 2)
        der(3, i, :) = r_zx(i, 3)
    end do

    ry_st = matmul(ry, x) + tp
    der(4, :, :) = matmul(drz, matmul(rx, ry_st))
    der(5, :, :) = matmul(rz, matmul(drx, ry_st))
    der(6, :, :) = matmul(r_zx, matmul(dry, x))
    append_der(1, :) = matmul(drz, matmul(r_xy, x0))
    append_der(2, :) = matmul(rz, matmul(drx, matmul(ry, x0)))
    append_der(3, :) = matmul(r_zx, matmul(dry, x0))
    
end subroutine rigid_transformation_der

subroutine ray_forward_trilinear(points_on_ray, n_rays, n_points, recon, nx, ny, nz, det_img)
    
    implicit none

    integer(kind=4),                                 intent(in)  :: n_rays, n_points, nx, ny, nz
    real(kind=4),    dimension(3, n_rays, n_points), intent(in)  :: points_on_ray
    real(kind=4),    dimension(nx*ny*nz),            intent(in)  :: recon
    real(kind=4),    dimension(n_rays),              intent(out) :: det_img

    integer(kind=4), dimension(3, n_rays, n_points)  :: floor_points
    real(kind=4),    dimension(3, n_rays, n_points)  :: w_floor
    integer(kind=4) :: r, p
    integer(kind=4) :: fx, fy, fz, cx, cy, cz
    integer(kind=4) :: dat_ind
    real(kind=4)    :: wt, wt_fx, wt_fy, wt_fz, wt_cx, wt_cy, wt_cz
    
    det_img(:) = 0._4
    floor_points = floor(points_on_ray)
    w_floor = 1._4 - (points_on_ray - real(floor_points))
    
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
            wt_cx = 1._4 - wt_fx
            wt_cy = 1._4 - wt_fy
            wt_cz = 1._4 - wt_fz
            
            if (fx >= 1 .and. fx <= nx .and. fy >= 1 .and. fy <= ny .and. fz >=1 .and. fz <= nz) then
                dat_ind = (fx-1)*ny*nz + (fy-1)*nz + fz-1 + 1
                wt = wt_fx * wt_fy * wt_fz
                det_img(r) = det_img(r) + recon(dat_ind)*wt
            end if
            
            if (fx >= 1 .and. fx <= nx .and. fy >= 1 .and. fy <= ny .and. cz >= 1 .and. cz <= nz) then
                dat_ind = (fx-1)*ny*nz + (fy-1)*nz + cz-1 + 1
                wt = wt_fx * wt_fy * wt_cz
                det_img(r) = det_img(r) + recon(dat_ind)*wt
            end if
            
            if (fx >= 1 .and. fx <= nx .and. cy >= 1 .and. cy <= ny .and. fz >= 1 .and. fz <= nz) then
                dat_ind = (fx-1)*ny*nz + (cy-1)*nz + fz-1 + 1
                wt = wt_fx * wt_cy * wt_fz
                det_img(r) = det_img(r) + recon(dat_ind)*wt
            end if
            
            if (fx >= 1 .and. fx <= nx .and. cy >= 1 .and. cy <= ny .and. cz >= 1 .and. cz <= nz) then
                dat_ind = (fx-1)*ny*nz + (cy-1)*nz + cz-1 + 1
                wt = wt_fx * wt_cy * wt_cz
                det_img(r) = det_img(r) + recon(dat_ind)*wt
            end if
            
            if (cx >= 1 .and. cx <= nx .and. fy >= 1 .and. fy <= ny .and. fz >=1 .and. fz <= nz) then
                dat_ind = (cx-1)*ny*nz + (fy-1)*nz + fz-1 + 1
                wt = wt_cx * wt_fy * wt_fz
                det_img(r) = det_img(r) + recon(dat_ind) * wt
            end if
            
            if (cx >= 1 .and. cx <= nx .and. fy >= 1 .and. fy <= ny .and. cz >= 1 .and. cz <= nz) then
                dat_ind = (cx-1)*ny*nz + (fy-1)*nz + cz-1 + 1
                wt = wt_cx * wt_fy * wt_cz
                det_img(r) = det_img(r) + recon(dat_ind) * wt
            end if
            
            if (cx >= 1 .and. cx <= nx .and. cy >= 1 .and. cy <= ny .and. fz >= 1 .and. fz <= nz) then
                dat_ind = (cx-1)*ny*nz + (cy-1)*nz + fz-1 + 1
                wt = wt_cx * wt_cy * wt_fz
                det_img(r) = det_img(r) + recon(dat_ind)*wt
            end if
            
            if (cx >= 1 .and. cx <= nx .and. cy >= 1 .and. cy <= ny .and. cz >= 1 .and. cz <= nz) then
                dat_ind = (cx-1)*ny*nz + (cy-1)*nz + cz-1 + 1
                wt = wt_cx * wt_cy * wt_cz
                det_img(r) = det_img(r) + recon(dat_ind)*wt
            end if
        end do
    end do
    
end subroutine ray_forward_trilinear

subroutine ray_forward_der_trilinear(points_on_ray, n_rays, n_points, nx, ny, nz, recon, &
                                step, der, append_der, det_img, grad_det_img)
    
    implicit none
    
    integer(kind=4),                                 intent(in)  :: n_rays, n_points, nx, ny, nz
    real(kind=4),    dimension(3, n_rays, n_points), intent(in)  :: points_on_ray
    real(kind=4),    dimension(nx*ny*nz),            intent(in)  :: recon
    real(kind=4),    dimension(n_rays, n_points),    intent(in)  :: step
    real(kind=4),    dimension(6, 3, n_rays),        intent(in)  :: der
    real(kind=4),    dimension(3, 3),                intent(in)  :: append_der
    real(kind=4),    dimension(n_rays),              intent(out) :: det_img
    real(kind=4),    dimension(6, n_rays),           intent(out) :: grad_det_img
    
    integer(kind=4), dimension(3, n_rays, n_points)  :: floor_points
    real(kind=4),    dimension(3, n_rays, n_points)  :: w_floor
    integer(kind=4) :: r, p
    integer(kind=4) :: fx, fy, fz, cx, cy, cz
    integer(kind=4) :: dat_ind
    real(kind=4)    :: wt, wt_fx, wt_fy, wt_fz, wt_cx, wt_cy, wt_cz
    
    real(kind=4), dimension(6, 3) :: g
    real(kind=4), dimension(6)    :: g1, g2, g3
    
    floor_points = floor(points_on_ray)
    w_floor = 1._4 - (points_on_ray - real(floor_points))
    det_img(:) = 0._4
    grad_det_img(:, :) = 0._4
    g = 0._4
    
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
            wt_cx = 1._4 - wt_fx
            wt_cy = 1._4 - wt_fy
            wt_cz = 1._4 - wt_fz
            
            g(:, :) = der(:, :, r)
            g(4, :) = g(4, :) + step(r, p) * append_der(1, :)
            g(5, :) = g(5, :) + step(r, p) * append_der(2, :)
            g(6, :) = g(6, :) + step(r, p) * append_der(3, :)
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
end subroutine ray_forward_der_trilinear