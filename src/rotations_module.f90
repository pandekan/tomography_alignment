module rotations_module
    
    implicit none
    contains
    
    subroutine rot_z(angle, rot_mat)
        real(kind=4), intent(in) :: angle
        real(kind=4), dimension(3, 3), intent(out) :: rot_mat
        real(kind=4) :: c, s
        
        rot_mat = 0._4
        c = cos(angle)
        s = sin(angle)
        
        rot_mat(1,1) = c
        rot_mat(1,2) = -s
        rot_mat(2,1) = s
        rot_mat(2,2) = c
        rot_mat(3,3) = 1._4
        
    end subroutine rot_z
    
    subroutine rot_x(angle, rot_mat)
        real(kind=4), intent(in) :: angle
        real(kind=4), dimension(3,3), intent(out) :: rot_mat
        real(kind=4) :: c, s
        
        rot_mat = 0._4
        c = cos(angle)
        s = sin(angle)
        
        rot_mat(1,1) = 1._4
        rot_mat(2,2) = c
        rot_mat(2,3) = -s
        rot_mat(3,2) = s
        rot_mat(3,3) = c
        
    end subroutine rot_x
    
    subroutine rot_y(angle, rot_mat)
        real(kind=4), intent(in) :: angle
        real(kind=4), dimension(3,3), intent(out) :: rot_mat
        real(kind=4) :: c, s
        
        rot_mat = 0._4
        c = cos(angle)
        s = sin(angle)
        
        rot_mat(1,1) = c
        rot_mat(1,3) = s
        rot_mat(2,2) = 1._4
        rot_mat(3,1) = -s
        rot_mat(3,3) = c
    end subroutine rot_y
    
    subroutine der_rot_z(angle, der_mat)
        real(kind=4), intent(in) :: angle
        real(kind=4), dimension(3,3), intent(out):: der_mat
        real(kind=4) :: c, s
        
        der_mat = 0._4
        c = cos(angle)
        s = sin(angle)
        
        der_mat(1,1) = -s
        der_mat(1,2) = -c
        der_mat(2,1) = c
        der_mat(2,2) = -s
        
    end subroutine der_rot_z
    
    subroutine der_rot_x(angle, der_mat)
        real(kind=4), intent(in) :: angle
        real(kind=4), dimension(3,3), intent(out):: der_mat
        real(kind=4) :: c, s
        
        der_mat = 0._4
        c = cos(angle)
        s = sin(angle)
        
        der_mat(2,2) = -s
        der_mat(2,3) = -c
        der_mat(3,2) = c
        der_mat(3,3) = -s
        
    end subroutine der_rot_x
    
    subroutine der_rot_y(angle, der_mat)
        real(kind=4), intent(in) :: angle
        real(kind=4), dimension(3,3), intent(out):: der_mat
        real(kind=4) :: c, s
        
        der_mat = 0._4
        c = cos(angle)
        s = sin(angle)
        
        der_mat(1,1) = -s
        der_mat(1,3) = c
        der_mat(3,1) = -c
        der_mat(3,3) = -s
        
    end subroutine der_rot_y
end module rotations_module
