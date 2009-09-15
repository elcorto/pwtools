! 
! Copyright (c) 2008, Steve Schmerler <mefx@gmx.net>.
! The pydos package. 
! 


subroutine vacf(v, m, c, method, natoms, nstep, use_m)
    
    ! Normalized vacf: c_vv(t) = C_vv(t) / C_vv(0)
    !
    ! method=1: loops
    ! method=2: vectorized, but makes a copy of `v`
         
    
    use omp_lib
    !f2py threadsafe

    implicit none
    integer :: natoms, nstep, t, i, j, k, method, use_m
    double precision, intent(in) :: v(0:natoms-1, 0:nstep-1, 0:2)
    double precision, intent(in) :: m(0:natoms-1)
    double precision, intent(out) :: c(0:nstep-1)
    
    ! for mass vector stuff in method 2
    double precision :: vv(0:natoms-1, 0:nstep-1, 0:2)
        
    !f2py intent(in, out) c
    
    if (method == 1) then 
        
        ! Code dup b/c of use_m doesn't look like good practice. But this way
        ! we assure that in case use_m==0, `m` can be *any* kind of crap, as
        ! long as it's 1d and length natoms. W/o `use_m`, `m` would have to be 
        ! [1,1,1,...] and we would do usless multiplications by 1.0 in the
        ! inner loop.
        ! The other possibility, an if-condition in the inner loop, seems
        ! stupid.


        if (use_m == 1) then
            !$omp parallel num_threads(4)
            !$omp do
            do t = 0,nstep-1
                do j = 0,nstep - t - 1
                    do i = 0,natoms-1
                        ! poor man's unrolled dot
                        c(t) = c(t) + ( v(i,j,0) * v(i,j+t,0)  &
                                    +   v(i,j,1) * v(i,j+t,1)  &
                                    +   v(i,j,2) * v(i,j+t,2) ) * m(i)
                    end do                    
                end do                    
            end do
            !$omp end do
            !$omp end parallel
        else if (use_m == 0) then
            !$omp parallel num_threads(4)
            !$omp do
            do t = 0,nstep-1
                do j = 0,nstep - t - 1
                    do i = 0,natoms-1
                        ! poor man's unrolled dot
                        c(t) = c(t) + ( v(i,j,0) * v(i,j+t,0)  &
                                    +   v(i,j,1) * v(i,j+t,1)  &
                                    +   v(i,j,2) * v(i,j+t,2) )
                    end do                    
                end do                    
            end do
            !$omp end do
            !$omp end parallel
        else
            call error("_flib.so, vacf", "illegal value for 'use_m'")
            return
        end if
        c = c / c(0)
        return
    
    ! Vectorzied version of the loops.
    else if (method == 2) then        
        
        ! Slightly faster b/c of vectorization, but makes a copy of `v`. Use
        ! only if you have enough memory.  Don't know how to implement numpy's
        ! newaxis+broadcast magic into Fortran, but numpy will also make temp
        ! copies, for sure. 
        ! We need a copy of `v` b/c `v` is "intent(in) v" and Fortran does not
        ! allow manipulation of input args. Moreover, we don't want to modify
        ! `v` directly should it become "intent(in,out)" one day.
        !
        ! Clever multiplication with the mass vector:
        ! Element-wise multiply each vector of length `natoms` vv(:,j,k) with
        ! the sqrt(m). In the dot product of the velocities
        ! vv(i,j,:) . vv(i,j+t,:), we get m(i) back.
        
        if (use_m == 1) then
            !$omp parallel num_threads(4)
            !$omp do
            do j = 0,nstep-1
                do k = 0,2
                    vv(:,j,k) = dsqrt(m) * v(:,j,k)
                end do    
            end do
            !$omp end do
            !$omp end parallel
            call vect_loops(vv, natoms, nstep, c)
        else if (use_m == 0) then        
            call vect_loops(v, natoms, nstep, c)
        else
            call error("_flib.so, vacf", "illegal value for 'use_m'")
            return
        end if
        c = c / c(0)
        return
    else        
        call error("_flib.so, vacf", "illegal value for 'method'")
        return
    end if
end subroutine vacf

!-----------------------------------------------------------------------------

subroutine vect_loops(v, natoms, nstep, c)
    
    use omp_lib
    implicit none
    
    !f2py threadsafe
    
    integer :: t, nstep, natoms
    ! v(:, :, :) and v(0:, 0:, 0:) results in c = [NaN, ..., NaN]. 
    ! v(0:, 0:nstep-1, 0:) is not allowed (at least, ifort complains).
!!    double precision, intent(in) :: v(:, :, :)
    double precision, intent(in) :: v(0:natoms-1, 0:nstep-1, 0:2)
    double precision, intent(out) :: c(0:nstep-1)
    
    !$omp parallel num_threads(4)
    !$omp do
    do t = 0,nstep-1
        c(t) = sum(v(:,:(nstep-t-1),:) * v(:,t:,:))
    end do
    !$omp end do
    !$omp end parallel

end subroutine vect_loops

!-----------------------------------------------------------------------------

subroutine error(what, msg)
    
    implicit none
    character(len=*) :: msg, what
    
    write(unit=0,fmt=*) "******* [", what, "] ERROR: ", msg    

end subroutine error

!-----------------------------------------------------------------------------

subroutine acorr(v, c, nstep, method)
    
    ! Normalized vacf: c_vv(t) = C_vv(t) / C_vv(0)
    
    implicit none
    integer :: nstep, t, j, method
    double precision, intent(in) :: v(0:nstep-1)
    double precision, intent(out) :: c(0:nstep-1)
    
    !f2py intent(in, out) c

    if (method == 1) then 
       
        do t = 0,nstep-1
            do j = 0,nstep - t - 1
                c(t) = c(t) + v(j) * v(j+t)
            end do                    
        end do                    
        c = c / c(0)
        return
    
    else if (method == 2) then        
        
        do t = 0,nstep-1
            c(t) = sum(v(:(nstep-t-1)) * v(t:))
        end do
        c = c / c(0)
        return
    end if

end subroutine acorr

