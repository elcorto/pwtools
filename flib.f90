! Fortran extension module to calculate the velocity autocorrelation function.
! See Makefile for compilation, pydos.fvacf() for usage.
!
! Remove "!verbose" below to get some OpenMP debugging infos.
!
! stdout/stderr
! -------------
! write(*,*) or write(6,*) write to stdout, write(0,*) to stderr. This works for
! redirecting text in the shell:
!   ./a.out > foo
!   ./a.out > foo 2>&1
! But it does not work for Fortran extensions. For instance, nose (testing) still
! prints stdout messages from Fortran extensions instead of catching them. AFAIK,
! there is no eays (read: f2py builtin) solution. If you know better, send me an
! email -- or a patch!

#define stdout 6
#define stderr 0

subroutine vacf(v, m, c, method, use_m, nthreads, natoms, nstep)
    
    ! Normalized vacf: c_vv(t) = C_vv(t) / C_vv(0) for atomic velocities stored
    ! in 3d array `v`.
    !
    ! method=1: loops
    ! method=2: vectorized, but makes a copy of `v`
         
#ifdef __OPENMP    
    use omp_lib
    !f2py threadsafe
#endif    

    implicit none
    integer, intent(in) :: natoms, nstep, method, use_m
    integer, intent(in), optional ::  nthreads
    double precision, intent(in) :: v(0:natoms-1, 0:2, 0:nstep-1)
    double precision, intent(in) :: m(0:natoms-1)
    double precision, intent(out) :: c(0:nstep-1)
    character(len=*), parameter :: this='[_flib.so:vacf] '
    integer ::  t, i, j, k
    ! for mass vector stuff in method 2
    double precision :: vv(0:natoms-1, 0:2, 0:nstep-1)
        
    !f2py intent(in, out) c

#ifdef __OPENMP
    ! Check if env vars are recognized. This seems to work (uncomment the lines
    ! starting with `!!`). getenv() is implemented in gfortran and ifort.
    !
    !! character(100) :: OMP_NUM_THREADS
    !! call getenv('OMP_NUM_THREADS', OMP_NUM_THREADS)
    !! write(*,*) 'OMP_NUM_THREADS: ', OMP_NUM_THREADS
    
    ! With f2py, "if (present(nthreads)) then ..." doesn't work for 'optional'
    ! input args. nthreads is alawys present. When it is not
    ! supplied, the if clause *should* evaluate to .false. . Instead, 
    ! nthreads is simply 0. This is not very pretty ...
!verbose    write(stdout,*) this, "nthreads input: ", nthreads
    
    if (nthreads /= 0) then
!verbose        write(stdout,*) this, "setting nthreads to ", nthreads
        call omp_set_num_threads(nthreads)
!verbose    else        
!verbose        write(stdout,*) this, "number of threads controlled by OMP_NUM_THREADS or &
!verbose        number of cores"        
    end if        
!verbose    write(stdout,*) this, "num threads:", omp_get_max_threads()
!verbose#else
!verbose    if (nthreads /= 0) then
!verbose        write(stdout,*) this, "warning: nthreads is ignored, not compiled with &
!verbose        OpenMP support"
!verbose    end if
#endif    

    if (method == 1) then 
        
        ! Code dup b/c of use_m doesn't look like good practice. But this way
        ! we assure that in case use_m==0, `m` can be *any* kind of crap, as
        ! long as it's 1d and length natoms. W/o `use_m`, `m` would have to be 
        ! [1,1,1,...] and we would do usless multiplications by 1.0 in the
        ! inner loop.
        ! The other possibility, an if-condition in the inner loop, seems
        ! stupid.


        if (use_m == 1) then
            !$omp parallel
            !$omp do
            do t = 0,nstep-1
                do j = 0,nstep - t - 1
                    do i = 0,natoms-1
                        ! poor man's unrolled dot
                        c(t) = c(t) + ( v(i,0,j) * v(i,0,j+t)  &
                                    +   v(i,1,j) * v(i,1,j+t)  &
                                    +   v(i,2,j) * v(i,2,j+t) ) * m(i)
                    end do                    
                end do                    
            end do
            !$omp end do
            !$omp end parallel
        else if (use_m == 0) then
            !$omp parallel
            !$omp do
            do t = 0,nstep-1
                do j = 0,nstep - t - 1
                    do i = 0,natoms-1
                        ! poor man's unrolled dot
                        c(t) = c(t) + ( v(i,0,j) * v(i,0,j+t)  &
                                    +   v(i,1,j) * v(i,1,j+t)  &
                                    +   v(i,2,j) * v(i,2,j+t) ) 
                    end do                    
                end do                    
            end do
            !$omp end do
            !$omp end parallel
        else
            write(stderr,*) this, "ERROR: illegal value for 'use_m'"
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
        ! sqrt(m). In the dot product of the velocities
        ! vv(i,j,:) . vv(i,j+t,:), we get m(i) back.
        
        if (use_m == 1) then
            !$omp parallel
            !$omp do
            do j = 0,nstep-1
                do k = 0,2
                    vv(:,k,j) = dsqrt(m) * v(:,k,j)
                end do    
            end do
            !$omp end do
            !$omp end parallel
            call vect_loops(vv, natoms, nstep, c)
        else if (use_m == 0) then        
            call vect_loops(v, natoms, nstep, c)
        else
            write(stderr,*) this, "ERROR: illegal value for 'use_m'"
            return
        end if
        c = c / c(0)
        return
    else        
        write(stderr,*) this, "ERROR: illegal value for 'method'"
        return
    end if
end subroutine vacf


subroutine vect_loops(v, natoms, nstep, c)
#ifdef __OPENMP
    use omp_lib
    !f2py threadsafe
#endif    
    implicit none
    integer :: t, nstep, natoms
    ! v(:, :, :) and v(0:, 0:, 0:) results in c = [NaN, ..., NaN]. 
    ! v(0:, 0:nstep-1, 0:) is not allowed (at least, ifort complains).
    !!double precision, intent(in) :: v(:, :, :)
    double precision, intent(in) :: v(0:natoms-1, 0:2, 0:nstep-1)
    double precision, intent(out) :: c(0:nstep-1)
    !$omp parallel
    !$omp do
    do t = 0,nstep-1
        c(t) = sum(v(:,:,:(nstep-t-1)) * v(:,:,t:))
    end do
    !$omp end do
    !$omp end parallel
end subroutine vect_loops


subroutine acorr(v, c, nstep, method, norm)
    ! (Normalized) 1d-vacf: c_vv(t) = C_vv(t) / C_vv(0)
    ! This is a reference implementation.
    implicit none
    integer :: nstep, t, j, method, norm
    double precision, intent(in) :: v(0:nstep-1)
    double precision, intent(out) :: c(0:nstep-1)
    !f2py intent(in, out) c
    if (method == 1) then 
        do t = 0,nstep-1
            do j = 0,nstep - t - 1
                c(t) = c(t) + v(j) * v(j+t)
            end do                    
        end do                    
    else if (method == 2) then        
        do t = 0,nstep-1
            c(t) = sum(v(:(nstep-t-1)) * v(t:))
        end do
    end if
    if (norm == 1) then
        c = c / c(0)
    end if        
    return
end subroutine acorr

subroutine cdistsq(arrx, arry, dist, nx, ny, ndim)
    implicit none
    integer :: ii, jj, nx, ny, ndim
    double precision :: arrx(nx, ndim), arry(ny, ndim)
    double precision :: dist(nx,ny)
    !f2py intent(in, out) dist
    ! note row-major loop order -> speed!
    !$omp parallel
    !$omp do
    do jj=1,ny
        do ii=1,nx
            dist(ii,jj) = sum((arrx(ii,:) - arry(jj,:))**2.0)
        end do
    end do        
    !$omp end parallel
end subroutine cdistsq
