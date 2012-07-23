!vim:comments+=fb:!
!
! Fortran extension module. Here we code speed-critical stuff.
!
! Compilation / usage
! -------------------
! See Makefile for compilation, pydos.fvacf() and flib_wrap.py usage.
!
! You may use the f2py Python wrapper function directly as _flib.<function>(...)
! or the top-level wrapper, e.g. in flib_wrap.py if there is one.
!
! To see the function signature of the f2py wrapper function, do:
!   $ python -c 'from pwtools import _flib; print _flib.<function>.__doc__'
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
!
! f2py notes
! ----------
! Doc:
! [1] http://cens.ioc.ee/projects/f2py2e/usersguide/index.html
! [2] http://www.sagemath.org/doc/numerical_sage/f2py.html
! 
! A subroutine needs to get input and pre-allocated result arrays. All args are
! "!f2py intent(in)" by default.
!
! All variables declared as "intent(out) foo" will be returned in a tuple (if
! there is more then one). Note that there must be one "intent(out)" line for
! *each* arg. Also, the order in the tuple is the one of the input args.
!
! subroutine foo(a, b, c, d, n)
!     implicit none
!     integer :: n
!     double precision :: a(n,n), b(n,n), c(n,n), d(n,n)
!     !f2py intent(out) :: c
!     !f2py intent(out) :: d
!     c = matmul(a,b)
!     d = matmul(a,b)*2.0d0
! end subroutine foo
!
! There are basically two ways of wrapping a subroutine: 
!
! (1) the python wrapper doesn't pre-allocate and pass result arrays. In this
! case, the signature would look like this:: 
!      c,d = foo(a,b,[n]) 
! This is the most easy and pythonic way.
!
! (2) Explicitely allocate result arrays on python side and pass in. Change
! the f2py statement to "!f2py intent(in,out)". Then::  
!     c,d = foo(a,b,c,d[n])
! In that case, there are two usage pattern in Python: one can call the wrapper
! with or without return args b/c the result arrays get overwritten in-place.
!     c = np.empty(..., order='F')
!     d = np.empty(..., order='F')
!     _flib(a,b,c,d)        # variant 1  
!     c,d = _flib(a,b,c,d)  # varaint 2
!
! By default, the f2py wrapper will make a copy of each numpy array which
!   * has rank >= 2
!   * order='C'
! This can be a real bottleneck for big arrays, sometimes performing much slower
! than the actual calculations! Use f2py -DF2PY_REPORT_ON_ARRAY_COPY=1 ... if
! you really want to know when.
!
! According to [2], the best way to avoid array copies between Python and
! Fortran is to use method (2) and allocate an array in Python in F-order (e.g.
! np.empty(<shape>, order='F')). Pass that as argument and use e.g. "intent
! (in,out)" or "intent (in,out,overwrite)" for arrays where results are written
! to. The latter case will add an arg ``overwrite_foo`` which defaults to 1
! (True).

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

subroutine distsq(arrx, arry, dsq, nx, ny, ndim)
#ifdef __OPENMP    
    !f2py threadsafe
#endif    
    implicit none
    integer :: ii, jj, nx, ny, ndim
    double precision :: arrx(nx, ndim), arry(ny, ndim), dsq(nx,ny) 
    !f2py intent(in, out) dsq
    ! note row-major loop order -> speed!
    !$omp parallel
    !$omp do
    do jj=1,ny
        do ii=1,nx
            dsq(ii,jj) = sum((arrx(ii,:) - arry(jj,:))**2.0d0)
        end do
    end do        
    !$omp end parallel
end subroutine distsq

subroutine angles_from_idx(distvecs, dists, angleidx, deg, angles, natoms, nang)
    ! Angles requested in `angleidx`.
    !
    ! Parameters
    ! ----------
    ! distvecs : 3d array w/ cartesian distance vectors
    ! dists : 2d array with distances: dists(i,j) = norm(distvecs(i,j,:))
    ! angleidx : 2d array (nang,3)
    !   index array into `angles`, each angleidx(i,:) = (/ii,jj,kk/) are the
    !   indices of atoms forming an angle angles(i), where `ii` is the central
    !   atom, note that you need to use `angles + 1` if you pass a numpy array
    ! deg : {0,1}
    !   whether to return angles in degrees (1) or cosine (0)
    ! angles : dummy, result array
    ! natoms : dummy, number of atoms
    ! nang : dummy, number of angles = angles.shape[0]
    !
    ! Returns
    ! -------
    ! angles : 1d array (nang,)
    implicit none
    integer :: ii, jj, kk, idx, nang, natoms, deg
    integer :: angleidx(nang, 3)
    double precision :: distvecs(natoms, natoms, 3)
    double precision :: dists(natoms, natoms)
    double precision, intent(out) :: angles(nang)
    double precision, parameter :: pi=acos(-1.0d0)
    double precision, parameter :: eps=2.2d-16, ceps=1.0d0 - eps
    !f2py intent(out) angles
    do idx=1,nang
        ii = angleidx(idx,1)
        jj = angleidx(idx,2)
        kk = angleidx(idx,3)
        angles(idx) = dot_product(distvecs(ii,jj,:), distvecs(ii,kk,:)) &
                   / dists(ii,jj) / dists(ii,kk)
    end do
    if (deg==1) then
        ! handle numerical corner cases where acos() can return NaN,
        ! note that eps is hardcoded currently
        where (angles > ceps) angles = 1.0d0
        where (angles < -ceps) angles = -1.0d0
        angles = acos(angles) * 180.0d0 / pi
    end if
end subroutine angles_from_idx

subroutine angles_from_loop(distvecs, dists, mask_val, deg, anglesijk, angles, angleidx, natoms)
    ! All angles.
    !
    ! Appart from 1d array `angles` with *all* angles, we also return a 3d array
    ! with angles for easy indexing anglesijk[ii,jj,kk]. Angles in deg are
    ! acos(anglesijk[ii,jj,kk])*180/pi . Not all array entries are used. Entries
    ! where ii==jj or ii==kk or jj==kk are set to fill value `mask_val`.
    !
    ! The Python-returned `anglesijk` array is in C-order. Not sure if the f2py
    ! wrapper allocates `anglesijk` in F-order and then returns a C-order copy.
    !
    ! Parameters
    ! ----------
    ! distvecs : 3d array w/ cartesian distance vectors
    ! dists : 2d array with distances: dists(i,j) = norm(distvecs(i,j,:))
    ! mask_val : float
    !   Fill value for anglesijk(ii,jj,kk) where ii==jj or ii==kk or
    !   jj==kk, i.e. no angle defined. Note that if you want to calculate
    !   acos(anglesijk) later (i.e. `anglesijk` holds cosine values), then
    !   `mask_val` must be in [0,1].
    ! deg : int, {0,1}
    !   whether to return angles in degrees (1) or cosine values (0)
    ! anglesijk : dummy result array
    ! angles : dummy result array
    ! angleidx : dummy result array
    ! natoms : int, dummy, number of atoms
    !
    ! Returns
    ! -------
    ! anglesijk : 3d array (natoms,)*3
    ! angles : 1d array (nang,), nang = natoms*(natoms-1)*(natoms-2)
    ! angleidx : 2d array (nang,3)
    !   index array into `anglesijk` and `angles`, each angleidx(i,:) = (/ii,jj,kk/) are the
    !   indices of atoms forming an angle, where `ii` is the central one
    !
    implicit none
    integer :: ii, jj, kk, idx, natoms, deg
    double precision :: distvecs(natoms, natoms, 3)
    double precision :: dists(natoms, natoms)
    double precision :: mask_val, cang
    integer, intent(out) :: angleidx(natoms*(natoms-1)*(natoms-2), 3)
    double precision, intent(out) :: anglesijk(natoms,natoms,natoms)
    double precision, intent(out) :: angles(natoms*(natoms-1)*(natoms-2))
    double precision, parameter :: pi=acos(-1.0d0)
    double precision, parameter :: eps=2.2d-16, ceps=1.0d0 - eps
    !f2py intent(out) anglesijk
    !f2py intent(out) angles
    !f2py intent(out) angleidx
    idx = 1
    do ii=1,natoms
        do jj=1,natoms
            do kk=1,natoms
                if (ii /= jj .and. ii /= kk .and. jj /= kk) then
                    cang = dot_product(distvecs(ii,jj,:), distvecs(ii,kk,:)) &
                        / dists(ii,jj) / dists(ii,kk)
                    !!if (cang > ceps) then
                    !!    cang = 1.0d0
                    !!else if (cang < -ceps) then
                    !!    cang = -1.0d0
                    !!end if    
                    anglesijk(ii,jj,kk) = cang
                    angles(idx) = cang
                    angleidx(idx,:) = (/ ii,jj,kk /)
                    idx = idx + 1
                else
                    anglesijk(ii,jj,kk) = mask_val
                end if                        
            end do                        
        end do                        
    end do
    if (deg==1) then
        ! handle numerical corner cases where acos() can return NaN, 
        ! note that eps is hardcoded currently
        where (anglesijk /= mask_val .and. anglesijk > ceps) anglesijk = 1.0d0
        where (anglesijk /= mask_val .and. anglesijk < -ceps) anglesijk = -1.0d0
        where (angles > ceps) angles = 1.0d0
        where (angles < -ceps) angles = -1.0d0
        anglesijk = acos(anglesijk) * 180.0d0 / pi
        angles = acos(angles) * 180.0d0 / pi
    end if
end subroutine angles_from_loop

subroutine distsq_frac(coords_frac, cell, pbc, distsq, distvecs, distvecs_frac, natoms)
    ! Special purpose routine to calculate distance vectors, squared distances
    ! and apply minimum image convention (pbc) for fractional atom coords.
    ! 
    ! Parameters
    ! ----------
    ! coords_frac : (natoms,3)
    ! cell : (3,3)
    ! pbc : int
    !   {0,1}
    ! distsq : dummy input
    ! distvecs : dummy input  
    ! distvecs_frac : dummy input  
    ! natoms : dummy input  
    !
    ! Returns
    ! -------
    ! distsq : (natoms,natoms)
    !   squared cartesien distances
    ! distvecs : (natoms,natoms,3)
    !   cartesian distance vectors  
    ! distvecs_frac : (natoms,natoms,3)
    !   fractional distance vectors, PBC applied if pbc=1
    implicit none
    integer :: natoms, pbc, ii,jj,kk
    double precision :: coords_frac(natoms,3), cell(3,3)
    double precision :: distsq(natoms,natoms), distvecs_frac(natoms, natoms, 3), &
                        distvecs(natoms, natoms, 3)
    !f2py intent(out) distsq
    !f2py intent(out) distvecs
    !f2py intent(out) distvecs_frac
    do jj=1,natoms
        do ii=1,natoms
            distvecs_frac(ii,jj,:) = coords_frac(ii,:) - coords_frac(jj,:)
        end do
    end do

    if (pbc == 1) then
        do kk=1,3
            do jj=1,natoms
                do ii=1,natoms
                    do while (distvecs_frac(ii,jj,kk) >= 0.5d0)
                        distvecs_frac(ii,jj,kk) = distvecs_frac(ii,jj,kk) - 1.0d0
                    end do    
                    do while (distvecs_frac(ii,jj,kk) < -0.5d0)
                        distvecs_frac(ii,jj,kk) = distvecs_frac(ii,jj,kk) + 1.0d0
                    end do    
                end do
            end do
        end do
    end if
    
    do kk=1,3
        do jj=1,natoms
            do ii=1,natoms
                distvecs(ii,jj,kk) = dot_product(distvecs_frac(ii,jj,:), cell(:,kk))
            end do
        end do
    end do

    do jj=1,natoms
        do ii=1,natoms
            distsq(ii,jj) = sum(distvecs(ii,jj,:)**2.0d0)
        end do
    end do
end subroutine distsq_frac
