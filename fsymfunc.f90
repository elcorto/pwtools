#define stdout 6
#define stderr 0

subroutine cutfunc(dists, rcut, ret, nn)
    ! Cut off fucntion for distances from Behler paper.
    ! Use stand-alone or in symfunc_45().
    !
    ! Parameters
    ! ----------
    ! dists : 2d array (nn,nn), where nn=natoms usually
    !   Cartesian distances.
    ! rcut : float
    !   Cutoff.
    ! ret : dummy result (nn,nn)
    ! nn : integer
    !
    implicit none
    integer :: ii,jj,nn
    double precision, parameter :: pi=acos(-1.0d0)
    double precision :: dists(nn,nn), ret(nn,nn), rcut
    intent(out) :: ret
    !f2py intent(out) ret
    ret = 0.5d0 * (cos(pi*dists/rcut) + 1.0d0)
    do jj=1,nn
        do ii=1,nn
            if (dists(ii,jj) > rcut) then
                ret(ii,jj) = 0.0d0
            end if                
        end do
    end do        
end subroutine cutfunc    

subroutine symfunc_45(distsq, cos_anglesijk, ret, params, what, natoms, npsets, nparams)
    ! Calculate sym funcs g4 or g5, based on `what`.
    !
    ! Parameters
    ! ----------
    ! distsq : 2d array (natoms, natoms)
    !   *squared* cartesian distances
    ! cos_anglesijk : (natoms,natoms,natoms)
    !   cos_anglesijk(i,j,k) = cos(angle(d_ij, d_ik))
    ! ret : 2d array (natoms, npsets)
    !   result array to overwrite
    ! params : 2d array (npsets, nparams)
    !   array with `npsets` parameter sets (vectors) of length `nparams`, i.e.
    !   (9,4) for 9 parameter sets for 4 parameters (called p0,p1,p2,p3 below)
    ! what : integer
    !   4 or 5
    ! 
    ! Returns
    ! -------
    ! ret : see args, note that this gets overwritten, so you can do
    ! >>> ret = np.empty(...)
    ! >>> _flib.symfunc(..., ret, ...)
    ! 
    implicit none
    integer :: natoms, ii, jj, kk, &
               ipset, npsets, nparams, what
    double precision :: p0, p1, p2, p3, val, sm
    double precision :: distsq(natoms, natoms), tmp2(natoms,natoms)
    double precision :: cf(natoms, natoms) 
    double precision :: ret(natoms, npsets)
    double precision :: params(npsets, nparams)
    double precision :: cos_anglesijk(natoms, natoms, natoms)
    double precision :: tmp(natoms,natoms,natoms)
    !f2py intent(in, out, overwrite) ret
    !f2py intent(in) params
    if (what == 4) then
        do ipset=1,npsets
            p0 = params(ipset,1)
            p1 = params(ipset,2)
            p2 = params(ipset,3)
            p3 = params(ipset,4)
            call cutfunc(sqrt(distsq), p0, cf, natoms)
            tmp = 2.0d0**(1.0d0-p1) * (1.0d0+p2*cos_anglesijk)**p1
            tmp2 = exp(-p3*distsq) * cf
            do ii=1,natoms
                sm = 0.0d0
                do kk=1,natoms
                    do jj=1,natoms
                        if (ii /= jj .and. ii /= kk .and. jj /= kk) then
                            !!val = tmp(ii,jj,kk) * &
                            !!      exp(-p3*(distsq(ii,jj) + &
                            !!               distsq(ii,kk) + &
                            !!               distsq(jj,kk))) * &
                            !!      cf(ii,jj)*cf(ii,kk)*cf(jj,kk)
                            val = tmp(ii,jj,kk) * tmp2(ii,jj) * tmp2(ii,kk) * &
                                  tmp2(jj,kk)
                            sm = sm + val
                        end if
                    end do                        
                end do
                ret(ii,ipset) = sm
            end do
        end do
   else if (what == 5) then
        do ipset=1,npsets
            p0 = params(ipset,1)
            p1 = params(ipset,2)
            p2 = params(ipset,3)
            p3 = params(ipset,4)
            call cutfunc(sqrt(distsq), p0, cf, natoms)
            tmp = 2.0d0**(1.0d0-p1) * (1.0d0+p2*cos_anglesijk)**p1
            tmp2 = exp(-p3*distsq) * cf
            do ii=1,natoms
                sm = 0.0d0
                do kk=1,natoms
                    do jj=1,natoms
                        if (ii /= jj .and. ii /= kk .and. jj /= kk) then
                            !!val = tmp(ii,jj,kk) * &
                            !!      exp(-p3*(distsq(ii,jj) + &
                            !!               distsq(ii,kk))) * &
                            !!      cf(ii,jj)*cf(ii,kk)
                            val = tmp(ii,jj,kk) * tmp2(ii,jj) * tmp2(ii,kk)
                            sm = sm + val
                        end if                        
                    end do                        
                end do                        
                ret(ii,ipset) = sm
            end do
        end do 
    else
        write(stderr,*) "error: illegal value for 'what'"
        return
    end if
end subroutine symfunc_45

subroutine symfunc_45_fast(distvecs, dists, ret, params, what, natoms, npsets, nparams)
    ! Calculate sym funcs g4 or g5, based on `what`. 
    !
    ! In contrast to symfunc_45(), we calculate angles and cutfunc in the loop,
    ! thus we don't calculate a potentially big `cos_anglesijk` array (big: 1G
    ! for 500 atoms) before and pass that in. Serial execution is slower than
    ! symfunc_45(), but with OpenMP and >= 4 cores, we are faster.
    !
    ! Parameters
    ! ----------
    ! distvecs : 3d array w/ cartesian distance vectors, maybe already PBC
    !   applied
    ! dists : (natoms,natoms)
    !   Cartesian distances, np.sqrt((distvecs**2.0).sum(axis=2))
    ! ret : 2d array (natoms, npsets)
    !   result array to overwrite
    ! params : 2d array (npsets, nparams)
    !   array with `npsets` parameter sets (vectors) of length `nparams`, i.e.
    !   (9,4) for 9 parameter sets for 4 parameters (called p0,p1,p2,p3 below)
    ! what : integer
    !   4 or 5
    ! 
    ! Returns
    ! -------
    ! ret : see args, note that this gets overwritten, so you can do
    ! >>> ret = np.empty(...)
    ! >>> _flib.symfunc(..., ret, ...)
    !
    ! Notes
    ! -----
    ! p0 = rcut
    ! p1 = zeta
    ! p2 = lambda
    ! p3 = eta
    !
    ! Speed
    ! -----
    ! All tests: 200 random atoms, npsets=25
    ! * We need to set rcut = rmax_smith, which is < dists.maax() (L/2 for cubic
    !   box). Only then, we skip enough dists to make the *_fast() routines
    !   faster. If we use rcut=dists.max(), then the *_fast() versions are
    !   slower, even with OpenMP.
    ! * OpenMP scales, don't expect too much speedup on many cores. Tested up to 4.
    !   In fact, I'm surprised that it scales at all :)
    ! * In serial execution, symfunc_45() is faster (5.7s vs. 10.5s), but we need to
    !   calculate cos_anglesijk before, which is fast but the array may be big for
    !   many atoms (1G for 500 atoms). The main advantage of symfunc_45_fast()
    !   implementation is that we don't need to calculate that array ever.
    ! * The "if" clauses in the inner loop look like a horroible idea, but
    !   actually they don't hit performance.
    ! * Passing in `distvecs` and `dists` is faster than calculating them in the
    !   loop -> array lookup faster than calculation
    ! * The unrolled dot_product() for cos_angles is faster
    ! * We parallelize over the "ii"-loop (__SF45_INNER). Using the outermost
    !   loop "ipset" instead (__SF45_OUTER) makes no difference, which I don't
    !   really understand. Must be some load balancing issue. Maybe if we use
    !   modulo(npsets, ncores) == 0 ? Also, we would need at least as many npsets
    !   as we have cores.
    ! * At the moment with OpenMP, we get almost linear speedup (tested up to 4
    !   cores). On cartman (4 cores, Intel Core i7 with hyperthreading), we get
    !   even a speedup using 8 cores instead of 4!
    ! * Also important is that we calculate `val` and cutfunc only when needed
    !   (second "if" clause).
    ! * With OpenMP, it is very very importand to declate *all* private
    !   variables as private! Otherwise, performance suffers dramatically.
#ifdef __OPENMP    
    use omp_lib
    !f2py threadsafe
#endif    
    implicit none
    integer :: natoms, ii, jj, kk, &
               ipset, npsets, nparams, what
    double precision, parameter :: pi=acos(-1.0d0)
    double precision :: p0, p1, p2, p3, val, sm, & 
                        distvecs(natoms,natoms,3), dists(natoms,natoms), dij, dik, djk, &
                        dvij(3), dvik(3), dvjk(3), cos_angle, &
                        dijc, dikc, djkc
    double precision :: ret(natoms, npsets)
    double precision :: params(npsets, nparams)
    !f2py intent(in, out, overwrite) ret

    if (what == 4) then
#ifdef __SF45_OUTER
        !$omp parallel do private(dvij,dvik,dvjk,dij,dik,djk,cos_angle, &
        !$omp dijc,dikc,djkc,val,sm,ii,jj,kk,ipset,p0,p1,p2,p3)
#endif    
        do ipset=1,npsets
            p0 = params(ipset,1)
            p1 = params(ipset,2)
            p2 = params(ipset,3)
            p3 = params(ipset,4)
#ifdef __SF45_INNER
            !$omp parallel do private(dvij,dvik,dvjk,dij,dik,djk,cos_angle, &
            !$omp dijc,dikc,djkc,val,sm,ii,jj,kk)
#endif    
            do ii=1,natoms
                sm = 0.0d0
                do kk=1,natoms
                    do jj=1,natoms
                        if (ii /= jj .and. ii /= kk .and. jj /= kk) then
                            dvij = distvecs(ii,jj,:)
                            dvik = distvecs(ii,kk,:)
                            dvjk = distvecs(jj,kk,:)
                            dij = dists(ii,jj)
                            dik = dists(ii,kk)
                            djk = dists(jj,kk)
                            if (dij <= p0 .and. dik <= p0 .and. djk <= p0) then
                                cos_angle = (dvij(1)*dvik(1) + dvij(2)*dvik(2) &
                                             + dvij(3)*dvik(3)) / dij / dik
                                ! cutfunc hard coded                                             
                                dijc = 0.5d0 * (cos(pi*dij/p0) + 1.0d0)
                                dikc = 0.5d0 * (cos(pi*dik/p0) + 1.0d0)
                                djkc = 0.5d0 * (cos(pi*djk/p0) + 1.0d0)
                                val = 2.0d0**(1.0d0-p1) * &
                                      (1.0d0 + p2*cos_angle)**p1 &
                                      * exp(-p3*(dij**2.0d0 + dik**2.0d0 + djk**2.0d0)) &
                                      * dijc * dikc * djkc
                                sm = sm + val
                            end if
                        end if
                    end do
                end do
                ret(ii,ipset) = sm
            end do
#ifdef __SF45_INNER
            !$omp end parallel do
#endif    
        end do
#ifdef __SF45_OUTER
        !$omp end parallel do
#endif    
    else if (what == 5) then
#ifdef __SF45_OUTER
        !$omp parallel do private(dvij,dvik,dij,dik,cos_angle, &
        !$omp dijc,dikc,val,sm,ii,jj,kk,ipset,p0,p1,p2,p3)
#endif    
        do ipset=1,npsets
            p0 = params(ipset,1)
            p1 = params(ipset,2)
            p2 = params(ipset,3)
            p3 = params(ipset,4)
            !$omp parallel do private(dvij,dvik,dij,dik,cos_angle, &
            !$omp dijc,dikc,val,sm,ii,jj,kk)
            do ii=1,natoms
                sm = 0.0d0
                do kk=1,natoms
                    do jj=1,natoms
                        if (ii /= jj .and. ii /= kk .and. jj /= kk) then
                            dvij = distvecs(ii,jj,:)
                            dvik = distvecs(ii,kk,:)
                            dij = dists(ii,jj)
                            dik = dists(ii,kk)
                            if (dij <= p0 .and. dik <= p0) then
                                cos_angle = (dvij(1)*dvik(1) + dvij(2)*dvik(2) &
                                             + dvij(3)*dvik(3)) / dij / dik
                                ! cutfunc hard coded                                             
                                dijc = 0.5d0 * (cos(pi*dij/p0) + 1.0d0)
                                dikc = 0.5d0 * (cos(pi*dik/p0) + 1.0d0)
                                val = 2.0d0**(1.0d0-p1) * &
                                      (1.0d0 + p2*cos_angle)**p1 &
                                      * exp(-p3*(dij**2.0d0 + dik**2.0d0)) &
                                      * dijc * dikc
                                sm = sm + val
                            end if
                        end if
                    end do                        
                end do
                ret(ii,ipset) = sm
            end do
#ifdef __SF45_INNER
            !$omp end parallel do
#endif            
        end do
#ifdef __SF45_OUTER
        !$omp end parallel do
#endif          
    else
        write(stderr,*) "error: illegal value for 'what'"
        return
    end if
end subroutine symfunc_45_fast
