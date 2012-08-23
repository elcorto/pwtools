#define stdout 6
#define stderr 0

subroutine cutfunc(dists, rcut, ret, nn)
    ! Cut off fucntion for distances from Behler paper.
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
    ! Reptrns
    ! -------
    ! ret : see args, note that this gets overwritten, so you can do
    ! >>> ret = np.empty(...)
    ! >>> _flib.symfunc(..., ret, ...)
    ! 
#ifdef __OPENMP    
    use omp_lib
    !f2py threadsafe
#endif    
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
