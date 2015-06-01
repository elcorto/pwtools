! vim:tw=79
!
! Tools for reading binary DCD files (tested: lammps, cp2k). 
!
! Endianess
! ---------
! Other tools which we found to read dcd deal with endianess, but we don't. All
! of our current machines are only x86-64 (i.e. Intel or AMD) clusters, all
! little-endian, so we don't care.
!
! Other tools (some even Python, but tons of dependencies or too lazy to
! read the documentation):
! * http://code.google.com/p/mdanalysis/
! * http://www.csb.pitt.edu/prody/
! * and of course VMD, but we can't wrap that due to a lacking Python API AFAIK

subroutine open_file(filename, unt)
    implicit none
    character(len=1024), intent(in) :: filename
    integer, intent(out) :: unt
    integer :: iost
    open(unit=unt, file=trim(filename), status='old', form='unformatted', &
         action='read', iostat=iost)
    if (iost /= 0) then
        write(*,*) "open_file: error when opening file: '", trim(filename), &
                   "' with iostat=", iost
        stop 1
    end if
end subroutine open_file

subroutine read_dcd_header_from_unit(unt, nstep, natoms, timestep)
    ! Read header from DCD file.
    !
    ! Parameters
    ! ----------
    ! unt : int
    !   open file unit (or whatever the Fortran name is)
    ! nstep, natoms, timestep : int, int, real
    !   dummy input
    !
    ! Returns
    ! -------
    ! nstep,natoms,timestep
    !
    ! Notes
    ! -----
    ! nstep = 0 in case of cp2k style files 
    ! 
    ! lammps
    ! ^^^^^^
    !
    ! The VMD documentation below helped us to implement the parser for the
    ! lammps-flavor dcd format.
    ! 
    ! Thanks to:
    ! http://www.ks.uiuc.edu/Research/vmd/plugins/molfile/dcdplugin.html
    ! http://www.ks.uiuc.edu/Research/namd/wiki/?ReadingDCDinFortran
    !
    ! The header has this format, according to the VMD dcdplugin docs:
    !
    ! HDR     NSET    ISTRT   NSAVC   5-ZEROS NATOM-NFREAT    DELTA   9-ZEROS
    ! `CORD'  #files  step 1  step    zeroes  (zero)          timestep  (zeroes)
    !                         interval
    ! C*4     INT     INT     INT     5INT    INT             DOUBLE  9INT
    ! ==========================================================================
    ! NTITLE          TITLE
    ! INT (=2)        C*MAXTITL
    !                 (=32)
    ! ==========================================================================
    ! NATOM
    ! #atoms
    ! INT
    ! ==========================================================================
    ! X(I), I=1,NATOM         (DOUBLE)
    ! Y(I), I=1,NATOM         
    ! Z(I), I=1,NATOM         
    ! ==========================================================================
    !
    ! Reading the header only works if hdr,nstep,...,9-zeros are read in *one*
    ! read statement. The same goes for ntitle,title. 
    !
    ! It turns out that for lammps, the last 9 integers (all zero) are actually
    ! 10 and the last one is 24. Same as in cp2k, so at least that's
    ! consistent. Hooray for undocumented binary formats! We should really push
    ! for a simple and extensible HDF5-based format.
    !
    ! Names:
    ! nstep     = NSET
    ! timestep  = DELTA
    !
    ! CP2K 
    ! ^^^^
    ! 
    ! From v2.6.1, src/motion_utils.F:
    !
    ! section_ref = "MOTION%PRINT%"//TRIM(my_pk_name)//"%EACH%"//TRIM(id_label)
    ! iskip = section_get_ival(root_section,TRIM(section_ref),error=error)
    ! WRITE (UNIT=traj_unit) id_dcd,0,it,iskip,0,0,0,0,0,0,REAL(dtime,KIND=sp),&
    !                        1,0,0,0,0,0,0,0,0,24
    ! remark1 = "REMARK "//id_dcd//" DCD file created by "//TRIM(cp2k_version)//&
    !           " (revision "//TRIM(compile_revision)//")"
    ! remark2 = "REMARK "//TRIM(r_user_name)//"@"//TRIM(r_host_name)
    ! WRITE (UNIT=traj_unit) 2,remark1,remark2
    ! WRITE (UNIT=traj_unit) nat
    ! 
    ! where
    !   id_dcd  = 'CORD'
    !   it      = is NOT nstep
    !   iskip   = ???
    !   dtime   = timestep
    ! 
    ! nstep (or NSET in the VMD docs) is 0 here -> We need to walk thru the
    ! file until we hit the bottom, just as cp2k/tools/dumpdcd.f90 does it.
    !
    implicit none 
    integer, intent(in) :: unt
    integer :: istrt, nsvac, dummyi, natomnfreat, ntitle, i
    character(len=4) :: hdr
    character(len=80) :: remark1, remark2
    integer, intent(out) :: nstep, natoms
    real, intent(out) :: timestep
    
    read(unt) hdr, nstep, istrt, nsvac, (dummyi, i=1,5), natomnfreat, &
              timestep, (dummyi, i=1,10)
    read(unt) ntitle, remark1, remark2
    read(unt) natoms
    !!write(*,*), "hdr        : ", hdr
    !!write(*,*), "nstep      : ", nstep
    !!write(*,*), "istrt      : ", istrt
    !!write(*,*), "nsvac      : ", nsvac
    !!write(*,*), "natomnfreat: ", natomnfreat
    !!write(*,*), "timestep   : ", timestep
    !!write(*,*), "ntitle     : ", ntitle
    !!write(*,*), "remarks    : ", remark1, remark2
    !!write(*,*), "natoms     : ", natoms
end subroutine read_dcd_header_from_unit


subroutine read_dcd_data_from_unit(unt, cryst_const, coords, nstep, natoms, &
                                   convang)
    ! After reading the file to the end of the header, this routine reads out
    ! the data part.
    !
    ! Note that cryst_const is double and coords is real! `nstep` and `natoms`
    ! must be obtained by reading the header first, and then call this routine.
    !
    ! Parameters
    ! ----------
    ! unt : int
    !   open file unit
    ! cryst_const : double precision (nstep,6)
    !   dummy
    ! coords : real (nstep,natoms,3)
    !   dummy
    ! nstep,natoms : int
    ! convang : int
    !   1 - convert angles from cosine to degree
    !
    ! Returns
    ! -------
    ! cryst_const, coords
    !
    implicit none
    integer, intent(in) :: nstep, natoms, unt, convang
    integer :: istep, iatom, k
    double precision :: cryst_const_dcd(6)
    double precision, parameter :: pi=acos(-1.0d0)
    real :: x(natoms), y(natoms), z(natoms)
    double precision, intent(out) :: cryst_const(nstep,6)
    real, intent(out) :: coords(nstep, natoms, 3)
    
    ! cryst_const_dcd: The DCD way to store the unit cell parameters is
    ! completely brain damaged. (1) They only store cryst_const and not the
    ! vectors. We must therefore assume that the MD code uses the cell as
    ! [[x,0,0],[xy,y,0],[xz,yz,z]], i.e. the cell is always aliged, also for
    ! deformations in NP{T,E,H}. cp2k has dcd_aligned_cell for that. Dunno
    ! what's with lammps. (2) the order of values in `cryst_const_dcd` is [a,
    ! cos(gamma), b, cos(beta), cos(alpha), c]. What have you been smoking!??
    ! Plus, there seems to be no formal definition of the format. I had to look
    ! at VMD's dcdplugin.c. Thank you, VMD! But: parser code is NO spec!
    do istep=1,nstep
        read(unt) (cryst_const_dcd(k), k=1,6)
        read(unt) (x(iatom), iatom=1,natoms)
        read(unt) (y(iatom), iatom=1,natoms)
        read(unt) (z(iatom), iatom=1,natoms)
        ! transform dcd unit cell to [a,b,c,alpha,beta,gamma]
        cryst_const(istep,1) = cryst_const_dcd(1)
        cryst_const(istep,2) = cryst_const_dcd(3)
        cryst_const(istep,3) = cryst_const_dcd(6)
        if (convang == 1) then
            cryst_const(istep,4) = acos(cryst_const_dcd(5))*180.0d0/pi
            cryst_const(istep,5) = acos(cryst_const_dcd(4))*180.0d0/pi
            cryst_const(istep,6) = acos(cryst_const_dcd(2))*180.0d0/pi
        else
            cryst_const(istep,4) = cryst_const_dcd(5)
            cryst_const(istep,5) = cryst_const_dcd(4)
            cryst_const(istep,6) = cryst_const_dcd(2)
        end if
        do iatom=1,natoms
            coords(istep,iatom,1) = x(iatom)
            coords(istep,iatom,2) = y(iatom)
            coords(istep,iatom,3) = z(iatom)
        end do
    end do
end subroutine read_dcd_data_from_unit


subroutine read_dcd_data(filename, cryst_const, coords, nstep, natoms, convang)
    ! To be wrapped by f2py and used from Python. Read data from DCD file. You
    ! need to call get_dcd_file_info() first to get nstep and natoms.
    !
    ! Parameters
    ! ----------
    ! filename : character
    ! nstep, natoms, timestep : int, int, real
    !   dummy input
    ! convang : int
    !   1 - convert angles from cosine to degree
    !
    ! Returns
    ! -------
    ! cryst_const, coords
    !
    ! Example
    ! -------
    ! >>> nstep,natoms,timestep = _dcd.get_dcd_file_info(filename,nstephdr)
    ! >>> cryst_const,coords = _dcd.read_dcd_data(filename,nstep,natoms,convang)
    ! 
    ! Notes
    ! ------
    ! Splitting the reading up like this (first get_dcd_file_info(), then
    ! read_dcd_data()) is complicated but I see no other way since we first
    ! need to obtain nstep,natoms and then allocate cryst_const+coords with
    ! them in read_dcd_data(). Maybe there is another way in Fortran, with
    ! allocatable and such? If so, send me a mail, or a patch!
    implicit none
    character(len=1024), intent(in) :: filename
    integer, intent(in) :: nstep, natoms, convang
    integer :: dummy_nstep, dummy_natoms, unt
    real :: dummy_timestep
    double precision, intent(out):: cryst_const(nstep,6)
    real, intent(out) :: coords(nstep, natoms, 3)
    
    if ((.not. natoms > 0) .or. (.not. nstep > 0)) then
        write(*,*) "read_dcd_data: error: nstep or natoms not > 0: ", &
                   "nstep=", nstep, "natoms=", natoms
        stop 1
    end if     
    unt = 1
    call open_file(filename, unt)
    ! read header again only to advance in file to the data section
    ! dummy_*: make sure we use nstep and natoms from input
    call read_dcd_header_from_unit(unt, dummy_nstep, dummy_natoms, &
                                   dummy_timestep)
    call read_dcd_data_from_unit(unt, cryst_const, coords, nstep, natoms, &
                                 convang)
    close(unt)
end subroutine read_dcd_data


subroutine get_dcd_file_info(filename, nstep, natoms, timestep, nstephdr)
    ! To be wrapped by f2py and used from Python. Read header (natoms,
    ! timestep) from dcd file `filename`. Read nstep from header (nstephdr=1,
    ! lammps) or find by walking thru the file once (nstephdr=0, cp2k, more
    ! general, should always work, but may be slow for large files).
    !
    ! Parameters
    ! ----------
    ! filename : character
    ! nstep, natoms, timestep : int, int, real
    !   dummy input
    ! nstephdr : int
    !   read nstep from header instead of walking file
    !
    ! Returns
    ! -------
    ! nstep, natoms, timestep
    !
    ! Example
    ! -------
    ! >>> nstep,natoms,timestep = _dcd.get_dcd_file_info(filename, nstephdr)

    implicit none
    character(len=1024), intent(in) :: filename
    integer, intent(in) :: nstephdr
    integer :: iatom, k, unt, iost
    double precision :: cryst_const_dcd(6)
    real,allocatable :: x(:), y(:), z(:)
    integer, intent(out) :: nstep, natoms
    real, intent(out) :: timestep
    
    unt = 1
    call open_file(filename, unt)    
    
    ! nstephdr = 1: lammps
    !   nstep, natoms, timestep known by call to read_dcd_header_from_unit()
    ! nstephdr = 0: cp2k
    !   natoms, timestep known by call to read_dcd_header_from_unit(),
    !   nstep = 0 in header, must walk thru data section in file until end 
    !   to find out, ugly but works
    call read_dcd_header_from_unit(unt, nstep, natoms, timestep)
    if (nstephdr == 0) then
        allocate(x(natoms), y(natoms), z(natoms))
        nstep = 0
        do
            read(unt, iostat=iost) (cryst_const_dcd(k), k=1,6)
            ! iost < 0: end of file
            if (iost < 0) then
                exit
            else if (iost > 0) then
                write(*,*) "get_dcd_file_info: read error in file", &
                           trim(filename)
                deallocate(x,y,z)
                stop 1
            end if        
            nstep = nstep + 1 
            read(unt, iostat=iost) (x(iatom), iatom=1,natoms)
            read(unt, iostat=iost) (y(iatom), iatom=1,natoms)
            read(unt, iostat=iost) (z(iatom), iatom=1,natoms)
        end do
        deallocate(x,y,z)
    end if
    close(unt)
    if (.not. nstep > 0) then
       write(*,*) "get_dcd_file_info: error: nstep is not > 0: nstep =", nstep
       stop 1
    end if    
end subroutine get_dcd_file_info
