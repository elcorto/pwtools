! Tools for reading binary DCD files (written by lammps, cp2k not tested yet). 
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
    ! Names:
    ! nstep     = NSET
    ! timestep  = DELTA
 
    implicit none 
    integer, intent(in) :: unt
    integer, intent(out) :: nstep, natoms
    real, intent(out) :: timestep
    ! dummy header content
    integer :: istrt, nsvac, dummyi, natomnfreat, ntitle, i
    character(len=4) :: hdr
    character(len=32) :: title

    read(unt) hdr, nstep, istrt, nsvac, (dummyi, i=1,5), natomnfreat, &
        timestep, (dummyi, i=1,9)
    read(unt) ntitle, title
    read(unt) natoms
    !!write(*,*), hdr
    !!write(*,*), nstep
    !!write(*,*), istrt
    !!write(*,*), nsvac
    !!write(*,*), natomnfreat
    !!write(*,*), timestep
    !!write(*,*), ntitle
    !!write(*,*), title
    !!write(*,*), natoms
end subroutine read_dcd_header_from_unit


subroutine read_dcd_data_from_unit(unt, cryst_const, coords, nstep, natoms)
    ! After reading the file to the end of the header, this routine reads out
    ! the data part.
    !
    ! Note that cryst_const is double and coords is real! `nstep` and `natoms`
    ! must be obtained by reading the header first, and then call this routine.
    !
    ! Parameters
    ! ----------
    ! unt : int
    !   file
    ! cryst_const : double precision (nstep,6)
    !   dummy
    ! coords : real (nstep,natoms,3)
    !   dummy
    ! nstep,natoms : int
    !
    ! Returns
    ! -------
    ! cryst_const, coords
    !
    implicit none
    integer :: nstep, natoms, istep, iatom, k, unt
    double precision :: unitcell(6)
    double precision, parameter :: pi=acos(-1.0d0)
    real :: x(natoms), y(natoms), z(natoms)
    double precision, intent(out) :: cryst_const(nstep,6)
    real, intent(out) :: coords(nstep, natoms, 3)
    
    ! unitcell: The DCD way to store the unit cell is completely brain damaged.
    ! (1) They only store cryst_const and not the vectors. We must therefore
    ! assume that the MD code uses the cell as [[x,0,0],[xy,y,0],[xz,yz,z]],
    ! i.e. the cell is always aliged, also for deformations in NP{T,E,H}. 
    ! (2) the order of values in `unitcell` is [a, cos(gamma), b, cos(beta),
    ! cos(alpha), c]. What have you been smoking!?? Plus, there seems to be no
    ! formal definition of the format. I had to look at VMD's dcdplugin.c. Thank
    ! you, VMD! 
    do istep=1,nstep
        read(unt) (unitcell(k), k=1,6)
        read(unt) (x(iatom), iatom=1,natoms)
        read(unt) (y(iatom), iatom=1,natoms)
        read(unt) (z(iatom), iatom=1,natoms)
        ! transform dcd unit cell to [a,b,c,alpha,beta,gamma]
        cryst_const(istep,1) = unitcell(1)
        cryst_const(istep,2) = unitcell(3)
        cryst_const(istep,3) = unitcell(6)
        cryst_const(istep,4) = acos(unitcell(5))*180.0d0/pi
        cryst_const(istep,5) = acos(unitcell(4))*180.0d0/pi
        cryst_const(istep,6) = acos(unitcell(2))*180.0d0/pi
        do iatom=1,natoms
            coords(istep,iatom,1) = x(iatom)
            coords(istep,iatom,2) = y(iatom)
            coords(istep,iatom,3) = z(iatom)
        end do
    end do
end subroutine read_dcd_data_from_unit


subroutine read_dcd_header(filename, nstep, natoms, timestep)
    ! To be wrapped by f2py and used from Python. Read header from dcd file
    ! `filename`.
    !
    ! Parameters
    ! ----------
    ! filename : character
    ! nstep, natoms, timestep : int, int, real
    !   dummy input
    !
    ! Returns
    ! -------
    ! nstep, natoms, timestep
    !
    ! Example
    ! -------
    ! >>> nstep,natoms,timestep = _dcd.read_dcd_header(filename)
    !
    character(len=1024) :: filename
    integer, intent(out) :: nstep, natoms
    real, intent(out) :: timestep

    open(1, file=trim(filename), status='old', form='unformatted', action='read')
    call read_dcd_header_from_unit(1, nstep, natoms, timestep)
    close(1)
end subroutine read_dcd_header


subroutine read_dcd_data(filename, cryst_const, coords, nstep, natoms)
    ! To be wrapped by f2py and used from Python. Read data from DCD file. You
    ! need to call read_dcd_header() first to get nstep and natoms.
    !
    ! Parameters
    ! ----------
    ! filename : character
    ! nstep, natoms, timestep : int, int, real
    !   dummy input
    !
    ! Returns
    ! -------
    ! cryst_const, coords
    !
    ! Example
    ! -------
    ! >>> nstep,natoms,timestep = _dcd.read_dcd_header(filename)
    ! >>> cryst_const,coords = _dcd.read_dcd_data(filename,nstep,natoms)
    ! 
    ! Notes
    ! ------
    ! Splitting the reading up like this (first read_dcd_header(), then
    ! read_dcd_data()) is complicated but I see no other way since we first need
    ! to obtain nstep,natoms and then allocate cryst_const+coords with them in
    ! read_dcd_data(). Maybe there is another way in Fortran? If so, send me a
    ! mail, or a patch!
    implicit none
    character(len=1024) :: filename
    integer :: nstep, natoms
    double precision, intent(out):: cryst_const(nstep,6)
    real, intent(out) :: coords(nstep, natoms, 3)
    real :: timestep

    open(1, file=trim(filename), status='old', form='unformatted', action='read')
    ! read header again only to advance in file to the data section
    call read_dcd_header_from_unit(1, nstep, natoms, timestep)
    call read_dcd_data_from_unit(1, cryst_const, coords, nstep, natoms)
    close(1)
end subroutine read_dcd_data
