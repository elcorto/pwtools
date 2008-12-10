! vim:let fortran_more_precise=1:let fortran_free_source=1

subroutine foo(a, nx, ny, nz)
    implicit none
    integer, intent(in) :: nx, ny, nz
    double precision, intent(in) :: a(nx, ny, nz)
    return
end subroutine foo


