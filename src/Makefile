# vim:ts=4:sw=4:noet
#
# See doc/ for more information on building the extension modules.
	    
F2PY=f2py3
F2PY_FLAGS=--opt='-O3' \
			--f90exec=$(F90) \
			--f77exec=$(F90) \
			--arch="$(ARCH)" \
			--f90flags="$(F90FLAGS) $(OMP_F90_FLAGS)" \
			--f77flags="$(F90FLAGS) $(OMP_F90_FLAGS)" \
			$(F2PY_OMP_F90_FLAGS) \
			$(LAPACK) \
##			-DF2PY_REPORT_ON_ARRAY_COPY=1 \

LAPACK=-llapack
OMP_F90_FLAGS=
F2PY_OMP_F90_FLAGS=

# compiler specific variables
ifort: F90=ifort
ifort: F90FLAGS=-fpp -no-prec-div -fast-transcendentals
ifort: ARCH=-xHost
ifort-omp: OMP_F90_FLAGS=-openmp -D__OPENMP
ifort-omp: F2PY_OMP_F90_FLAGS=-liomp5

gfortran: F90=gfortran
gfortran: F90FLAGS=-x f95-cpp-input -Wall
gfortran: ARCH=-mmmx -msse2
gfortran-omp: OMP_F90_FLAGS=-fopenmp -D__OPENMP
gfortran-omp: F2PY_OMP_F90_FLAGS=-lgomp

# MKL_LIB should be something like
# 	/sw/global/compilers/intel/2013/composer_xe_2013.1.117/mkl/lib/intel64
# i.e. the path to all the MKL libs	
gfortran-mkl: LAPACK=-L$(MKL_LIB) -lmkl_gf_lp64  -lmkl_sequential -lmkl_core -lmkl_def
ifort-mkl: LAPACK=-L$(MKL_LIB) -lmkl_intel_lp64  -lmkl_sequential -lmkl_core -lmkl_def

# user targets (e.g. make gfortran)
gfortran: libs
ifort: libs
gfortran-omp: gfortran
ifort-omp: ifort
gfortran-mkl: gfortran
ifort-mkl: ifort

help:
	@echo "make gfortran            # gfortran, default"
	@echo "make gfortran-omp        # gfortran + OpenMP"
	@echo "make gfortran-mkl        # gfortran, Intel MKL lapack, set MKL_LIB"
	@echo "make ifort               # ifort"
	@echo "make ifort-omp           # ifort + OpenMP"
	@echo "make ifort-mkl           # ifort, Intel MKL lapack, set MKL_LIB"

# internal targets
libs: _flib*.so _dcd*.so

# http://www.cprogramming.com/tutorial/makefiles_continued.html
# %   = flib
# $*  = flib
# $?  = flib.f90
#
# For names such as _flib.cpython-36m-x86_64-linux-gnu.so:
# %*  = flib*
# $** = flib*
#
# Need the last involved copy command since we need to
# 	cp _flib.cpython-36m-x86_64-linux-gnu.so ../pwtools/_flib.so
_%*.so: %.f90
	CC=gcc CXX=g++ $(F2PY) -h $*.pyf $? -m _$* --overwrite-signature; \
	CC=gcc CXX=g++ $(F2PY) -c $*.pyf $? $(F2PY_FLAGS); \
	cp -v _$**.so ../pwtools/_$*.so

clean:
	rm -rvf *.so *.pyf
