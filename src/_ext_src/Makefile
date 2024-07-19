# vim:ts=4:sw=4:noet
#
# See doc/ for more information on building the extension modules.

F2PY=f2py
LAPACK=-llapack
OMP_F90_FLAGS=
OMP_F2PY_FLAGS=

# compiler specific variables
ifort: F90=ifort
ifort: F90FLAGS=-fpp -no-prec-div -fast-transcendentals
ifort: ARCH=-xHost
ifort: OPT=-O2
ifort-omp: OMP_F90_FLAGS=-openmp -D__OPENMP
ifort-omp: OMP_F2PY_FLAGS=-liomp5

gfortran: F90=gfortran
gfortran: F90FLAGS=-x f95-cpp-input -Wall
gfortran: ARCH=-march=native
gfortran: OPT=-O2
gfortran-omp: OMP_F90_FLAGS=-fopenmp -D__OPENMP
gfortran-omp: OMP_F2PY_FLAGS=-lgomp

# MKL_LIB should be something like
#	/sw/global/compilers/intel/2013/composer_xe_2013.1.117/mkl/lib/intel64
# i.e. the path to all the MKL libs
gfortran-mkl: LAPACK=-L$(MKL_LIB) -lmkl_gf_lp64  -lmkl_sequential -lmkl_core -lmkl_def
ifort-mkl: LAPACK=-L$(MKL_LIB) -lmkl_intel_lp64  -lmkl_sequential -lmkl_core -lmkl_def

# We assume that the path to libopenblas.so is in e.g. LD_LIBRARY_PATH
gfortran-openblas: LAPACK=-lopenblas
# untested
ifort-openblas: LAPACK=-lopenblas

# user targets (e.g. make gfortran)
gfortran: libs
ifort: libs
gfortran-omp: gfortran
ifort-omp: ifort
gfortran-mkl: gfortran
ifort-mkl: ifort
gfortran-openblas: gfortran
ifort-openblas: ifort

help:
	@echo "make gfortran            # gfortran, default"
	@echo "make gfortran-omp        # gfortran + OpenMP"
	@echo "make gfortran-mkl        # gfortran, Intel MKL lapack, set MKL_LIB"
	@echo "make gfortran-openblas   # gfortran, OpenBLAS"
	@echo "make ifort               # ifort"
	@echo "make ifort-omp           # ifort + OpenMP"
	@echo "make ifort-mkl           # ifort, Intel MKL lapack, set MKL_LIB"

# Target files. OBJ are the copy target ../pwtools/{_flib.so,_dcd.so} .
# OBJ_LOCAL are the result of the compilation done here: _flib.so, _dcd.so . f2py
# generates files such as _foo.cpython-36m-x86_64-linux-gnu.so, which we rename
# below to _foo.so in the compile target.
COPY_TGT_DIR = ../pwtools/
OBJ_LOCAL = _dcd.so _flib.so
OBJ = $(addprefix $(COPY_TGT_DIR),$(OBJ_LOCAL))

# internal targets
libs: $(OBJ)

# Copy target only to copy compiled artifacts away. Depends on artifacts and
# triggers compilation if needed. We could skip all the copy and complex
# dependency gymnastics below if we could tell f2py the target file names in
# $OBJ directly. #bummer
#
# $? = _foo.so _bar.so (all dependencies)
$(OBJ): $(OBJ_LOCAL)
	cp $? $(COPY_TGT_DIR)

# http://www.cprogramming.com/tutorial/makefiles_continued.html
# %   = foo
# $*  = foo
# $?  = foo.f90
#
# For names such as _foo.cpython-36m-x86_64-linux-gnu.so (result of f2py):
# %*  = foo*
# $** = foo*
#
# When we copy or move files, we need to rm the cp/mv target first, else make
# thinks that the cp/mv target is a directory. Hmmpf. However, we need to do
# that only here, not above in the OBJ target.
#
# All compiler flags (-O2 -xHost ...) have to go into FFLAGS. All linked libs
# (-llapack -lgomp ...) have to be appended to the f2py command. See
# https://numpy.org/doc/stable/f2py/buildtools/distutils-to-meson.html
#
# Here we use a make feature called target pattern. Fancy, eh?
$(OBJ_LOCAL): _%.so: %.f90
	CC=gcc CXX=g++ CFLAGS=$(OPT) \
		FC=$(F90) FFLAGS="$(F90FLAGS) $(ARCH) $(OMP_F90_FLAGS) $(OPT)" \
		$(F2PY) -c $? -m _$* --backend meson $(LAPACK) $(OMP_F2PY_FLAGS)
	rm -f $@; mv _$**.so $@


.PHONY: clean
clean:
	rm -rvf *.so $(OBJ_LOCAL) $(OBJ)
