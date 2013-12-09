# vim:ts=4:sw=4:noet
#
# See doc/ for more information on building the extension modules.
	    
F2PY=f2py
F2PY_FLAGS=--opt='-O3' \
			--f90exec=$(F90) \
			--f77exec=$(F90) \
			--arch="$(ARCH)" \
			--f90flags="$(F90FLAGS) $(OMP_F90_FLAGS)" \
			--f77flags="$(F90FLAGS) $(OMP_F90_FLAGS)" \
			$(F2PY_OMP_F90_FLAGS) \
			-llapack \
##			-DF2PY_REPORT_ON_ARRAY_COPY=1 \

OMP_F90_FLAGS=
F2PY_OMP_F90_FLAGS=

# compiler specific variables
ifort: F90=ifort
ifort: F90FLAGS=-fpp -no-prec-div -fast-transcendentals
ifort: ARCH=-xHost
ifort-omp: OMP_F90_FLAGS=-openmp -D__OPENMP
ifort-omp: F2PY_OMP_F90_FLAGS=-liomp5

gfortran: F90=gfortran
gfortran: F90FLAGS=-x f95-cpp-input
gfortran: ARCH=-mmmx -msse2
gfortran-omp: OMP_F90_FLAGS=-fopenmp -D__OPENMP
gfortran-omp: F2PY_OMP_F90_FLAGS=-lgomp

# user targets (e.g. make gfortran)
gfortran: libs
ifort: libs
gfortran-omp: gfortran
ifort-omp: ifort

help:
	@echo "make gfortran            # gfortran, default"
	@echo "make gfortran-omp        # gfortran + OpenMP"
	@echo "make ifort               # ifort"
	@echo "make ifort-omp           # ifort + OpenMP"

# internal targets
libs: _flib.so

# http://www.cprogramming.com/tutorial/makefiles_continued.html
# %  = flib
# $* = flib
# $? = flib.f90
_%.so: %.f90
	mkdir -pv build; cp -v $*.f90 build/; cd build; \
	$(F2PY) -h $*.pyf $? -m _$* --overwrite-signature; \
	$(F2PY) -c $*.pyf $? $(F2PY_FLAGS); \
	pwd; cp -v _$*.so ../ 

clean:
	rm -rvf *.so build
