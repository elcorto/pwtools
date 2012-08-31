# vim:ts=4:sw=4:noet
	    
# Compile Fortran extensions. Generates *.so and *.pyf (f2py interface) files.
# 
# usage:  
#   make [-B]
# 
# You need:
# * numpy
# * a Fortran compiler
# * Python headers (for Linux: usually a package python-dev or python-devel,
#   see the package manager of your distro)
# 
# The module is compiled with f2py (currently part of numpy, tested with numpy
# 1.1.0 .. 1.6.x). 
# 
# Just try 
#     $ make
#
# Compiler
# --------
# Instead of letting numpy.distutils pick a compiler + special flags, which is
# not trivial and therefore almost never works, it is much easier to simply
# define the compiler to use + architecture-specific flags. See F90 and ARCH
# below.
# 
# OpenMP 
# ------
# We managed to speed up the calculations by sprinkling some OpenMP
# pragmas in flib.f90. This works pretty good. If you wanna try, uncomment 
# *OMP_F90_FLAGS below. If all went well, _flib.so should be linked to libgomp
# (or libiomp for ifort). Check with
# 	
# 	$ ldd _flib.so
# 
# Setting the number of threads:  
# 	
# 	$ export OMP_NUM_THREADS=2
# 	$ python -c "import numpy as np; from pwtools.pydos import fvacf; \
# 	             fvacf(np.random.rand(1000,3,5000))"
# 
# If this env var is NOT set, then OpenMP uses all available cores (e.g. 4 on a
# quad-core box).
# 
# IMPORTANT: 
# 	Note that we may have found a f2py bug (see test/test_f2py_flib_openmp.py)
# 	re. OMP_NUM_THREADS. We have a workaround for that in pydos.fvacf().
#
# There is also an optional arg 'nthreads' to _flib.vacf(). If this is
# supplied, then it will override OMP_NUM_THREADS. Currently, this is the
# safest way to set the number of threads.


# f2py executable. On some systems (Debian), you may have
#	/usr/bin/f2py -> f2py2.6
#	/usr/bin/f2py2.5
#	/usr/bin/f2py2.6
# and such. Here you define the correct executable for your environment.
# Usually, just "f2py" should be fine.	
F2PY=f2py

# ARCH below is for Intel Core i7 / Xeon. If you don't know what your CPU is
# capable of (hint: see /proc/cpuinfo) then use "ARCH=".
#
# Wanny try OpenMP? Then uncomment OMP_F90_FLAGS and F2PY_OMP_F90_FLAGS.

# gfortran
##F90=gfortran
##F90FLAGS=-x f95-cpp-input 
##ARCH=-mmmx -msse4.2 
##OMP_F90_FLAGS=-fopenmp -D__OPENMP
##F2PY_OMP_F90_FLAGS=-lgomp

# ifort 11.x
F90=ifort
F90FLAGS=-fpp -no-prec-div -fast-transcendentals
ARCH=-xHost
OMP_F90_FLAGS=-openmp -D__OPENMP 
F2PY_OMP_F90_FLAGS=-liomp5

# no OpenMP
##OMP_F90_FLAGS=
##F2PY_OMP_F90_FLAGS=

# f2py stuff
#
# numpy.distutils has default -03 for fcompiler. --f90flags="-02" does NOT
# override this. We get "-O3 -O2" and a compiler warning. We have to use f2py's
# --opt= flag.
F2PY_FLAGS=--opt='-O3' \
			--f90exec=$(F90) \
			--f77exec=$(F90) \
			--arch="$(ARCH)" \
			--f90flags="$(F90FLAGS) $(OMP_F90_FLAGS)" \
			--f77flags="$(F90FLAGS) $(OMP_F90_FLAGS)" \
			$(F2PY_OMP_F90_FLAGS) \
##			-DF2PY_REPORT_ON_ARRAY_COPY=1 \

all: _flib.so _fsymfunc.so

_flib.so: flib.f90
	$(F2PY) -h flib.pyf flib.f90 -m _flib --overwrite-signature
	$(F2PY) -c flib.pyf flib.f90 $(F2PY_FLAGS)

_fsymfunc.so: fsymfunc.f90
	$(F2PY) -h fsymfunc.pyf fsymfunc.f90 -m _fsymfunc --overwrite-signature
	$(F2PY) -c fsymfunc.pyf fsymfunc.f90 $(F2PY_FLAGS)


clean:
	rm -vf *.so *.pyf
