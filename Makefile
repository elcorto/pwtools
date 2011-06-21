# vim:ts=4:sw=4:noet
	    
# Compile Fortran extension (_flib.so).
# 
# usage:  
#   make [-B]
# 
# Further Notes:
#
# For the velocity autocorrelation function (VACF), we use an extension module
# _flib.so written in Fortran (flib.f90). You need
#     - numpy
#     - a Fortran compiler
#     - Python headers (for Linux: usually a package python-dev or python-devel,
#       see the package manager of your distro)
# 
# The module is compiled with f2py (currently part of numpy, tested with numpy
# 1.1.0 .. 1.4.x). 
# 
# Just try 
#     $ make
# It should result in a file "_flib.so".
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

#--- variables ---------------------------------------------------------------

# files
FILE=flib
FORT=$(FILE).f90
PYF=$(FILE).pyf
EXT_MODULE=_$(FILE)
SO=$(EXT_MODULE).so

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
# Wanny try OpenMP? Then uncomment *_OMP_F90_FLAGS.

# gfortran
F90=gfortran
F90FLAGS=-x f95-cpp-input 
ARCH=-mmmx -msse4.2 
OMP_F90_FLAGS=-fopenmp -D__OPENMP
F2PY_OMP_F90_FLAGS=-lgomp

# ifort 11.1
##F90=ifort
##F90FLAGS=-fpp
##ARCH=-xHost
##OMP_F90_FLAGS=-openmp -D__OPENMP 
##F2PY_OMP_F90_FLAGS=-liomp5

# no OpenMP
##OMP_F90_FLAGS=
##F2PY_OMP_F90_FLAGS=

# f2py stuff
#
# numpy.distutils has default -03 for fcompiler. --f90flags="-02" does NOT
# override this. We get "-O3 -O2" and a compiler warning. We have to use f2py's
# --opt= flag.
F2PY_FLAGS=--opt='-O2' \
			--f90exec=$(F90) \
			--arch="$(ARCH)" \
			--f90flags="$(F90FLAGS) $(OMP_F90_FLAGS)" \
			$(F2PY_OMP_F90_FLAGS) \
##			-DF2PY_REPORT_ON_ARRAY_COPY=1 \

#--- targets ----------------------------------------------------------------

all: $(SO)

pyf: $(PYF)

clean:
	rm -f $(SO) $(PYF)

#--- internal targets -------------------------------------------------------

# Make .pyf file and overwrite old one. We could also do
#   f2py -c $(FORT) -m $(FILE)
# in one run to create the extension module. But this woudn't keep the .pyf
# file around.  
$(PYF): $(FORT)
	$(F2PY) -h $(PYF) $(FORT) -m $(EXT_MODULE) --overwrite-signature

# make shared lib 
$(SO): $(PYF) $(FORT)
	$(F2PY) -c $(PYF) $(FORT) $(F2PY_FLAGS)
