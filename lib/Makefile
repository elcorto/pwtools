# vim:ts=4:sw=4:noet
	    
# 
# Copyright (c) 2008, Steve Schmerler <mefx@gmx.net>.
# The pydos package. 
# 

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
# 1.1.0 with f2py 2_5237). 
# 
# Just try 
#     
#     $ make
# 
# It should result in a file "_flib.so".
# No Fortran compiler is explicitly named. f2py tries to find one on your
# system.
# 
# To see if f2py (numpy.distutils actually) picks up the correct compiler or if
# you want to specify a different one, try
# 
#     $ f2py -c --help-fcompiler
# 
# On my system, I get
# 
#     $ f2py -c --help-fcompiler
#     IntelEM64TFCompiler instance properties:
#       archiver        = ['/opt/intel/fce/9.1.036/bin/ifort', '-cr']
#       compile_switch  = '-c'
#       compiler_f77    = ['/opt/intel/fce/9.1.036/bin/ifort', '-FI', '-w90', '-
#                         w95', '-KPIC', '-cm', '-O3', '-unroll']
#       compiler_f90    = ['/opt/intel/fce/9.1.036/bin/ifort', '-FR', '-KPIC', '-
#                         cm', '-O3', '-unroll']
#       compiler_fix    = ['/opt/intel/fce/9.1.036/bin/ifort', '-FI', '-KPIC', '-
#                         cm', '-O3', '-unroll']
#       libraries       = []
#       library_dirs    = []
#       linker_exe      = None
#       linker_so       = ['/opt/intel/fce/9.1.036/bin/ifort', '-shared', '-
#                         shared', '-nofor_main']
#       object_switch   = '-o '
#       ranlib          = ['/opt/intel/fce/9.1.036/bin/ifort']
#       version         = LooseVersion ('9.1')
#       version_cmd     = ['/opt/intel/fce/9.1.036/bin/ifort', '-FI', '-V', '-c',
#                         '/tmp/tmp1s5-5d/ZDLAG6.f', '-o', '/tmp/tmp1s5-
#                         5d/ZDLAG6.o']
#     Fortran compilers found:
#       --fcompiler=intelem  Intel Fortran Compiler for EM64T-based apps (9.1)
#     Compilers available for this platform, but not found:
#       --fcompiler=absoft  Absoft Corp Fortran Compiler
#       --fcompiler=compaq  Compaq Fortran Compiler
#       --fcompiler=g95     G95 Fortran Compiler
#       --fcompiler=gnu     GNU Fortran 77 compiler
#       --fcompiler=gnu95   GNU Fortran 95 compiler
#       --fcompiler=intel   Intel Fortran Compiler for 32-bit apps
#       --fcompiler=intele  Intel Fortran Compiler for Itanium apps
#       --fcompiler=lahey   Lahey/Fujitsu Fortran 95 Compiler
#       --fcompiler=nag     NAGWare Fortran 95 Compiler
#       --fcompiler=pg      Portland Group Fortran Compiler
#       --fcompiler=vast    Pacific-Sierra Research Fortran 90 Compiler
#     Compilers not available on this platform:
#       --fcompiler=hpux     HP Fortran 90 Compiler
#       --fcompiler=ibm      IBM XL Fortran Compiler
#       --fcompiler=intelev  Intel Visual Fortran Compiler for Itanium apps
#       --fcompiler=intelv   Intel Visual Fortran Compiler for 32-bit apps
#       --fcompiler=mips     MIPSpro Fortran Compiler
#       --fcompiler=none     Fake Fortran compiler
#       --fcompiler=sun      Sun or Forte Fortran 95 Compiler
#     For compiler details, run 'config_fc --verbose' setup command.
# 
# So it has correctly found my ifort 9.1 and uses that by default. If you want
# another compiler, e.g. gfortran, modify F2PY_FLAGS to use
# --fcompiler=gnu95 or set --f90exec=/usr/bin/gfortran directly.     
# 
# There is also a pythonic way to build the extension: setup.py (i.e. using
# Python's distutils).  See
#     
#     $ python setup.py help 
#     $ python setup.py build --help-fcompiler
#
# and comments in setup.py .
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
# 	$ pydos.py ...
# 
# does work. If this env var is NOT set, then OpenMP uses all available cores
# (e.g. 4 on a quad-core box).
# 
# IMPORTANT: 
# 	Note that we may have found an f2py bug (see test/test_f2py_flib_openmp.py)
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
EXE=$(FILE).x
EXT_MODULE=_$(FILE)
SO=$(EXT_MODULE).so

# ARCH below is for Intel Core i7
F90=gfortran
F90FLAGS=-x f95-cpp-input 
ARCH=-mmmx -msse4.2 
#
##F90=ifort
##F90FLAGS=-fpp
##ARCH=-xSSE4.2

# Wanny try OpenMP? Then uncomment below.
#
# gfortran
OMP_F90_FLAGS=-fopenmp -D__OPENMP
F2PY_OMP_F90_FLAGS=-lgomp
#
# ifort 11.1
##OMP_F90_FLAGS=-openmp -D__OPENMP 
##F2PY_OMP_F90_FLAGS=-liomp5
#
# no OpenMP
##OMP_F90_FLAGS=
##F2PY_OMP_F90_FLAGS=

# f2py stuff
#
# numpy.distutils has default -03 for fcompiler. --f90flags="-02" does NOT
# override this. We get "-O3 -O2" and a compiler warning. We have to use f2py's
# --opt= flag.
# 
# On cartman (AMD X4 Phenom), numpy.distutils falsely sets "-march=k6-2".
# So we set f2py's --arch flag manually.
F2PY_FLAGS=--opt='-O3' \
			-DF2PY_REPORT_ON_ARRAY_COPY=1 \
			--f90exec=$(F90) \
			--arch="$(ARCH)" \
			--f90flags="$(F90FLAGS) $(OMP_F90_FLAGS)" $(F2PY_OMP_F90_FLAGS)

#--- targets ----------------------------------------------------------------

all: $(SO)

pyf: $(PYF)

clean:
	rm -f $(SO) $(PYF)

main: $(EXE)

#--- internal targets -------------------------------------------------------

# Make .pyf file and overwrite old one. We could also do
#   f2py -c $(FORT) -m $(FILE)
# in one run to create the extension module. But this woudn't keep the .pyf
# file around.  
$(PYF): $(FORT)
	f2py -h $(PYF) $(FORT) -m $(EXT_MODULE) --overwrite-signature

# make shared lib 
$(SO): $(PYF) $(FORT)
	f2py -c $(PYF) $(FORT) $(F2PY_FLAGS)

# In case we have a Fortran main program (e.g. "program main") in the source of
# the extension for testing.
$(EXE): $(FORT)
	$(F90) -O2 -o $(EXE) $(FORT)
