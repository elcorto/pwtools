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
# another compiler, e.g.  gfortran, modify F2PY_FLAGS to use
# "--fcompiler=gnu95".    
# 
# There is also a pythonic way to build the extension: setup.py (i.e. using
# Python's distutils).  See
#     
#     $ python setup.py help 
#     $ python setup.py build --help-fcompiler
#
# and comments in setup.py .

#--- variables ---------------------------------------------------------------

# files
FILE=flib
FORT=$(FILE).f90
PYF=$(FILE).pyf
EXE=$(FILE).x
EXT_MODULE=_$(FILE)
SO=$(EXT_MODULE).so

# f2py stuff
#
# numpy.distutils has default -03 for fcompiler. --f90flags="-02" does NOT
# override this. We get "-O3 -O2" and a compiler warning.
F2PY_FLAGS=--opt='-O2' -DF2PY_REPORT_ON_ARRAY_COPY=1 # -DF2PY_REPORT_ATEXIT

# compiler 
#
# It is NOT needed for buiding the extension. No need to change that unless you
# want to compile $(EXE).
F90=ifort

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
