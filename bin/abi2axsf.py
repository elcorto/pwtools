#!/usr/bin/python

# Parse abinit MD-like output file and write .axsf file for xcrysden/VMD.
# 
# usage:
#   abi2axsf.py abi.out abi.axsf

# This script (and the used parsing classes) have been tested extensively, but
# you should nevertheless verify the output by yourself!

import sys
import optparse
from textwrap import dedent as dd
from pwtools import parse, crys, constants, io, verbose
verbose.VERBOSE = False

parser = optparse.OptionParser(description=dd("""\
    Convert Abinit MD-like output to animated XSF."""),
    usage="%prog [options] abi.out abi.axsf")
parser.add_option("-t", "--mdtype", default='vcmd',
    help=dd("""\
    type of MD: vcmd (ionmov 13 + optcell 0,1,2),
    md (ionmov 2 + optcell 0,1,2; ionmov 8)
    [%default]"""))
opts, args = parser.parse_args()

filename = args[0]
outfile = args[1]
if opts.mdtype == 'vcmd':
    parse_class = parse.AbinitVCMDOutputFile
elif opts.mdtype == 'md':     
    parse_class = parse.AbinitMDOutputFile
else:
    raise StandardError("illegal mdtype: %s" %repr(opts.mdtype))
ppout = parse_class(filename)

#               Abinit      XSF         
# cell          Bohr        Ang
# cart. forces  Ha/Bohr     Ha/Ang

# Use ppout.get_*() to parse only what we need.
print "parsing ..."
coords_frac = ppout.get_coords_frac()
symbols = ppout.get_symbols()
forces = ppout.get_forces()
if forces is not None:
    forces /= constants.a0_to_A

# For variable cell MD-like output, cell.shape = (3,3,nstep). If optcell=0,
# then no cell info is printed at every time step. In that case, cell.shape =
# (3,3,1), i.e. only the start cell is parsed. It may also happen that cell is
# None. In that case, try using AbinitSCFOutputFile. Then `cell` is 2d, i.e.
# fixed cell over the trajectory, which io.write_axsf() can handle.
cell = ppout.get_cell()
if cell is not None:
    if cell.shape == (3,3,1):
        cell = cell[...,0]
else:
    ppin = parse.AbinitSCFOutputFile(outfile)
    cell = ppin.get_cell()
    if cell is None:
        raise StandardError("cannot determine `cell` from '%s'" %outfile)
    assert cell.shape == (3,3), ("cell obtained from AbinitSCFOutputFile "
                                  "doesn't have shape (3,3)")        
cell *= constants.a0_to_A 
print "... ready"

print "writing ..."
io.write_axsf(filename=outfile, 
              coords_frac=coords_frac,
              cell=cell,
              forces=forces,
              symbols=symbols)
print "... ready"
