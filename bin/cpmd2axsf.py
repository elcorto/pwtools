#!/usr/bin/python

# Parse CPMD MD output file and write .axsf file for xcrysden/VMD.
# 
# usage:
#   cpmd2axsf.py cpmd.out cpmd.axsf

# This script (and the used parsing classes) have been tested extensively, but
# you should nevertheless verify the output by yourself!

import sys
import optparse
from textwrap import dedent as dd
from pwtools import parse, constants, io, verbose, crys, common
verbose.VERBOSE = False

def parse_repeat(rep):
    return [int(x) for x in rep.split(',')]

parser = optparse.OptionParser(description=dd("""\
    Convert CPMD MD output to animated XSF."""),
    usage="%prog [options] cpmd.out cpmd.axsf")
parser.add_option("-r", "--repeat", default="1,1,1",
    help=dd("""\
    build nx,ny,nz supercell of the trajectory
    [%default]"""))
parser.add_option("-f", "--force-scaling", default=1.0, type="float",
    help=dd("""\
    Forces scaling factor.
    [%default]"""))
parser.add_option("-t", "--timeslice", default=':',
    help=dd("""\
    Slice for time axis, e.g. '2000:'
    [%default]"""))
opts, args = parser.parse_args()

timeslice = common.toslice(opts.timeslice)
parse_class = parse.CpmdMDOutputFile
filename = args[0]
outfile = args[1]
ppout = parse_class(filename)

#               CPMD        XSF         
# cell          Bohr        Ang
# cart. forces  Ha/Bohr     Ha/Ang

# Use ppout.get_*() to parse only what we need.
print "parsing ..."
coords_frac = ppout.get_coords_frac()
symbols = ppout.get_symbols()
forces = ppout.get_forces()
cell = ppout.get_cell() * constants.a0_to_A
if forces is not None:
    forces /= constants.a0_to_A
    assert coords_frac.shape == forces.shape
if cell.ndim == 3:
    assert cell.shape[-1] == coords_frac.shape[-1]
nstep = coords_frac.shape[-1]    
print "... ready"

repeat = parse_repeat(opts.repeat)
if repeat != [1,1,1]:
    sc = crys.scell3d(coords_frac, cell, repeat, symbols)
    coords_frac = sc['coords']
    cell = sc['cell']
    symbols = sc['symbols']
    if forces is not None:
        sc = crys.scell3d(forces, cell, repeat)
        forces = sc['coords']

if forces is not None:
    _forces = forces[...,timeslice] * opts.force_scaling
if cell.ndim == 3:
    _cell = cell[...,timeslice]
else:
    _cell = cell

print "writing ..."
io.write_axsf(filename=outfile, 
              coords=coords_frac[...,timeslice],
              cell=_cell,
              forces=_forces,
              symbols=symbols)
print "... ready"
