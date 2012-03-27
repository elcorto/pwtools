#!/usr/bin/python

# Parse CPMD MD output file and write .axsf file for xcrysden/VMD.
#
# This script (and the used parsing classes) have been tested extensively, but
# you should nevertheless verify the output by yourself!

import sys
import optparse
from textwrap import dedent as dd
from pwtools import parse, io, verbose, crys, common
from pwtools.constants import Bohr, Angstrom
from pwtools.crys import Trajectory
verbose.VERBOSE = False

def parse_repeat(rep):
    return [int(x) for x in rep.split(',')]

outfile_default = 'pwtools.axsf'
parser = optparse.OptionParser(description=dd("""\
    Convert CPMD MD output to animated XSF."""),
    usage="""%prog [options] cpmd.out [cpmd.axsf]
    
args:
-----
cpmd.out : CPMD output file (cpmd.x ... > cpmd.out)
cpmd.axsf : name of output file [default: pwtools.axsf]
""")
parser.add_option("-r", "--repeat", default="1,1,1",
    help=dd("""\
    build nx,ny,nz supercell of the trajectory
    [%default]"""))
parser.add_option("-f", "--force-scaling", default=1.0, type="float",
    help=dd("""\
    Forces scaling factor. Can be used to scale the length of force vectors in
    visualization (e.g. xcrysden) if forces are all big (or small).
    [%default]"""))
parser.add_option("-t", "--timeslice", default=':',
    help=dd("""\
    Slice for time axis, e.g. '2000:' = from step 2000 to end
    [%default]"""))
opts, args = parser.parse_args()

# units
# -----
#               CPMD        XSF         
# cell          Bohr        Ang
# cart. forces  Ha/Bohr     Ha/Ang

timeslice = common.toslice(opts.timeslice)
filename = args[0]
outfile = args[1] if len(args) == 2 else outfile_default
pp = parse.CpmdMDOutputFile(filename,
                            units={'length': Bohr/Angstrom,
                                   'forces': Angstrom/Bohr \
                                             * opts.force_scaling})


print "parsing ..."
traj = pp.get_traj()
print "... ready"

repeat = parse_repeat(opts.repeat)
if repeat != [1,1,1]:
    sc = crys.scell3d(traj, tuple(repeat))
else:
    sc = traj

if sc.forces is not None:
    _forces = sc.forces[timeslice,...]
else:
    _forces = None
_cell = sc.cell[timeslice,...]
_coords_frac = sc.coords_frac[timeslice,...]

print "writing ..."
io.write_axsf(outfile, Trajectory(coords_frac=_coords_frac,
                                  cell=_cell,
                                  symbols=sc.symbols,
                                  forces=_forces))
print "... ready"
