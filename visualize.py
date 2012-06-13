import tempfile, os
from pwtools import io, common

def view_xcrysden(obj, logfile='/tmp/pwtools_view_xcrysden.log', disp=False):
    """
    View Structure or Trajectory as AXSF file interactively with xcrysden.

    args:
    -----
    obj : Structure or Trajectory
    logfile : str
        Filename of a logfile for xcrysden's text output.
    disp : bool
        Display text output.
    """
    fd,fn = tempfile.mkstemp(dir='/tmp', prefix='pwtools_view_xcrysden')
    io.write_axsf(fn, obj)
    if disp:
        cmd = "xcrysden --axsf %s 2>&1 | tee %s" %(fn, logfile)
    else:       
        cmd = "xcrysden --axsf %s > %s 2>&1" %(fn, logfile)
    common.system(cmd)
    os.unlink(fn)
