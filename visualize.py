from tempfile import mkstemp
import os
from pwtools import io, common

def _set_dummy_symbols(obj):
    if obj.symbols is None:
        print("object has no symbols, setting all to 'H'")
        obj.symbols = ['H']*obj.natoms

def view_xcrysden(obj, logfile=None, structfile=None, disp=False,
                  keepfiles=False):
    """
    View Structure or Trajectory as AXSF file interactively with xcrysden.

    args:
    -----
    obj : Structure or Trajectory
    logfile : str, optional
        Filename of a logfile for xcrysden's text output.
    structfile : str, optional
        Filename of a file to write the structure to.
    disp : bool
        Display text output (i.e. `logfile`'s content).
    keepfiles : bool
        Keep `structfile` and `logfile` on disk.
    """
    _set_dummy_symbols(obj)
    if structfile is None:
        fd1,structfile = mkstemp(dir='/tmp',
                                 prefix='pwtools_view_xcrysden_struct_',
                                 suffix='.axsf')

    if logfile is None:
        fd2,logfile = mkstemp(dir='/tmp', 
                              prefix='pwtools_view_xcrysden_log_',
                              suffix='.log')
    io.write_axsf(structfile, obj)
    if disp:
        cmd = "xcrysden --axsf %s 2>&1 | tee %s" %(structfile, logfile)
    else:       
        cmd = "xcrysden --axsf %s > %s 2>&1" %(structfile, logfile)
    common.system(cmd)
    if not keepfiles:
        os.unlink(structfile)
        os.unlink(logfile)

def view_jmol(obj, logfile=None, structfile=None, disp=False,
              keepfiles=False):
    """
    View Structure interactively with jmol.

    args:
    -----
    obj : Structure
    logfile : str, optional
        Filename of a logfile for xcrysden's text output.
    structfile : str, optional
        Filename of a file to write the structure to.
    disp : bool
        Display text output (i.e. `logfile`'s content).
    keepfiles : bool
        Keep `structfile` an `logfile` on disk.
    """
    assert obj.is_struct, ("input is not Structure instance")
    _set_dummy_symbols(obj)
    if structfile is None:
        fd1,structfile = mkstemp(dir='/tmp',
                                 prefix='pwtools_view_jmol_struct_',
                                 suffix='.cif')
    if logfile is None:
        fd2,logfile = mkstemp(dir='/tmp', 
                              prefix='pwtools_view_jmol_log_',
                              suffix='.log')
    io.write_cif(structfile, obj)
    if disp:
        cmd = "jmol %s 2>&1 | tee %s" %(structfile, logfile)
    else:       
        cmd = "jmol %s > %s 2>&1" %(structfile, logfile)
    common.system(cmd)
    if not keepfiles:
        os.unlink(structfile)
        os.unlink(logfile)

def view_avogadro(obj, logfile=None, structfile=None, disp=False,
                  keepfiles=False):
    """
    View Structure interactively with avogadro.

    args:
    -----
    obj : Structure
    logfile : str, optional
        Filename of a logfile for xcrysden's text output.
    structfile : str, optional
        Filename of a file to write the structure to.
    disp : bool
        Display text output (i.e. `logfile`'s content).
    keepfiles : bool
        Keep `structfile` an `logfile` on disk.
    """
    assert obj.is_struct, ("input is not Structure instance")
    _set_dummy_symbols(obj)
    if structfile is None:
        fd1,structfile = mkstemp(dir='/tmp',
                                 prefix='pwtools_view_avogadro_struct_',
                                 suffix='.cif')
    if logfile is None:
        fd2,logfile = mkstemp(dir='/tmp', 
                              prefix='pwtools_view_avogadro_log_',
                              suffix='.log')
    io.write_cif(structfile, obj)
    if disp:
        cmd = "avogadro %s 2>&1 | tee %s" %(structfile, logfile)
    else:       
        cmd = "avogadro %s > %s 2>&1" %(structfile, logfile)
    common.system(cmd)
    if not keepfiles:
        os.unlink(structfile)
        os.unlink(logfile)
