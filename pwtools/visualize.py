"""
Interfaces for molecular viewers
================================

The viewers defined here are meant to be used for quick interactive structure
and trajectory display. All viewer functions are actually callable instances of
:class:`ViewFactory`, and thus have the same call signature:

.. automethod:: ViewFactory.__call__

VMD
---

Use :func:`view_vmd_xyz` or :func:`view_vmd_axsf`. To execute a tcl script
after the struct has loaded (in the shell: ``vmd -e script.tcl foo.xyz``), use::

>>> tr = crys.Trajectory(...)
>>> view_vmd_axsf(tr, options='-e script.tcl')

Note that the viewer (in this case VMD) is simply called like ``vmd
structfile``, which can take very long for big MD data b/c the VMD default
is to use smth like ``mol new ... waitfor 1``, which is slow. In that case you
want to call VMD directly::

    $ vmd
    vmd > mol new /tmp/foo.axsf type xsf waitfor all
    vmd > set molid 0
    vmd > source ~/work/vmd/ca_salt.tcl

The trick here is ``... waitfor all``. See the VMD manual for the ``mol``
command. Or you place these lines in a script and use::

    $ vmd script_to_execute.tcl
"""

from tempfile import mkstemp
import os
from pwtools import io, common


class ViewFactory(object):
    """Factory for creating interface functions to external molecular
    viewers."""
    def __init__(self, cmd=None, assert_cmd=None, suffix='.axsf',
                 writer=io.write_axsf):
        """
        Parameters
        ----------
        cmd : str
            Shell command to call the viewer. Used as ``<cmd>
            <structfile>``. Example: 'jmol', 'xcrysden --axsf'.
        assert_cmd : callable
            Function which accepts a single arg. Called as ``assert_cmd(obj)``
            where `obj` = Structure or Trajectory instance usually). Will be
            called early. Use to make additional tests on `obj`.
        suffix : str
            File end for written structure file.
        writer : callable
            Called as ``writer(obj, structfile)``. Write struct file to read by
            viewer.
        
        Examples
        --------
        >>> viewer = ViewFactory(...)
        >>> viewer(struct)
        >>> # To start more than one viewer, use bg=True to send the spawned
        >>> # process to the background. Will leave temp files on disk.
        >>> viewer(struct1, bg=True)
        >>> viewer(struct2, bg=True)
        """
        self.cmd = cmd
        self.assert_cmd = assert_cmd
        self.suffix = suffix
        self.writer = writer
    
    def __call__(self, obj, logfile=None, structfile=None, disp=False,
                 keepfiles=False, tmpdir='/tmp', wait=True, bg=False,
                 options=''):
        """
        Call viewer. 
        
        The executed shell command is::
            
            <cmd> <options> <structfile> > <logfile>
        
        Parameters
        ----------
        obj : Structure or Trajectory
        logfile : str, optional
            Filename of a logfile for the viewer's text output.
        structfile : str, optional
            Filename of a file to write the structure to.
        disp : bool
            Display text output (i.e. `logfile`'s content).
        keepfiles : bool
            Keep `structfile` and `logfile` on disk.
        tmpdir : str, optional
            Directory where temp files are written to.
        wait : bool, optional
            `wait` passed to common.system(), wait (or not) for command to exit
        bg : bool
            Background mode. If True then this is an alias for `wait=False` +
            `keepfiles=True`. The latter is needed b/c with just `wait=False`,
            temp files will be deleted right after the shell call and the
            viewer program may complain.
        """        
        if bg:
            wait = False
            keepfiles = True
        if self.assert_cmd is not None:
            self.assert_cmd(obj)    
        self._set_dummy_symbols(obj)
        if structfile is None:
            fd1,structfile = mkstemp(dir=tmpdir,
                                     prefix='pwtools_view_struct_',
                                     suffix=self.suffix)

        if logfile is None:
            fd2,logfile = mkstemp(dir=tmpdir, 
                                  prefix='pwtools_view_log_')
        self.writer(structfile, obj)
        if disp:
            cmd_str = "%s %s %s 2>&1 | tee %s" %(self.cmd, options, structfile, logfile)
        else:       
            cmd_str = "%s %s %s > %s 2>&1" %(self.cmd, options, structfile, logfile)
        common.system(cmd_str, wait=wait)
        if not keepfiles:
            os.unlink(structfile)
            os.unlink(logfile)

    def _set_dummy_symbols(self, obj):
        if obj.symbols is None:
            print("object has no symbols, setting all to 'H'")
            obj.symbols = ['H']*obj.natoms


def assert_struct(obj):
    assert obj.is_struct, ("input is not Structure instance")


view_xcrysden = \
    ViewFactory(cmd='xcrysden --axsf',
                suffix='.axsf',
                writer=io.write_axsf)

view_vmd_axsf = \
    ViewFactory(cmd='vmd',
                suffix='.axsf',
                writer=io.write_axsf)

view_vmd_xyz = \
    ViewFactory(cmd='vmd',
                suffix='.xyz',
                writer=io.write_xyz)

view_jmol = \
    ViewFactory(cmd='jmol',
                suffix='.cif',
                writer=io.write_cif,
                assert_cmd=assert_struct)

view_avogadro = \
    ViewFactory(cmd='avogadro',
                suffix='.cif',
                writer=io.write_cif,
                assert_cmd=assert_struct)


