import os
import shutil
import numpy as np
from pwtools import common
from pwtools.sql import SQLEntry, SQLiteDB
from pwtools.verbose import verbose
pj = os.path.join

class Machine(object):
    """This is a container for machine specific stuff. Most of the
    machine-specific settings can (and should) be placed in the corresponding
    job template file. But some settings need to be in sync between files (e.g.
    outdir (pw.in) and scratch (job.*). For this stuff, we use this class."""
    def __init__(self, name=None, subcmd=None, scratch=None, bindir=None, 
                 pseudodir=None, jobfn=None):
        # attr_lst
        self.name = name
        self.subcmd = subcmd
        self.scratch = scratch
        self.bindir = bindir
        self.pseudodir = pseudodir
        # extra
        self.jobfn = jobfn

        self.jobtempl = FileTemplate(self.jobfn)

        self.attr_lst = ['subcmd', 
                         'scratch',
                         'bindir',
                         'pseudodir',
                         ]

    def get_sql_record(self):
        dct = {}
        for key in self.attr_lst:
            val = getattr(self, key)
            if val is not None:
                dct[key] = SQLEntry(sqltype='TEXT', sqlval=val)
        return dct
    

class FileTemplate(object):
    """Class to represent a template file in parameter studies.
    
    placeholders
    ------------
    Each template file is supposed to contain a number of placeholder strings
    (e.g. XXXFOO or @foo@ or whatever other form). The `dct` passed to
    self.write() is a dict which contains key-value pairs for replacement (e.g.
    {'foo': 1.0, 'bar': 'say-what'}, keys=dct.keys()). Each key is converted to
    a placeholder by `func`.
    
    We use common.template_replace(..., mode='txt'). dict-style placeholders
    like "%(foo)s %(bar)i" will not work.

    example
    -------
    This will take a template file calc.templ/pw.in, replace the placeholders
    "@prefix@" and "@ecutwfc@" with some values and write the file to
    calc/0/pw.in .

    >>> templ = FileTemplate(basename='pw.in',
    >>>                      keys=['prefix', 'ecutwfc'],
    >>>                      dir='calc.templ',
    >>>                      func=lambda x: "@%s@" %x)
    >>>
    >>> dct = {}
    >>> dct['prefix'] = 'foo_run_1'
    >>> dct['ecutwfc'] = 23.0
    >>> templ.write(dct, 'calc/0')
    >>>
    # Not specifying keys will instruct write() to replace all placeholders in
    # the template file which match the placeholders defined by dct.keys().
    >>>
    >>> templ2 = FileTemplate(basename='pw.in',
    >>>                       dir='calc.templ', 
    >>>                       func=lambda x: "@%s@" %x)
    >>> templ2.write(dct, 'calc/0')
    >>>
    # or with SQL foo in a parameter study
    >>>
    >>> from sql import SQLEntry
    >>> dct = {}                     
    >>> dct['prefix']  = SQLEntry(sqltype='text',  sqlval='foo_run_1')
    >>> sct['ecutwfc'] = SQLEntry(sqltype='float', sqlval=23.0)
    >>> templ2.writesql(dct, 'calc/0')
    """
    
    def __init__(self, basename='pw.in', keys=None, dir='calc.templ',
                 func=lambda x:'XXX'+x.upper()):
        """
        args
        ----
        basename : string
            The name of the template file and target file.
            example: basename = pw.in
                template = calc.templ/pw.in
                target   = calc/0/pw.in
        keys : {None, list of strings)
            keys=None: All keys dct.keys() in self.write() are used. This is
                useful if you have a dict holding many keys, whose placeholders
                are spread across files. Then this will just replace every
                match in each file. This is what most people want.
            keys=[<key1>, <key2>, ...] : Each string is a key. Each key is
                connected to a placeholder in the template. See func. This is
                for binding keys to template files, i.e. replace only these
                keys.
            keys=[]: The template file is simply copied to `calc_dir` (see
                self.write()).
        dir : dir where the template lives (e.g. calc.templ)
        func : callable
            A function which takes a string (key) and returns a string, which
            is the placeholder corresponding to that key.
            example: (this is actually default)
                key = "lala"
                placeholder = "XXXLALA"
                func = lambda x: "XXX" + x.upper()
        """
        self.keys = keys
        self.dir = dir
        
        # We hardcode the convention that template and target files live in
        # different dirs and have the same name ("basename") there.
        #   template = <dir>/<basename>
        #   target   = <calc_dir>/<basename>
        # e.g.
        #   template = calc.templ/pw.in
        #   target   = calc/0/pw.in
        # Something like
        #   template = ./pw.in.templ
        #   target   = ./pw.in
        # is not possible.
        self.basename = basename
        self.filename = pj(self.dir, self.basename)
        self.func = func
        
        self._get_placeholder = self.func
        
    def write(self, dct, calc_dir='calc', mode='dct'):
        """Write file self.filename (e.g. calc/0/pw.in) by replacing 
        placeholders in the template (e.g. calc.templ/pw.in).
        
        args:
        -----
        dct : dict 
            key-value pairs, dct.keys() are converted to placeholders
        calc_dir : str
            the dir where to write the target file to
        mode : str, {'dct', 'sql'}
            mode='dct': replacement values are dct[<key>]
            mode='sql': replacement values are dct[<key>].fileval and every
                dct[<key>] is an SQLEntry instance
        """
        assert mode in ['dct', 'sql'], ("Wrong 'mode' kwarg, use 'dct' "
                                        "or 'sql'")
        # copy_only : bypass reading the file and passing the text thru the
        # replacement machinery and getting the text back, unchanged. While
        # this works, it is slower and useless.
        if self.keys == []:
            _keys = None
            txt = None
            copy_only = True
        else:
            if self.keys is None:
                _keys = dct.iterkeys()
                warn_not_found = False
            else:
                _keys = self.keys
                warn_not_found = True
            txt = common.file_read(self.filename)
            copy_only = False
        
        tgt = pj(calc_dir, self.basename)
        verbose("write: %s" %tgt)
        if copy_only:    
            verbose("write: ignoring input, just copying file: %s -> %s"
                    %(self.filename, tgt))
            shutil.copy(self.filename, tgt)
        else:            
            rules = {}
            for key in _keys:
                if mode == 'dct':
                    rules[self._get_placeholder(key)] = dct[key]
                elif mode == 'sql':                    
                    # dct = sql_record, a list of SQLEntry's
                    rules[self._get_placeholder(key)] = dct[key].fileval
                else:
                    raise StandardError("'mode' must be wrong")
            new_txt = common.template_replace(txt, 
                                              rules, 
                                              mode='txt',
                                              conv=True,
                                              warn_not_found=warn_not_found)
            common.file_write(tgt, new_txt) 
                                  
    def writesql(self, sql_record, calc_dir='calc'):
        self.write(sql_record, calc_dir=calc_dir, mode='sql')



def conv_table(xx, yy, ffmt="%15.4f", sfmt="%15s"):
    """Convergence table. Assume that quantity `xx` was varied, resulting in
    `yy` values. Return a string (table) listing x,y and diff(y), where
    each row of diff(y) is the difference to the *next* row.
    
    Useful for quickly viewing the results of a convergence study.

    args:
    -----
    xx : 1d sequence (need not be numpy array)
    yy : 1d sequence, len(xx), must be convertable to numpy float array
    ffmt, sfmt : str
        Format strings for strings and floats
    
    example:
    --------
    >>> kpoints = ['2 2 2', '4 4 4', '8 8 8']
    >>> etot = [-300.0, -310.0, -312.0]
    >>> print conv_table(kpoints, etot)
          2 2 2      -300.0000       -10.0000
          4 4 4      -310.0000        -2.0000
          8 8 8      -312.0000         0.0000
    """
    yy = np.asarray(yy, dtype=np.float)
    lenxx = len(xx)
    dyy = np.zeros((lenxx,), dtype=np.float)
    dyy[:-1] = np.diff(yy)
    st = ""
    fmtstr = "%s%s%s\n" %(sfmt,ffmt,ffmt)
    for idx in range(lenxx):
        st += fmtstr %(xx[idx], yy[idx], dyy[idx])
    return st


# Settings for the machines which we frequently use.
adde = Machine(name='adde',
               subcmd='qsub',
               scratch='/local/scratch/schmerler',
               bindir='/home/schmerler/soft/lib/espresso/current/bin',
               pseudodir='/home/schmerler/soft/lib/espresso/pseudo/pseudo_espresso',
               jobfn='job.sge.adde')

mars = Machine(name='mars',
               subcmd='bsub <',
               scratch='/fastfs/schmerle',
               bindir='/home/schmerle/mars/soft/lib/espresso/current/bin',
               pseudodir='/home/schmerle/soft/lib/espresso/pseudo/pseudo_espresso',
               jobfn='job.lsf.mars')

deimos = Machine(name='deimos',
               subcmd='bsub <',
               scratch='/fastfs/schmerle',
               bindir='/home/schmerle/deimos/soft/lib/espresso/current/bin',
               pseudodir='/home/schmerle/soft/lib/espresso/pseudo/pseudo_espresso',
               jobfn='job.lsf.deimos')

local = Machine(name='local',
               subcmd='bash',
               scratch='/tmp',
               bindir='/home/schmerler/soft/lib/espresso/current/bin',
               pseudodir='/home/schmerler/soft/lib/espresso/pseudo/pseudo_espresso',
               jobfn='job.local')

