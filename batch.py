import os
import shutil
import numpy as np
from pwtools import common
from pwtools.sql import SQLEntry, SQLiteDB
from pwtools.verbose import verbose
pj = os.path.join

class Machine(object):
    """This is a container for machine-specific stuff. Most of the
    machine-specific settings can (and should) be placed in the corresponding
    job template file. But some settings need to be in sync between files (e.g.
    outdir (pw.in) and scratch (job.*). For this stuff, we use this class.
    
    Useful to predefine commonly used machines.

    Note that the template file for the jobfile is not handled by this class.
    This must be done outside by FileTemplate. We only provide a method to
    store the jobfile name (`jobfn`).

    methods:
    --------
    get_sql_record() : Return a dict of SQLEntry instances. Each key is a
        attr name from self.attr_lst.        
    """
    def __init__(self, name=None, subcmd=None, scratch=None, bindir=None, 
                 pseudodir=None, jobfn=None, home=None):
        """
        args:
        -----
        name : st
            machine name ('mars', 'local', ...)
        subcmd : str
            shell command to submit jobfiles (e.g. 'bsub <', 'qsub')
        scratch : str
            scratch dir
        bindir : str
            dir where binaries live
        pseudodir : str
            esp. for pwscf: dir where pseudopotentials live
        jobfn : str
            basename of jobfile, can be used as FileTemplate(basename=jobfn)
        home : str
            $HOME
        """
        # attr_lst
        self.name = name
        self.subcmd = subcmd
        self.scratch = scratch
        self.bindir = bindir
        self.pseudodir = pseudodir
        self.home = home
        # extra
        if os.sep in jobfn:
            raise StandardError("Path separator in `jobfn`: '%s', "
                  "this should be a basename." %jobfn)
        self.jobfn = jobfn
                
        self.attr_lst = ['name',
                         'subcmd', 
                         'scratch',
                         'bindir',
                         'pseudodir',
                         'jobfn',
                         'home'
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
    ...                      keys=['prefix', 'ecutwfc'],
    ...                      templ_dir='calc.templ',
    ...                      func=lambda x: "@%s@" %x)
    >>> dct = {}
    >>> dct['prefix'] = 'foo_run_1'
    >>> dct['ecutwfc'] = 23.0
    >>> templ.write(dct, 'calc/0')
    #
    # Not specifying keys will instruct write() to replace all placeholders in
    # the template file which match the placeholders defined by dct.keys().
    #
    >>> templ2 = FileTemplate(basename='pw.in',
    ...                       templ_dir='calc.templ', 
    ...                       func=lambda x: "@%s@" %x)
    >>> templ2.write(dct, 'calc/0')
    #
    # or with SQL foo in a parameter study
    #
    >>> from sql import SQLEntry
    >>> dct = {}                     
    >>> dct['prefix']  = SQLEntry(sqltype='text',  sqlval='foo_run_1')
    >>> sct['ecutwfc'] = SQLEntry(sqltype='float', sqlval=23.0)
    >>> templ2.writesql(dct, 'calc/0')
    """
    
    def __init__(self, basename='pw.in', keys=None, templ_dir='calc.templ',
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
        templ_dir : dir where the template lives (e.g. calc.templ)
        func : callable
            A function which takes a string (key) and returns a string, which
            is the placeholder corresponding to that key.
            example: (this is actually default)
                key = "lala"
                placeholder = "XXXLALA"
                func = lambda x: "XXX" + x.upper()
        """
        self.keys = keys
        self.templ_dir = templ_dir
        
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
        self.filename = pj(self.templ_dir, self.basename)
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


class Calculation(object):
    """A single calculation, e.g. in dir calc_mars/0/ .
    
    methods:
    --------
    get_sql_record : Return a dict of SQLEntry instances. Each key is a
        candidate placeholder for the FileTemplates.
    write_input : For each template in `templates`, write an input file to
        `calc_dir`.

    notes:
    ------
    The dir where file templates live is defined in the FileTemplates (usually
    'calc.templ').
    
    Default keys in sql_record:
        'idx' : self.idx
        'prefix' : self.prefix
    """
    # XXX ATM, the `params` argument is a list of SQLEntry instances which is
    # converted to a dict self.sql_record. This is fine in the context of
    # ParameterStudy (esp. with helper functions like sql.sql_column()) and
    # also necessary b/c ParameterStudy must know about sqlite types. (The fact
    # that it is a list rather then a dict in the first place is that the
    # parameter set usually comes from comb.nested_loops(), which returns
    # lists.)
    #
    # But this may be overly complicated if this class is used alone, where a
    # single Calculation will not write a sqlite database, just deal with
    # `templates`. Then a simple dict (without SQLEntry instances) to pass in
    # the params might do it. Keep in mind that in this situation, Calculation
    # should have no kowledge of sql at all. This is kept strictly in
    # ParameterStudy. Maybe allow `params` to be either a dict or a list of
    # SQLEntry instances. In the dict case, let get_sql_record() raise a
    # warning.
    def __init__(self, machine, templates, params, prefix='calc',
                 idx=0):
        """
        args:
        -----
        machine : instance of batch.Machine
            The get_sql_record() method is used to add machine-specific
            parameters to the FileTemplates.
        templates : dict 
            Dict of FileTemplate instances. 
        params : sequence of SQLEntry instances
            A single "parameter set". The `key` attribute (=sql column name) of
            each SQLEntry will be converted to a placeholder in each
            FileTemplate and an attempt to replacement in the template files is
            made.
        prefix : str, optional
            Unique string identifying this calculation.
        idx : int, optional
            The number of this calculation. Useful in ParameterStudy.
        """
        self.machine = machine
        self.templates = templates
        self.params = params
        self.prefix = prefix
        self.idx = idx
        
        self.sql_record = {}
        self.sql_record['idx']      = SQLEntry('integer'  , self.idx)
        self.sql_record['prefix']   = SQLEntry('text'     , self.prefix)
        self.sql_record.update(self.machine.get_sql_record())
        for entry in self.params:
            self.sql_record[entry.key] = entry
    
    def get_sql_record(self):
        return self.sql_record

    def write_input(self, calc_dir):
        """
        args:
        ----
        calc_dir : str
            Calculation directory to which input files are written.
        """            
        if not os.path.exists(calc_dir):
            os.makedirs(calc_dir)
        for templ in self.templates.itervalues():
            templ.writesql(self.sql_record, calc_dir)


class ParameterStudy(object):
    """Class to represent a (pwscf) parameter study, i.e. a number of
    Calculations.
    
    methods:
    --------
    write_input : Create calculation dir(s) for each parameter set and write
        input files. Write sqlite database storing all relevant parameters.
        Write (bash) shell script to start all calculations (run locally or
        submitt batch job file, depending on machine.subcmd).
    
    notes:
    ------
    The basic idea is to assemble all to-be-varied parameters in a script
    outside (`params_lst`) and pass these to this class along with a list
    of input and job file `templates`. Then, a simple loop over the parameter
    sets is done and input files are written. 
    
    Calculation dirs are numbered automatically. The default is
    (write_input()):
        calc_dir = <root>/<calc_dir_prefix>_<machine>
    and each calculation for each parameter set
        calc_dir/0
        calc_dir/1
        calc_dir/2
        ...
    
    A sqlite database calc_dir/<db_name> is written. If this class operates
    on a calc_dir where such a database already exists, then new
    calculations are added and the numbering of calc dirs continues at the
    end. 
    
    Rationale:
    B/c the pattern in which (any number of) parameters will be varied may be
    arbitrary complex, it is up to the user to prepare the parameter sets. Each
    calculation (each parameter set) gets its own dir. Calculations should be
    simply numbered. No fancy naming conventions. Parameters (and results) can
    then be extracted using SQL in any number of ways. Especially wenn adding
    calculations later to an already performed study, we just extend the sqlite
    database.
    
    example:
    --------
    See test/test_parameter_study.py
    """
    def __init__(self, machine, templates, params_lst, prefix='calc',
                 db_name='calc.db', db_table='calc'):
        """                 
        args:
        -----
        machine, templates : see Calculation
        params_lst : list of lists
            The "parameter sets". Each sublist is a set of calculation
            parameters as SQLEntry instances: 
                [[SQLEntry(...), SQLEntry(...), ...], # calc_dir/0
                 [SQLEntry(...), SQLEntry(...), ...], # calc_dir/1
                 ...
                ] 
            For each sublist, a separate calculation dir is created and
            populated with files based on `templates`. The `key` attribute of
            each SQLEntry will be converted to a placeholder in each
            FileTemplate and an attempt to replacement in the template files is
            made. Note: Each sublist (parameter set) is flattened, so that it
            can in fact be a nested list, e.g. params_lst = the result of a
            complex comb.nested_loops() call.
        prefix : str, optional
            Calculation name. From this, the prefix for pw.in files and job
            name etc. will be built. By default, a string "_run<idx>" is
            appended to create a unique name.
        db_name : str, optional
            Basename of the sqlite database.
        db_table : str, optional
            Name of the sqlite database table.
        """            
        self.machine = machine
        self.templates = templates
        self.params_lst = params_lst
        self.prefix = prefix
        self.db_name = db_name
        self.db_table = db_table
    
    def write_input(self, calc_dir=None, root=os.curdir,
                    calc_dir_prefix='calc', backup=True, sleep=2):
        """
        args:
        -----
        calc_dir : str, optional
            Top calculation dir (e.g. 'calc_mars' and each calc in
            'calc_mars/0, ...').
            If None then default is <root>/<calc_dir_prefix>_<machine>/
        root : str, optional
            Root of all dirs.
        calc_dir_prefix : str, optional
            Prefix for the top calculation dir (e.g. 'calc' for 'calc_mars').
        backup : bool, optional
            do a backup of `calc_dir` if it already exists
        sleep : int, optional
            For the script to start (submitt) all jobs: time in seconds for the
            shell sleep(1) commmand.
        """
        # calc_mars/, calc_deimos/, ...
        if calc_dir is None:
            calc_dir = pj(root, calc_dir_prefix + \
                          '_%s' %self.machine.name)
        if os.path.exists(calc_dir):
            if backup:
                common.backup(calc_dir)
        else:        
            os.makedirs(calc_dir)
        dbfn = pj(calc_dir, self.db_name)
        have_new_db = not os.path.exists(dbfn)
        sqldb = SQLiteDB(dbfn, table=self.db_table)
        # max_idx: counter for calc dir numbering
        if have_new_db:
            max_idx = -1
        else:
            if sqldb.has_column('idx'):
                max_idx = sqldb.execute("select max(idx) from %s" \
                %self.db_table).fetchone()[0]
            else:
                raise StandardError("database '%s': table '%s' has no "
                      "column 'idx', don't know how to number calcs"
                      %(self.db_name, self.db_table))
         
        run_txt = "here=$(pwd)\n"
        sql_records = []
        for _idx, params in enumerate(self.params_lst):
            params = common.flatten(params)
            idx = max_idx + _idx + 1
            calc_subdir = pj(calc_dir, str(idx))
            calc = Calculation(machine=self.machine,
                               templates=self.templates,
                               params=params,
                               prefix=self.prefix + "_run%i" %idx,
                               idx=idx)
            calc.write_input(calc_subdir)                               
            sql_records.append(calc.get_sql_record())
            run_txt += "cd %i && %s %s && cd $here && sleep %i\n" %(idx,\
                        self.machine.subcmd, self.machine.jobfn, sleep)
        common.file_write(pj(calc_dir, 'run.sh'), run_txt)
        
        record = sql_records[0]
        if have_new_db:
            header = [(key, entry.sqltype) for key, entry in record.iteritems()]
            sqldb.create_table(header)
        else:
            for key, entry in record.iteritems():
                if not sqldb.has_column(key):
                    sqldb.add_column(key, entry.sqltype)
        for record in sql_records:
            sqlvals = ",".join(str(entry.sqlval) for entry in record.itervalues())
            cmd = "insert into %s (%s) values (%s)" %(self.db_table,
                                                      ",".join(record.keys()), 
                                                      sqlvals)
            sqldb.execute(cmd) 
        sqldb.commit()


def sql_column(key, sqltype, lst, sqlval_func=lambda x: x, fileval_func=lambda x: x):
    """Convert `lst` to list of SQLEntry instances of the same column name
    `key` and `sqltype`. 

    See ParameterStudy for applications.
     
    args:
    -----
    key : str
        sql column name
    sqltype : str
        sqlite type
    lst : sequence of arbitrary values, these will be SQLEntry.sqlval
    sqlval_func : callable, optional
        Function to transform each entry lst[i] to SQLEntry.sqlval
        Default is sqlval = lst[i].
    fileval_func : callable, optional
        Function to transform each entry lst[i] to SQLEntry.fileval
        Default is fileval = lst[i].
        example:
            lst[i] = '23'
            fileval = 'value = 23'
            fileval_func = lambda x: "value = %s" %str(x)
    
    example:
    --------
    >>> _vals = [25,50,75]
    >>> vals = sql_column('ecutfwc', 'float', _vals, 
    ...                   fileval_func=lambda x: 'ecutfwc=%s'%x)
    >>> for v in vals:
    ...     print v.key, v.sqltype, v.sqlval, v.fileval
    ecutfwc float 25 ecutfwc=25
    ecutfwc float 50 ecutfwc=50
    ecutfwc float 75 ecutfwc=75
    """
    return [SQLEntry(key=key, 
                     sqltype=sqltype, 
                     sqlval=sqlval_func(x), 
                     fileval=fileval_func(x)) for x in lst]


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
               home='/home/schmerler',
               jobfn='job.sge.adde')

mars = Machine(name='mars',
               subcmd='bsub <',
               scratch='/fastfs/schmerle',
               home='/home/schmerle',
               jobfn='job.lsf.mars')

deimos = Machine(name='deimos',
               subcmd='bsub <',
               scratch='/fastfs/schmerle',
               home='/home/schmerle',
               jobfn='job.lsf.deimos')

local = Machine(name='local',
               subcmd='bash',
               scratch='/tmp',
               home='/home/schmerler',
               jobfn='job.local')

