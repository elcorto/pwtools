import os, copy, shutil, warnings
import numpy as np
from pwtools import common
from pwtools.sql import SQLEntry, SQLiteDB
from pwtools.verbose import verbose
import collections
pj = os.path.join

# backwd compat
from pwtools.sql import sql_column, sql_matrix

class Machine(object):
    """Container for machine-specific stuff.

    Most of the machine-specific settings can (and should) be placed in the
    corresponding job template file. But some settings need to be in sync
    between files like the scratch dir, the $HOME etc.

    Useful to predefine commonly used machines.
    """
    def __init__(self, hostname=None, subcmd=None, scratch=None,
                 home=None, template=None, filename=None):
        """
        Parameters
        ----------
        hostname : st
            machine name ('mars', 'local', ...)
        subcmd : str
            shell command to submit jobfiles (e.g. 'bsub <', 'qsub')
        scratch : str
            scratch dir (like '/scratch')
        home : str
            $HOME
        template : FileTemplate
        filename : str
            full jobfile name for FileTemplate (e.g.
            'calc.templ/job.local'), use either this or `template`
        """
        assert None in [filename, template], ("use either `filename` "
                                              "or `template`")
        self.hostname = hostname
        self.subcmd = subcmd
        self.scratch = scratch
        self.home = home
        self.template = template
        self.filename = filename
        self.attr_lst = ['subcmd',
                         'scratch',
                         'home',
                         ]
        if (self.template is None) and (self.filename is not None):
            self.template = FileTemplate(filename=self.filename)

    def get_jobfile_basename(self):
        if self.template is not None:
            return self.template.basename
        else:
            raise Exception("cannot get job file name")

    def get_sql_record(self):
        """Return a dict of SQLEntry instances. Each key is a attr name from
        self.attr_lst. """
        dct = {}
        for key in self.attr_lst:
            val = getattr(self, key)
            dct[key] = SQLEntry(key=key, sqlval=val)
        return dct


class FileTemplate(object):
    """Class to represent a template file in parameter studies.

    Each template file is supposed to contain a number of placeholder strings
    (e.g. XXXFOO or @foo@ or whatever other form). The `dct` passed to
    self.write() is a dict which contains key-value pairs for replacement (e.g.
    {'foo': 1.0, 'bar': 'say-what'}, keys=dct.keys()). Each key is converted to
    a placeholder by `func`.

    We use common.template_replace(..., mode='txt'). dict-style placeholders
    like "%(foo)s %(bar)i" will not work.

    Examples
    --------
    This will take a template file calc.templ/pw.in, replace the placeholders
    "@prefix@" and "@ecutwfc@" with some values and write the file to
    calc/0/pw.in .

    Fist, set up a dictionary which maps placeholder to values. Remember,
    that the placeholders in the files will be obtained by processing the
    dictionary keys with `func`. In the example, this will be::

        'prefix' -> '@prefix@'
        'ecutwfc' -> '@ecutwfc@'

    ::

        >>> dct = {}
        >>> dct['prefix'] = 'foo_run_1'
        >>> dct['ecutwfc'] = 23.0
        >>>
        >>> # Not specifying the `keys` agrument to FileTemplate will instruct the
        >>> # write() method to replace all placeholders in the template file which
        >>> # match the placeholders defined by dct.keys(). This is the most simple
        >>> # case.
        >>> #
        >>> templ = FileTemplate(filename='calc.templ/pw.in',
        ...                      func=lambda x: "@%s@" %x)
        >>> templ.write(dct, 'calc/0')
        >>>
        >>> # Now with `keys` explicitely.
        >>> templ = FileTemplate(filename='calc.templ/pw.in',
        ...                      keys=['prefix', 'ecutwfc'],
        ...                      func=lambda x: "@%s@" %x)
        >>> templ.write(dct, 'calc/0')
        >>>
        >>> # or with SQL foo in a parameter study
        >>> from sql import SQLEntry
        >>> dct = {}
        >>> dct['prefix']  = SQLEntry(sqlval='foo_run_1')
        >>> sct['ecutwfc'] = SQLEntry(sqlval=23.0)
        >>> templ.writesql(dct, 'calc/0')
    """
    def __init__(self, basename=None, txt=None, keys=None, templ_dir='calc.templ',
                 filename=None, func=lambda x:'XXX'+x.upper()):
        """
        Parameters
        ----------
        basename : string
            | The name of the template file and target file.
            | example: basename = pw.in
            |     template = calc.templ/pw.in
            |     target   = calc/0/pw.in
        txt : string, optional
            Text of the template file. If None, then we assume a file
            ``templ_dir/basename`` and read that.
        keys : {None, list of strings, []}
            | keys=None: All keys dct.keys() in self.write() are used. This is
            |     useful if you have a dict holding many keys, whose placeholders
            |     are spread across files. Then this will just replace every
            |     match in each file. This is what most people want.
            | keys=[<key1>, <key2>, ...] : Each string is a key. Each key is
            |     connected to a placeholder in the template. See func. This is
            |     for binding keys to template files, i.e. replace only these
            |     keys.
            | keys=[]: The template file is simply copied to `calc_dir` (see
            |     self.write()).
        templ_dir : dir where the template lives (e.g. 'calc.templ')
        filename : str
            full file name of the template, then templ_dir=os.path.dirname(filename)
            and basename=os.path.basename(filename)
        func : callable
            | A function which takes a string (key) and returns a string, which
            | is the placeholder corresponding to that key.
            | example: (this is actually default)
            |     key = "lala"
            |     placeholder = "XXXLALA"
            |     func = lambda x: "XXX" + x.upper()
        """
        assert (filename is None) or (basename is None), \
               ("use either `filename` or `templ_dir` + `basename`")
        self.keys = keys
        self.txt = txt
        # We hardcode the convention that template and target files live in
        # different dirs and have the same name ("basename") there.
        #   template = <templ_dir>/<basename>
        #   target   = <calc_dir>/<basename>
        # e.g.
        #   template = calc.templ/pw.in
        #   target   = calc/0/pw.in
        # Something like
        #   template = ./pw.in.templ
        #   target   = ./pw.in
        # is not possible.
        if filename is None:
            self.templ_dir = templ_dir
            self.basename = basename
            self.filename = pj(templ_dir, basename)
        else:
            self.filename = filename
            self.templ_dir = os.path.dirname(filename)
            self.basename = os.path.basename(filename)
        self.func = func

    def write(self, dct, calc_dir='calc', mode='dct'):
        """Write file self.filename (e.g. calc/0/pw.in) by replacing
        placeholders in the template (e.g. calc.templ/pw.in).

        Parameters
        ----------
        dct : dict
            key-value pairs, dct.keys() are converted to placeholders with
            self.func()
        calc_dir : str
            the dir where to write the target file to
        mode : str, {'dct', 'sql'}
            | mode='dct': replacement values are dct[<key>]
            | mode='sql': replacement values are dct[<key>].fileval and every
            |     dct[<key>] is an SQLEntry instance
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
                _keys = dct.keys()
                warn_not_found = False
            else:
                _keys = self.keys
                warn_not_found = True
            if self.txt is None:
                txt = common.file_read(self.filename)
            else:
                txt = self.txt
            copy_only = False

        tgt = pj(calc_dir, self.basename)
        verbose("write: %s" %tgt)
        if copy_only:
            verbose("write: ignoring input, just copying file to %s"
                    %(self.filename, tgt))
            shutil.copy(self.filename, tgt)
        else:
            rules = {}
            for key in _keys:
                if mode == 'dct':
                    rules[self.func(key)] = dct[key]
                elif mode == 'sql':
                    # dct = sql_record, a list of SQLEntry's
                    rules[self.func(key)] = dct[key].fileval
                else:
                    raise Exception("'mode' must be wrong")
            new_txt = common.template_replace(txt,
                                              rules,
                                              mode='txt',
                                              conv=True,
                                              warn_not_found=warn_not_found,
                                              warn_mult_found=False,
                                              disp=False)
            common.file_write(tgt, new_txt)

    def writesql(self, sql_record, calc_dir='calc'):
        self.write(sql_record, calc_dir=calc_dir, mode='sql')


class Calculation(object):
    """Represent a single calculation.

    The calculation data will be placed, e.g., in dir calc_foo/0/. A
    Calculation is bound to one Machine. This class is usually not used on it's
    own, but only by ParameterStudy.

    The dir where file templates live is defined in the FileTemplates (usually
    'calc.templ').
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
    def __init__(self, machine=None, templates=[], params=[],
                 calc_dir='calc_dir'):
        """
        Parameters
        ----------
        machine : Machine, optional
            Used to add machine specific attributes. Pulled from
            Machine.get_sql_record(). Used for FileTemplate replacement.
        templates : sequence
            Sequence of FileTemplate instances.
        params : sequence of SQLEntry instances
            A single "parameter set". The `key` attribute (=sql column name) of
            each SQLEntry will be converted to a placeholder in each
            FileTemplate and an attempt to replacement in the template files is
            made.
        calc_dir : str, optional
            Calculation directory to which input files are written (e.g.
            'calc_foo/0/')
        """
        self.machine = machine
        self.templates = templates
        self.params = params
        self.calc_dir = calc_dir
        self.sql_record = dict((sqlentry.key, sqlentry) \
                                for sqlentry in self.params)
        # add machine-specific stuff only for replacement, doesn't go into
        # database
        self.sql_record_write = copy.deepcopy(self.sql_record)
        if self.machine is not None:
            self.sql_record_write.update(self.machine.get_sql_record())
            # use machine.template if it has one
            if self.machine.template is not None:
                self.templates.append(self.machine.template)

    def get_sql_record(self):
        """Return a dict of SQLEntry instances. Each key is a candidate
        placeholder for the FileTemplates.
        """
        return self.sql_record

    def write_input(self):
        """For each template in ``self.templates``, write an input file to
        ``self.calc_dir``.
        """
        if not os.path.exists(self.calc_dir):
            os.makedirs(self.calc_dir)
        for templ in self.templates:
            templ.writesql(self.sql_record_write, self.calc_dir)


class ParameterStudy(object):
    """Class to represent a parameter study (a number of Calculations,
    based on template files).

    The basic idea is to assemble all to-be-varied parameters in a script
    outside (`params_lst`) and pass these to this class along with a list of
    `templates` files, usually software input files. Then, a simple loop over
    the parameter sets is done and input files are written.

    Calculation dirs are numbered automatically. The default is

        ``calc_dir = <calc_root>/<calc_dir_prefix>_<machine.hostname>``, e.g.
        ``./calc_foo``

    and each calculation for each parameter set

        | ``./calc_foo/0``
        | ``./calc_foo/1``
        | ``./calc_foo/2``
        | ``...``

    A sqlite database ``calc_dir_root/<db_name>`` is written. If this class
    operates on a `calc_dir_root` where such a database already exists, then
    the default is to append new calculations. The numbering of calc dirs
    continues at the end. This can be changed with the `mode` kwarg of
    :meth:`~ParameterStudy.write_input`. By default, a sql column "revision" is
    added which numbers each addition.

    Examples
    --------
    >>> # Here are some examples for constructing `params_lst`.
    >>>
    >>> # Vary two (three, ...) parameters on a 2d (3d, ...) grid: In fact, the
    >>> # way you are constructing params_lst is only a matter of zip() and
    >>> # comb.nested_loops()
    >>> par1 = sql.sql_column('par1', [1,2,3])
    >>> par2 = sql.sql_column('par2', ['a','b'])
    >>> par3 = ...
    >>> # 3d grid
    >>> params_lst = comb.nested_loops([par1, par2, par3])
    >>> # which is the same as:
    >>> params_lst = []
    >>> for par1 in [1,2,3]:
    ...     for par2 in ['a','b']:
    ...         for par3 in [...]:
    ...             params_lst.append([sql.SQLEntry(key='par1', sqlval=par1),
    ...                                sql.SQLEntry(key='par2', sqlval=par2),
    ...                                sql.SQLEntry(key='par3', sqlval=par3),
    ...                                ])
    >>>
    >>> # vary par1 and par2 together, and par3 -> 2d grid w/ par1+par2 on one
    >>> # axis and par3 on the other
    >>> params_lst = comb.nested_loops([zip(par1, par2), par3], flatten=True)

    See ``examples/parameter_study/`` for complete examples, as well as
    ``test/test_parameter_study.py``.

    See Also
    --------
    itertools.product
    :func:`pwtools.comb.nested_loops`
    :func:`pwtools.sql.sql_column`
    """
    # more notes
    # ----------
    # The `params_lst` list of lists is a "matrix" which in fact represents the
    # sqlite database table.
    #
    # The most simple case is when we vary only one parameter::
    #
    #     [[SQLEntry(key='foo', sqlval=1.0)],
    #      [SQLEntry(key='foo', sqlval=2.0)]]
    #
    # The sqlite database would have one column and look like this::
    #
    #     foo
    #     ---
    #     1.0   # calc_foo/0
    #     2.0   # calc_foo/1
    #
    # Note that you have one entry per row [[...], [...]], like in a
    # column vector, b/c "foo" is a *column* in the database and b/c each
    # calculation is represented by one row (record).
    #
    # Another example is a 2x2 setup (vary 2 parameters 'foo' and 'bar')::
    #
    #     [[SQLEntry(key='foo', sqlval=1.0), SQLEntry(key='bar', sqlval='lala')],
    #      [SQLEntry(key='foo', sqlval=2.0), SQLEntry(key='bar', sqlval='huhu')]]
    #
    # Here we have 2 parameters "foo" and "bar" and the sqlite db would
    # thus have two columns::
    #
    #     foo   bar
    #     ---   ---
    #     1.0   lala  # calc_foo/0
    #     2.0   huhu  # calc_foo/1
    #
    # Each row (or record in sqlite) will be one Calculation, getting
    # it's own dir.
    def __init__(self, machines=[], templates=[], params_lst=[], study_name='calc',
                 db_name='calc.db', db_table='calc', calc_root=os.curdir,
                 calc_dir_prefix='calc'):
        """
        Parameters
        ----------
        machines : sequence or Machine
            List of Machine instances or a single Machine
        templates : see Calculation
        params_lst : list of lists
            The "parameter sets". Each sublist is a set of calculation
            parameters as SQLEntry instances::

                [[SQLEntry(...), SQLEntry(...), ...], # calc_foo/0
                 [SQLEntry(...), SQLEntry(...), ...], # calc_foo/1
                 ...
                ]

            For each sublist, a separate calculation dir is created and
            populated with files based on `templates`. The `key` attribute of
            each SQLEntry will be converted to a placeholder in each
            FileTemplate and an attempt to replacement in the template files is
            made. Thus, the way placeholders are created is defined in
            FileTemplate, not here!
            Note: Each sublist (parameter set) is flattened, so that it
            can in fact be a nested list, e.g. params_lst = the result of a
            complex comb.nested_loops() call. Also, sublists need not have the
            same length or `key` attributes per entry ("incomplete parameter
            sets"). The sqlite table header is compiled from all distinct
            `key`s found.
        study_name : str, optional
            Name for the parameter study. From this, the `calc_name` for input
            files and job name etc. will be built. By default, a string
            "_run<idx>" is appended to create a unique name.
        db_name : str, optional
            Basename of the sqlite database.
        db_table : str, optional
            Name of the sqlite database table.
        calc_root : str, optional
            Root of all calc dirs.
        calc_dir_prefix : str, optional
            Prefix for the top calculation dir (e.g. 'calc' for 'calc_foo').
        """
        self.machines = machines if (type([]) == type(machines)) else [machines]
        self.templates = templates
        self.params_lst = params_lst
        self.study_name = study_name
        self.db_name = db_name
        self.db_table = db_table
        self.calc_root = calc_root
        self.calc_dir_prefix = calc_dir_prefix
        self.dbfn = pj(self.calc_root, self.db_name)

    def write_input(self, mode='a', backup=True, sleep=0, excl=True):
        """
        Create calculation dir(s) for each parameter set and write input files
        based on ``templates``. Write sqlite database storing all relevant
        parameters. Write (bash) shell script to start all calculations (run
        locally or submitt batch job file, depending on ``machine.subcmd``).

        Parameters
        ----------
        mode : str, optional
            Fine tune how to write input files (based on ``templates``) to calc
            dirs calc_foo/0/, calc_foo/1/, ... . Note that this doesn't change
            the base dir calc_foo at all, only the subdirs for each calc.
            {'a', 'w'}

            | 'a': Append mode (default). If a previous database is found, then
            |     subsequent calculations are numbered based on the last 'idx'.
            |     calc_foo/0 # old
            |     calc_foo/1 # old
            |     calc_foo/2 # new
            |     calc_foo/3 # new
            | 'w': Write mode. The target dirs are purged and overwritten. Also,
            |     the database (self.dbfn) is overwritten. Use this to
            |     iteratively tune your inputs, NOT for working on already
            |     present results!
            |     calc_foo/0 # new
            |     calc_foo/1 # new
        backup : bool, optional
            Before writing anything, do a backup of self.calc_dir if it already
            exists.
        sleep : int, optional
            For the script to start (submitt) all jobs: time in seconds for the
            shell sleep(1) commmand.
        excl : bool
            If in append mode, a file <calc_root>/excl_push with all indices of
            calculations from old revisions is written. Can be used with
            ``rsync --exclude-from=excl_push`` when pushing appended new
            calculations to a cluster.
        """
        assert mode in ['a', 'w'], "Unknown mode: '%s'" %mode
        if os.path.exists(self.dbfn):
            if backup:
                common.backup(self.dbfn)
            if mode == 'w':
                os.remove(self.dbfn)
        have_new_db = not os.path.exists(self.dbfn)
        common.makedirs(self.calc_root)
        # this call creates a file ``self.dbfn`` if it doesn't exist
        sqldb = SQLiteDB(self.dbfn, table=self.db_table)
        # max_idx: counter for calc dir numbering
        revision = 0
        if have_new_db:
            max_idx = -1
        else:
            if mode == 'a':
                if sqldb.has_column('idx'):
                    max_idx = sqldb.execute("select max(idx) from %s" \
                    %self.db_table).fetchone()[0]
                else:
                    raise Exception("database '%s': table '%s' has no "
                          "column 'idx', don't know how to number calcs"
                          %(self.dbfn, self.db_table))
                if sqldb.has_column('revision'):
                    revision = int(sqldb.get_single("select max(revision) \
                        from %s" %self.db_table)) + 1
            elif mode == 'w':
                max_idx = -1
        sql_records = []
        hostnames = []
        for imach, machine in enumerate(self.machines):
            hostnames.append(machine.hostname)
            calc_dir = pj(self.calc_root, self.calc_dir_prefix + \
                          '_%s' %machine.hostname)
            if os.path.exists(calc_dir):
                if backup:
                    common.backup(calc_dir)
                if mode == 'w':
                    common.system("rm -r %s" %calc_dir, wait=True)
            run_txt = "here=$(pwd)\n"
            for _idx, params in enumerate(self.params_lst):
                params = common.flatten(params)
                idx = max_idx + _idx + 1
                calc_subdir = pj(calc_dir, str(idx))
                extra_dct = \
                    {'revision': revision,
                     'study_name': self.study_name,
                     'idx': idx,
                     'calc_name' : self.study_name + "_run%i" %idx,
                     }
                extra_params = [SQLEntry(key=key, sqlval=val) for key,val in \
                                extra_dct.items()]
                # templates[:] to copy b/c they may be modified in Calculation
                calc = Calculation(machine=machine,
                                   templates=self.templates[:],
                                   params=params + extra_params,
                                   calc_dir=calc_subdir,
                                   )
                if mode == 'w' and os.path.exists(calc_subdir):
                    shutil.rmtree(calc_subdir)
                calc.write_input()
                run_txt += "cd %i && %s %s && cd $here && sleep %i\n" %(idx,\
                            machine.subcmd, machine.get_jobfile_basename(), sleep)
                if imach == 0:
                    sql_records.append(calc.get_sql_record())
            common.file_write(pj(calc_dir, 'run.sh'), run_txt)
        for record in sql_records:
            record['hostname'] = SQLEntry(sqlval=','.join(hostnames))
        # for incomplete parameters: collect header parts from all records and
        # make a set = unique entries
        raw_header = [(key, entry.sqltype.upper()) for record in sql_records \
            for key, entry in record.items()]
        header = list(set(raw_header))
        if have_new_db:
            sqldb.create_table(header)
        else:
            for record in sql_records:
                for key, entry in record.items():
                    if not sqldb.has_column(key):
                        sqldb.add_column(key, entry.sqltype.upper())
        for record in sql_records:
            cmd = "insert into %s (%s) values (%s)"\
                %(self.db_table,
                  ",".join(list(record.keys())),
                  ",".join(['?']*len(list(record.keys()))))
            sqldb.execute(cmd, tuple(entry.sqlval for entry in record.values()))
        if excl and revision > 0 and sqldb.has_column('revision'):
            old_idx_lst = [str(x) for x, in sqldb.execute("select idx from calc where \
                                                          revision < ?", (revision,))]
            common.file_write(pj(self.calc_root, 'excl_push'),
                              '\n'.join(old_idx_lst))
        sqldb.finish()


def conv_table(xx, yy, ffmt="%15.4f", sfmt="%15s", mode='last', orig=False,
               absdiff=False):
    """Convergence table. Assume that quantity `xx` was varied, resulting in
    `yy` values. Return a string (table) listing::

        x, dy1, dy2, ...

    Useful for quickly viewing the results of a convergence study, where we
    assume that the sequence of `yy` values converges to a constant value.

    Parameters
    ----------
    xx : 1d sequence
    yy : 1d sequence, nested 1d sequences, 2d array
        Values varied with `xx`. Each row is one parameter.
    ffmt, sfmt : str
        Format strings for floats (`ffmt`) and strings (`sfmt`)
    mode : str
        'next' or 'last'. Difference to the next value ``y[i+1] - y[i]`` or to
        the last ``y[-1] - y[i]``.
    orig : bool
        Print original `yy` data as well.
    absdiff : bool
        absolute values of differences

    Examples
    --------
    >>> kpoints = ['2 2 2', '4 4 4', '8 8 8']
    >>> etot = [-300.0, -310.0, -312.0]
    >>> forces_rms = [0.3, 0.2, 0.1]
    >>> print(batch.conv_table(kpoints, etot, mode='last'))
          2 2 2       -12.0000
          4 4 4        -2.0000
          8 8 8         0.0000
    >>> print(batch.conv_table(kpoints, [etot,forces_rms], mode='last'))
          2 2 2       -12.0000        -0.2000
          4 4 4        -2.0000        -0.1000
          8 8 8         0.0000         0.0000
    >>> print(batch.conv_table(kpoints, [etot,forces_rms], mode='last', orig=True))
          2 2 2       -12.0000      -300.0000        -0.2000         0.3000
          4 4 4        -2.0000      -310.0000        -0.1000         0.2000
          8 8 8         0.0000      -312.0000         0.0000         0.1000
    >>> print(batch.conv_table(kpoints, np.array([etot,forces_rms]), mode='next'))
          2 2 2       -10.0000        -0.1000
          4 4 4        -2.0000        -0.1000
          8 8 8         0.0000         0.0000
    """
    npoints = len(xx)
    yy = np.asarray(yy).copy()
    if yy.ndim == 1:
        yy = yy[:,None]
    else:
        yy = yy.T
    ny = yy.shape[1]
    dyy = np.empty_like(yy)
    for iy in range(ny):
        if mode == 'next':
            dyy[-1,iy] = 0.0
            dyy[:-1,iy] = np.diff(yy[:,iy])
        elif mode == 'last':
            dyy[:,iy] = yy[-1,iy] - yy[:,iy]
        else:
            raise Exception("unknown mode")
    if absdiff:
        dyy = np.abs(dyy)
    if orig:
        fmtstr = ("%s"*(2*ny+1) + "\n") %((sfmt,) + (ffmt,)*2*ny)
    else:
        fmtstr = ("%s"*(ny+1)   + "\n") %((sfmt,) + (ffmt,)*ny)
    st = ''
    for idx in range(npoints):
        if orig:
            repl = (xx[idx],) + \
                tuple(common.flatten([dyy[idx,iy],yy[idx,iy]] for iy in range(ny)))
        else:
            repl = (xx[idx],) + tuple(dyy[idx,iy] for iy in range(ny))
        st += fmtstr %repl
    return st


def default_repl_keys():
    """Return a dict of default keys for replacement in a
    :class:`ParameterStudy`. Each key will in practice be processed by
    :class:`FileTemplate`, such that e.g. the key 'foo' becomes the placeholder
    'XXXFOO'.

    Each of these placeholders can be used in any parameter study in any file
    template, indenpendent from `params_lst` to :class:`ParameterStudy`.

    Notes
    -----
    If this function is called, dummy files are written to a temp dir, the
    datsbase is read, and the file are deleted.
    """
    # HACK!!! Due to the way ParameterStudy.write_input() is coded, we really
    # need to set up a dummy ParameterStudy and read out the database to get
    # the replacement keys :-)
    import tempfile
    calc_root = tempfile.mkdtemp()
    jobfn_templ = pj(calc_root, 'foo.job')
    common.file_write(jobfn_templ, 'dummy job template, go away!')
    m = Machine(hostname='foo',
                template=FileTemplate(basename='foo.job',
                                      templ_dir=calc_root))
    print("writing test files to: %s, will be deleted" %calc_root)
    params_lst = [[SQLEntry(key='dummy', sqlval=1)]]
    study = ParameterStudy(machines=m, params_lst=params_lst,
                           calc_root=calc_root)
    study.write_input()
    db = SQLiteDB(study.dbfn, table='calc')
    db_keys = [str(x[0]) for x in db.get_header()]
    for kk in ['dummy', 'hostname']:
        db_keys.pop(db_keys.index(kk))
    db.finish()
    ret = {'ParameterStudy' : db_keys,
           'Machine' : list(m.get_sql_record().keys())}
    shutil.rmtree(calc_root)
    return ret


class Case(object):
    """General purpose container class, supporting only keywords in the
    constructor. This is essentially the same as a dictionary, just with
    another syntax for attribute access, i.e. ``case.foo`` instead of
    ``case['foo']``.

    Sometimes you want to do more than storing values. Then, you need to
    subclass Case and do stuff in ``__init__``. But don't call
    ``Case.__init__`` as in ::

        class MyCase(Case):
            def __init__(self, *args, **kwds):
                super(Case, self).__init__(*args, **kwds)
                    self.z = self.x + self.y

    Instead, define a method called ``init``, which is automatically called if
    it exists. ::

        class MyCase(Case):
            def init(self):
                self.z = self.x + self.y

    Examples
    --------
    >>> from pwtools.batch import Case
    >>> case1 = Case(x=1, y=2)
    >>> case2 = Case(x=11, y=22)
    >>> for case in [case1, case2]: print(case.x, case.y)
    """
    def __init__(self, **kwds):
        for k,v in kwds.items():
            setattr(self, k, v)

        if hasattr(self, 'init') and isinstance(self.init, collections.Callable):
            eval('self.init()')
