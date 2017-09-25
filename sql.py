import sqlite3, types
import numpy as np
from pwtools import common

import os


def get_test_db():
    """Return in-memory database for playing around."""
    db = SQLiteDB(':memory:', table='calc')
    db.create_table([('a', 'TEXT'), ('b', 'FLOAT')])
    db.execute("insert into calc (a,b) values ('lala', 1.0)")
    db.execute("insert into calc (a,b) values ('huhu', 2.0)")
    db.commit()
    return db


def find_sqltype(val):
    """
    Find sqlite data type which matches the type of `val`.

    Parameters
    ----------
    val : any python type
    
    Returns
    -------
    sqltype : str
        String with sql type which can be used to set up a sqlile table
    """        
    mapping = {\
        type(None): 'NULL',
        int:        'INTEGER',
        float:      'REAL',  # 'FLOAT' also works
        str:        'TEXT',  
        memoryview: 'BLOB'}
    for typ in mapping:
        if isinstance(val, typ):
            return mapping[typ]
    raise Exception("type '%s' unknown, cannot find mapping "
        "to sqlite3 type" %str(type(val)))


def fix_sqltype(sqltype):
    """Fix `sqltype` string. Force uppercase string and  
    change 'FLOAT' -> 'REAL'.

    Parameters
    ----------
    sqltype : str
    
    Returns
    -------
    st : string
    """
    st = sqltype.upper()
    if st == 'FLOAT':
        st = 'REAL'
    return st        


def fix_sql_header(header):
    """fix_sqltype() applied to any sqltype in `header`"""
    return [(x[0], fix_sqltype(x[1])) for x in header]


class SQLEntry(object):
    """
    Represent an entry in a SQLite database. An entry is one single
    value of one column and record (record = row). 
    
    This class is ment to be used in parameter studies where a lot of
    parameters are vaired (e.g. in pw.x input files) and entered in a
    SQLite database. 
    
    There is the possibility that the entry has a slightly different value
    in the db and in the actual input file. See fileval.
    """
    def __init__(self, sqlval=None, sqltype=None, fileval=None, key=None):
        """
        Parameters
        ----------
        sqlval : Any Python type (str, unicode, float, integer, ...)
            The value of the entry which is entered into the database.
        sqltype : {str, None}, optional
            A string (not case sentitive) which determines the sqlite type of the
            entry: 'integer', 'real', 'null', ... If None then automatic type
            detection will be attempted. Only default types are supported, see
            notes below. This is needed to create a sqlite table like in::

                create table calc (foo integer, bar real)
        fileval : {None, <anything>}, optional
            If not None, then this is the value of the entry that it has in
            another context (actually used in the input file). If None, then
            fileval = val. 
            Example: K_POINTS in pw.x input file::

                sqlval: '2 2 2 0 0 0'
                fileval: 'K_POINTS automatic\\n2 2 2 0 0 0'

        key : optional, {None, str}, optional
            An optional key. This key should refer to the column name in the 
            database table, as in::

                % create table calc (key1 sqltype1, key2 sqltype2, ...)
            For example::

                % create table calc (idx integer, ecutwfc float, ...)

        Notes
        -----
        SQLite types from the Python docs of sqlite3::

            Python type     SQLite type
            -----------     -----------
            None            NULL
            int             INTEGER
            long            INTEGER
            float           REAL
            str (UTF8-encoded)  TEXT
            unicode         TEXT
            buffer          BLOB
        """
        self.sqltype = find_sqltype(sqlval) if sqltype is None else \
                       fix_sqltype(sqltype)
        self.sqlval = sqlval
        self.fileval = sqlval if fileval is None else fileval
        self.key = key


# Implementation of multiple tables support: It works perfect, but new code
# changes will need good testing. This is b/c we explicitely pass
# `table` around as arg to every method which used to use self.table. That 
# affects all "convenience" methods like "has_column()" etc. Now it is very
# easy to introduce bugs, if passing `table` is forgotten. 
#
# The reason for doing this is that SQLiteDB was first developed with the
# assumption that we always use one db + one table in it. But often, the
# `table` is not even needed, i.e. for all statements like "select * from
# <table>" where the user passes the table name in the command, which is
# actually the most common use case. The only convenience method which is
# really used often is add_column().
#
# We may need to add a class SQLiteTable, which behaves like the old SQLiteDB
# with enforced `table` and use self.table everywhere in methods which need it.
# SQLiteDB would be composed of multiple SQLiteTable instances. But this sounds
# to fancy and complicated. We just wanted a *simpler* interface to sqlite3,
# not another implementation. So, lets not add more convenience methods. Things
# like attach_column() are already as complicated as it gets and almost not
# used, even though they work!
class SQLiteDB(object):
    """Interface class which wraps the sqlite3 module. It abstacts away the
    connecting to the database and cursor setup and adds some convenience
    methods.   
    
    Examples
    --------
    >>> db = SQLiteDB('test.db', table='calc')
    >>> db.create_table([('a', 'float'), ('b', 'text')])
    >>> db.execute("insert into %s ('a', 'b') values (1.0, 'lala')" %db.table)
    >>> db.execute("insert into %s ('a', 'b') values (?,?)" %db.table, (2.0, 'huhu'))
    >>> # iterator
    >>> for record in db.execute("select * from calc"):
    ...     print record
    (1.0, u'lala')
    (2.0, u'huhu')
    >>> # list
    >>> print db.execute("select * from calc").fetchall()
    [(1.0, u'lala'), (2.0, u'huhu')]
    >>> db.get_list1d("select a from calc")
    [1.0, 2.0]
    >>> db.get_list1d("select b from calc")
    [u'lala', u'huhu']
    >>> db.get_array1d("select a from calc")
    array([ 1.,  2.])
    >>> db.add_column('c', 'float')
    >>> db.execute("update calc set c=5.0")
    >>> db.get_array("select a,c from calc")
    array([[ 1.,  5.],
           [ 2.,  5.]])
    >>> # db in memory, attach and query over multiple databases, assume both
    >>> # databases have a table named 'calc'
    >>> db = SQLiteDB(':memory:')
    >>> db.executescript("attach 'foo.db' as foo; attach 'bar.db' as bar;")
    >>> db.get_array("select foo.calc.a,bar.calc.b from foo.calc,bar.calc "
    ... "where foo.calc.idx==bar.calc.idx and foo.calc.c not like '%gohome%'")

    Notes
    -----
    There are actually 2 methods to put entries into the db:

    1) Use sqlite3 placeholder syntax. This is recommended. Here, automatic
       type conversion Python -> sqlite is done by the sqlite3 module. For
       instance, double numbers (i.e. Python float type) will be correctly
       stored as double by SQLite default.
       
       >>> db.execute("insert into calc ('a', 'b') values (?,?)", (1.0, 'lala'))
    
    2) Write values directly into sql command. Here all values are actually
       strings.

       >>> db.execute("insert into calc ('a', 'b') values (1.0, 'lala')")
       >>> db.execute("insert into calc ('a', 'b') values (%e, '%s')" %(1.0, 'lala')")
       
       There are some caveats. For example, the string ('lala' in the example)
       must appear *qsingle-quoted* in the sqlite cmd to be recognized as such.
       Also aviod things like `"... %s" %str(1.0)`. This will truncate the
       float after less then 16 digits and thus store the 8-byte float with
       less precision! 
    """
    def __init__(self, db_fn, table=None):
        """
        Parameters
        ----------
        db_fn : str
            database filename
        table : str, optional
            name of the database table, you can also use :meth:`set_table`
            later or the `table` keyword in all methods which need to know to
            which database table you're talking
        """            
        self.db_fn = db_fn
        self.conn = sqlite3.connect(db_fn)
        self.cur = self.conn.cursor()
        self.set_table(table)
    
    def _get_table(self, table):
        if table is None:
            if self.table is None:
                raise Exception("table and self.table are None")
            else:
                return self.table
        else:
            return table

    def set_table(self, table):
        """Set the table name (aka switch to another table).

        Parameters
        ----------
        table : str
            table name
        """            
        self.table = table
    
    def get_table(self):
        """Return string self.table."""
        return self.table

    def execute(self, *args, **kwargs):
        """This calls self.cur.execute()."""
        return self.cur.execute(*args, **kwargs)
    
    def executemany(self, *args, **kwargs):
        """This calls self.cur.executemany()."""
        return self.cur.executemany(*args, **kwargs)
    
    def executescript(self, *args, **kwargs):
        """This calls self.cur.executescript(). 
        
        Returns what `executescript` returns. Calling
        ``executescript().fetchall()`` returns []."""
        return self.cur.executescript(*args, **kwargs)
    
    def has_table(self, table):
        """Check if a table named `table` already extists."""
        assert table is not None, ("table is None")
        return self.execute("pragma table_info(%s)" %table).fetchall() != []

    def has_column(self, col, table=None):
        """Check if `table`  already has the column `col`.
        
        Parameters
        ----------
        col : str
            column name in the database
        table : str, optional            
        """            
        for entry in self.get_header(self._get_table(table)):
            if entry[0] == col:
                return True
        return False                
    
    def add_column(self, col, sqltype, table=None):
        """Add column `col` with type `sqltype` to the header. To actually put
        data into that, use :meth:`execute` and standard sql statements or see 
        :meth:`attach_column` or :meth:`fill_column`.
        
        Parameters
        ----------
        col : str
            column name
        sqltype : str
            sqlite data type (see :class:`SQLEntry`)
        table : str, optional            
        """
        table = self._get_table(table)
        if not self.has_column(col, table=table):
            self.execute("ALTER TABLE %s ADD COLUMN %s %s" \
                        %(table, col, fix_sqltype(sqltype)))
    
    def add_columns(self, header, table=None):
        """Convenience function to add multiple columns from `header`. See
        :meth:`get_header`.
        
        Parameters
        ----------
        header : sequence
            see :meth:`get_header`
        table : str, optional            

        Examples
        --------
        >>> db.add_columns([('a', 'text'), ('b', 'real')])
        # is the same as
        >>> db.add_column('a', 'text')
        >>> db.add_column('b', 'real')
        """
        kwds = {'table': self._get_table(table)}
        for entry in fix_sql_header(header):
            self.add_column(*entry, **kwds)
    
    def get_max_rowid(self, table=None):
        """Return max(rowid), which is equal to the number of rows in
        `table`.
        
        Parameters
        ----------
        table : str, optional            
        """
        return self.get_single("select max(rowid) from %s" %self._get_table(table))

    def fill_column(self, col, values, start=1, extend=True, 
                    overwrite=False, table=None):
        """Fill existing column `col` with values from `values`, starting from
        rowid `start`. "rowid" is a special sqlite column which is always
        present and which numbers all rows. 

        The column must already exist. To add a new column and fill it, see
        :meth:`attach_column`.
        
        Parameters
        ----------
        col : str
            Column name.
        values : sequence
            Values to be inserted.
        start : int
            sqlite rowid value to start at (first row: start=1)
        extend : bool
            If `extend=True` and `len(values)` extends the last row, then
            continue to add values. All other column entries will be NULL. If
            False, then we silently stop inserting at the last row.
        overwrite : bool
            Whether to overwrite entries which are not NULL (None in Python).
        table : str, optional            
        """
        # The operation "update <table> ..." works only as long as there is at
        # least one column with a non-NULL entry. After that, rowid is not
        # defined and nothing gets inserted. Then, we need to use "insert into
        # ..." to appand rows to the bottom.
        table = self._get_table(table)
        maxrowid = self.get_max_rowid(table)
        assert self.has_column(col, table=table), "column missing: %s" %col
        if not extend:
            assert start <= maxrowid, "start > maxrowid"
        rowid = start
        for val in values:
            if rowid <= maxrowid:
                if not overwrite:
                    _val = self.get_single("select %s from %s where rowid==?" \
                            %(col, table), (rowid,))
                    assert _val is None, ("value for column '%s' at rowid "
                        "%i is not NULL (%s)" %(col, rowid, repr(_val)))
                self.execute("update %s set %s=? where rowid==?" \
                             %(table, col), (val, rowid))
            else:
                if extend:
                    self.execute("insert into %s (%s) values (?)" %(table,
                        col,), (val,))
            rowid += 1                
    
    def attach_column(self, col, values, sqltype=None, table=None, **kwds):
        """Attach (add) a new column named `col` of `sqltype` and fill it with
        `values`. With overwrite=True, allow writing into existing columns,
        i.e. behave like :meth:`fill_column`.
        
        This is a short-cut method which essentially does 
        ``add_column(...); fill_column(...)``

        Parameters
        ----------
        col : str
            Column name.
        values : sequence
            Values to be inserted.
        sqltype : str, optional
            sqlite type of values in `values`, obtained from values[0] if None
        table : str, optional            
        **kwds : 
            additional keywords passed to :meth:`fill_column`,
            default: `start=1, extend=True, overwrite=False`
        """
        table = self._get_table(table)
        current_kwds = {'start':1, 'extend': True, 'overwrite': False, 
                        'table': table}
        current_kwds.update(kwds)
        if not current_kwds['overwrite']:
            assert not self.has_column(col, table=table), \
                       ("column already present: %s, use " 
                        "overwrite=True" %col)
        if sqltype is None:
            sqltype = find_sqltype(values[0])
        self.add_column(col, sqltype, table=table)
        self.fill_column(col, values, **current_kwds)

    def get_header(self, table=None):
        """Return the header of the `table`:

        Parameters
        ----------
        table : str, optional

        Examples
        --------
        >>> db = SQLiteDB('test.db')
        >>> db.execute("create table foo (a text, b real)")
        >>> db.get_header('foo') 
        [('a', 'text'), ('b', 'real')]
        """
        return [(x[1], x[2]) for x in \
                self.execute("PRAGMA table_info(%s)" %self._get_table(table))]
    
    def create_table(self, header, table=None):
        """Create a `table` from `header`. `header` is in the
        same format which :meth:`get_header` returns.
        
        Parameters
        ----------
        header : list of lists/tuples
            [(colname1, sqltype1), (colname2, sqltype2), ...]
        table : str, optional
        """
        self.execute("CREATE TABLE %s (%s)" %(self._get_table(table), 
                                            ','.join("%s %s" %(x[0], x[1]) \
                                            for x in fix_sql_header(header))))
    
    def get_list1d(self, *args, **kwargs):
        """Shortcut for commonly used functionality. If one extracts a single
        column, then ``self.cur.fetchall()`` returns a list of tuples like
        ``[(1,), (2,)]`` We call ``fetchall()`` and return the flattened list. 
        """
        return common.flatten(self.execute(*args, **kwargs).fetchall())
    
    def get_single(self, *args, **kwargs):
        """Return single entry from the table."""
        ret = self.get_list1d(*args, **kwargs)
        assert len(ret) > 0, ("nothing returned")
        assert len(ret) == 1, ("no unique result")
        return ret[0]

    def get_array1d(self, *args, **kwargs):
        """Same as :meth:`get_list1d`, but return numpy array."""
        return np.array(self.get_list1d(*args, **kwargs))
    
    def get_array(self, *args, **kwargs):
        """Return result of ``self.execute().fetchall()`` as numpy array. 
        
        Usful for 2d arrays, i.e. convert result of extracting >1 columns to
        numpy 2d array. The result depends on the data types of the columns."""
        return np.array(self.execute(*args, **kwargs).fetchall())
    
    def get_dict(self, *args, **kwargs):
        """For the provided select statement, return a dict where each key is
        the column name and the column is a list. Column names are obtained
        from the ``Cursor.description`` attribute.
        
        Examples
        --------
        >>> db.get_dict("select foo,bar from calc")
        {'foo': [1,2,3],
         'bar': ['x', 'y', 'z']}
        """
        cur = self.execute(*args, **kwargs)
        # ['col0', 'col1, ...]
        cols = [entry[0] for entry in cur.description]
        # [(val0_0, val0_1, ...), # row 0
        #  (val1_0, val1_1, ...), # row 1
        #  ...]
        ret = cur.fetchall()
        dct = dict((col, []) for col in cols)
        # {'col0': [val0_0, val1_0, ...], 
        #  'col1': [val0_1, val1_1, ...], 
        #  ...}
        for row in ret:
            for idx, col in enumerate(cols):
                dct[col].append(row[idx])
        return dct                

    def commit(self):
        """Commit changes to connection."""
        self.conn.commit()
    
    def finish(self):
        """Commit and close cursor."""
        self.commit()
        self.cur.close()

    def __del__(self):
        self.finish()


# XXX old behavior in argument list: key, sqltype, lst. If you want this, then
# either replace sql_column -> sql_column_old in your script or explitely use 
# sql_column(key=..., sqltype=..., lst=...).
def sql_column_old(key, sqltype, lst, sqlval_func=lambda x: x, fileval_func=lambda x: x):
    """
    Examples
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


def sql_column(key, lst, sqltype=None, sqlval_func=lambda x: x, fileval_func=lambda x: x):
    """Convert a list `lst` of values of the same type (i.e. all floats) to a
    list of SQLEntry instances of the same column name `key` and `sqltype`
    (e.g. 'float'). 

    See ParameterStudy for applications.
     
    Parameters
    ----------
    key : str
        sql column name
    lst : sequence of arbitrary values, these will be SQLEntry.sqlval
    sqltype : str, optional
        sqlite type, if None then it is determined from the first entry in
        `lst` (possibly modified by sqlval_func)
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
    
    Examples
    --------
    >>> vals = sql_column('ecutfwc', [25.0, 50.0, 75.0], 
    ...                   fileval_func=lambda x: 'ecutfwc=%s'%x)
    >>> for v in vals:
    ...     print v.key, v.sqltype, v.sqlval, v.fileval
    ecutfwc REAL 25.0 ecutfwc=25.0 
    ecutfwc REAL 50.0 ecutfwc=50.0
    ecutfwc REAL 75.0 ecutfwc=75.0
    """
    sqlval_lst = [sqlval_func(x) for x in lst]
    fileval_lst = [fileval_func(x) for x in lst]
    types = [type(x) for x in sqlval_lst]
    assert len(set(types)) == 1, ("after sqlval_func(), not all entries in "
        "sqlval_lst have the same type: %s" %str(types))
    _sqltype = find_sqltype(sqlval_lst[0]) if sqltype is None \
        else sqltype.upper()
    return [SQLEntry(key=key, 
                     sqltype=_sqltype, 
                     sqlval=sv, 
                     fileval=fv) for sv,fv in \
                        zip(sqlval_lst, fileval_lst)]


def sql_matrix(lists, header=None, colnames=None, sqlval_funcs=None, fileval_funcs=None):
    """Convert each entry in a list of lists ("matrix" = sql table) to an
    SQLEntry based on `header`. This can be used to quickly convert the result
    of comb.nested_loops() (nested lists) to input `params_lst` for
    ParameterStudy.
    
    The entries in the lists can have arbitrary values, but each "column"
    should have the same type. Each sublist (= row) can be viewed as a record in
    an sql database, each column as input for sql_column().
    
    If you provide `header`, then tyes for each column are taken from that. If
    `colnames` are used, then types are fetched (by find_sqltype()) from the
    first row. May not work for "incomplete" datasets, where some entries in
    the first row are None (NULL in sqlite).

    Parameters
    ----------
    lists : list of lists 
    header : sequence, optional 
        [('foo', 'integer'), ('bar', 'float'), ...], see
        sql.SQLiteDB
    colnames : sequence os strings, optional
        Use either `colnames` or `header`.
    sqlval_funcs, fileval_funcs: {None, dict}
        For certain (or all) columns, you can specify a sqlval_func /
        fileval_func. They have the same meaning as in sql_column().
        E.g. sql_matrix(..., fileval_funcs={'foo': lambda x: str(x)+'-value'})
        would set fileval_func for the whole column 'foo'.
    
    Returns
    -------
    list of lists

    Examples
    --------
    >>> lists=comb.nested_loops([[1.0,2.0], zip(['a']*2, [777,888])],flatten=True)
    >>> lists
    [[1.0, 'a', 777], 
     [1.0, 'a', 888], 
     [2.0, 'a', 777], 
     [2.0, 'a', 888]]
    >>> # use explicit header 
    >>> header=[('col0', 'float'), ('col1', 'text'), ('col2', 'integer')]
    >>> m=sql.sql_matrix(lists, header=header)
    >>> for row in m: print [(xx.key, xx.fileval, xx.sqltype) for xx in row]
    [('col0', 1.0, 'REAL'), ('col1', 'a', 'TEXT'), ('col2', 777, 'INTEGER')]
    [('col0', 1.0, 'REAL'), ('col1', 'a', 'TEXT'), ('col2', 888, 'INTEGER')]
    [('col0', 2.0, 'REAL'), ('col1', 'a', 'TEXT'), ('col2', 777, 'INTEGER')]
    [('col0', 2.0, 'REAL'), ('col1', 'a', 'TEXT'), ('col2', 888, 'INTEGER')]
    >>> # use colnames -> automatic sqltype detected, also use fileval_funcs
    >>> m=sql.sql_matrix(lists, colnames=['col0','col1','col2'], fileval_funcs={'col0': lambda x: x*100})
    >>> for row in m: print [(xx.key, xx.fileval, xx.sqltype) for xx in row]
    [('col0', 100.0, 'REAL'), ('col1', 'a', 'TEXT'), ('col2', 777, 'INTEGER')]
    [('col0', 100.0, 'REAL'), ('col1', 'a', 'TEXT'), ('col2', 888, 'INTEGER')]
    [('col0', 200.0, 'REAL'), ('col1', 'a', 'TEXT'), ('col2', 777, 'INTEGER')]
    [('col0', 200.0, 'REAL'), ('col1', 'a', 'TEXT'), ('col2', 888, 'INTEGER')]
 
    """
    if header is None:
        assert colnames is not None, ("colnames is None")
        sqltypes = [find_sqltype(xx) for xx in lists[0]]
        header = list(zip(colnames, sqltypes))
    ncols = len(header)
    ncols2 = len(lists[0])
    keys = [entry[0] for entry in header]
    assert ncols == ncols2, ("number of columns differ: lists (%i), "
        "header (%i)" %(ncols2, ncols))
    _sqlval_funcs = dict([(key, lambda x: x) for key in keys])
    _fileval_funcs = dict([(key, lambda x: x) for key in keys])
    if sqlval_funcs is not None:
        _sqlval_funcs.update(sqlval_funcs)
    if fileval_funcs is not None:
        _fileval_funcs.update(fileval_funcs)
    newlists = []        
    for row in lists:
        newrow = []
        for ii, entry in enumerate(row):
            key = header[ii][0]
            sqltype = header[ii][1]
            newrow.append(SQLEntry(key=key,
                                   sqltype=sqltype,
                                   sqlval=_sqlval_funcs[key](entry),
                                   fileval=_fileval_funcs[key](entry)))
        newlists.append(newrow)                                   
    return newlists   


def makedb(filename, lists, colnames, table=None, mode='a', close=True, **kwds):
    """ Create sqlite db `filename` (mode='w') or append to existing db
    (mode='a'). The database is build up from `lists` and `colnames`, see 
    sql_matrix().

    In append mode, rows are simply added to the bottom of the table and only
    column names (`colnames`) which are already in the table are allowed.
    `colnames` can contain a subset of the original header, in which case the
    other entries are NULL by default.

    If the datsbase file doesn't exist, then mode='a' is the same as mode='w'.
    
    By default `close=True`, i.e. a db with a closed connection is returned.
    For interactive use, `close=False` is what you want. That gives you a db
    which can be used right away.

    Parameters
    ----------
    lists : list of lists, see sql_matrix()
    colnames : list of column names, see sql_matrix()
    table : str, optional
        String with table name. If None then we try to set a default name based
        on `filename`.
    mode : str
        'w': write new db, 'a': append
    close : bool, optional
        Close cursor after db has been filled with values.
    **kwds : passed to sql_matrix()

    Returns
    -------
    db : SQLiteDB instance

    Examples
    --------
    >>> lists=zip([1,2,3],['a','b','c'])
    >>> db=sql.makedb('/tmp/foo.db', lists, ['col0', 'col1'], mode='w',
    ...               table='calc', close=False)
    >>> db.get_dict('select * from calc')
    {'col0': [1, 2, 3], 'col1': [u'a', u'b', u'c']}
    """
    sufs = ['.db', '.sqlite', '.sqlite3']
    if table is None:
        for suffix in sufs:
            if filename.endswith(suffix):
                table = os.path.basename(filename.replace(suffix, ''))
                break
    assert table is not None, ("table name missing or could not determine "
                               "from filename")
    assert len(colnames) == len(lists[0]), ("len(colnames) != length of "
                                            "first list")        
    if mode == 'w':        
        if os.path.exists(filename):
            os.remove(filename)
    sqltypes = [find_sqltype(xx) for xx in lists[0]]
    header = list(zip(colnames, sqltypes))
    db = SQLiteDB(filename, table=table)
    if not db.has_table(table):
        db.create_table(header)
    else:
        db_header = db.get_header()
        db_colnames = [x[0] for x in db_header]
        db_types = [x[1] for x in db_header]
        for col,typ in header:
            assert col in db_colnames, ("col '%s' not in db header" %col)
            db_typ = db_types[db_colnames.index(col)]
            assert typ == db_typ, ("col: '%s': "
                "types differ, db: '%s', here: '%s'" %(col, db_typ, typ))
    sql_lists = sql_matrix(lists, header=header, **kwds)
    for row in sql_lists:
        ncols = len(row)
        names = ','.join(colnames)
        values = ','.join(['?']*ncols)
        cmd = 'insert into %s (%s) values (%s)' %(table, names, values)
        db.execute(cmd, [entry.sqlval for entry in row])
    if close:    
        db.finish() 
    else:
        db.commit()
    return db
