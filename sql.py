import sqlite3, types
import numpy as np
from pwtools import common
from itertools import izip

def get_test_db():
    db = SQLiteDB(':memory:', table='calc')
    db.create_table([('a', 'TEXT'), ('b', 'FLOAT')])
    db.execute("insert into calc (a,b) values ('lala', 1.0)")
    db.execute("insert into calc (a,b) values ('huhu', 2.0)")
    db.finish()
    return db

def find_sqltype(val):
    mapping = {\
        types.NoneType:    'NULL',
        types.IntType:     'INTEGER',
        types.LongType:    'INTEGER',
        types.FloatType:   'REAL',  # 'FLOAT' also works
        types.StringTypes: 'TEXT',  # StringType + UnicodeType
        types.BufferType:  'BLOB'}
    for typ in mapping.keys():
        if isinstance(val, typ):
            return mapping[typ]
    raise StandardError("type '%s' unknown, cannot find mapping "
        "to sqlite3 type" %str(type(val)))

def fix_sqltype(sqltype):
    st = sqltype.upper()
    if st == 'FLOAT':
        st = 'REAL'
    return st        

def fix_sql_header(header):
    return [(x[0], fix_sqltype(x[1])) for x in header]


class SQLEntry(object):
    def __init__(self, sqlval=None, sqltype=None, fileval=None, key=None):
        """Represent an entry in a SQLite database. An entry is one single
        value of one column and record (record = row). 
        
        This class is ment to be used in parameter studies where a lot of
        parameters are vaired (e.g. in pw.x input files) and entered in a
        SQLite database. 
        
        There is the possibility that the entry has a slightly different value
        in the db and in the actual input file. See fileval.
        
        args:
        -----
        sqlval : Any Python type (str, unicode, float, integer, ...)
            the value of the entry
        sqltype : {str, None}
            A string (not case sentitive) which determines the sqlite type of the
            entry: 'integer', 'real', 'null', ... If None then automatic type
            detection will be attempted. Only default types are supported, see
            notes below.
        fileval : {None, <anything>}
            If not None, then this is the value of the entry that it has in
            another context (actually used in the input file). If None, then
            fileval = val. 
            Example: K_POINTS in pw.x input file:
                sqlval: '2 2 2 0 0 0'
                fileval: 'K_POINTS automatic\n2 2 2 0 0 0'
        key : optional, {None, str}
            An optional key. This key should refer to the column name in the 
            database table, as in:
                % create table calc (key1 sqltype1, key2 sqltype2, ...)
            For example:
                % create table calc (idx integer, ecutwfc float, ...)

        notes:
        ------
        SQLite types from the Python docs of sqlite3:
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
    

class SQLiteDB(object):
    """Small convenience inerface class for sqlite3. It abstacts away the
    connecting to the database etc. It simplifies the usage of connection and
    cursor objects (bit like the "shortcuts" already defined in sqlite3).   
    
    exported methods:
    -----------------
    self.cur.execute() -> execute()
    self.conn.commit() -> commit()
    where self.cur  -> sqlite3.Cursor
          self.conn -> sqlite3.Connection
    
    example:
    --------
    >>> db = SQLiteDB('test.db', table='calc')
    >>> db.create_table([('a', 'float'), ('b', 'text')])
    >>> db.execute("insert into %s ('a', 'b') values (1.0, 'lala')" %db.table)
    >>> db.execute("insert into %s ('a', 'b') values (?,?)" %db.table, (2.0, 'huhu'))
    # iterator
    >>> for record in db.execute("select * from calc"):
    ...     print record
    (1.0, u'lala')
    (2.0, u'huhu')
    # list
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
    
    notes:
    ------
    There are actually 2 methods to put entries into the db. Fwiw, this is a
    general sqlite3 (module) note.

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
        args:
        -----
        db_fn : str
            database filename
        table : str
            name of the database table 
        """            
        self.db_fn = db_fn
        self.conn = sqlite3.connect(db_fn)
        self.cur = self.conn.cursor()
        self.table = table
        if self.table is None:
            raise StandardError("missing table name for database")
    
    def set_table(self, table):
        """Set the table name (aka switch to another table).

        args:
        -----
        table : str
            table name
        """            
        self.table = table
    
    def get_table(self):
        return self.table

    def execute(self, *args, **kwargs):
        """This calls self.cur.execute()"""
        return self.cur.execute(*args, **kwargs)

    def has_column(self, col):
        """Check if table self.table already has the column `col`.
        
        args:
        -----
        col : str
            column name in the database
        """            
        has_col = False
        for entry in self.get_header():
            if entry[0] == col:
                has_col = True
                break
        return has_col                            
    
    def add_column(self, col, sqltype):
        """Add column `col` with type `sqltype`. 
        
        args:
        -----
        col : str
            column name
        sqltype : str
            sqite data type (see SQLEntry)
        """
        if not self.has_column(col):
            self.execute("ALTER TABLE %s ADD COLUMN %s %s" \
                        %(self.table, col, fix_sqltype(sqltype)))
    
    def add_columns(self, header):
        """Convenience function to add multiple columns from `header`. See
        get_header()."""
        for entry in fix_sql_header(header):
            self.add_column(*entry)

    def get_header(self):
        """Return the "header" of the db:

        example:
        --------
        >>> db = SQLiteDB('test.db', table='foo')
        >>> db.execute("create table foo (a text, b real)"
        >>> db.get_header() 
        [('a', 'text'), ('b', 'real')]
        """            
        return [(x[1], x[2]) for x in \
                self.execute("PRAGMA table_info(%s)" %self.table)]
    
    def create_table(self, header):
        """Create a table named self.table from `header`. `header` is in the
        same format which get_header() returns.
        
        args:
        -----
        header : list of lists/tuples
            [(colname1, sqltype1), (colname2, sqltype2), ...]
        """
        self.execute("CREATE TABLE %s (%s)" %(self.table, 
                                            ','.join("%s %s" %(x[0], x[1]) \
                                            for x in fix_sql_header(header))))
    
    def get_list1d(self, *args, **kwargs):
        """Shortcut for commonly used functionality: If one extracts a single
        column, then self.cur.fetchall() returns a list of tuples like 
            [(1,), (2,)]. 
        We call fetchall() and return the flattened list. 
        """
        return common.flatten(self.execute(*args, **kwargs).fetchall())

    def get_array1d(self, *args, **kwargs):
        """Same as get_list1d, but return numpy array."""
        return np.array(self.get_list1d(*args, **kwargs))
    
    def get_array(self, *args, **kwargs):
        """Return result of self.execute().fetchall() as numpy array. 
        
        Usful for 2d arrays, i.e. convert result of extracting >1 columns to
        numpy 2d array. The result depends on the data types of the columns."""
        return np.array(self.execute(*args, **kwargs).fetchall())
    
    def get_dict(self, cols=None):
        dct = {}
        cols = [x[0] for x in self.get_header()] if cols is None else cols
        for entry in cols:
            col = entry[0]
            dct[str(col)] = self.get_list1d("select %s from %s" %(col, self.table))
        return dct            

    def commit(self):
        self.conn.commit()
    
    def finish(self):
        self.commit()
        self.cur.close()

    def __del__(self):
        self.finish()


# XXX old behavior in argument list: key, sqltype, lst. If you want this, then
# either replace sql_column -> sql_column_old in your script or explitely use 
# sql_column(key=..., sqltype=..., lst=...).
def sql_column_old(key, sqltype, lst, sqlval_func=lambda x: x, fileval_func=lambda x: x):
    """
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


def sql_column(key, lst, sqltype=None, sqlval_func=lambda x: x, fileval_func=lambda x: x):
    """Convert a list `lst` of values of the same type (i.e. all floats) to a
    list of SQLEntry instances of the same column name `key` and `sqltype`
    (e.g. 'float'). 

    See ParameterStudy for applications.
     
    args:
    -----
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
    
    example:
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
                        izip(sqlval_lst, fileval_lst)]


def sql_matrix(lists, header, sqlval_funcs=None, fileval_funcs=None):
    """Convert each entry in a list of lists ("matrix" = sql table) to an
    SQLEntry based on `header`. This can be used to quickly convert the result
    of comb.nested_loops() (nested lists) to input `params_lst` for
    ParameterStudy.
    
    The entries in the lists can have arbitrary values, but each "column"
    should have the same type. Each sublist (= row) can be viewed as a row in
    an sql database, each column as input for sql_column().
    
    Automatic sql type detection is not implemented. You must specify the
    column type in `header`.

    args:
    -----
    lists : list of lists 
    header : sequence 
        [('foo', 'integer'), ('bar', 'float'), ...], see
        sql.SQLiteDB
    sqlval_funcs, fileval_funcs: {None, dict}
        For certain (or all) columns, you can specify a sqlval_func /
        fileval_func. They have the same meaning as in sql_column().
        E.g. sql_matrix(..., fileval_funcs={'foo': lambda x: str(x)+'-value'})
        would set fileval_func for the whole column 'foo'.
    
    returns:
    --------
    list of lists

    example:
    --------
    >>> lists=comb.nested_loops([[1.0,2.0,3.0], zip(['a', 'b'], [888, 999])],
    ...                         flatten=True)
    >>> lists
    [[1.0, 'a', 888],
     [1.0, 'b', 999],
     [2.0, 'a', 888],
     [2.0, 'b', 999],
     [3.0, 'a', 888],
     [3.0, 'b', 999]]
    >>> header=[('col0', 'float'), ('col1', 'text'), ('col2', 'integer')]
    >>> m=batch.sql_matrix(lists, header)
    >>> for row in m:
    ...     print [(xx.key, xx.fileval) for xx in row]
    ...
    [('col0', 1.0), ('col1', 'a'), ('col2', 888)]
    [('col0', 1.0), ('col1', 'b'), ('col2', 999)]
    [('col0', 2.0), ('col1', 'a'), ('col2', 888)]
    [('col0', 2.0), ('col1', 'b'), ('col2', 999)]
    [('col0', 3.0), ('col1', 'a'), ('col2', 888)]
    [('col0', 3.0), ('col1', 'b'), ('col2', 999)]
    >>> m=batch.sql_matrix(lists, header, fileval_funcs={'col0': lambda x: x*100})
    >>> for row in m:
    ...     print [(xx.key, xx.fileval) for xx in row]
    ...
    [('col0', 100.0), ('col1', 'a'), ('col2', 888)]
    [('col0', 100.0), ('col1', 'b'), ('col2', 999)]
    [('col0', 200.0), ('col1', 'a'), ('col2', 888)]
    [('col0', 200.0), ('col1', 'b'), ('col2', 999)]
    [('col0', 300.0), ('col1', 'a'), ('col2', 888)]
    [('col0', 300.0), ('col1', 'b'), ('col2', 999)]
    """
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
