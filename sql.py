import sqlite3

class SQLEntry(object):
    def __init__(self, sqltype=None, sqlval=None, fileval=None, key=None, sql_type=None,
                 sql_val=None, file_val=None):
        """Represent an entry in a SQLite database. An entry is one single
        value of one column and record (record = row). 
        
        This class is ment to be used in parameter studies where a lot of
        parameters are vaired (e.g. in pw.x input files) and entered in a
        SQLite database. 
        
        There is the possibility that the entry has a slightly different value
        in the db and in the actual input file. See fileval.
        
        args:
        -----
        sqltype : str
            A string (not case sentitive) which determines the SQL type of the
            entry: 'integer', 'real', ...
        sqlval : the value of the entry
            If it is a string, it is automatically quoted to match SQLite
            syntax rules, e.g. 'lala' -> "'lala'", which appears as 'lala' in
            the db.
        fileval : optional, {None, <anything>}
            If not None, then this is the value of the entry that it has in
            another context (actually used in the input file). If None, then
            fileval = val. 
            Example: K_POINTS in pw.x input file:
                sqlval: '2 2 2 0 0 0'
                fileval: 'K_POINTS automatic\n2 2 2 0 0 0'
        key : optional, {None, str}
            An optional key. This key should refer to the column name in the 
            database table, as in:
                % create table calc (key1 sqltype1, key2    sqltype2, ...)
            For example:
                % create table calc (idx  integer,   ecutwfc float,     ...)


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
        # backwd compat
        assert None in [sqltype, sql_type], ("you use old syntax, one of " 
               "[sql_type, sqltype] must not be None")
        assert None in [sqlval, sql_val], ("you use old syntax, one of " 
               "[sql_val, sqlval] must not be None")
        if sql_type is not None:
            sqltype = sql_type
        if sql_val is not None:
            sqlval = sql_val
        if file_val is not None:
            fileval = file_val
        
        self.sqltype = sqltype
        self.fileval = sqlval if fileval is None else fileval
        self.sqlval = self._fix_sqlval(sqlval)
        self.key = key

        # backwd compat
        self.sql_val = self.sqlval
        self.sql_type = self.sqltype
        self.file_val = self.fileval
    
    def _fix_sqlval(self, sqlval):
        if isinstance(sqlval, str):
            # "lala" -> "'lala'"
            # 'lala' -> "'lala'"
            return repr(sqlval)
        else:
            return sqlval


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
    >>> db.create_table([('a', 'FLOAT'), ('b', 'TEXT')])
    >>> db.execute("INSERT INTO %s ('a', 'b') VALUES (1.0, 'lala')" %db.table)
    >>> db.execute("INSERT INTO %s ('a', 'b') VALUES (2.0, 'huhu')" %db.table)
    # iterator
    >>> for record in db.execute("SELECT * FROM calc"):
    ...     print record
    (1.0, u'lala')
    (2.0, u'huhu')
    # list
    >>> print db.execute("SELECT * FROM calc").fetchall()
    [(1.0, u'lala'), (2.0, u'huhu')]
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
                        %(self.table, col, sqltype))
    
    def add_columns(self, header):
        """Convenience function to add multiple columns from `header`. See
        get_header()."""
        for entry in header:
            self.add_column(*entry)

    def get_header(self):
        """Return the "header" of the db:

        example:
        --------
        db = SQLiteDB('test.db', table='foo')
        db.execute("create table foo (a text, b real)"
        db.get_header() -> [('a', 'text'), ('b', 'real')]
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
                                            for x in header)))

    def commit(self):
        self.conn.commit()

    def __del__(self):
        self.commit()
        self.cur.close()


