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
    >>> sql = SQLiteDB('test.db')
    >>> for record in sql.execute("SELECT * FROM calc"):
    ...     print record
    >>> records = sql.execute("SELECT * FROM calc").fetchall()        
    """
    def __init__(self, db_fn):
        """
        args:
        -----
        db_fn : str
            Database filename
        """            
        self.db_fn = db_fn
        self.conn = sqlite3.connect(db_fn)
        self.cur = self.conn.cursor()
            
    def execute(self, *args, **kwargs):
        """This calls self.cur.execute()"""
        return self.cur.execute(*args, **kwargs)

    def has_column(self, table, col):
        """Check if table `table` already has the column `col`.

        args:
        -----
        table : str
            table name in the database
        col : str
            column name in the database
        """            
        has_col = False
        for entry in self.execute("PRAGMA table_info(%s)" %table):
            if entry[1] == col:
                has_col = True
                break
        return has_col                            
    
    def commit(self):
        self.conn.commit()

    def __del__(self):
        self.commit()
        self.cur.close()


