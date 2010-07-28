import sqlite3

class SQLEntry(object):
    def __init__(self, sql_type, sql_val, file_val=None, key=None):
        """Represent an entry in an SQLite database. An entry is one single
        value of one column and record (record = row). 
        
        This class is ment to be used in parameter studies where a lot of
        parameters are vaired (e.g. in pw.x input files) and entered in a
        SQLite database. 
        
        There is the possibility that the entry has a slightly different value
        in the db and in the actual input file. See file_val.
        
        args:
        -----
        sql_type : str
            A string (not case sentitive) which determines the SQL type of the
            entry: 'integer', 'real', ...
        sql_val : the value of the entry
            If it is a string, it is automatically quoted to match SQLite
            syntax rules, e.g. 'lala' -> "'lala'", which appears as 'lala' in
            the db.
        file_val : optional, {None, <anything>}
            If not None, then this is the value of the entry that it has in
            another context (actually used in the input file). If None, then
            file_val = val. 
            Example: K_POINTS in pw.x input file:
                sql_val: '2 2 2 0 0 0'
                file_val: 'K_POINTS automatic\n2 2 2 0 0 0'
        key : optional, {None, str}
            An optional key. This key should refer to the column name in the 
            database table, as in:
                % create table calc (key1 sql_type1, key2    sql_type2, ...)
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
        self.sql_type = sql_type
        self.file_val = sql_val if file_val is None else file_val
        self.sql_val = self._fix_sql_val(sql_val)
        self.key = key
    
    def _fix_sql_val(self, sql_val):
        if isinstance(sql_val, str):
            # "lala" -> "'lala'"
            # 'lala' -> "'lala'"
            return repr(sql_val)
        else:
            return sql_val


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
            
    def execute(self, cmd):
        """
        args:
        -----
        cmd : str
            SQLite command
        """            
        return self.cur.execute(cmd)

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


