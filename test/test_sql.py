# Test the sql module (and Python's sqlite3). We create one db, and inside,
# one table named "calc". SQL statements can be lowercase or uppercase. i.e.
# "SELECT * FROM calc WHERE idx==1" == "select * from calc where idx==1".

import os
import numpy as np
from pwtools.sql import SQLiteDB, SQLEntry
from pwtools import common
from testenv import testdir
pj = os.path.join

def test():

    # --- SQLiteDB ----------------------------------------------------
    dbfn = pj(testdir, 'test.db')
    if os.path.exists(dbfn):
        os.remove(dbfn)

    header = [('idx', 'INTEGER'), ('foo', 'FLOAT'), ('bar', 'TEXT')]
    db = SQLiteDB(dbfn, table='calc')
    db.execute("CREATE TABLE calc (%s)" %','.join("%s %s" %(x[0], x[1]) \
                                                  for x in header)) 

    vals = [[0, 1.1, 'a'],
            [1, 2.2, 'b'],
            [2, 3.3, 'c']]
    for lst in vals:
        db.execute("INSERT INTO calc (idx, foo, bar) VALUES (?,?,?)", tuple(lst))
    db.commit()

    assert header == db.get_header()
    # call sqlite3, the cmd line interface
    assert common.backtick("sqlite3 %s 'select * from calc'" %dbfn) \
        == '0|1.1|a\n1|2.2|b\n2|3.3|c\n'

    # ret = 
    # [(0, 1.1000000000000001, u'a'),
    # (1, 2.2000000000000002, u'b'),
    # (2, 3.2999999999999998, u'c')]
    ret = db.execute("select * from calc").fetchall()
    for idx, lst in enumerate(vals):
        assert list(ret[idx]) == lst

    # generator object, yields
    # tup = (0, 1.1000000000000001, u'a')
    # tup = (1, 2.2000000000000002, u'b')
    # tup = (2, 3.2999999999999998, u'c')
    itr = db.execute("select * from calc")
    for idx, tup in enumerate(itr):
        assert list(tup) == vals[idx]

    # [(0, 1.1000000000000001, u'a')]
    assert db.execute("select * from calc where idx==0").fetchall() == \
        [tuple(vals[0])]
    # (0, 1.1000000000000001, u'a')
    assert db.execute("select * from calc where idx==0").fetchone() == \
        tuple(vals[0])

    assert db.execute("select bar from calc where idx==0").fetchone()[0] == \
        'a'
    
    # get_list1d(), get_array1d(), get_array()
    assert db.get_list1d("select idx from calc") == [0,1,2]
    np.testing.assert_array_equal(db.get_array1d("select idx from calc"),
                                  np.array([0,1,2]))
    np.testing.assert_array_equal(db.get_array("select idx from calc"), 
                                  np.array([0,1,2])[:,None])
    np.testing.assert_array_equal(db.get_array("select idx,foo from calc"), 
                                  np.array(vals)[:,:2].astype(float))

    # add_column(), fill with values
    db.add_column('baz', 'TEXT')
    add_header = [('baz', 'TEXT')]
    header += add_header
    assert db.get_header() == header
    db.execute("UPDATE %s SET baz='xx' where idx==0" %db.table)
    db.execute("UPDATE %s SET baz='yy' where idx==1" %db.table)
    db.execute("UPDATE %s SET baz='zz' where idx==2" %db.table)
    db.commit()
    print common.backtick("sqlite3 %s 'select * from calc'" %dbfn)
    print db.execute("select baz from calc").fetchall()
    assert db.execute("select baz from calc").fetchall() == \
        [(u'xx',), (u'yy',), (u'zz',)]
    
    # add even more cols with add_columns()
    add_header = [('bob', 'TEXT'), ('alice', 'BLOB')]
    header += add_header
    db.add_columns(add_header)
    assert db.get_header() == header
    
    # create_table()
    dbfn2 = pj(testdir, 'test2.db')
    header2 = [('a', 'FLOAT'), ('b', 'TEXT')]
    db2 = SQLiteDB(dbfn2, table='foo')
    db2.create_table(header2)
    assert db2.get_header() == header2

    # --- SQLEntry ----------------------------------------------------
    x = SQLEntry(1, 'integer')
    assert x.sqlval == 1
    assert x.sqltype == 'INTEGER'
    assert x.fileval == 1
    x = SQLEntry(sqltype='INTEGER', sqlval=1)
    assert x.sqlval == 1
    assert x.sqltype == 'INTEGER'
    assert x.fileval == 1
    x = SQLEntry(sqltype='text', sqlval='lala', fileval='xx\nlala')
    assert x.sqlval == "'lala'"
    assert x.sqltype == 'TEXT'
    assert x.fileval == 'xx\nlala'
    # auto type detection
    mapping = \
        [('NULL',     None),    
         ('INTEGER',  1),       
         ('INTEGER',  long(1)), 
         ('REAL',     1.0),     
         ('TEXT',     'xx'),    
         ('TEXT',     u'xx'),   
         ('BLOB',     np.array([1,2,3]).data)]
    for sqltype, val in mapping:
        print val, sqltype
        x = SQLEntry(sqlval=val)
        assert x.sqltype == sqltype
    
