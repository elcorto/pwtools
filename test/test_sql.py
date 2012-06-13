# Test the sql module (and Python's sqlite3). We create one db, and inside,
# one table named "calc". SQL statements can be lowercase or uppercase. i.e.
# "SELECT * FROM calc WHERE idx==1" == "select * from calc where idx==1".

import os
import numpy as np
from pwtools.sql import SQLiteDB, SQLEntry
from pwtools import sql
from pwtools import common
from testenv import testdir
pj = os.path.join

def test_sql():

    # --- SQLiteDB ----------------------------------------------------
    dbfn = pj(testdir, 'test.db')
    if os.path.exists(dbfn):
        os.remove(dbfn)

    header = [('idx', 'INTEGER'), ('foo', 'REAL'), ('bar', 'TEXT')]
    db = SQLiteDB(dbfn, table='calc')
    db.execute("CREATE TABLE calc (%s)" %','.join("%s %s" %(x[0], x[1]) \
                                                  for x in header)) 

    vals = [[0, 1.1, 'a'],
            [1, 2.2, 'b'],
            [2, 3.3, 'c']]
    for lst in vals:
        db.execute("INSERT INTO calc (idx, foo, bar) VALUES (?,?,?)", tuple(lst))
    db.commit()
    
    # get_max_rowid
    assert db.get_max_rowid() == 3

    # has_table
    assert db.has_table('calc')
    assert not db.has_table('foo')
    
    # has_column
    assert db.has_column('idx')
    assert not db.has_column('grrr')

    # get_single
    assert float(db.get_single("select foo from calc where idx==0")) == 1.1

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
    db.execute("UPDATE %s SET baz=? where idx==2" %db.table, ('zz',))
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
    header2 = [('a', 'REAL'), ('b', 'TEXT')]
    db2 = SQLiteDB(dbfn2, table='foo')
    db2.create_table(header2)
    assert db2.get_header() == header2
    
    # get_dict()
    dct = db.get_dict("select foo,bar from calc")
    cols = [x[0] for x in db.get_header()]
    for key in ['foo', 'bar']:
        assert key in cols
    foo = db.get_list1d("select foo from calc")
    bar = db.get_list1d("select bar from calc")
    assert foo == dct['foo']
    assert bar == dct['bar']
    
    # sql_matrix
    lists = [['a', 1.0], ['b', '2.0']]
    colnames = ['foo', 'bar']
    types = ['TEXT', 'REAL']
    sql_lists = sql.sql_matrix(lists=lists, colnames=colnames)
    assert sql_lists[0][0].sqltype == 'TEXT'
    assert sql_lists[1][0].sqltype == 'TEXT'
    assert sql_lists[0][1].sqltype == 'REAL'
    assert sql_lists[1][1].sqltype == 'REAL'
    for ii, row in enumerate(sql_lists):
        for jj, entry in enumerate(row):
            assert entry.sqlval == lists[ii][jj]
    sql_lists = sql.sql_matrix(lists=lists, colnames=colnames,
        fileval_funcs={'foo': lambda x: str(x)+'-lala'} )
    for ii, row in enumerate(sql_lists):
        for jj, entry in enumerate(row):
            if entry.key == 'foo':
                assert entry.fileval == str(lists[ii][jj]) + '-lala'
            else:
                assert entry.fileval == lists[ii][jj]
            assert entry.sqlval == lists[ii][jj]
    
    # makedb
    dbfn = pj(testdir, 'test3.db')
    sql.makedb(filename=dbfn, lists=lists, colnames=colnames, mode='w')
    db = sql.SQLiteDB(dbfn, table='test3')
    dct =  db.get_dict("select * from test3")
    assert dct['foo'] == [u'a', u'b']
    assert dct['bar'] == [1.0, 2.0]
    sql.makedb(filename=dbfn, lists=lists, colnames=colnames, mode='a')
    db = sql.SQLiteDB(dbfn, table='test3')
    dct =  db.get_dict("select * from test3")
    assert dct['foo'] == [u'a', u'b']*2
    assert dct['bar'] == [1.0, 2.0]*2
    
    # attach_column, fill_column
    db.attach_column('baz', values=[1,2,3,4,5,6], 
                     extend=False, start=1)
    for col, typ in db.get_header():
        if col == 'baz':
            assert typ == 'INTEGER'
    assert db.get_max_rowid() == 4                     
    assert db.get_list1d('select baz from test3') == [1,2,3,4]
    # need `extend` b/c up to now, table has 4 rows
    db.fill_column('baz', values=[5,6], 
                    extend=True, start=5)
    assert db.get_max_rowid() == 6                     
    assert db.get_list1d('select baz from test3') == [1,2,3,4,5,6]
    assert db.get_list1d("select foo from test3") == [u'a', u'b']*2 + [None]*2
    assert db.get_list1d("select bar from test3") ==  [1.0, 2.0]*2 + [None]*2
    # `extend` kwd not needed b/c table already has 6 rows
    db.attach_column('baz2', values=[1,2,3,4,5,6], 
                     start=1)
    assert db.get_list1d('select baz2 from test3') == [1,2,3,4,5,6]
    db.fill_column('baz2', values=[1,4,9,16], 
                    overwrite=True, start=1)
    assert db.get_list1d('select baz2 from test3') == [1,4,9,16,5,6]

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
    assert x.sqlval == 'lala'
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
    
    # uppercase type magic
    assert sql.fix_sqltype('integer') == 'INTEGER'
    assert sql.fix_sqltype('float') == 'REAL'
    assert sql.fix_sql_header([('a', 'text'), ('b', 'float')]) == \
           [('a', 'TEXT'), ('b', 'REAL')]

