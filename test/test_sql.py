# Test the sql module (and Python's sqlite3). We create one db, and inside,
# one table named "calc". SQL statements can be lowercase or uppercase. i.e.
# "SELECT * FROM calc WHERE idx==1" == "select * from calc where idx==1".

import os
from pwtools.sql import SQLiteDB, SQLEntry
from pwtools import common
pj = os.path.join

dr = '/tmp/pwtools_test'
dbfn = pj(dr, 'test.db')
if not os.path.exists(dr):
    os.makedirs(dr)
if os.path.exists(dbfn):
    os.remove(dbfn)

db = SQLiteDB(dbfn)
db.execute("CREATE TABLE calc (idx INTEGER, foo FLOAT, bar TEXT)")

vals = [[0, 1.1, 'a'],
        [1, 2.2, 'b'],
        [2, 3.3, 'c']]
for lst in vals:
    db.execute("INSERT INTO calc (idx, foo, bar) VALUES (?,?,?)", tuple(lst))
db.commit()

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

x = SQLEntry('integer', 1)
assert x.sqlval == 1
assert x.sqltype == 'integer'
assert x.fileval == 1
x = SQLEntry(sqltype='integer', sqlval=1)
assert x.sqlval == 1
assert x.sqltype == 'integer'
assert x.fileval == 1
x = SQLEntry(sqltype='text', sqlval='lala', fileval='xx\nlala')
assert x.sqlval == "'lala'"
assert x.sqltype == 'text'
assert x.fileval == 'xx\nlala'
