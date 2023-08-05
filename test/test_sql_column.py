from pwtools.sql import sql_column
import pytest


def test_sql_column():
    x = sql_column(key='foo',
                   sqltype='integer',
                   lst=[1,2,3])
    for num, xx in zip([1,2,3], x):
        assert xx.sqlval == num
        assert xx.fileval == num

    x = sql_column(key='foo',
                   sqltype='integer',
                   lst=[1,2,3],
                   fileval_func=lambda z: "k=%i"%z)
    for num, xx in zip([1,2,3], x):
        assert xx.sqlval == num
        assert xx.fileval == "k=%i" %num

    x = sql_column(key='foo',
                   sqltype='integer',
                   lst=[1,2,3],
                   sqlval_func=lambda z: z**2,
                   fileval_func=lambda z: "k=%i"%z)
    for num, xx in zip([1,2,3], x):
        assert xx.sqlval == num**2
        assert xx.fileval == "k=%i" %num

    for xx in sql_column('foo', [1,2]):
        assert xx.sqltype == 'INTEGER'
    for xx in sql_column('foo', [1.0,2.0]):
        assert xx.sqltype == 'REAL'


def test_sql_column_fail_for_mixed_types():
    with pytest.raises(AssertionError):
        s = sql_column('foo', [1,2.0])
