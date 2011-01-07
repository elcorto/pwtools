from pwtools.batch import sql_column

def test():
    x = sql_column('foo', 'integer', [1,2,3])
    for num, xx in zip([1,2,3], x):
        assert xx.sqlval == num
        assert xx.fileval == num
    
    x = sql_column('foo', 'integer', [1,2,3], fileval_func=lambda z: "k=%i"%z)
    for num, xx in zip([1,2,3], x):
        assert xx.sqlval == num
        assert xx.fileval == "k=%i" %num 

    x = sql_column('foo', 
                   'integer', 
                   [1,2,3], 
                   sqlval_func=lambda z: z**2,
                   fileval_func=lambda z: "k=%i"%z)
    for num, xx in zip([1,2,3], x):
        assert xx.sqlval == num**2
        assert xx.fileval == "k=%i" %num        
