import os
from pwtools.batch import FileTemplate
from pwtools.common import file_write, file_read
from pwtools import sql
from testenv import testdir
pj = os.path.join

def test():
    templ_dir = pj(testdir, 'calc.templ')
    templ_fn = pj(templ_dir, 'foo.in')
    tgt_dir = pj(testdir, 'calc')
    tgt_fn = pj(tgt_dir, 'foo.in')
    for dr in [templ_dir, tgt_dir]:
        if not os.path.exists(dr):
            os.makedirs(dr)
    
    templ_txt = "XXXFOO XXXBAR XXXBAZ"
    file_write(templ_fn, templ_txt)
    
    # specify keys
    templ = FileTemplate(basename='foo.in', 
                         keys=['foo', 'bar'],
                         templ_dir=templ_dir)
    rules = {'foo': 1, 'bar': 'lala', 'baz': 3}
    templ.write(rules, calc_dir=tgt_dir)
    assert file_read(tgt_fn).strip() == "1 lala XXXBAZ"
    
    # no keys
    templ = FileTemplate(basename='foo.in', 
                         templ_dir=templ_dir)
    rules = {'foo': 1, 'bar': 'lala', 'baz': 3}
    templ.write(rules, calc_dir=tgt_dir)
    assert file_read(tgt_fn).strip() == "1 lala 3"
    
    # sql
    rules = {'foo': sql.SQLEntry(sqltype='integer', sqlval=1),
             'bar': sql.SQLEntry(sqltype='text', sqlval='lala'),
             'baz': sql.SQLEntry(sqltype='integer', sqlval=3)}
    templ.writesql(rules, calc_dir=tgt_dir)
    assert file_read(tgt_fn).strip() == "1 lala 3"
    
    # non-default placefolders
    templ_txt = "@foo@ @bar@"
    file_write(templ_fn, templ_txt)
    templ = FileTemplate(basename='foo.in', 
                         templ_dir=templ_dir,
                         func=lambda x: "@%s@" %x)
    rules = {'foo': 1, 'bar': 'lala'}
    templ.write(rules, calc_dir=tgt_dir)
    assert file_read(tgt_fn).strip() == "1 lala"
