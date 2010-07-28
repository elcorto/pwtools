#!/usr/bin/env python

import shutil
import os
pj = os.path.join

import numpy as np
from pwtools import version
assert version.version >= version.tov('0.6.1')

from pwtools import common as com
from pwtools import comb as comb
from pwtools.sql import SQLEntry, SQLiteDB
fpj = com.fpj


if __name__ == '__main__':
    
    #------------------------------------------------------------------------
    # customize here
    #------------------------------------------------------------------------
    prefix_prefix = "zval"
    
    # list of SQLEntry's for each variable to vary
    nelec = [SQLEntry(sql_type='float', 
                     sql_val=x, 
                     key='nelec') \
            for x in [7,8,9, 9.2]]
    
    # List all template files. If keys=[], then nothing will be replaced in the
    # file. It will just be copied to the computation dir.
    templates = \
        {'pwin': com.FileTemplate(basename='pw.in', 
                                  keys=['prefix', 'nelec']),
         'upf':  com.FileTemplate(basename='Si.upf', 
                                  keys=[]),
        }                                
    
    # This is a list of lists. Each entry is a parameter set. For example, you
    # vary a = [1,2,3], b=['foo', 'bar'] and c=[11, 33]. Then this will be:
    #   >>> nested_loops([a,b,c])
    #   [[1, 'foo', 11],
    #    [1, 'foo', 33],
    #    [1, 'bar', 11],
    #    [1, 'bar', 33],
    #    [2, 'foo', 11],
    #    ...
    # 
    # Use zip to vary a and b together. The entries will be flattened below.
    #   >>> nested_loops([zip(a,b), c])
    #
    # Use this even if you have only one list, e.g. one parameter to vary.
    #   >>> nested_loops([a])
    loop_lists = comb.nested_loops([nelec])
    #------------------------------------------------------------------------
    

    calc_dir = 'calc'
    if not os.path.exists(calc_dir):
        os.makedirs(calc_dir)
    
    db_fn = pj(calc_dir, 'calc.db')
    have_new_db = not os.path.exists(db_fn)
    sql = SQLiteDB(db_fn)
    if not have_new_db and sql.has_column('calc', 'idx'):
        max_idx = sql.execute("select max(idx) from calc").fetchall()[0][0]
    else:
        max_idx = -1
    
    run_txt = "here=$(pwd)\n"
    sql_records =[]
    for _idx, lst in enumerate(loop_lists):
        idx = max_idx + _idx + 1
        dir = pj(calc_dir, str(idx))
        if not os.path.exists(dir):
            os.makedirs(dir)
        
        #--------------------------------------------- 
        prefix = prefix_prefix + "_run%i" %idx
        lst = com.flatten(lst)
        sql_record = {}
        
        # canonical entries
        sql_record['prefix'] = SQLEntry('text', prefix)
        sql_record['idx'] = SQLEntry('integer', idx)
        
        for sql_entry in lst:
            sql_record[sql_entry.key] = sql_entry
        #---------------------------------------------            
        
        for templ in templates.itervalues():
            templ.writesql(sql_record, dir)

        sql_records.append(sql_record)
        ##run_txt += "cd %i && qsub job.sge && sleep 1 && cd $here\n" %idx
        run_txt += "cd %i; mpirun -np 4 pw.x -input pw.in | tee pw.out; cd $here\n" %idx
    com.file_write(pj(calc_dir, 'run.sh'), run_txt)
    record = sql_records[0]
    keys = ",".join(record.keys())
    if have_new_db:
        header = ",".join("%s %s" %(key, entry.sql_type) for key, entry in
                          record.iteritems())
        sql.execute("create table calc (%s)" %header)
    for record in sql_records:
        sql_vals = ",".join(str(entry.sql_val) for entry in record.itervalues())
        cmd = "insert into calc (%s) values (%s)" %(keys, sql_vals)
        print cmd
        sql.execute(cmd) 
    sql.commit()        
    com.system("sqlite3 -header %s 'select * from calc' | column -t -s '|' > "
               "%s.txt" %(db_fn, db_fn))
