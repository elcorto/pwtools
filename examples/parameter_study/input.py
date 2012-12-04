#!/usr/bin/env python

import os
import numpy as np
from pwtools import common, batch, sql, comb
pj = os.path.join

# step 1
# ------
# Define prefix = calculation name usually. Use any string you like, like
# 'my_calc'.
#
# step 2
# -------
# Create Machine instance. In this case, just the local machine, which will run
# the job script 'job.local' by bash. See calc_local/run.sh. 
#
# step 3
# ------
# Create list of templates. Here we use FileTemplate's `txt` keyword to pass
# the template text. In FileTemplate, the default placeholders are 'XXXFOO' for
# parameter 'foo'.
#
# Note that each arg to Machine (scratch, home, ...) will be converted to a
# placeholder 'XXXSCRATCH', 'XXXHOME', ... below in ParameterStudy (by using
# FileTemplate) and can be used in template files.
#
# We could also write the template files to a directory (default is
# templ_dir='calc.templ)' and use ``FileTemplate(basename='input.in')``.
#
##if not os.path.exists('calc.templ'):
##    os.makedirs('calc.templ')
##common.file_write('calc.templ/job.local', txt_job)
##common.file_write('calc.templ/input.in', txt_in)
#
# step 4
# -------
# Define parameters to vary. Will result in this table:
#
# idx         ecutwfc     pseudo      scratch    
# ----------  ----------  ----------  --------------
# 0           50.0        Si.pp       /tmp/my_calc/0
# 1           50.0        Si.paw      /tmp/my_calc/1
# 2           60.0        Si.pp       /tmp/my_calc/2
# 3           60.0        Si.paw      /tmp/my_calc/3
# 4           70.0        Si.pp       /tmp/my_calc/4
# 5           70.0        Si.paw      /tmp/my_calc/5
# 6           80.0        Si.pp       /tmp/my_calc/6
# 7           80.0        Si.paw      /tmp/my_calc/7
#
#
# step 5
# ------
# Write files for each parameter set. The default dir to write to is
# 'calc_local' b/c Machine(name='local',...) was used.
#
# Creates dirs 
#   calc_local/0
#   ...
#   calc_local/7
# and write input.in and job.local with parameters replaced. 
#
# The parameter names 'ecutwfc' and 'pseudo' will be converted to 'XXXECUTWFC'
# and 'XXXPSEUDO' in ParameterStudy by FileTemplate and replaced in each file
#   calc_local/0/input.in
#   calc_local/0/job.local
#   calc_local/1/input.in 
#   ...
# XXXSCRATCH will get a default value.  

# step 1
##prefix = 'job_' + os.path.basename(common.fullpath('.'))
prefix = 'my_calc'

# step 2
local = batch.Machine(hostname='local',
                      subcmd='bash',
                      scratch='/tmp',
                      jobfn='job.local',
                      home='/home/schmerler')

# step 3
txt_in = """
my_path=XXXHOME/calculations
prefix=XXXPREFIX
scratch=XXXSCRATCH
ecutwfc=XXXECUTWFC
pseudo=XXXPSEUDO
"""
txt_job = """
scratch=XXXSCRATCH
jobname=XXXPREFIX
my_app.x < input.in > output.out
"""
templates = [batch.FileTemplate(basename=fn,txt=txt) for fn,txt in \
    zip(['input.in', 'job.local'],[txt_in, txt_job])]    

# step 4
ecutwfc = sql.sql_column('ecutwfc', np.array([50,60,70,80.0]))
pseudo = sql.sql_column('pseudo', ['Si.pp', 'Si.paw'])
params_lst = comb.nested_loops([ecutwfc,pseudo])

# step 5
calc = batch.ParameterStudy(machine=local,
                            templates=templates,
                            params_lst=params_lst, 
                            prefix=prefix,
                            )
calc.write_input(sleep=0, backup=True, mode='a')

# Load written sqlite DB and print table.
db = sql.SQLiteDB('calc_local/calc.db', table='calc')
print common.backtick("sqlite3 -column -header calc_local/calc.db \
    'select idx,ecutwfc,pseudo,scratch from calc'")
