#!/usr/bin/env python

"""
Example session for using pwtools.batch. This script is self-contained. Just
run it and inspect the created files.

step 1
------
Define study_name. Use any string you like, like 'my_calc'.

step 2
-------
Create Machine instances. For each machine, calculations are written to
calc_host1/ and calc_host2/, if requested in ParameterStudy.

step 3
------
Create list of templates. In FileTemplate, the default placeholders are
'XXXFOO' for replacement key 'foo'. Use batch.default_repl_keys() to get a list
of all supported default replacement keys, which are defined in ParameterStudy,
Calculation and Machine.

step 4
-------
Define your parameters to vary (ecutwfc and pseudo). They are added to
the default replacements.

step 5
------
Write files for each parameter set. Replacements are done in each specified
FileTemplate, with keys from batch.default_repl_keys() and `params_lst`.

step 6
------
Extend the study, this time writing for both machines. Since for one machine,
input is already there, a backup will be made (calc_host1.0). Observe the
changes to calc.db: revision=1, column "hostname" mentions both machines.
calc_*/run.sh refers only to the new calculations. A file "excl_push" is
written, which lists all old calculation indices. Can be used with ``rsync
--exclude-from=excl_push``.
"""

import os
import numpy as np
from pwtools import common, batch, sql, comb
pj = os.path.join

# step 1
##study_name = 'job_' + os.path.basename(common.fullpath('.'))
study_name = 'my_calc'

# step 2
host1 = \
    batch.Machine(hostname='host1',
                  subcmd='bash',
                  scratch='/tmp',
                  jobfn='job.host1',
                  home='/home/schmerler')

host2 = \
    batch.Machine(hostname='host2',
                  subcmd='qsub',
                  scratch='/scratch/schmerler',
                  jobfn='job.host2',
                  home='/home/schmer42')

# step 3
txt_in = """
# input file for super fast simulation code
data_dir    =   XXXHOME/share/pseudo
calc_name   =   XXXCALC_NAME
my_path     =   XXXHOME/calculations
study_name  =   XXXSTUDY_NAME
scratch     =   /scratch/XXXSTUDY_NAME/XXXIDX
ecutwfc     =   XXXECUTWFC
pseudo      =   XXXPSEUDO
"""
txt_job_host1 = """
# job file for host1
#PBS -N XXXCALC_NAME
#PBS -q short.q
data_dir    =   XXXHOME/share/pseudo
scratch     =   /scratch/XXXSTUDY_NAME/XXXIDX
# running job
mkdir -pv $scratch
my_app.x < input.in > output.out
"""
txt_job_host2 = """
# job file for host2
#
#BSUB -N XXXCALC_NAME
#BSUB -q long.q
data_dir    =   XXXHOME/share/pseudo
scratch     =   /big/share/fastfs/XXXSTUDY_NAME/XXXIDX
# running job
mkdir -pv $scratch
/path/to/apps/bin/my_app.x < input.in > output.out
"""
# FileTemplate using `txt`
#
# templates = [batch.FileTemplate(basename='input.in',txt=txt_in)] 
# for fn, txt in zip([host1.jobfn, host2.jobfn],[txt_job_host1, txt_job_host2]):
#     templates.append(batch.FileTemplate(basename=fn, txt=txt))
#
# usual case: read templates from disk (in this example we write them first)
if not os.path.exists('calc.templ'):
    os.makedirs('calc.templ')
common.file_write('calc.templ/job.host1', txt_job_host1)
common.file_write('calc.templ/job.host2', txt_job_host2)
common.file_write('calc.templ/input.in', txt_in)
machines = [host1, host2]
templates = [batch.FileTemplate(basename=m.jobfn) for m in machines] + \
            [batch.FileTemplate(basename='input.in')]


# step 4
#
# ecutwfc = sql.sql_column('ecutwfc', np.array([50,60,70,80.0]))
# pseudo = sql.sql_column('pseudo', ['Si.pp', 'Si.paw'])
# params_lst = comb.nested_loops([ecutwfc,pseudo])
#
params_lst = []
for ecutwfc in np.array([50,60.0]):
    for pseudo in ['Si.pp', 'Si.paw']:
        params_lst.append([sql.SQLEntry(key='ecutwfc', sqlval=ecutwfc),
                           sql.SQLEntry(key='pseudo', sqlval=pseudo)])

# step 5
calc = batch.ParameterStudy(machines=host1,
                            templates=templates,
                            params_lst=params_lst, 
                            study_name=study_name,
                            )
calc.write_input(sleep=0, backup=True, mode='a')

# step 6
params_lst = []
for ecutwfc in np.array([70,80.0]):
    for pseudo in ['Si.paw']:
        params_lst.append([sql.SQLEntry(key='ecutwfc', sqlval=ecutwfc),
                           sql.SQLEntry(key='pseudo', sqlval=pseudo)])

calc = batch.ParameterStudy(machines=[host1,host2],
                            templates=templates,
                            params_lst=params_lst, 
                            study_name=study_name,
                            )
calc.write_input(sleep=0, backup=True, mode='a')

# Load written sqlite DB and print table.
print common.backtick("sqlite3 -column -header calc.db \
                       'select * from calc'")
# Some example db query using Python.    
db = sql.SQLiteDB('calc.db', table='calc')
print db.get_dict("select idx,ecutwfc,pseudo from calc where ecutwfc <= 60")
