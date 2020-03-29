#!/usr/bin/env python

"""
Example session for using pwtools.batch. This script is self-contained. Just
run it and inspect the created files. Only the files in calc.templ/ are needed.

Notes
=====

Another way to specify the parameters in step 4
-----------------------------------------------

::

    ecutwfc = sql.sql_column('ecutwfc', np.array([50,60,70,80.0]))
    pseudo = sql.sql_column('pseudo', ['Si.pp', 'Si.paw'])
    params_lst = comb.nested_loops([ecutwfc,pseudo])


File templates
--------------

It is not mandatory to have template files in a dir `templ_dir` on disk
as in::

    FileTemplate(basename='job.host1', templ_dir='calc.templ')

which is assumed if you do::

    Machine(..., filename='calc.templ/job.host1')

You can also pass the text content of the template file directly::

    FileTemplate(basename='job.host1', txt=...)
    Machine(..., template=FileTemplate(basename='job.host1', txt=...))

The same applies to FileTemplate instances passed to ParameterStudy with the
`templates` keyword.
"""

import os
import numpy as np
from pwtools import common, batch, sql, comb
pj = os.path.join

# step 1
# ------
#
# Define study_name. Use any string you like.
#
study_name = 'job_' + os.path.basename(common.fullpath('.'))

# step 2
# ------
#
# Create Machine instances. For each machine, calculations are written to
# calc_host0/ and calc_host1/, if requested in ParameterStudy. Job template files
# are defined by `filename` or `template`. These are usually shell scripts which
# you use on a cluster to submit the job to a batch system such as LSF, PBS,
# SLURM, ...
#
host0 = \
    batch.Machine(hostname='host0',
                  subcmd='bash',
                  scratch='/tmp',
                  filename='calc.templ/job.host0',
                  home='/home/schmerler')

host1 = \
    batch.Machine(hostname='host1',
                  subcmd='qsub',
                  scratch='/scratch/schmerler',
                  filename='calc.templ/job.host1',
                  home='/home/schmer42')
machines = [host0, host1]

# step 3
# ------
#
# Create list of templates for calculation input. In FileTemplate, the default
# placeholders are 'XXXFOO' for replacement key 'foo'. Use
# batch.default_repl_keys() to get a list of all supported default replacement
# keys, which are defined in ParameterStudy, Calculation and Machine. Templates
# and placeholders defined in Calculation and Machine are automatically added
# later in ParameterStudy.
#
# The default dir where template files are supposed to be found is
# FileTemplate(..., templ_dir='calc.templ'). Since we have that, passing a
# `basename` is sufficient.
#
templates = [batch.FileTemplate(basename='input.in')]


# step 4
# ------
#
# Define your parameters to vary (ecutwfc and pseudo). They are added to
# the default replacements.
#
params_lst = []
for ecutwfc in np.array([50,60.0]):
    for pseudo in ['Si.pp', 'Si.paw']:
        params_lst.append([sql.SQLEntry(key='ecutwfc', sqlval=ecutwfc),
                           sql.SQLEntry(key='pseudo', sqlval=pseudo)])


# step 5
# ------
#
# Write files for each parameter set. Replacements are done in each specified
# FileTemplate, with keys from batch.default_repl_keys() and `params_lst`.
#
calc = batch.ParameterStudy(machines=host0,
                            templates=templates,
                            params_lst=params_lst,
                            study_name=study_name,
                            )
calc.write_input(sleep=0, backup=True, mode='a')


# step 6
# ------
#
# Now, we extend the study. This time we write input for both machines. Since
# for one machine, input is already there, a backup will be made
# (calc_host0.0). Observe the changes to calc.db: revision=1, column "hostname"
# mentions both machines. calc_*/run.sh refers only to the new calculations. A
# file "excl_push" is written, which lists all old calculation indices. Can be
# used with ``rsync --exclude-from=excl_push``.
#
params_lst = []
for ecutwfc in np.array([70,80.0]):
    for pseudo in ['Si.paw']:
        params_lst.append([sql.SQLEntry(key='ecutwfc', sqlval=ecutwfc),
                           sql.SQLEntry(key='pseudo', sqlval=pseudo)])

calc = batch.ParameterStudy(machines=[host0,host1],
                            templates=templates,
                            params_lst=params_lst,
                            study_name=study_name,
                            )
calc.write_input(sleep=0, backup=True, mode='a')


# Load written sqlite DB and print table.
print(common.backtick("sqlite3 -column -header calc.db \
                       'select * from calc'"))

# Some example db query using Python.
db = sql.SQLiteDB('calc.db', table='calc')
print(db.get_dict("select idx,ecutwfc,pseudo from calc where ecutwfc <= 60"))
