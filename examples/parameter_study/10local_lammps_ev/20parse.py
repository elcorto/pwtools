#!/usr/bin/env python3

# Parse each lammps output and write results/idx/traj.pk

from pwtools import sql, io

db = sql.SQLiteDB('calc.db', table='calc')

for idx in db.get_list1d("select idx from calc"):
    print(idx)
    tr = io.read_lammps_md_txt('calc/%i/log.lammps' %idx)
    tr.dump('results/%i/traj.pk' %idx)
