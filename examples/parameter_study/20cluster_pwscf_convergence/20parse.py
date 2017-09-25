#!/usr/bin/python

# Parse each pw.out and write results/idx/struct.pk

from pwtools import sql, io

db = sql.SQLiteDB('calc.db', table='calc')

for idx in db.get_list1d("select idx from calc"):
    print(idx)
    st = io.read_pw_scf('calc/%i/pw.out' %idx)
    st.dump('results/%i/struct.pk' %idx)
