#!/usr/bin/python

# Load parsed results and put some values in the database.

from pwtools import sql, io, num

db = sql.SQLiteDB('calc.db', table='calc')
idx_lst = db.get_list1d("select idx from calc")

cols = [('etot', 'float'),          # eV
        ('pressure', 'float'),      # GPa
        ('volume', 'float'),        # Ang**3
        ('forces_rms', 'float'),    # eV / Ang
        ('sxx', 'float'),           # GPa
        ('syy', 'float'),           # GPa
        ('szz', 'float'),           # GPa
        ]
db.add_columns(cols)

for idx in idx_lst:
    print idx
    struct = io.cpickle_load('results/%i/traj.pk' %idx)[-1]
    db.execute("update calc set etot=? where idx==?", (struct.etot, idx))
    db.execute("update calc set volume=? where idx==?", (struct.volume, idx))
    db.execute("update calc set pressure=? where idx==?", (struct.pressure, idx))
    db.execute("update calc set sxx=? where idx==?", (struct.stress[0,0], idx))
    db.execute("update calc set syy=? where idx==?", (struct.stress[1,1], idx))
    db.execute("update calc set szz=? where idx==?", (struct.stress[2,2], idx))
    db.execute("update calc set forces_rms=? where idx==?",
               (num.rms(struct.forces), idx))
db.commit()
