#!/usr/bin/python3

# Plot E(V) curve.

from pwtools import sql, mpl, io

db = sql.SQLiteDB('calc.db')
data = db.get_array("select volume,etot from calc order by volume")
natoms = io.cpickle_load('results/0/traj.pk').natoms

# plotting
fig,ax = mpl.fig_ax()
ax.plot(data[:,0]/float(natoms), data[:,1]/float(natoms))
ax.set_ylabel('energy/atom [eV]')
ax.set_xlabel('volume/atom [Ang^3]')

mpl.plt.show()
