#!/usr/bin/python

# Print result of convergence study: differences of etot, pressure

from pwtools import sql, batch, mpl

db = sql.SQLiteDB('calc.db')
etot_fac = 1000.0/4 # eV -> meV/atom, 4 atoms
data = db.get_array("select ecutwfc,etot,pressure from calc order by ecutwfc")
print "ecutwfc, diff(etot) [meV/atom], diff(pressure) [GPa]"
print batch.conv_table(data[:,0],
                      [data[:,1]*etot_fac, data[:,2]],
                       mode='last', orig=False)

# plotting
fig,ax = mpl.fig_ax()
ax.plot(data[:,0], (data[:,1]-data[-1,1])*etot_fac, label='etot', color='b')
ax.set_ylabel('diff(etot) [meV/atom]')
ax.set_xlabel('ecutwfc [Ry]')
ax.legend()

mpl.plt.show()
