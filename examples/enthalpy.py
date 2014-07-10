"""
Clarify the difference between the enthalpies
    H = E + P * V 
with
    P = P(V) = -dE/dV
and
    H = E + Pconst * V 
In the latter case, min_V [ E + Pconst * V ] provides V_opt and
P(V_opt) == Pconst.
"""

import numpy as np
from pwtools import mpl, num

pl = mpl.Plot()
ax = pl.ax
pl.ax2 = pl.ax.twinx(); 
ax2 = pl.ax2

v=np.linspace(1,5,20) 
e=(v-3)**2+1; 
p=-2*(v-3); 
Pconst = 2.0
ax.plot(v,e, label='E(V)'); 
ax.plot([3], [1], 'bo')

# E at each point + P at each point * V = minimal enthalpy at each point
# (minimal w.r.t. the target pressure at each point), i.e. the real enthalpy
ax.plot(v, e+p*v, label='H=E+P(V)*V');

# E at each point  + constant P * V -> minimize this H(V) to find optimal
# V and minimal H for *only this* pressure
ax.plot(v, e+Pconst*v, label='H=E+Pconst*V');
ax.plot([2], [6], 'ro')


ax.grid();
ax2.plot(v, p, 'k', label='P=-dE/dV');
ax2.hlines(Pconst, *ax2.get_xlim(), color='m', label='Pconst=%g' %Pconst)
ax2.plot([2],[2], 'mo')

ax.set_ylabel('H')
ax.set_xlabel('V')
ax2.set_ylabel('P')
pl.legend(legaxname='ax', axnames=['ax','ax2'], loc='lower left')
mpl.plt.show()

