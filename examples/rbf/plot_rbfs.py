#!/usr/bin/env python3

import numpy as np

from pwtools import rbf, mpl
plt = mpl.plt

fig,ax = plt.subplots()
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

r = np.linspace(-5,5,200)
for color, tup in zip(colors, rbf.rbf_dct.items()):
    name, func = tup
    for p,ls in [(1, "-"), (0.1, "--")]:
        ax.plot(r, func(r**2, p=p), label=f"{name} p={p}", color=color,
                ls=ls)

ax.set_xlabel("r")
ax.set_ylabel("$\phi(r)$")
ax.legend()
fig.savefig('/tmp/rbfs.png')
plt.show()
