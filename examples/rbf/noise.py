#!/usr/bin/python3

import numpy as np
from pwtools import mpl, rbf
plt = mpl.plt

def go(nn, ax):
    x = np.linspace(0, 10, nn)
    xi = np.linspace(x[0], x[-1], 300) 
    y = np.sin(x) + np.random.rand(len(x))
    ax.plot(x, y, 'o', alpha=0.3)
    for name in ['gauss', 'multi']:
        f = rbf.RBFInt(x[:,None], y, rbf=name) 
        ax.plot(xi, f(xi[:,None]), label=name)

if __name__ == '__main__':
    
    fig,axs = plt.subplots(2, 1, sharex=True)
    go(15,  axs[0])
    go(100, axs[1])
    axs[1].legend()
    plt.show()
