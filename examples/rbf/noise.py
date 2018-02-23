import numpy as np
from matplotlib import cm
from pwtools import mpl, rbf, num
plt = mpl.plt

# multi and inv_multi always behave almost identical
rbfs = {
    'gauss': rbf.RBFGauss,
    'multi': rbf.RBFMultiquadric,
##    'inv_multi': rbf.RBFInverseMultiquadric,
    }

def go(nn, ax, opt=False, seed=None):
    x = np.linspace(0, 10, nn)
    xi = np.linspace(x[0], x[-1], 300) 
    rnd = np.random.RandomState(seed=seed)
    y = np.sin(x) + rnd.rand(len(x))
    ax.plot(x, y, 'o', alpha=0.3)
    for name,func in rbfs.items():
        f = rbf.RBFInt(x[:,None], y, rbf=func()) 
        if opt:
            p0 = f.get_param('est')
            f.fit_opt_param()
            p1 = f.rbf.param
            print("{}: p0={} p1={}".format(name, p0, p1))
        else:
            f.fit()
        ax.plot(xi, f(xi[:,None]), label=name)

if __name__ == '__main__':
    
    ri = lambda: np.random.randint(0, 9999)
    seed_lo = ri()
    seed_hi = ri()
    for opt in [True, False]:
        fig,axs = plt.subplots(2, 1, sharex=True)
        go(15,  axs[0], opt=opt, seed=seed_lo)
        go(100, axs[1], opt=opt, seed=seed_hi)
        axs[1].legend()
##        fig.savefig('/tmp/rbf_1d_opt_{}.png'.format(opt))
        axs[0].set_title("opt={}".format(opt))
    plt.show()
