#!/usr/bin/env python3

"""
* define 2D scalar field z=f(x,y)
* sample random (x,y) points from f
* fit with RBF, plot and compare with original function, plot contour of
  difference -> fit is almost perfect
* calculate and plot the x-derivative of the fit, compare to analytic deriv ->
  also almost perfect match
"""

import numpy as np
from matplotlib import cm
from pwtools import mpl, rbf, num
plt = mpl.plt
rand = np.random.rand

plt.rcParams["figure.autolayout"] = True

# create shiny polished plot for documentation
export = False
##export = True
if export:
    savefig_opts = dict(bbox_inches='tight', pad_inches=0)
    plt.rcParams['font.size'] = 10


class SurfaceData:
    def __init__(self, xlim, ylim, nx, ny, mode):
        self.xlim = xlim
        self.ylim = ylim
        self.nx = nx
        self.ny = ny
        self.xg, self.yg = self.get_xy_grid()
        self.XG, self.YG = num.meshgridt(self.xg, self.yg)
        self.XY = self.make_XY(mode)

    def make_XY(self, mode='grid'):
        if mode == 'grid':
            XY = np.empty((self.nx * self.ny,2))
            XY[:,0] = self.XG.flatten()
            XY[:,1] = self.YG.flatten()
            ##for i in range(self.nx):
            ##    for j in range(self.ny):
            ##        XY[i*self.ny+j,0] = self.xg[i]
            ##        XY[i*self.ny+j,1] = self.yg[j]
        elif mode == 'rand':
            XY = rand(self.nx * self.ny, 2)
            XY[:,0] = XY[:,0] * (self.xlim[1] - self.xlim[0]) + self.xlim[0]
            XY[:,1] = XY[:,1] * (self.ylim[1] - self.ylim[0]) + self.ylim[0]
        return XY

    def get_xy_grid(self):
        x = np.linspace(self.xlim[0], self.xlim[1], self.nx)
        y = np.linspace(self.ylim[0], self.ylim[1], self.ny)
        return x,y

    def func(self, XY):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        if 'der' in kwargs:
            der = kwargs['der']
            kwargs.pop('der')
            if der == 'x':
                return self.deriv_x(*args, **kwargs)
            elif der == 'y':
                return self.deriv_y(*args, **kwargs)
            else:
                raise Exception("der != 'x' or 'y'")
        else:
            return self.func(*args, **kwargs)


class MexicanHat(SurfaceData):
    def func(self, XY):
        r = np.sqrt((XY**2).sum(axis=1))
        return np.sin(r)/r

    def deriv_x(self, XY):
        r = np.sqrt((XY**2).sum(axis=1))
        x = XY[:,0]
        return x * np.cos(r) / r**2 - x * np.sin(r) / r**3.0

    def deriv_y(self, XY):
        r = np.sqrt((XY**2).sum(axis=1))
        y = XY[:,1]
        return y * np.cos(r) / r**2 - y * np.sin(r) / r**3.0


class UpDown(SurfaceData):
    def func(self, XY):
        x = XY[:,0]
        y = XY[:,1]
        return x*np.exp(-x**2-y**2)


class SinExp(SurfaceData):
    def func(self, XY):
        x = XY[:,0]
        y = XY[:,1]
        return np.sin(np.exp(x)) * np.cos(y) + 0.5*y


class Square(SurfaceData):
    def func(self, XY):
        x = XY[:,0]
        y = XY[:,1]
        return (x**2 + y**2)


if __name__ == '__main__':

    # Some nice 2D examples

    fu = MexicanHat([-10,20], [-10,15], 20, 20, 'rand')
##    fu = UpDown([-2,2], [-2,2], 20, 20, 'grid')
##    fu = SinExp([-1,2.5], [-2,2], 40, 30, 'rand')
##    fu = Square([-1,1], [-1,1], 20, 20, 'grid')

    Z = fu(fu.XY)

    rbfi = rbf.Rbf(fu.XY, Z, rbf='gauss', r=1e-11, p=5)

    dati = SurfaceData(fu.xlim, fu.ylim, fu.nx*2, fu.ny*2, 'grid')

    ZI_func = fu(dati.XY)
    ZI_rbf = rbfi(dati.XY)
    ZG_func = ZI_func.reshape((dati.nx, dati.ny))
    ZG_rbf = ZI_rbf.reshape((dati.nx, dati.ny))
    zlim = [ZI_func.min(), ZI_func.max()]

    fig, ax = mpl.fig_ax3d(clean=True)

    ax.scatter(fu.XY[:,0], fu.XY[:,1], Z, color='b', label='f(x,y) samples')
    dif = np.abs(ZI_func - ZI_rbf).reshape((dati.nx, dati.ny))
    if not export:
        ax.plot_wireframe(dati.XG, dati.YG, ZG_func, cstride=1, rstride=1,
                          color='g', label='f(x,y)', alpha=0.5)
    ax.plot_wireframe(dati.XG, dati.YG, ZG_rbf, cstride=1, rstride=1,
                      color='r', label='rbf(x,y)', alpha=0.5)
    cont = ax.contour(dati.XG, dati.YG, dif, offset=zlim[0],
                      levels=np.linspace(dif.min(), dif.max(), 20),
                      cmap=cm.plasma)
    if not export:
        fig.colorbar(cont, aspect=5, shrink=0.5, format="%.3f")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim3d(dati.xlim)
    ax.set_ylim3d(dati.ylim)
    ax.set_zlim3d(zlim)
    fig.tight_layout()
    if export:
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.zaxis.set_ticklabels([])
        for ext in ['png', 'pdf']:
            fig.savefig(f'/tmp/rbf_2d_surface.{ext}', **savefig_opts)
    else:
        ax.legend()

    # derivs only implemented for MexicanHat
    ZI_func = fu(dati.XY, der='x')
    ZI_rbf = rbfi(dati.XY, der=1)[:,0]
    ZG_func = ZI_func.reshape((dati.nx, dati.ny))
    ZG_rbf = ZI_rbf.reshape((dati.nx, dati.ny))

    zlim = [ZI_func.min(), ZI_func.max()]

    fig, ax = mpl.fig_ax3d(clean=True)

    dif = np.abs(ZI_func - ZI_rbf).reshape((dati.nx, dati.ny))
    ax.plot_wireframe(dati.XG, dati.YG, ZG_func, cstride=1, rstride=1,
                      color='g', label='df/dx', alpha=0.5)
    ax.plot_wireframe(dati.XG, dati.YG, ZG_rbf, cstride=1, rstride=1,
                      color='r', label='d(rbf)/dx', alpha=0.5)
    cont = ax.contour(dati.XG, dati.YG, dif, offset=zlim[0],
                      levels=np.linspace(dif.min(), dif.max(), 20),
                      cmap=cm.plasma)
    if not export:
        fig.colorbar(cont, aspect=5, shrink=0.5, format="%.3f")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim3d(dati.xlim)
    ax.set_ylim3d(dati.ylim)
    ax.set_zlim3d(zlim)
    ax.legend()
    fig.tight_layout()
    if export:
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.zaxis.set_ticklabels([])
        for ext in ['png', 'pdf']:
            fig.savefig(f'/tmp/rbf_2d_surface_deriv.{ext}', **savefig_opts)

    plt.show()
