import numpy as np
from pwtools import mpl, rbf
plt = mpl.plt
rand = np.random.rand

class SurfaceData(object):
    def __init__(self, xlim, ylim, nx, ny, mode):
        self.xlim = xlim
        self.ylim = ylim
        self.nx = nx
        self.ny = ny
        self.xg, self.yg = self.get_xy_grid()
        self.XG, self.YG = mpl.meshgridt(self.xg, self.yg)
        self.X = self.gen_coords(mode)
        
    def gen_coords(self, mode='grid'):
        if mode =='grid':
            X = np.empty((self.nx * self.ny,2))
            X[:,0] = self.XG.flatten()
            X[:,1] = self.YG.flatten()
            ##for i in range(self.nx):
            ##    for j in range(self.ny):
            ##        X[i*self.ny+j,0] = self.xg[i]
            ##        X[i*self.ny+j,1] = self.yg[j]
        elif mode == 'rand':
            X = rand(self.nx * self.ny, 2)
            X[:,0] = X[:,0] * (self.xlim[1] - self.xlim[0]) + self.xlim[0]
            X[:,1] = X[:,1] * (self.ylim[1] - self.ylim[0]) + self.ylim[0]
        return X
    
    def get_xy_grid(self):
        x = np.linspace(self.xlim[0], self.xlim[1], self.nx)
        y = np.linspace(self.ylim[0], self.ylim[1], self.ny)
        return x,y

    def get_X(self, X=None):
        return self.X if X is None else X

    def func(self, X=None):
        X = self.get_X(X)
        return None

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
    def func(self, X=None):
        X = self.get_X(X)
        r = np.sqrt((X**2).sum(axis=1))
        return np.sin(r)/r
    
    def deriv_x(self, X=None):
        X = self.get_X(X)
        r = np.sqrt((X**2).sum(axis=1))
        x = X[:,0]
        y = X[:,1]
        return x * np.cos(r) / r**2 - x * np.sin(r) / r**3.0

    def deriv_y(self, X=None):
        X = self.get_X(X)
        r = np.sqrt((X**2).sum(axis=1))
        x = X[:,0]
        y = X[:,1]
        return y * np.cos(r) / r**2 - y * np.sin(r) / r**3.0

class UpDown(SurfaceData):
    def func(self, X=None):
        X = self.get_X(X)
        x = X[:,0]
        y = X[:,1]
        return x*np.exp(-x**2-y**2)

class SinExp(SurfaceData):
    def func(self, X=None):
        X = self.get_X(X)
        x = X[:,0]
        y = X[:,1]
        return np.sin(np.exp(x)) * np.cos(y) + 0.5*y

class Square(SurfaceData):
    def func(self, X=None):
        X = self.get_X(X)
        x = X[:,0]
        y = X[:,1]
        return (x**2 + y**2)

if __name__ == '__main__':
    
    # Some nice 2D examples
    
    fu = MexicanHat([-10,20], [-10,15], 20, 20, 'rand')
##    fu = UpDown([-2,2], [-2,2], 20, 20, 'grid')
##    fu = SinExp([-1,2.5], [-2,2], 40, 30, 'rand')
##    fu = Square([-1,1], [-1,1], 20, 20, 'grid')
    
    X = fu.X
    Z = fu(X)

    rbfi = rbf.RBFInt(X, Z, X, rbf=rbf.RBFMultiquadric(), verbose=True)
    print("train linalg single ...")
    rbfi.train('linalg')
    print("param:", rbfi.rbf.param)
    print("... ready train")
    
##    print "train param ..."
##    rbfi = train_param(X, Z, pattern='rand', randskip=0.2)
##    print "param:", rbfi.rbf.param
##    print "... ready train"
    
    dati = SurfaceData(fu.xlim, fu.ylim, fu.nx*2, fu.ny*2, 'grid')

    ZI_func = fu(dati.X)
    ZI_rbf = rbfi(dati.X)
    ZG_func = ZI_func.reshape((dati.nx, dati.ny))
    ZG_rbf = ZI_rbf.reshape((dati.nx, dati.ny))
    zlim = [ZI_func.min(), ZI_func.max()]

    fig, ax = mpl.fig_ax3d()
    ax.scatter(X[:,0],X[:,1],Z, color='r')
    dif = np.abs(ZI_func - ZI_rbf).reshape((dati.nx, dati.ny))
    wf = ax.plot_wireframe(dati.XG, dati.YG, ZG_rbf, cstride=1, rstride=1, color='g')
    wf.set_alpha(0.5)
    wf2 = ax.plot_wireframe(dati.XG, dati.YG, ZG_func, cstride=1, rstride=1, color='m')
    wf2.set_alpha(0.5)
    cont = ax.contour(dati.XG, dati.YG, dif, offset=zlim[0], 
                      levels=np.linspace(dif.min(), dif.max(), 20))
    fig.colorbar(cont, aspect=5, shrink=0.5, format="%.3g")    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim3d(dati.xlim)
    ax.set_ylim3d(dati.ylim)
    ax.set_zlim3d(zlim)
    
    # x-derivs, only implemented for MexicanHat
    ZI_func = fu(dati.X, der='x')
    ZI_rbf = rbfi(dati.X, der=1)[:,0]
    print(ZI_func.shape)
    print(ZI_rbf.shape)
    ZG_func = ZI_func.reshape((dati.nx, dati.ny))
    ZG_rbf = ZI_rbf.reshape((dati.nx, dati.ny))

    zlim = [ZI_func.min(), ZI_func.max()]

    fig, ax = mpl.fig_ax3d()
    dif = np.abs(ZI_func - ZI_rbf).reshape((dati.nx, dati.ny))
    wf = ax.plot_wireframe(dati.XG, dati.YG, ZG_rbf, cstride=1, rstride=1, color='g')
    wf.set_alpha(0.5)
    wf2 = ax.plot_wireframe(dati.XG, dati.YG, ZG_func, cstride=1, rstride=1, color='m')
    wf2.set_alpha(0.5)
    cont = ax.contour(dati.XG, dati.YG, dif, offset=zlim[0], 
                      levels=np.linspace(dif.min(), dif.max(), 20))
    fig.colorbar(cont, aspect=5, shrink=0.5, format="%.3g")    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim3d(dati.xlim)
    ax.set_ylim3d(dati.ylim)
    ax.set_zlim3d(zlim)
    
    plt.show()
