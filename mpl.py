# mpl.py
# 
# Plotting stuff for matplotlib: layouts, predefined markers etc.

import itertools
import sys
import os
from pwtools import common

import matplotlib
from matplotlib import pyplot as plt
import numpy as np
try:
    from mpl_toolkits.axes_grid.parasite_axes import SubplotHost, \
        ParasiteAxes
except ImportError:
    print("mpl.py: cannot import from mpl_toolkits.axes_grid")
# This is with mpl < 1.0
try:
    from mpl_toolkits.mplot3d import Axes3D
except ImportError:
    print("mpl.py: cannot import from mpl_toolkits.mplot3d")

#----------------------------------------------------------------------------
# mpl helpers, boilerplate stuff
#----------------------------------------------------------------------------

def meshgridt(x, y):
    """A version of 
        X,Y = numpy.meshgrid(x,y) 
    which returns X and Y transposed, i.e. (nx, ny) instead (ny, nx) 
    where nx,ny = len(x),len(y).

    This is useful for dealing with 2D splines in 
    scipy.interpolate.bisplev(), which also returns a (nx,ny) array.
    
    args:
    -----
    x,y : 1d arrays
    """
    X,Y = np.meshgrid(x,y)
    return X.T, Y.T


def plotlines3d(ax3d, x,y,z, *args, **kwargs):
    """Plot x-z curves stacked along y.
    
    args:
    -----
    ax3d : Axes3D instance
    x : nd array
        1d (x-axis) or 2d (x-axes are the columns)
    y : 1d array        
    z : nd array with "y"-values
        1d : the same curve will be plotted len(y) times against x (1d) or
             x[:,i] (2d) 
        2d : each column z[:,i] will be plotted against x (1d) or each x[:,i]
             (2d)
    *args, **kwargs : additional args and keywords args passed to ax3d.plot()

    returns:
    --------
    ax3d

    example:
    --------
    >>> x = linspace(0,5,100)
    >>> y = arange(1.0,5) # len(y) = 4
    >>> z = np.repeat(sin(x)[:,None], 4, axis=1)/y # make 2d 
    >>> fig,ax = fig_ax3d()
    >>> plotlines3d(ax, x, y, z)
    >>> show()
    """
    assert y.ndim == 1
    if z.ndim == 1:
        zz = np.repeat(z[:,None], len(y), axis=1)
    else:
        zz = z
    if x.ndim == 1:
        xx = np.repeat(x[:,None], zz.shape[1], axis=1)
    else:
        xx = x
    assert xx.shape == zz.shape
    assert len(y) == xx.shape[1] == zz.shape[1]
    for j in range(xx.shape[1]):
        ax3d.plot(xx[:,j], np.ones(xx.shape[0])*y[j], z[:,j], *args, **kwargs)
    return ax3d        

def fig_ax():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    return fig, ax

def fig_ax3d():
    fig = plt.figure()
    try: 
        ax = fig.add_subplot(111, projection='3d')
    except:
        # mpl < 1.0.0
        ax = Axes3D(fig)
    return fig, ax


class Plot(object):
    """Container for a plot figure with (by default one) axis `ax`.

    You can add more axes with twinx() etc and operate on them.
    
    examples:
    ---------
    # same as Plot(*mpl.fig_ax()), i.e. default is 2d plot
    >>> pp = mpl.Plot() 
    >>> pp.ax.plot([1,2,3], label='ax')
    >>> pp.ax2 = pp.ax.twinx()
    >>> pp.ax2.plot([2,2,1], 'r', label='ax2')
    >>> # legend on `ax` (default legaxname='ax') with all lines from `ax` and
    >>> # `ax2`
    >>> pp.legend(['ax', 'ax2'])
    >>> pp.fig.savefig('lala.png')
    >>> pp.fig.savefig('lala.pdf')
    >>> # or
    >>> pp.savefig('lala', ext=['png', 'pdf'])
    """
    def __init__(self, fig=None, ax=None, projection='2d'):
        """
        args:
        -----
        fig, ax : Figure + Axis instance (e.g. from fig_ax())
        projection : str, optional
            If fig+ax not given, use this to call fig_ax() or fig_ax3d(), else
            ignored.
        """
        if [fig, ax] == [None]*2:
            if projection == '2d':
                func = fig_ax
            elif projection == '3d':        
                func = fig_ax3d
            else:   
                raise StandardError("unknown projection: %s" %projection)
            self.fig, self.ax = func()
        elif [fig, ax].count(None) == 1:
            raise StandardError("one of fig,ax is None")
        else:            
            self.fig = fig
            self.ax = ax
    
    def collect_legends(self, axnames=['ax']):
        """If self has more then one axis object attached, then collect legends
        from all axes specified in axnames. Useful for handling legend entries
        of lines on differend axes (in case of twinx, for instance).

        args:
        -----
        axnames : sequence of strings

        returns:
        --------
        tuple of lines and labels
            ([line1, line2, ...], ['foo', 'bar', ...])
        where lines and labels are taken from all axes. Use this as input for
        any axis's legend() method.
        """            
        axhls = [getattr(self, axname).get_legend_handles_labels() for axname in
                 axnames]
        ret = [common.flatten(x) for x in zip(*tuple(axhls))]
        return ret[0], ret[1]
    
    # XXX This is not completly transparent. This 
    #   >>> plot = Plot(...)
    #   >>> plot.ax.plot(...)
    #   >>> plot.legend(...)
    # does only behave as ax.legend() if only kwargs are used. For anything
    # else, use 
    #   >>> plot.ax.legend() directly.
    def legend(self, axnames=None, legaxname='ax', **kwargs):
        """Collect legend entries from all axes in axnames and place legend on
        the axis named with legaxname.
        """
        ax = getattr(self, legaxname)
        if axnames is None:
            ax.legend(**kwargs)
        else:
            ax.legend(*self.collect_legends(axnames), **kwargs)

    def savefig(self, base, ext=['png']):
        for ex in ext:
            self.fig.savefig(base + '.' + ex)

def prepare_plots(names, projection='2d'):
    """Return a dict of Plot instances.
    
    args:
    -----
    names : sequence of strings (keys for the dict)
    projection : str
        type of plot; {'2d','3d'}

    example:
    --------
    >>> plots = prepare_plots(['etot', 'ekin'])
    >>> plots['etot'].ax.plot(etot)
    >>> plots['ekin'].ax.plot(ekin)
    >>> for key,pp in plots.iteritems():
    ...     pp.ax.set_title(key)
    ...     pp.fig.savefig(key+'.png')
    """
    assert projection in ['2d', '3d'], ("unknown projection, allowed: "
        "'2d', '3d'")
    if projection == '2d':
        func = fig_ax
    elif projection == '3d':        
        func = fig_ax3d
    plots = {}
    for nn in names:
        plots[nn] = Plot(*func())
    return plots        


class Data3D(object):
    """Container which converts between different x-y-z data formats frequently
    used by scipy.interpolate.bispl{rep,ev} and mpl_toolkits.mplot3d fuctions.
    """
    def __init__(self, x=None, y=None, xx=None, yy=None, zz=None, X=None,
                 Y=None, Z=None):
        """
        args:
        -----
        x,y : 1d arrays, shape (nx,) (ny,)
            These are the raw x and y "axes".
        X,Y,Z: meshgrid-like 2d arrays (nx, ny)
        xx,yy,zz : 1d arrays (nx*ny)
            "Double-loop" versions of x,y,Z, input for ax3d.scatter() or
            bisplrep(). 
        
        example:
        --------
        x = linspace(...,5)
        y = linspace(...,5)
        X,Y = np.meshgridt(x,y)
        Z = X**2+Y**2
        data = Data3D(x=x,y=y,Z=Z)
        xi = linspace(...,50)
        yi = linspace(...,50)
        ZI = bisplev(xi,yi,bisplrep(data.xx, data.yy, data.zz))
        spline = Data3D(x=xi, y=yi, Z=ZI)
        ax3d.scatter(data.xx, data.yy, data.zz)
        ax3d.plot_wireframe(data.X, data.Y, data.Z)
        ax3d.plot_surface(spline.X, spline.Y, spline.Z, cstride=1, rstride=1)

        notes:
        ------
        Shape of X,Y,Z:

        In 
            X,Y = meshgridt(x,y), 
        X and Y are the *transposed* versions of  
            X,Y = numpy.meshgrid()
        which returns (ny,nx). The shape (nx,ny), which we use, is more
        intuitive and also used in ax3d.plot_surface() etc. The output of
        scipy.interpolate.bisplev() is also (nx,ny).        
        
        xx,yy,zz:

            nx = 10
            ny = 5
            x = linspace(...,nx)
            y = linspace(...,ny)
        To calculate z=f(x,y) on the x,y-grid, use meshgridt() or X.T, Y.T
        from numpy.meshgrid(). 
            X,Y = meshgridt(x,y)
            Z = X**2 + Y**2
        X,Y,Z are good for data generation and plotting (ax3d.plot_wireframe()). But
        the input to bisplrep() must be flat X,Y,Z (xx,yy,zz) like so:
            xx = X.flatten()
            yy = Y.flatten()
            zz = Z.flatten()
        The same, as explicit loops:
            xx = np.empty((nx*ny), dtype=float)
            yy = np.empty((nx*ny), dtype=float)
            zz = np.empty((nx*ny), dtype=float)
            for ii in range(nx):
                for jj in range(ny):
                    idx = ii*ny+jj
                    xx[idx] = x[ii]
                    yy[idx] = y[jj]
                    zz[idx] = x[ii]**2 + y[jj]**2
        Construct the spline and evaluate
            spl = bisplrep(xx,yy,zz,...)
            ZI = bisplev(x,y)
        Note that for evaluation, we must use the "axes" x and y, not xx and yy! 
        ZI has the correct shape: (nx, ny), which is the shape of np.outer(x,y).
        """
        self.x = x
        self.y = y
        self.xx = xx
        self.yy = yy
        self.zz = zz
        self.X = X
        self.Y = Y
        self.Z = Z
        self.update()

    def update(self):
        if [self.x,self.y] != [None]*2:
            self.X,self.Y = meshgridt(self.x, self.y)
            self.xx = self.X.flatten()
            self.yy = self.Y.flatten()
        else:
            if self.X is not None:
                self.x = self.X[:,0]
                self.xx = self.X.flatten()
            if self.Y is not None:
                self.y = self.Y[0,:]
                self.yy = self.Y.flatten()
        if self.Z is not None:
            self.zz = self.Z.flatten()


def get_3d_testdata():
    x = np.linspace(-5,5,20)
    y = np.linspace(-5,5,20)
    X,Y = meshgridt(x,y)
    Z = np.sin(X) + np.cos(Y)
    return Data3D(X=X, Y=Y, Z=Z)


#----------------------------------------------------------------------------
# color and marker iterators
#----------------------------------------------------------------------------

# Typical matplotlib line/marker colors and marker styles. See help(plot).
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
markers = ['o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H', '+', 'x', 'D']

colors_markers = []
for mark in markers:
    for col in colors:
        colors_markers.append(col+mark)

# Iterators which infinitely repeat each sequence. 
cycle_colors = itertools.cycle(colors)
cycle_markers = itertools.cycle(markers)
cycle_colors_markers = itertools.cycle(colors_markers)
cc = cycle_colors
cm = cycle_markers
ccm = cycle_colors_markers

def smooth_color(idx, niter):
    """
    example:
    --------
    # change color smoothly blue -> red
    for ii in range(10):
        z = smooth_color(ii,10)
        plot(..., color=(z, 0, 1-z)
    """
    niter = float(niter)
    idx = float(idx)
    return idx / niter

#----------------------------------------------------------------------------
# Layout defs
#----------------------------------------------------------------------------

def check_ax_obj(ax):
    if not isinstance(ax, matplotlib.axes.Axes):
        raise StandardError("argument `ax` %s is no instance of "
            "matplotlib.axes.Axes" %repr(ax))


def get_rctarget(rctarget, default=plt):
    """If `rctarget` is None, return `default` (the module-level `plt` module)
    for modifications with it's rc() method. See set_plot_layout*()."""
    return default if rctarget is None else rctarget


def set_plot_layout(rctarget=None, layout='latex_hs'):
    """Set mpl rc parameters.
    
    args:
    -----
    rctarget : something with an rc() method
    
    example:
    ------
    >>> from matplotlib import pyplot as plt
    >>> set_plot_layout(plt)
    >>> plt.plot(...)
    """
    rctarget = get_rctarget(rctarget)
    if not hasattr(rctarget, 'rc'):
        raise AttributeError("argument %s has no attribute 'rc'" %repr(rctarget))
    # This is good for presentations and tex'ing files in a PhD Thesis
    if layout == 'latex_hs':
        rctarget.rc('text', usetex=True) # this may not work on windows systems !!
        rctarget.rc('font', weight='bold')
        rctarget.rc('font', size='20')
        rctarget.rc('lines', linewidth=3.0)
        rctarget.rc('lines', markersize=8)
        rctarget.rc('savefig', dpi=150)
        rctarget.rc('lines', dash_capstyle='round') # default 'butt'
    else:
        raise StandardError("unknown layout '%s'" % layout)    


def set_tickline_width(ax, xmin=1.0,xmaj=1.5,ymin=1.0,ymaj=1.5):
    """Set the ticklines (minors and majors) to the given values.
     Looks more professional in Papers and is an Phys.Rev.B. like Style.

        ax -- an Axes object (e.g. ax=gca() or ax=fig.add_subplot(111))
     """
    check_ax_obj(ax)
    for tick in ax.xaxis.get_major_ticks():
        tick.tick1line.set_markeredgewidth(xmaj)
        tick.tick2line.set_markeredgewidth(xmaj)
    for tick in ax.xaxis.get_minor_ticks():
        tick.tick1line.set_markeredgewidth(xmin)
        tick.tick2line.set_markeredgewidth(xmin)
    for tick in ax.yaxis.get_major_ticks():
        tick.tick1line.set_markeredgewidth(ymaj)
        tick.tick2line.set_markeredgewidth(ymaj)
    for tick in ax.yaxis.get_minor_ticks():
        tick.tick1line.set_markeredgewidth(ymin)
        tick.tick2line.set_markeredgewidth(ymin)


def set_plot_layout_phdth(rctarget=None):
    rctarget = get_rctarget(rctarget)
    set_plot_layout(rctarget)
    rctarget.rc('legend', borderpad=0.2)
    # figure
    rctarget.rc('figure', figsize=(11,10))
    rctarget.rc('savefig', dpi=100)

def set_plot_layout_talk(rctarget=None):
    rctarget = get_rctarget(rctarget)
    rctarget.rc('legend', borderpad=0.2)
    # minimal possible spacing, labels do not overlap
    rctarget.rc('legend', labelspacing=0)
    rctarget.rc('savefig', dpi=100)
    rctarget.rc('font', size=13)
    rctarget.rc('lines', linewidth=2)
    rctarget.rc('lines', markersize=6)
    # Equal whitespace left and right in case of twinx(). And even if not, we
    # want all subplots to have the same aspect ratio.
    rctarget.rc('figure.subplot', left=0.125)
    rctarget.rc('figure.subplot', right=0.875)
    rctarget.rc('figure.subplot', bottom=0.125)
    rctarget.rc('mathtext', default='regular')


def set_plot_layout_paper(rctarget=None):
    rctarget = get_rctarget(rctarget)
    rctarget.rc('legend', borderpad=0.2)
    # minimal possible spacing, labels do not overlap
    rctarget.rc('legend', labelspacing=0)
    rctarget.rc('savefig', dpi=100)
    rctarget.rc('font', size=16)
    rctarget.rc('lines', linewidth=2)
    rctarget.rc('lines', markersize=6)
    # Equal whitespace left and right in case of twinx(). And even if not, we
    # want all subplots to have the same aspect ratio.
    rctarget.rc('figure.subplot', left=0.125)
    rctarget.rc('figure.subplot', right=0.875)
    rctarget.rc('figure.subplot', bottom=0.125)
    rctarget.rc('mathtext', default='regular')

#----------------------------------------------------------------------------
# new axis line
#----------------------------------------------------------------------------

# works with mpl 0.99
#
# XXX This is probably superseded by ax.spine or gridspec (in 1.0) now. Have
#     not tested both, but looks good.
def new_axis(fig, hostax, off=50, loc='bottom', ticks=None, wsadd=0.1,
             label='', sharex=False, sharey=False):
    """Make a new axis line using mpl_toolkits.axes_grid's SubplotHost and
    ParasiteAxes. The new axis line will be an instance of ParasiteAxes
    attached to `hostax`. You can do twinx()/twiny() type axis (off=0) or
    completely free-standing axis lines (off > 0).

    args:
    -----
    fig : mpl Figure
    hostax : Instance of matplotlib.axes.HostAxesSubplot. This is the subplot
        of the figure `fig` w.r.t which all new axis lines are placed. See
        make_axes_grid_fig().
    off : offset in points, used with parax.get_grid_helper().new_fixed_axis
    loc : one of 'left', 'right', 'top', 'bottom', where to place the new axis
        line
    ticks : sequence of ticks (numbers)
    wsadd : Whitespace to add at the location `loc` to make space for the new
        axis line (only useful if off > 0). The number is a relative unit
        and is used to change the bounding box: hostax.get_position().
    label : str, xlabel (ylabel) for 'top','bottom' ('left', 'right')
    sharex, sharey : bool, share xaxis (yaxis) with `hostax`
    
    returns:
    --------
    (fig, hostax, parax)
    fig : the Figure
    hostax : the hostax
    parax : the new ParasiteAxes instance

    notes:
    ------
    * The sharex/sharey thing may not work correctly.
    """

    # Create ParasiteAxes, an ax which overlays hostax.
    if sharex and sharey:
        parax = ParasiteAxes(hostax, sharex=hostax, sharey=hostax)
    elif sharex:        
        parax = ParasiteAxes(hostax, sharex=hostax)
    elif sharey:        
        parax = ParasiteAxes(hostax, sharey=hostax)
    else:        
        parax = ParasiteAxes(hostax)
    hostax.parasites.append(parax)
    
    # If off != 0, the new axis line will be detached from hostax, i.e.
    # completely "free standing" below (above, left or right of) the main ax
    # (hostax), so we don't need anything visilbe from it b/c we create a
    # new_fixed_axis from this one with an offset anyway. We only need to
    # activate the label.
    for _loc in ['left', 'right', 'top', 'bottom']:
        parax.axis[_loc].set_visible(False)
        parax.axis[_loc].label.set_visible(True)
    
    # Create axis line. Free-standing if off != 0, else overlaying one of hostax's
    # axis lines. In fact, with off=0, one simulates twinx/y(). 
    new_axisline = parax.get_grid_helper().new_fixed_axis
    if loc == 'top':
        offset = (0, off)
        parax.set_xlabel(label)
    elif loc == 'bottom':
        offset = (0, -off)
        parax.set_xlabel(label)
    elif loc == 'left':
        offset = (-off, 0)
        parax.set_ylabel(label)
    elif loc == 'right':
        offset = (off, 0)
        parax.set_ylabel(label)
    newax = new_axisline(loc=loc, offset=offset, axes=parax)
    # name axis lines: bottom2, bottom3, ...
    n=2
    while parax.axis.has_key(loc + str(n)):
        n += 1
    parax.axis[loc + str(n)] = newax
    
    # set ticks
    if ticks is not None:
        newax.axis.set_ticks(ticks)

    # Read out whitespace at the location (loc = 'top', 'bottom', 'left',
    # 'right') and adjust whitespace.
    #
    # indices of the values in the array returned by ax.get_position()
    bbox_idx = dict(left=[0,0], bottom=[0,1], right=[1,0], top=[1,1])
    old_pos = np.array(hostax.get_position())[bbox_idx[loc][0], bbox_idx[loc][1]] 
    if loc in ['top', 'right']:
        new_ws = old_pos - wsadd
    else:
        new_ws = old_pos + wsadd
    # hack ...        
    cmd = "fig.subplots_adjust(%s=%f)" %(loc, new_ws)
    eval(cmd)
    return fig, hostax, parax


def make_axes_grid_fig(num=None):
    """Create an mpl Figure and add to it an axes_grid.SubplotHost subplot
    (`hostax`).
    
    returns:
    --------
    fig, hostax
    """
    if num is not None:
        fig = plt.figure(num)
    else:        
        fig = plt.figure()
    hostax = SubplotHost(fig, 111)
    fig.add_axes(hostax)
    return fig, hostax



if __name__ == '__main__':
    
    #-------------------------------------------------------------------------
    # colors and markers  
    #-------------------------------------------------------------------------

    fig0 = plt.figure(0)
    # All combinations of color and marker
    for col_mark in colors_markers: 
        plt.plot(np.random.rand(10), col_mark+'-')
        # The same
        ## plot(rand(10), col_mark, linestyle='-')
    
    fig1 = plt.figure(1)
    # Now use one of those iterators
    t = np.linspace(0, 2*np.pi, 100)
    for f in np.linspace(1,2, 14):
        plt.plot(t, np.sin(2*np.pi*f*t)+10*f, ccm.next()+'-')        
    
    #-------------------------------------------------------------------------
    # new axis lines, works with mpl 0.99 
    #-------------------------------------------------------------------------
    
    try:
        from pwtools.common import flatten
    except ImportError:
        from matplotlib.cbook import flatten
    
    # Demo w/ all possible axis lines.
    
    x = np.linspace(0,10,100)
    y = np.sin(x)

    fig3, hostax = make_axes_grid_fig(3)
    
    hostax.set_xlabel('hostax bottom')
    hostax.set_ylabel('hostax left')

    # {'left': (off, wsadd),
    # ...}
    off_dct = dict(left=(60, .1), 
                   right=(60, .1), 
                   top=(60, .15), 
                   bottom=(50, .15))

    for n, val in enumerate(off_dct.iteritems()):
        loc, off, wsadd = tuple(flatten(val))
        fig3, hostax, parax = new_axis(fig3, hostax=hostax, 
                                       loc=loc, off=off, label=loc, 
                                       wsadd=wsadd)
        parax.plot(x*n, y**n)
    
    new_axis(fig3, hostax=hostax, loc='right', off=0, wsadd=0, 
             label="hostax right, I'm like twinx()")
    
    new_axis(fig3, hostax=hostax, loc='top', off=0, wsadd=0, 
             label="hostax top, I'm like twiny()")
    

    # many lines 

    fig4, hostax = make_axes_grid_fig(4)
    off=40
    for n in range(1,5):
        fig4, hostax, parax = new_axis(fig4, 
                                       hostax=hostax, 
                                       off=n*off, 
                                       ticks=np.linspace(0,10*n,11),
                                       loc='bottom')
    hostax.plot(x, y, label='l1') 
    hostax.set_title('many axis lines yeah yeah')
    
    hostax.legend()
    
    
    #-------------------------------------------------------------------------
    # plotlines3d
    #-------------------------------------------------------------------------

    fig4, ax3d = fig_ax3d()
    x = np.linspace(0,5,100)
    y = np.arange(1.0,5) # len(y) = 4
    z = np.repeat(np.sin(x)[:,None], 4, axis=1)/y # make 2d 
    plotlines3d(ax3d, x, y, z)
    ax3d.set_xlabel('x')
    ax3d.set_ylabel('y')
    ax3d.set_zlabel('z')
    
    plt.show()
