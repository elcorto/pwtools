"""
Example for changing axis color (pure matplotlib, just as reference)
---------------------------------------------------------------------
From http://stackoverflow.com/questions/4761623/changing-the-color-of-the-axis-ticks-and-labels-for-a-plot-in-matplotlib

>>> import matplotlib.pyplot as plt
>>> fig = plt.figure()
>>> ax = fig.add_subplot(111)

>>> ax.plot(range(10))
>>> ax.set_xlabel('X-axis')
>>> ax.set_ylabel('Y-axis')

>>> ax.spines['bottom'].set_color('red')
>>> ax.spines['top'].set_color('red')
>>> ax.xaxis.label.set_color('red')
>>> ax.tick_params(axis='x', colors='red')

>>> plt.show()
"""

import itertools
from pwtools import common, num
import warnings
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
try:
    from mpl_toolkits.axes_grid1.parasite_axes import SubplotHost, \
        ParasiteAxes
except ImportError:
    warnings.warn("cannot import from mpl_toolkits.axes_grid")
# This is with mpl < 1.0
try:
    from mpl_toolkits.mplot3d import Axes3D
except ImportError:
    warnings.warn("cannot import from mpl_toolkits.mplot3d")

#----------------------------------------------------------------------------
# mpl helpers, boilerplate stuff
#----------------------------------------------------------------------------

def plotlines3d(ax3d, x,y,z, *args, **kwargs):
    """Plot x-z curves stacked along y.

    Parameters
    ----------
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

    Returns
    -------
    ax3d

    Examples
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


def fig_ax(**kwds):
    """``fig,ax = fig_ax()``"""
    return plt.subplots(**kwds)


def fig_ax3d(clean=False, **kwds):
    """``fig,ax3d = fig_ax()``

    Parameters
    ----------
    clean : bool
        see :func:`clean_ax3d`
    """
    fig = plt.figure(**kwds)
    try:
        ax = fig.add_subplot(111, projection='3d')
    except:
        # mpl < 1.0.0
        ax = Axes3D(fig)
    if clean:
        clean_ax3d(ax)
    return fig, ax


def clean_ax3d(ax):
    """On ``Axes3DSubplot`` `ax`, set x,y,z pane color to white and remove
    grid."""
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.grid(False)


class Plot(object):
    """Container for a plot figure with (by default one) axis `ax`.

    You can add more axes with twinx() etc and operate on them.

    Examples
    --------
    # same as Plot(*mpl.fig_ax()), i.e. default is 2d plot
    >>> pp = mpl.Plot()
    >>> pp.ax.plot([1,2,3], label='ax')
    >>> pp.ax2 = pp.ax.twinx()
    >>> pp.ax2.plot([3,2,1], 'r', label='ax2')
    >>> # legend on `ax` (default legaxname='ax') with all lines from `ax` and
    >>> # `ax2`
    >>> pp.legend(['ax', 'ax2'], loc='lower left')
    >>> pp.fig.savefig('lala.png')
    >>> pp.fig.savefig('lala.pdf')
    >>> # or
    >>> pp.savefig('lala', ext=['png', 'pdf'])
    """
    def __init__(self, fig=None, ax=None, projection='2d', **kwds):
        """
        Parameters
        ----------
        fig, ax : Figure + Axis instance (e.g. from fig_ax())
        projection : str, optional
            If fig+ax not given, use this to call fig_ax() or fig_ax3d(), else
            ignored.
        kwds : keywords passed to fig_ax() or fig_ax3d() if `fig=None` and
            `ax=None`
        """
        if [fig, ax] == [None]*2:
            if projection == '2d':
                func = fig_ax
            elif projection == '3d':
                func = fig_ax3d
            else:
                raise Exception("unknown projection: %s" %projection)
            self.fig, self.ax = func(**kwds)
        elif [fig, ax].count(None) == 1:
            raise Exception("one of fig,ax is None")
        else:
            self.fig = fig
            self.ax = ax

    def collect_legends(self, axnames=['ax']):
        """If self has more then one axis object attached, then collect legends
        from all axes specified in axnames. Useful for handling legend entries
        of lines on differend axes (in case of twinx, for instance).

        Parameters
        ----------
        axnames : sequence of strings

        Returns
        -------
        tuple of lines and labels
            ([line1, line2, ...], ['foo', 'bar', ...])
        where lines and labels are taken from all axes. Use this as input for
        any axis's legend() method.
        """
        return collect_legends(*tuple(getattr(self, axname) for axname in
                                      axnames))

    def legend(self, axnames=None, legaxname='ax', **kwargs):
        """Collect legend entries from all axes in `axnames` and place legend on
        the axis named with `legaxname`.

        Parameters
        ----------
        axnames : None or list of axes names, optional
            e.g. ['ax'] or ['ax', 'ax2']. If None (default) then ax.legend() is
            called directly (if legaxname='ax').
        legaxname : string, optional
            The name of the axis where the legend is placed on. If you use
            things like twinx(), then you may want to choose top most the axis,
            i.e. the one in the foreground. For example:
            >>> pp = Plot(...)
            >>> pp.ax.plot(...)
            >>> pp.ax2 = pp.ax.twinx()
            >>> pp.ax2.plot(...)
            >>> pp.legend(axnames=['ax', 'ax2'], legaxname='ax2')

        Notes
        -----
        This is not completly transparent. This:
            >>> plot = Plot(...)
            >>> plot.ax.plot(...)
            >>> plot.legend(...)
        does only behave as ax.legend() if only kwargs are used. For anything
        else, use
            >>> plot.ax.legend()
        directly.
        """
        ax = getattr(self, legaxname)
        if axnames is None:
            leg = ax.legend(**kwargs)
        else:
            leg = ax.legend(*self.collect_legends(axnames), **kwargs)
        return leg

    def savefig(self, base, ext=['png'], **kwds):
        for ex in ext:
            self.fig.savefig(base + '.' + ex, **kwds)


def collect_legends(*axs):
    """
    Collect legend data from multiple axes, return input for legend().

    Examples
    --------
    >>> from pwtools import mpl
    >>> from numpy.random import rand
    >>> fig, ax = mpl.fig_ax()
    >>> ax.plot([1,2,3], label='ax line')
    >>> ax.bar([1,2,3], rand(3), label='ax bar')
    >>> ax2 = ax.twinx()
    >>> ax2.plot([3,2,1], 'r', label='ax2')
    >>> ax.legend(*mpl.collect_legends(ax, ax2))
    """
    axhls = tuple(ax.get_legend_handles_labels() for ax in
                  axs)
    ret = [itertools.chain(*x) for x in zip(*axhls)]
    return ret[0], ret[1]


def prepare_plots(names, projection='2d', **kwds):
    """Return a dict of Plot instances.

    Parameters
    ----------
    names : sequence
        keys for the dict, e.g. [1,2] or ['plot1', 'plot2']
    projection : str
        type of plot; {'2d','3d'}
    kwds : keywords passed to fig_ax() or fig_ax3d() which are internally used

    Examples
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
        plots[nn] = Plot(*func(**kwds))
    return plots


class Data2D(object):
    """Container which converts between different x-y-z data formats frequently
    used by ``scipy.interpolate.bispl{rep,ev}`` and ``mpl_toolkits.mplot3d``
    fuctions.

    2D because the data is a 2D scalar field, i.e. `z(x,y)`. See
    also :class:`~pwtools.num.Interpol2D`.

    Naming conventions:
    * lowercase: 1d array
    * uppercase: 2d array

    num.Interpol2D.points = num.PolyFit.points = mpl.Data2D.XY
    num.Interpol2D.values = num.PolyFit.values = mpl.Data2D.zz
    """
    def __init__(self, x=None, y=None, xx=None, yy=None, zz=None, X=None,
                 Y=None, Z=None, XY=None):
        """
        Use either `x`, `y` or `xx`, `yy` or `X`, `Y` or `XY` to define the x-y
        data. z-data is optional. For that, use `Z` or `zz`. Conversion to all
        forms is done automatically.

        Parameters
        ----------
        x,y : 1d arrays, (nx,) and (ny,)
            These are the raw x and y "axes".
        X,Y,Z : 2d arrays (nx, ny)
            Like ``np.meshgrid`` but transposed to have shape (nx,ny), see also
            :func:`~pwtools.num.meshgridt`
        xx,yy,zz : 1d arrays (nx*ny)
            "Double-loop" versions of x,y,Z, input for ax3d.scatter() or
            bisplrep().
        XY : np.array([xx,yy]).T

        Examples
        --------
        >>> from pwtools.mpl import Data2D,
        >>> from pwtools import num
        >>> from scipy.interpolate import bisplrep, bisplev
        >>> x = linspace(-5,5,10)
        >>> y = linspace(-5,5,10)
        >>> X,Y = num.meshgridt(x,y)
        >>> Z = X**2+Y**2
        >>> data = Data2D(x=x,y=y,Z=Z)
        >>> xi = linspace(-5,5,50)
        >>> yi = linspace(-5,5,50)
        >>> ZI = bisplev(xi,yi,bisplrep(data.xx, data.yy, data.zz))
        >>> spline = Data2D(x=xi, y=yi, Z=ZI)
        >>> fig,ax3d = mpl.fig_ax3d()
        >>> ax3d.scatter(data.xx, data.yy, data.zz, color='b')
        >>> ax3d.plot_wireframe(data.X, data.Y, data.Z, color='g')
        >>> ax3d.plot_surface(spline.X, spline.Y, spline.Z, cstride=1,
        ...                   rstride=1, color='r')

        Notes
        -----
        ``X,Y = num.meshgridt(x,y)`` are the *transposed* versions of ``X,Y =
        numpy.meshgrid()`` which returns shape (ny,nx). The shape (nx,ny),
        which we use, is more intuitive and also used in ``ax3d.plot_surface``
        etc. The output of ``scipy.interpolate.bisplev`` is also (nx,ny).

        ::

            nx = 10
            ny = 5
            x = linspace(...,nx)
            y = linspace(...,ny)

        To calculate z=f(x,y) on the x,y-grid, use num.meshgridt() or X.T, Y.T
        from numpy.meshgrid()::

            X,Y = num.meshgridt(x,y)
            Z = X**2 + Y**2

        X,Y,Z are good for data generation and plotting (ax3d.plot_wireframe()). But
        the input to bisplrep() must be flat X,Y,Z (xx,yy,zz) like so::

            xx = X.flatten()
            yy = Y.flatten()
            zz = Z.flatten()

        The same, as explicit loops::

            xx = np.empty((nx*ny), dtype=float)
            yy = np.empty((nx*ny), dtype=float)
            zz = np.empty((nx*ny), dtype=float)
            for ii in range(nx):
                for jj in range(ny):
                    idx = ii*ny+jj
                    xx[idx] = x[ii]
                    yy[idx] = y[jj]
                    zz[idx] = x[ii]**2 + y[jj]**2

        or::

            XY = np.array([k for k in itertools.product(x,y)])
            xx = XY[:,0]
            yy = XY[:,1]
            zz = xx**2 + yy**2

        Construct the spline and evaluate::

            spl = bisplrep(xx,yy,zz,...)
            ZI = bisplev(x,y)

        `ZI` has the correct shape: (nx, ny), which is the shape of
        ``np.outer(x,y)``.

        The "inverse meshgrid" operation to transform `xx`, `yy` to `x`, `y` is
        done by using ``numpy.unique``. We assumes that ``xx`` and ``yy`` are
        generated like in the nested loop above. For otherwise ordered `xx`,
        `yy` this will fail.
        """
        self.attr_lst = ['x', 'y', 'xx', 'yy', 'zz', 'X', 'Y', 'Z', 'XY']
        self.x = x
        self.y = y
        self.xx = xx
        self.yy = yy
        self.zz = zz
        self.X = X
        self.Y = Y
        self.Z = Z
        self.XY = XY
        self._update()

    @staticmethod
    def _unique(x):
        # numpy.unique(x) with preserved order
        # http://stackoverflow.com/questions/15637336/numpy-unique-with-order-preserved
        #
        # >>> y=array([1,3,-3,-7,-8])
        # >>> unique(y)
        # array([-8, -7, -3,  1,  3])
        # >>> mpl.Data2D._unique(y)
        # array([ 1,  3, -3, -7, -8])
        return x[np.sort(np.unique(x, return_index=True)[1])]

    def _update(self):
        if self.x is not None and self.y is not None:
            self.X,self.Y = num.meshgridt(self.x, self.y)
            self.xx = self.X.flatten()
            self.yy = self.Y.flatten()
            self.XY = np.array([self.xx, self.yy]).T
        elif self.X is not None and self.Y is not None:
            self.x = self.X[:,0]
            self.xx = self.X.flatten()
            self.y = self.Y[0,:]
            self.yy = self.Y.flatten()
            self.XY = np.array([self.xx, self.yy]).T
        elif self.xx is not None and self.yy is not None:
            self.x = self._unique(self.xx)
            self.y = self._unique(self.yy)
            self.X,self.Y = num.meshgridt(self.x, self.y)
            self.XY = np.array([self.xx, self.yy]).T
        elif self.XY is not None:
            self.xx = self.XY[:,0]
            self.yy = self.XY[:,1]
            self.x = self._unique(self.xx)
            self.y = self._unique(self.yy)
            self.X,self.Y = num.meshgridt(self.x, self.y)
        else:
            raise Exception("cannot determine x and y from input")
        # by now, we have all forms of x and y: x,y; xx,yy; X,Y; XY
        self.nx = len(self.x)
        self.ny = len(self.y)
        # Z is optional
        if self.Z is None:
            if self.zz is not None:
                self.Z = self.zz.reshape(len(self.x), len(self.y))
        else:
            self.zz = self.Z.flatten()

    def update(self, **kwds):
        """Update object with new or more input data. Input args are the same
        as in the constructor, i.e. `x`, `y`, `xx`, ..."""
        for key,val in kwds.items():
            assert key in self.attr_lst, ("'%s' not allowed" %key)
            setattr(self, key, val)
        self._update()

    def copy(self):
        """Copy object. numpy arrays are real copies, not views."""
        kwds = {}
        for name in self.attr_lst:
            attr = getattr(self, name)
            if attr is not None:
                kwds[name] = attr.copy()
            else:
                kwds[name] = None
        return Data2D(**kwds)

# XXX needed??
# backwd compat
Data3D = Data2D

def get_2d_testdata(n=20):
    """2D sin + cos data.

    Returns
    -------
    ret : :class:`Data2D`
    """
    x = np.linspace(-5, 5, n)
    X,Y = num.meshgridt(x, x)
    Z = np.sin(X) + np.cos(Y)
    return Data2D(X=X, Y=Y, Z=Z)


#----------------------------------------------------------------------------
# color and marker iterators
#----------------------------------------------------------------------------

# Typical matplotlib line/marker colors and marker styles. See help(plot).
# The naming is convention is foo_bar for
#
# lst_of_plot_styles = []
# foo = ['b', 'r']
# bar = ['-', '--']
# for x in foo:
#     for y in bar:
#         lst_of_plot_styles.append(x+y)
#
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
markers = ['o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H', '+', 'x', 'D']
linestyles = ['-', '--', ':', '-.']
colors_markers = [x[0]+x[1] for x in itertools.product(colors,markers)]
colors_linestyles = [x[0]+x[1] for x in itertools.product(colors,linestyles)]
markers_colors = [x[0]+x[1] for x in itertools.product(markers,colors)]
linestyles_colors = [x[0]+x[1] for x in itertools.product(linestyles,colors)]

# Iterators which raise StopIteration
iter_colors = iter(colors)
iter_markers = iter(markers)
iter_linestyles = iter(linestyles)
iter_colors_markers = iter(colors_markers)
iter_colors_linestyles = iter(colors_linestyles)
iter_markers_colors = iter(markers_colors)
iter_linestyles_colors = iter(linestyles_colors)

# Iterators which infinitely repeat each sequence.
cycle_colors = itertools.cycle(colors)
cycle_markers = itertools.cycle(markers)
cycle_colors_markers = itertools.cycle(colors_markers)
cycle_colors_linestyles = itertools.cycle(colors_linestyles)
cycle_markers_colors = itertools.cycle(markers_colors)
cycle_linestyles_colors = itertools.cycle(linestyles_colors)

# shortcuts
cc = cycle_colors
cm = cycle_markers
ccm = cycle_colors_markers
ccl = cycle_colors_linestyles
cmc = cycle_markers_colors
clc = cycle_linestyles_colors

ic = iter_colors
im = iter_markers
icm = iter_colors_markers
icl = iter_colors_linestyles
imc = iter_markers_colors
ilc = iter_linestyles_colors


def smooth_color(idx, niter):
    """Helper for creating color transitions in loops.

    Examples
    --------
    >>> # change color smoothly blue -> red
    >>> from pwtools import mpl
    >>> N = 10
    >>> for ii in range(N):
    ...     z = mpl.smooth_color(ii,N)
    ...     plot(rand(20)+ii, color=(z,0,1-z))
    """
    return float(idx) / float(niter - 1)


def smooth_color_func(niter, func):
    """Version of :func:`smooth_color` that accepts a function.

    Can be used to pre-calculate a color list outside of a loop.

    Parameters
    ----------
    niter : int
        number of iterations
    func : callable

    Examples
    --------
    >>> from pwtools import mpl
    >>> mpl.smooth_color_func(3, lambda z: (z,0,1-z))
    [(0.0, 0, 1.0), (0.5, 0, 0.5), (1.0, 0, 0.0)]

    >>> for ii,color in enumerate(mpl.smooth_color_func(10, lambda z: (z,0,1-z))):
    ...     plot(rand(20)+ii, color=color)
    """
    col = []
    fniter = float(niter) - 1
    for ii in range(niter):
        z = float(ii) / fniter
        col.append(func(z))
    return col

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

    Parameters
    ----------
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

    Returns
    -------
    (fig, hostax, parax)
    fig : the Figure
    hostax : the hostax
    parax : the new ParasiteAxes instance

    Notes
    -----
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
    while loc + str(n) in parax.axis:
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

    Returns
    -------
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
        plt.plot(t, np.sin(2*np.pi*f*t)+10*f, next(ccm)+'-')

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

    for n, val in enumerate(off_dct.items()):
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

# deprecation warnings
def meshgridt(*args, **kwds):
##    warnings.simplefilter('always')
    warnings.warn("mpl.meshgridt is deprecated, use num.meshgridt",
                  DeprecationWarning)
    return num.meshgridt(*args, **kwds)
