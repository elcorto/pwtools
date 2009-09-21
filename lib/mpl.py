# mpl.py
# 
# Plotting stuff for matplotlib: layouts, predefined markers etc.

import itertools
import sys
import os

import matplotlib 
from mpl_toolkits.axes_grid.parasite_axes import SubplotHost, \
    ParasiteAxes
from matplotlib import pyplot as plt
import numpy as np

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


#----------------------------------------------------------------------------
# Layout defs
#----------------------------------------------------------------------------

def check_ax_obj(ax):
    if not isinstance(ax, matplotlib.axes.Axes):
        raise StandardError("argument `ax` %s is no instance of "
            "matplotlib.axes.Axes" %repr(ax))

#----------------------------------------------------------------------------


def set_plot_layout(py, layout='latex_hs'):
    """Set up the pylab plotting subsystem for a specific task.
    py : somethiig w/ an rc() method
    """
    
    if not hasattr(py, 'rc'):
        raise AttributeError("argument %s has no attribute 'rc'" %repr(py))

    # This is good for presentations and tex'ing files in a PhD Thesis
    if layout == 'latex_hs':
        py.rc('text', usetex=True) # this may not work on windows systems !!
##        # text
##        pyl_obj.rc('font', weight='normal')
        py.rc('font', weight='bold')
        py.rc('font', size='20')
##        # lines
        py.rc('lines', linewidth=3.0)
        py.rc('lines', markersize=8)
##        # axes
##        py.rc('axes', titlesize=30)
##        py.rc('axes', labelsize=28)
##        py.rc('axes', linewidth=2.0)
##        # ticks
##        py.rc('xtick.major', size=12)
##        py.rc('ytick.major', size=12)
##        py.rc('xtick.minor', size=8)
##        py.rc('ytick.minor', size=8)
##        py.rc('xtick', labelsize=30)
##        py.rc('ytick', labelsize=30)
##        # legend
##        py.rc('legend', numpoints=3)
##        py.rc('legend', fontsize=25)
##        py.rc('legend', markerscale=0.8)
####        py.rc('legend', axespad=0.04)
##        py.rc('legend', shadow=False)
####        py.rc('legend', handletextsep=0.02)
####        py.rc('legend', pad=0.3)
##        # figure
####        py.rc('figure', figsize=(12,9))
        py.rc('savefig', dpi=150)
##        # then, set custom vals
##        py.rc('xtick', labelsize=30)
##        py.rc('ytick', labelsize=30)
##        py.rc('legend', fontsize=25)
        
        # fractional whitespace between legend entries and legend border
    ##    py.rc('legend', pad=0.1) # default 0.2
        # length of the legend lines, useful for plotting many 
        # dashed lines with slowly changing dash length
    ##    py.rc('legend', handlelen=0.1) # default 0.05
        py.rc('lines', dash_capstyle='round') # default 'butt'

        
        # Need this on Linux (Debian etch) for correct .eps bounding boxes. We
        # check for the OS here because it works without this on MacOS.
##        if sys.platform == 'linux2':
##            py.rc('ps', usedistiller='xpdf')
    else:
        raise StandardError("unknown layout '%s'" % layout)    

#-----------------------------------------------------------------------------

def _set_tickline_width(ax, xmin=1.0,xmaj=1.5,ymin=1.0,ymaj=1.5):
    """Set the ticklines (minors and majors) to the given values.
     Looks more professional in Papers and is an Phys.Rev.B. like Style.

        ax -- an Axes object (e.g. ax=gca() or ax=fig.add_subplot(111))
     """
    
    check_ax_obj(ax)

    # the axis object to use
    # get the x-ticklines
    for tick in ax.xaxis.get_major_ticks():
        tick.tick1line.set_markeredgewidth(xmaj)
        tick.tick2line.set_markeredgewidth(xmaj)
    #
    for tick in ax.xaxis.get_minor_ticks():
        tick.tick1line.set_markeredgewidth(xmin)
        tick.tick2line.set_markeredgewidth(xmin)
    #
    for tick in ax.yaxis.get_major_ticks():
        tick.tick1line.set_markeredgewidth(ymaj)
        tick.tick2line.set_markeredgewidth(ymaj)
    #
    for tick in ax.yaxis.get_minor_ticks():
        tick.tick1line.set_markeredgewidth(ymin)
        tick.tick2line.set_markeredgewidth(ymin)

#-----------------------------------------------------------------------------

def set_tickline_width(ax, xmin=1.0, xmaj=1.5, ymin=1.0, ymaj=1.5):
    check_ax_obj(ax)
    _set_tickline_width(ax, xmin=xmin, xmaj=xmaj, ymin=ymin, ymaj=ymaj) 

#-----------------------------------------------------------------------------

def set_plot_layout_phdth(pyl_obj):
    set_plot_layout(pyl_obj)
    pyl_obj.rc('legend', pad=0.2)
    # figure
    pyl_obj.rc('figure', figsize=(11,10))
    pyl_obj.rc('savefig', dpi=100)

#----------------------------------------------------------------------------
# new axis line
#----------------------------------------------------------------------------

def new_axis(fig, hostax, off=50, loc='bottom', ticks=None, ws_add=0.1,
             label='', sharex=False, sharey=False):
    """Make a new axis line using mpl_toolkits.axes_grid's SubplotHost and
    ParasiteAxes. The new axis line will be an instance of ParasiteAxes
    attached to `hostax`. You can do twinx()/twiny() type axis (off=0) or
    completely free-standing axis lines (off > 0).

    args:
    -----
    fig : mpl Figure
    hostax : Instance of matplotlib.axes.HostAxesSubplot. This is the subplot
        if the figure `fig` w.r.t which all new axis lines are placed. See
        make_axes_grid_fig().
    off : offset in points, used with parax.get_grid_helper().new_fixed_axis
    loc : one of 'left', 'right', 'top', 'bottom', where to place the new axis
        line
    ticks : sequence of ticks (numbers)
    ws_add : Whitespace to add at the locatopn `loc` to make space for the new
        axis line (only useful if off != 0). The number is a relative unit
        and is used to change the bounding box: hostax.get_position().
    label : str, xlabel (ylabel) for 'top','bottom' ('left', 'right')
    sharex, sharey : bool, share xaxis (yaxis) with `hostax`
    
    returns:
    --------
    (fig, hostax, parax)
    fig : the Figure
    hostax : the hostax
    parax : the new ParasiteAxes instance
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
        new_ws = old_pos - ws_add
    else:
        new_ws = old_pos + ws_add
    cmd = "fig.subplots_adjust(%s=%f)" %(loc, new_ws)
    eval(cmd)
    return fig, hostax, parax

#-----------------------------------------------------------------------------

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


#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------

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

    from pwtools.lib.common import flatten
    
    # Demo w/ all possible axis lines.
    
    x = np.linspace(0,10,100)
    y = np.sin(x)

    fig3, hostax = make_axes_grid_fig(3)
    
    hostax.set_xlabel('hostax bottom')
    hostax.set_ylabel('hostax left')

    # {'left': (off, ws_add),
    # ...}
    off_dct = dict(left=(60, .1), right=(60, .1), top=(60, .15), bottom=(50,
    .15))

    for n, val in enumerate(off_dct.iteritems()):
        loc, off, ws_add = tuple(flatten(val))
        fig3, hostax, parax = new_axis(fig3, hostax=hostax, 
                                       loc=loc, off=off, label=loc, 
                                       share=True, ws_add=ws_add)
        parax.plot(x*n, y**n)
    
    new_axis(fig3, hostax=hostax, loc='right', off=0, ws_add=0, 
             label="hostax right, I'm like twinx()")
    
    new_axis(fig3, hostax=hostax, loc='top', off=0, ws_add=0, 
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
    plt.show()
