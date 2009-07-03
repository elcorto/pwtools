# mpl.py
# 
# Plotting stuff for matplotlib: layouts, predefined markers etc.

import itertools
import sys
import os
import matplotlib 

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
    set_plot_layout_steve(pyl_obj)
    pyl_obj.rc('legend', pad=0.2)
    # figure
    pyl_obj.rc('figure', figsize=(11,10))
    pyl_obj.rc('savefig', dpi=100)

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    import numpy as np

    plt.figure(0)
    # All combinations of color and marker
    for col_mark in colors_markers: 
        plt.plot(np.random.rand(10), col_mark+'-')
        # The same
        ## plot(rand(10), col_mark, linestyle='-')
    
    plt.figure(1)
    # Now use one of those iterators
    t = np.linspace(0, 2*np.pi, 100)
    for f in np.linspace(1,2, 14):
        plt.plot(t, np.sin(2*np.pi*f*t)+10*f, ccm.next()+'-')        
    plt.show()        

