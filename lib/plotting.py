#!/usr/bin/env python
# encoding: utf-8
"""
plotting.py

Sets the plot Parameters of a pylab instance to specific values.

Created by Torsten Hahn on 2007-05-15.
Modified: B. Stechlich, John Doe, Fritz Pfiffig, 2007
Copyright (c) 2007 __MyCompanyName__. All rights reserved.
"""

import sys
import os
import subprocess as sp
import re

import matplotlib 

# Pyplot is new as of matplotlib 0.91.0 (see
# http://matplotlib.sourceforge.net/_static/CHANGELOG)
from matplotlib import pyplot as plt

import pycts.lib.utils as utils

#-----------------------------------------------------------------------------
# utilities
#-----------------------------------------------------------------------------

# aliases
pjoin = os.path.join 

def check_ax_obj(ax):
    if not isinstance(ax, matplotlib.axes.Axes):
        raise StandardError("argument `ax` %s is no instance of "
            "matplotlib.axes.Axes" %repr(ax))


#-----------------------------------------------------------------------------
# some matplotlib setups
#-----------------------------------------------------------------------------

def set_plot_layout(pyl_obj, layout='latex_hs'):
    """Set up the pylab plotting subsystem for a specific task."""
    py = pyl_obj
    
    if not hasattr(py, 'rc'):
        raise AttributeError("argument %s has no attribute 'rc'" %repr(py))

    # This is good for presentations and tex'ing files in a PhD Thesis
    if layout == 'latex_hs':
        py.rc('text', usetex=True) # this may not work on windows systems !!
        # lines
        py.rc('font', weight='bold')
        py.rc('lines', linewidth=3.0)
        py.rc('lines', markersize=8)
        #py.rc('patch', linewidth=1.5)        
        # axes
        py.rc('axes', titlesize=30)
        py.rc('axes', labelsize=28)
        py.rc('axes', linewidth=2.0)
        # ticks
        py.rc('xtick.major', size=12)
        py.rc('ytick.major', size=12)
        py.rc('xtick.minor', size=8)
        py.rc('ytick.minor', size=8)
        py.rc('xtick', labelsize=30)
        py.rc('ytick', labelsize=30)
        # legend
        py.rc('legend', numpoints=3)
        py.rc('legend', fontsize=25)
        py.rc('legend', markerscale=0.8)
        py.rc('legend', axespad=0.04)
        py.rc('legend', shadow=False)
        py.rc('legend', handletextsep=0.02)
        py.rc('legend', pad=0.3)
        # figure
        py.rc('figure', figsize=(12,9))
        py.rc('savefig', dpi=120)
        # Need this on Linux (Debian etch) for correct .eps bounding boxes. We
        # check for the OS here because it works without this on MacOS.
        if sys.platform == 'linux2':
            py.rc('ps', usedistiller='xpdf')
    else:
        pass

#-----------------------------------------------------------------------------

def set_tickline_width(ax, xmin=1.0,xmaj=1.5,ymin=1.0,ymaj=1.5):
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

def eps2pdf(f_name):
    """Converts a *.eps file to pdf using the epstopdf programm."""
    pc = sp.Popen('epstopdf ' + f_name, shell=True)
    pc.wait()

#-----------------------------------------------------------------------------

def savepdf(py, f_name):
    eps_fn = f_name + '.eps'
    print "Saving %s as pdf ..." %eps_fn
    py.savefig(eps_fn)
    eps2pdf(eps_fn)

#----------------------------------------------------------------------------- 
# Steve's custom stuff
#
    
def set_plot_layout_steve(pyl_obj, layout='latex_hs'):
    
    # first pass pyl_obj through Torstens func
    set_plot_layout(pyl_obj, layout=layout)
    
    # then, set custom vals
    pyl_obj.rc('font', weight='normal')
    pyl_obj.rc('xtick', labelsize=30)
    pyl_obj.rc('ytick', labelsize=30)
    pyl_obj.rc('legend', fontsize=25)
    # fractional whitespace between legend entries and legend border
    pyl_obj.rc('legend', pad=0.1) # default 0.2
    # length of the legend lines, useful for plotting many 
    # dashed lines with slowly changing dash length
    pyl_obj.rc('legend', handlelen=0.1) # default 0.05
    pyl_obj.rc('lines', dash_capstyle='round') # default 'butt'

#-----------------------------------------------------------------------------

def set_tickline_width_steve(ax, xmin=1.0, xmaj=1.5, ymin=1.0, ymaj=1.5):
    check_ax_obj(ax)
    set_tickline_width(ax, xmin=xmin, xmaj=xmaj, ymin=ymin, ymaj=ymaj) 

#-----------------------------------------------------------------------------

def set_plot_layout_phdth(pyl_obj):
    set_plot_layout_steve(pyl_obj)
    pyl_obj.rc('legend', pad=0.2)
    # figure
    pyl_obj.rc('figure', figsize=(11,10))
    pyl_obj.rc('savefig', dpi=100)

#-----------------------------------------------------------------------------

class Plot(object):
    
    def __init__(self, dir=None):
        # Dir where all to-be-plotted data is located.
        if dir is not None: 
            self.set_dir(dir)
        else:
            print("no dir, use set_dir() method")
        
        # Init the Figure to which we will plot everything. 
        self.init_fig()

    def set_dir(self, dir):
        """Set the data dir.
        
        example
        -------
        pl = Plot('/path/to/foo')
        pl.some_plot_function()
        pl.set_dir('/path/to/bar')
        pl.some_plot_function()
        """
        self.dir = utils.fullpath(dir)
        self.get_filenames()
        # Files that we load only once when we go to a dir.
        self.time = utils.load_binary_data(pjoin(self.dir, 'time.dat')) 
        self.keyrange = utils.load_binary_data(pjoin(self.dir, 'keyrange.dat'))

    def init_fig(self): 
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
    
    def get_filenames(self):
        """Determine all files to plot in a dir."""
        self.filenames = []
        for fn in os.listdir(self.dir):
            fn = pjoin(self.dir, fn)
            if os.path.isfile(fn):
                self.filenames.append(fn)
    
    def match_filename(self, fn, pattern):
        print("[match]: fn: %s" %fn)
        pattern = r'%s' %pattern
        if re.match(pattern, fn) is not None:
            return True
        else:
            return False

    def get_plotfilenames(self, pattern):
        # XXX expose to user as set_pattern or so
        self.plotfilenames = []
        for fn in self.filenames:
            if self.match_filename(fn, pattern):
                self.plotfilenames.append(fn)

    def plot_single(self, filename, outfilename, proc=None):
        """
        Load, process, plot, save.
        """
        data = utils.load_binary_data(filename)
        # XXX proc is an arbitrary function e.g.
        #
        # def dark_current(t,y):
        #   return t, y-y[0]
        # 
        # def transient(t, y, index=1000):
        #   return t[index:], y[index:]
        #
        if proc is not None:
            t, y = proc(self.time, data)
        else:
            t, y = self.time, data
        self.ax.plot(t, y)
        self.fig.savefig(outfilename)
    
    
    def plot_all(self):
        self.get_plotfilenames('.*level0.*\.dat')
        # XXX TEST, should be more general
        for fn in self.plotfilenames:
            self.plot_single(fn, fn.replace('.dat', '.png'))
        
        # XXX need this in non-interactive mode?
        pyplot.close()

