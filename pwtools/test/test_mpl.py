# If we work over ssh, we really want to run matplotlib in non-X mode by
# telling it to use a non-X backend by ``matplotlib.use('Agg')``. This works if
# we call this test alone::
#
#   $ ./runtests.sh test_mpl.py
#
# but if we run the whole test suite, nose does apparently import matplotlib
# earlier and we get the annoying warning::
#
#   $ ./runtests.sh
#
#   /usr/lib/pymodules/python2.7/matplotlib/__init__.py:923: UserWarning:  This
#   call to matplotlib.use() has no effect
#   because the the backend has already been chosen;
#   matplotlib.use() must be called *before* pylab, matplotlib.pyplot,
#   or matplotlib.backends is imported for the first time.
#
# This happens over ssh and on localhost, wich is a big PITA!
#
# The only way to turn that off is to NOT use ``use('Agg')``. Over ``ssh -X``,
# this test will then be very slow b/c the whole TkAgg (default backend)
# machinery is running for no reason.
#
# If you wish, disable the test:
#
#   $ ./runtests.sh -e 'test_mpl'

from pwtools.test import tools

def test_mpl():
    try:
        from pwtools import mpl
        try:
            import os
            print(os.environ['DISPLAY'])
            fig,ax = mpl.fig_ax(dpi=15,num=20)
            assert fig.dpi == 15
            assert fig.number == 20

            pl = mpl.Plot(dpi=15,num=20)
            assert pl.fig.dpi == 15
            assert pl.fig.number == 20

            dct = mpl.prepare_plots(['test'], dpi=15,num=20)
            assert dct['test'].fig.dpi == 15
            assert dct['test'].fig.number == 20

            fig, ax = mpl.fig_ax3d(dpi=15)
            assert fig.dpi == 15
        except KeyError:
            tools.skip("no DISPLAY environment variable, skipping test")
    except ImportError:
        tools.skipping("couldn't import matplotlib, skipping test")

