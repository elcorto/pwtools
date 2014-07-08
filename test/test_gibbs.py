"""
Compare thermo.Gibbs results with pre-calculated reference data. Need this in
case of code refactoring.

In the 2d case, we need to use fairly high tolerances (rtol). We have numerical
noise in the test/files/gibbs/**.h5 test data files when running
test/utils/gibbs_test_data.py on different machines, with the same numpy, scipy
etc versions! But only in the 2d case. All #opt results are affected. This is a
comparison of 

  files/gibbs/2d/cartman.h5
  files/gibbs/2d/kenny.h5
  
  value                 abs. err   rel. err
  -----                 --------   --------  
  /#opt/T/P/Cp          1.04e-10   4.18e-11
  /#opt/T/P/G              5e-16   5.36e-16  <<<
  /#opt/T/P/V           3.02e-07   9.78e-09
  /#opt/T/P/alpha_V     7.08e-10   0.000187  <<<
  /#opt/T/P/alpha_ax0   3.26e-10   0.000206  <<<
  /#opt/T/P/alpha_ax1   2.21e-10   0.000356  <<<
  /#opt/T/P/ax0         1.43e-08   4.74e-09
  /#opt/T/P/ax1         6.76e-09   1.98e-09
  /P/P                         0          0
  /T/P/ax0-ax1/G               0          0
  /T/T                         0          0
  /ax0-ax1/Etot                0          0
  /ax0-ax1/T/Cv                0          0
  /ax0-ax1/T/Evib              0          0
  /ax0-ax1/T/F                 0          0
  /ax0-ax1/T/Fvib              0          0
  /ax0-ax1/T/Svib              0          0
  /ax0-ax1/V                   0          0
  /ax0-ax1/ax0-ax1             0          0
   
Since all #opt results are affected, it seems that fmin(), which is used by
default in the 2d case to minimize G(ax0,...), is very sensitive to rounding
errors and therefore the results depend on the machine. But note that
/#opt/T/P/ax0 and ax1 have an error of about 1e-9, which is still OK. The only
big relative error of 1e-4 is in the alpha_* values! This is probably b/c the
errors in the ax0 and ax1 results are magnified by the 1st derivative. Similar
for Cp, where the error in G is 1e-16 and in Cp (*second* derivative) is 1e-11,
only that this is still pretty good!

For this reason, it is also a bad idea to use a log-scale T axis in this test.
At low T, the thermal expansion is zero and almost constant. Since we use a
Spline by default to fit the ax0(T) and ax1(T), which goes thru all points,
small numerical noise *on one machine* from the minimization (find
/#opt/T/P/ax0 and ax1) will be greatly magnified when taking the derivative,
especially if we have very small dT steps. This would be no problem if the
produced data was the same on each machine! Then we would conmpare the *same*
large fluctuations. However, due to the rounding behavior on different
machines, these (large) fluctuations are *different* for each machine, which
makes the resulting alpha values impossible to compare at low T. When using a
linear T scale, we kind of average out the fluctuations and therefore minimize
this effect.

In the 1d case, we have *zero* difference, i.e. different machines produce the
same test data.
"""

import numpy as np
import copy, os
from itertools import product
from pwtools.thermo import Gibbs
from pwtools.signal import gauss
from pwtools import num, crys, io, constants
from pwtools.test import tools


def compare_dicts_with_arrays(a, b):
    """For each numpy array in `a` and `b`, print absolute and relative
    difference."""
    for k in a.keys():
        if b.has_key(k) and b[k] is not None:
            df = abs(a[k]-b[k]).max()
            fmt = "{:20}  {:8.3g}   {:8.3g}"
            print fmt.format(k, df, df / abs(a[k]).max())
                

def test_gibbs():
    # number of varied axis points
    nax = 6
    # phonon freq axis
    freq = np.linspace(0,1000,300) # cm^-1
    T = np.linspace(5, 2000, 50)
    P = np.linspace(0,5,2)

    # 2d case
    case = '2d'
    cell_a = np.linspace(2.5,3.5,nax)
    cell_c = np.linspace(3,3.8,nax)
    volfunc_ax = lambda x: x[0]**2 * x[1]
    axes_flat = np.array([x for x in product(cell_a, cell_c)])
    V = np.array([volfunc_ax(x) for x in axes_flat])
    cell_a_mean = cell_a.mean()
    cell_c_mean = cell_c.mean()
    cell_a_min = cell_a.min()
    cell_c_min = cell_c.min()
    etot = np.array([(a-cell_a_mean)**2.0 + (c-cell_c_mean)**2.0 for a,c in axes_flat])
    phdos = []
    Vmax = V.max()
    # phonon dos (just a gaussian) shifted to lower (higher) freqs for higher
    # (lower) volume
    for ii in range(axes_flat.shape[0]):
        a,c = axes_flat[ii,:]
        fc = 550 - 50 * V[ii] / Vmax
        phdos.append(np.array([freq,gauss(freq-fc,100)*0.01]).T)

    gibbs = Gibbs(T=T, P=P, etot=etot, phdos=phdos, axes_flat=axes_flat,
                  volfunc_ax=volfunc_ax, case=case, dosarea=None)
    gibbs.set_fitfunc('C', lambda x,y: num.Spline(x,y,s=None,k=5, eps=1e-5))
    g = gibbs.calc_G(calc_all=True)
    
    dr = 'files/gibbs/2d'
    for name in os.listdir(dr):
        fn = '%s/%s' %(dr, name)
        gref = io.read_h5(fn)
        print "testing: %s" %fn
        compare_dicts_with_arrays(gref, g) 
        tools.assert_dict_with_all_types_almost_equal(gref, 
                                                      g, 
                                                      keys=gref.keys(),
                                                      atol=1e-8, rtol=1e-3)

    # 1d case
    case = '1d'
    V = np.linspace(10,20,nax)
    axes_flat = V**(1/3.) # cubic
    volfunc_ax = lambda x: x[0]**3.0
    etot = (V-V.mean())**2
    fcenter = 450 + 100*(axes_flat - axes_flat.min())
    # fake phonon dos data (Gaussian), shift to lower freq for higher volume
    phdos = [np.array([freq,gauss(freq-fc, 100)]).T for fc in
             fcenter[::-1]]

    gibbs = Gibbs(T=T, P=P, etot=etot, phdos=phdos, axes_flat=axes_flat,
                  volfunc_ax=volfunc_ax, case=case, dosarea=None)
    gibbs.set_fitfunc('C', lambda x,y: num.Spline(x,y,s=None,k=5, eps=1e-5))
    g = gibbs.calc_G(calc_all=True)
    
    dr = 'files/gibbs/1d'
    for name in os.listdir(dr):
        fn = '%s/%s' %(dr, name)
        gref = io.read_h5(fn)
        print "testing: %s" %fn
        compare_dicts_with_arrays(gref, g) 
        tools.assert_dict_with_all_types_almost_equal(gref, 
                                                      g, 
                                                      keys=gref.keys(),
                                                      atol=1e-14, rtol=1e-14)
    
