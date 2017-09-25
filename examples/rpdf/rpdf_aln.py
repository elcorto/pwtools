#!/usr/bin/python
#
# rpdf_aln.py
#
# Test pwtools.crys.rpdf() -- the radial pair distribution function -- on
# crystal data (rs-AlN).
# 
# Key results: 
# * Always use periodic boundary conditions to get the correct
#   number of nearest neighbors. 
# * Always use rmax < rmax_auto (results are only correct up to rmax_auto).
# * The function does work for non-orthorhombic boxes.

import os
import numpy as np
from pwtools import crys, io, mpl
plt = mpl.plt

class Struct(crys.Structure):
    def __init__(self, fnbase=None, tgtdir=None, **kwds):
        self.fnbase = fnbase
        self.tgtdir = tgtdir
        if (self.tgtdir is not None) and not os.path.exists(self.tgtdir):
            os.makedirs(self.tgtdir)
        crys.Structure.__init__(self, set_all_auto=True, **kwds)

    def _assert_fnbase(self):        
        assert (self.fnbase is not None), "self.fnbase is not set"
    
    def _get_fn(self, suffix):
        self._assert_fnbase()
        if self.tgtdir is None:
            fn = self.fnbase + suffix
        else:
            fn = os.path.join(self.tgtdir, self.fnbase + suffix)
        return fn                

    def write_cif(self):
        fn = self._get_fn('.cif')
        print(("writing: %s" %fn))
        crys.write_cif(fn, self)
    
    def savetxt(self):
        self._assert_fnbase()
        fn_coords = self._get_fn('.coords.txt')
        fn_cell = self._get_fn('.cell.txt')
        print(("writing: %s, %s" %(fn_coords, fn_cell)))
        np.savetxt(fn_coords, self.coords_frac)
        np.savetxt(fn_cell, self.cell)
                    

class Plot(object):
    pass


def plot_pl(*pls):
    # A `pl` is an instance of Plot(). Plot the RPDF and the number integral
    # with the same color and marker. This can be done b/c they look quite
    # different. 
    fig, ax = mpl.fig_ax()
    for pl in pls:
        pl.line1, = ax.plot(pl.rad, 
                            pl.hist, 
                            pl.color + pl.marker + '-',
                            )
        pl.line2, = ax.plot(pl.rad, 
                            pl.num_int, 
                            pl.color + pl.marker + '-', 
                            )
    ax.legend((pl.line1 for pl in pls), (pl.leg_label for pl in pls),
              loc='upper left')
    xlo, xhi = ax.get_xlim()
    ax.hlines(1, xlo, xhi, color='k', lw=2)
    return fig,ax

if __name__ == '__main__':
    
    #------------------------------------------------------------------------
    # Generate some structures and calculate their RPDF.
    #------------------------------------------------------------------------
    structs = {}
    tgtdir = '/tmp/rpdf_test' 

    # AlN ibrav=0 
    # -----------
    #
    # A rs-AlN crystal with ibrav=0 -- cubic unit cell (orthorhombic). We use
    # some bogus alat=5.0 (Angstrom) to have nicer rmax values. We build a
    # 2x2x2 supercell which has alat=10 and therefore rmax_auto = 5.0 = half
    # the box length.
    #
    # Nearest neighbor numbers are:
    #   shell   r   number int.     num. neibors in shell
    #   -----   -   -----------     ---------------------
    #
    #   1st     2.5     6           6
    #   2nd     3.5     18          12
    #   3rd     4.3     26          8
    #
    # This cell can also be tested in VMD.
    #
    coords_frac = \
        np.array([[0.0, 0.0, 0.0], 
                  [0.5, 0.0, 0.0],
                  [0.0, 0.5, 0.0],
                  [0.5, 0.5, 0.0],
                  [0.0, 0.0, 0.5],
                  [0.5, 0.0, 0.5],
                  [0.0, 0.5, 0.5],
                  [0.5, 0.5, 0.5],
                  ])
    symbols = ['Al', 'N', 'N', 'Al', 'N', 'Al', 'Al', 'N']
    alat = 5.0
    cell = np.identity(3) * alat
    struct = crys.Structure(coords_frac=coords_frac,
                            symbols=symbols,
                            cell=cell)
    sc = crys.scell(struct, (2,2,2))
    name = 'aln_ibrav0_sc'
    structs[name] = Struct(coords_frac=sc.coords_frac, 
                           cell=sc.cell,
                           symbols=sc.symbols,
                           fnbase=name,
                           tgtdir=tgtdir)
    print(structs[name].cryst_const)                            
##    structs[name].write_cif()
##    structs[name].write_axsf()
##    structs[name].savetxt()

    
    # AlN ibrav=2
    # -----------
    #
    # The same rs-AlN crystal, but with ibrav=2 -- fcc. The primitive cell is
    # non-orthorhombic. We build a 4x4x4 supercell which has an rmax_auto of
    # 5.8. For rmax up to 5, the RPDF and the number integral must match
    # exactly with that of the ibrav=0 case b/c the structure is the same.
    #
    coords_frac = \
        np.array([[0.0, 0.0, 0.0], 
                  [0.5, 0.5, 0.5],
                  ])
    symbols = ['Al', 'N']
    alat = 5.0
    cell = alat/2.0 * np.array([[-1,0,1.], [0,1,1], [-1,1,0]]) # pwscf
    struct = crys.Structure(coords_frac=coords_frac,
                            symbols=symbols,
                            cell=cell)
    sc = crys.scell(struct, (4,4,4))
    name = 'aln_ibrav2_sc'
    structs[name] = Struct(coords_frac=sc.coords_frac, 
                           cell=sc.cell,
                           symbols=sc.symbols,
                           fnbase=name,
                           tgtdir=tgtdir)
    print(structs[name].cryst_const)                            
##    structs[name].write_cif()
##    structs[name].write_axsf()
##    structs[name].savetxt()

    
    #------------------------------------------------------------------------
    # plotting
    #------------------------------------------------------------------------
    
    # For all structs, calculate rmax=5 (= rmax_auto for aln_ibrav0_sc for a
    # 2x2x2 cell) and rmax = 20 as well as pbc=True, False. Plot for 1 struct.
    plots = {}
    for struct in structs.values():
        cc = iter(['b', 'r', 'g', 'm'])
        mm = iter(['v', '^', '+', 'x'])
        for rmax in [5, 20]:
            for pbc in [True, False]:
                rmax_auto = crys.rmax_smith(struct.cell)
                out = \
                    crys.rpdf(struct, 
                              rmax=rmax, 
                              dr=0.05, 
                              pbc=pbc)
                pl = Plot()
                pl.name = struct.fnbase
                pl.rad = out[:,0]
                pl.hist = out[:,1]
                pl.num_int = out[:,2]
                pl.color = next(cc)                                                     
                pl.marker = next(mm)                                                     
                pl.leg_label = "pbc=%s, rmax=%i, rmax_auto=%.1f" %(pbc, rmax,
                                                                   rmax_auto)
                plots["%s-%i-%s" %(struct.fnbase, rmax, pbc)] = pl
    
    name='aln_ibrav0_sc'
    plot_pl(plots[name + "-5-True"],  plots[name + "-20-True"])
    plot_pl(plots[name + "-5-False"], plots[name + "-20-False"])
    plot_pl(plots[name + "-5-True"],  plots[name + "-5-False"])
    plot_pl(plots[name + "-20-True"], plots[name + "-20-False"])
    
    # For one setting which we know is correct (pbc=True), test AlN
    # ibrav=0,2. Results match up to the *smaller* rmax_auto of the two
    # structures, which is rmax=5 here for aln_ibrav0_sc. 
    #
    # This means that rmax > rmax_auto gives results which depend on the shape
    # of the box, which *should not be*! This is a result of the violation of
    # the minimum image convention! Always use rmax <= rmax_auto!
    pl1 = plots['aln_ibrav0_sc-20-True']
    pl1.color = 'r'
    pl1.marker = 'o'
    pl1.leg_label = pl1.name + ', ' + pl1.leg_label
    pl2 = plots['aln_ibrav2_sc-20-True']
    pl2.color = 'g'
    pl2.marker = '+'
    pl2.leg_label = pl2.name + ', ' + pl2.leg_label
    plot_pl(pl1,pl2)
    
    # g_{Al,N}(r): Al-N rpdf between 2 seelctions, 
    # num_int should be: 1st (6) = 6, 2nd (8) = 14
    name = 'aln_ibrav0_sc'
    rmax = 5
    pbc = True
    struct = structs[name]
    ##c1 = struct.coords[:,np.array(struct.symbols) == 'Al',:]
    ##c2 = struct.coords[:,np.array(struct.symbols) == 'N',:]
    sy = np.array(struct.symbols)
    amask = [sy=='Al', sy=='N']

    rmax_auto = crys.rmax_smith(struct.cell)
    out = \
        crys.rpdf(struct, 
                  rmax=rmax,
                  amask=amask,
                  dr=0.05, 
                  pbc=pbc)

    cc = iter(['b', 'r', 'g', 'm'])
    mm = iter(['v', '^', '+', 'x'])
    pl = Plot()
    pl.name = struct.fnbase
    pl.rad = out[:,0]
    pl.hist = out[:,1]
    pl.num_int = out[:,2]
    pl.color = next(cc)                                                     
    pl.marker = next(mm)                                                     
    pl.leg_label = "pbc=%s, rmax=%i, rmax_auto=%.1f, Al-N" %(pbc, rmax,
                                                             rmax_auto)
    plot_pl(pl)                                                   
    
    plt.show()
    
