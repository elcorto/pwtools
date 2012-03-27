# Test crys.rpdf() against reference results. Test 2 AlN structs from
# examples/rpdf/rpdf_aln.py and one random trajectory with selections (2 atom
# types) and time average.
#
# See examples/rpdf/ for more examples and VMD comparisons, esp. regarding
# nearest neighbors.
#
# See utils/gen_rpdf_ref.py for how the references were generated.
#
# Compare crys.vmd_measure_gofr() against crys.rpdf(). This was tested w/ VMD
# 1.9 and 1.8.7 . Note that these tests may fail for other VMD versions.

import os
import numpy as np
from pwtools import crys, parse, arrayio
from testenv import testdir
from pwtools.test.tools import aaae, aae
pj = os.path.join
rand = np.random.rand

doplot = False

if doplot:
    from matplotlib import pyplot as plt

have_vmd = os.system('which vmd > /dev/null 2>&1') == 0

def load_old(fn):
    # Load old ref data and reshape:
    # (natoms, 3, nstep) -> (nstep, natoms, 3)    
    arr = arrayio.readtxt(fn)
    arr2 = np.rollaxis(arr, 2, 0)
    for istep in range(arr.shape[-1]):
        assert (arr[...,istep] == arr2[istep,...]).all()
    return arr2        

def test():
    for name in ['rand_3d', 'aln_ibrav0_sc', 'aln_ibrav2_sc']:
        print("name: %s" %name)
        dd = 'files/rpdf'
        if name == 'rand_3d':
            # 2 Trajectory = 2 selections
            cell = np.loadtxt(pj(dd, name + '.cell.txt'))
            coords_frac = [load_old(pj(dd, name + '.coords0.txt')), 
                           load_old(pj(dd, name + '.coords1.txt'))]
            trajs = [crys.Trajectory(coords_frac=cf, cell=cell) for cf in
                     coords_frac]
            for tr in trajs:
                assert tr.coords_frac.shape == (20,10,3)
                assert tr.nstep == 20
                assert tr.natoms == 10
        else:
            # one Structure
            struct = parse.CifFile(pj(dd, name + '.cif')).get_struct()
            trajs = [struct]
            cell = struct.cell

        # rpdf() 
        ret = crys.rpdf(trajs, rmax=5.0, dr=0.05, pbc=True)
        results = {'rad':       ret[:,0],
                   'hist':      ret[:,1], 
                   'num_int':   ret[:,2],
                   'rmax_auto': np.array(crys.rmax_smith(cell)),
                   }
        for key, val in results.iteritems():
            print("    key: %s" %key)
            ref_fn = pj(dd, "result.%s.%s.txt" %(key, name))
            print("    reference file: %s" %ref_fn)
            ref = np.loadtxt(ref_fn)
            if doplot:
                plt.figure()
                plt.plot(ref, '.-', label='ref')
                plt.plot(val, '.-', label='val')
                plt.legend()
                plt.title(key)
            else:
                # decimal=3 b/c ref data created w/ older implementation,
                # slight numerical noise
                np.testing.assert_array_almost_equal(ref, val, decimal=3)
                print("    key: %s ... ok" %key)
        
        # API
        if name.startswith('aln_'):
            sy = np.array(trajs[0].symbols)
            ret1 = crys.rpdf(trajs, dr=0.1, amask=[sy=='Al', sy=='N'])
            ret2 = crys.rpdf(trajs, dr=0.1, amask=['Al', 'N'])
            aae(ret1, ret2)

    if have_vmd:                        
        # slicefirst and API
        print "vmd_measure_gofr: slicefirst ..."
        traj = crys.Trajectory(coords_frac=rand(100,20,3),
                               cell=np.identity(3)*20,
                               symbols=['O']*5+['H']*15)
        
        for first,last,step in [(0,-1,1), (20, 80, 10)]:
            ret = []
            for sf in [True, False]:
                print "first=%i, last=%i, step=%i, slicefirst=%s" %(first,
                    last, step, sf)
                tmp = crys.vmd_measure_gofr(traj, 
                                            dr=0.1,
                                            sel=['all', 'all'],
                                            slicefirst=sf,
                                            first=first,
                                            last=last,
                                            step=step,
                                            fntype='xsf',
                                            usepbc=1, datafn=None,
                                            scriptfn=None, logfn=None, xsffn=None,
                                            tmpdir=testdir,
                                            keepfiles=True,
                                            verbose=False,
                                            )
                ret.append(tmp)
        
        aaae(ret[0][:,0], ret[1][:,0])
        aaae(ret[0][:,1], ret[1][:,1])
        aaae(ret[0][:,2], ret[1][:,2])
        
        # compare results, up to L/2 = rmax_auto = 10 = rmax_smith(cell)
        
        # all-all, hist will differ
        rmax = 10
        vmd = crys.vmd_measure_gofr(traj, dr=0.1, sel=['all', 'all'], rmax=10)
        pwt = crys.rpdf(traj, dr=0.1, amask=None, rmax=10)
        aaae(vmd[:-1,0], pwt[:,0])  # rad
        ##aaae(vmd[:-1,1], pwt[:,1]) # hist
        aaae(vmd[:-1,2], pwt[:,2])  # num_int
        
        # 2 selections, all ok
        sy = np.array(traj.symbols)
        vmd = crys.vmd_measure_gofr(traj, dr=0.1, sel=['name O', 'name H'], rmax=10)
        pwt = crys.rpdf(traj, dr=0.1, amask=[sy=='O', sy=='H'], rmax=10)
        aaae(vmd[:-1,0], pwt[:,0])  # rad
        aaae(vmd[:-1,1], pwt[:,1])  # hist
        aaae(vmd[:-1,2], pwt[:,2])  # num_int
        
        if doplot:
            plt.show()
