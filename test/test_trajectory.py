# We assume all lengths in Angstrom. Only important for ASE comparison.
#
import numpy as np
from pwtools.crys import Trajectory
from pwtools import crys, constants
from pwtools.test.tools import aaae
from pwtools import num
rand = np.random.rand

def test():
    natoms = 10
    nstep = 100
    cell = rand(nstep,3,3)
    stress = rand(nstep,3,3)
    forces = rand(nstep,natoms,3)
    etot=rand(nstep),
    cryst_const = crys.cell2cc3d(cell, axis=0)
    coords_frac = np.random.rand(nstep,natoms,3)
    coords = crys.coord_trans3d(coords=coords_frac,
                                old=cell,
                                new=num.extend_array(np.identity(3),
                                                     nstep,axis=0),
                                axis=1,
                                timeaxis=0)                                                    
    assert cryst_const.shape == (nstep, 6)
    assert coords.shape == (nstep,natoms,3)
    symbols = ['H']*natoms
    
    # automatically calculated:
    #   coords
    #   cell
    #   pressure
    #   velocity (from coords)
    #   temperature (from ekin)
    #   ekin (from velocity)
    st = Trajectory(coords_frac=coords_frac,
                    cell=cell,
                    symbols=symbols,
                    forces=forces,
                    stress=stress,
                    etot=etot,
                    timestep=1,
                    )
    # Test if all getters work.
    for name in st.attr_lst:
        print name
        if name not in ['ase_atoms']:
            st.try_set_attr(name)
            assert getattr(st, name) is not None, "attr None: %s" %name
            assert eval('st.get_%s()'%name) is not None, "getter returns None: %s" %name
    aaae(coords_frac, st.coords_frac)
    aaae(cryst_const, st.cryst_const)
    aaae(np.trace(stress, axis1=1, axis2=2)/3.0, st.pressure)
    assert st.coords.shape == (nstep,natoms,3)
    assert st.cell.shape == (nstep,3,3)
    assert st.velocity.shape == (nstep-1, natoms, 3)
    assert st.temperature.shape == (nstep-1,)
    assert st.ekin.shape == (nstep-1,)
    assert st.nstep == nstep
    assert st.natoms == natoms

    st = Trajectory(coords_frac=coords_frac,
                    symbols=symbols,
                    cell=cell)
    aaae(coords, st.coords)
    
    # Cell calculated from cryst_const has defined orientation in space which may be
    # different from the original `cell`, but the volume and underlying cryst_const
    # must be the same.
    st = Trajectory(coords_frac=coords_frac,
                    symbols=symbols,
                    cryst_const=cryst_const)
    try:
        aaae(cell, st.cell)
    except AssertionError:
        print "KNOWNFAIL: differrnt cell orientation"
    np.testing.assert_almost_equal(crys.volume_cell3d(cell),
                                   crys.volume_cell3d(st.cell))
    aaae(cryst_const, crys.cell2cc3d(st.cell))
    
    # extend arrays
    cell2d = rand(3,3)
    cc2d = crys.cell2cc(cell2d)
    st = Trajectory(coords_frac=coords_frac,
                    cell=cell2d,
                    symbols=symbols)
    assert st.cell.shape == (nstep,3,3)
    assert st.cryst_const.shape == (nstep,6)
    for ii in range(st.nstep):
        assert (st.cell[ii,...] == cell2d).all()
        assert (st.cryst_const[ii,:] == cc2d).all()
    
    st = Trajectory(coords_frac=coords_frac,
                    cryst_const=cc2d,
                    symbols=symbols)
    assert st.cell.shape == (nstep,3,3)
    assert st.cryst_const.shape == (nstep,6)
    for ii in range(st.nstep):
        assert (st.cryst_const[ii,:] == cc2d).all()

    # units
    st = Trajectory(coords_frac=coords_frac,
                    cell=cell,
                    symbols=symbols,
                    stress=stress,
                    forces=forces,
                    units={'length': 2, 'forces': 3, 'stress': 4})
    aaae(2*coords, st.coords)                    
    aaae(3*forces, st.forces)                    
    aaae(4*stress, st.stress)                    
    
    # minimal input
    st = Trajectory(coords=coords, 
                    symbols=symbols,
                    timestep=1)
    not_none_attrs = [\
        'coords',
        'ekin',
        'mass',
        'natoms',
        'nspecies',
        'nstep',
        'ntypat',
        'order',
        'symbols',
        'symbols_unique',
        'temperature',
        'timestep',
        'typat',
        'velocity',
        'znucl',
        ]
    for name in not_none_attrs:
        assert getattr(st, name) is not None, "attr None: %s" %name
        assert eval('st.get_%s()'%name) is not None, "getter returns None: %s" %name
