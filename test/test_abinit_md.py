import numpy as np
from pwtools import parse, common, crys

def assrt_aae(*args, **kwargs): 
    np.testing.assert_array_almost_equal(*args, **kwargs)

def check(pp, none_attrs=[], extra_attrs=[]):
    """
    args:
    -----
    pp : instance of parsing class
    none_attrs : attrs which are in self.attr_lst but are None after calling
        self.parse().
    extra_attrs : attrs which are not in self.attr_lst but for which a getter
        exists
    """
    print ">>> attr_lst" 
    for attr_name in sorted(pp.attr_lst):
        if not attr_name in none_attrs:
            print("    attr not None %s" %attr_name)
            assert getattr(pp, attr_name) is not None
        else:
            print("    attr None:    %s" %attr_name)
    print "<<< attr_lst" 
    print ">>> extra_attrs" 
    for attr_name in sorted(extra_attrs):
        if not attr_name in none_attrs:
            print("    attr not None %s" %attr_name)
            assert eval("pp.get_%s()" %attr_name) is not None
        else:
            print("    attr None:    %s" %attr_name)
    print "<<< extra_attrs" 


def check_cons(pp):
    """Check consistency of redundant attrs. This is mainly for debugging.

    Check consistency between cell and cryst_const. Assume:
    pp.time_axis = -1
    cell.shape = (3,3,nstep)
    c1.shape = (nstep, 6)

    Check forces_rms. Assume:
    pp.time_axis = -1
    pp.forces_rms.shape = (nstep,)
    pp.forces.shape = (natoms, 3, nstep)
    """
    print ">>> cryst_const" 
    cc1 = pp.get_cryst_const()
    cc2 = pp.get_cryst_const_angles_lengths()
    if None not in [cc1, cc2]:
        assert cc1.shape[0] == cc2.shape[0]
        assrt_aae(cc1, cc2)
    print "<<< cryst_const" 
    print ">>> cryst_const cell" 
    cell = pp.get_cell()
    cc = pp.get_cryst_const()
    if cell is not None:
        nstep_cell = cell.shape[-1]
        if cc is not None:
            assert nstep_cell == cc.shape[0], ("cell and cryst_const have "
                "different nstep")
            for ii in range(nstep_cell):
                assrt_aae(crys.cell2cc(cell[...,ii]), cc[ii,:])
    print "<<< cryst_const cell" 
    print ">>> forces_rms"
    arr1 = pp.get_forces_rms()
    if arr1 is not None:
        forces = pp.get_forces()
        if forces is not None:
            arr2 = crys.rms3d(forces, axis=-1, nitems='all')
            assrt_aae(arr1, arr2)
    print "<<< forces_rms"
    print ">>> volume"
    arr1 = pp.get_volume()
    if arr1 is not None:
        cell = pp.get_cell()
        if cell is not None:
            arr2 = np.array([crys.volume_cell(pp.cell[...,ii]) for ii in \
                           range(pp.cell.shape[-1])])
            n1, n2 = arr1.shape[0], arr2.shape[0]
            nn = min(n1, n2)
            assrt_aae(arr1[-nn:], arr2[-nn:], decimal=4)
    print "<<< volume"


def run(cls, filename_base, none_attrs=[], extra_attrs=[]):
    """Parse file and check that all relevant attrs are not None, i.e. that the
    getter found something. 
    
    Note that we do not do verfication with reference data, that would be
    infinitely much work. Parsing all this weird files is complicated enough as
    it is. If you use this parsing machinery, verify results by yourself before
    using it for serious analyis.
    
    Only as a proof of concept, run the SCF parser on the MD output. No attr
    should be None. Then, the MD parser.
    """
    filename = 'files/abinit_md/%s' %filename_base
    print("#"*77)
    print("testing: %s" %filename)
    print("#"*77)
    common.system('gunzip %s.gz' %filename)
    print "--- scf ------------------------"
    scf = parse.AbinitSCFOutputFile(filename=filename)
    scf.parse()
    check(scf, none_attrs=[], extra_attrs=[])
    print "--- md -------------------------"
    md = cls(filename=filename)
    md.parse()
    check(md, none_attrs=none_attrs, extra_attrs=extra_attrs)
    check_cons(md)
    common.system('gzip %s' %filename)


def test():
    extra_attrs = [\
        ]
    
    filename = 'abi_ionmov13_optcell2.out'
    none_attrs = [\
##        'angles',
##        'cell',
##        'cryst_const',
##        'cryst_const_angles_lengths',
##        'ekin_vel',
##        'lengths',
##        'temperature',
##        'velocity',
        ]
    run(parse.AbinitVCMDOutputFile, filename, none_attrs=none_attrs,
        extra_attrs=extra_attrs)
    
    filename = 'abi_ionmov13_optcell1.out'
    none_attrs = [\
##        'angles',
##        'cell',
##        'cryst_const',
##        'cryst_const_angles_lengths',
##        'ekin_vel',
##        'lengths',
##        'temperature',
##        'velocity',
        ]
    run(parse.AbinitVCMDOutputFile, filename, none_attrs=none_attrs,
        extra_attrs=extra_attrs)

    filename = 'abi_ionmov13_optcell0.out'
    none_attrs = [\
##        'angles',
##        'cell',
##        'cryst_const',
##        'cryst_const_angles_lengths',
##        'ekin_vel',
##        'lengths',
##        'temperature',
##        'velocity',
        ]
    run(parse.AbinitVCMDOutputFile, filename, none_attrs=none_attrs,
        extra_attrs=extra_attrs)

    filename = 'abi_ionmov2_optcell2.out'
    none_attrs = [\
##        'angles',
##        'cell',
##        'cryst_const',
##        'cryst_const_angles_lengths',
        'ekin_vel',
##        'lengths',
        'temperature',
        'velocity',
        ]
    run(parse.AbinitMDOutputFile, filename, none_attrs=none_attrs,
        extra_attrs=extra_attrs)
    
    filename = 'abi_ionmov2_optcell1.out'
    none_attrs = [\
##        'angles',
##        'cell',
##        'cryst_const',
##        'cryst_const_angles_lengths',
        'ekin_vel',
##        'lengths',
        'temperature',
        'velocity',
        ]
    run(parse.AbinitMDOutputFile, filename, none_attrs=none_attrs,
        extra_attrs=extra_attrs)
    
    filename = 'abi_ionmov2_optcell0.out'
    none_attrs = [\
##        'angles',
##        'cell',
##        'cryst_const',
##        'cryst_const_angles_lengths',
        'ekin_vel',
##        'lengths',
        'temperature',
        'velocity',
        ]
    run(parse.AbinitMDOutputFile, filename, none_attrs=none_attrs,
        extra_attrs=extra_attrs)

    filename = 'abi_ionmov8_optcell0.out'
    none_attrs = [\
##        'angles',
##        'cell',
##        'cryst_const',
##        'cryst_const_angles_lengths',
##        'ekin_vel',
##        'lengths',
##        'temperature',
##        'velocity',
        ]
    run(parse.AbinitMDOutputFile, filename, none_attrs=none_attrs,
        extra_attrs=extra_attrs)

