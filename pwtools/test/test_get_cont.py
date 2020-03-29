from pwtools import parse, num
from pwtools.test import tools

def test_get_cont():
    filename = tools.unpack_compressed('files/pw.md.out.gz', prefix=__file__)
    pp = parse.PwMDOutputFile(filename=filename)
    tr1 = pp.get_traj()

    # Need new parser instance, since pp.cont is already used, i.e. set_all()
    # called -> all attrs set. Also units are already applied, thus won't be
    # applied again since self.units_applied=True.
    pp = parse.PwMDOutputFile(filename=filename)
    tr2 = pp.get_traj(auto_calc=False)

    # specific for the used pw.out file, None is everything which is not parsed
    # since nothing is calculated from parsed data
    none_attrs = [
        'coords_frac',
        'cryst_const',
        'pressure',
        'velocity',
        'volume',
        'mass',
        'mass_unique',
        'nspecies',
        'ntypat',
        'order',
        'symbols_unique',
        'typat',
        'time',
        'znucl',
        'znucl_unique',
        ]

    for name in tr1.attr_lst:
        a1 = getattr(tr1, name)
        a2 = getattr(tr2, name)
        if name in none_attrs:
            assert a1 is not None, ("a1 %s is None" %name)
            assert a2 is None, ("a2 %s is not None" %name)
        else:
            tools.assert_all_types_equal(a1, a2)
