from pwtools import parse
from pwtools.crys import UnitsHandler

parsers = [parse.CifFile,
           parse.CMLFile,
           parse.PDBFile,
           parse.PwSCFOutputFile,
           parse.PwMDOutputFile,
           parse.PwVCMDOutputFile,
           parse.CpmdSCFOutputFile,
           parse.CpmdMDOutputFile,
           parse.Cp2kSCFOutputFile,
           parse.Cp2kMDOutputFile,
           parse.LammpsTextMDOutputFile,
           parse.LammpsDcdMDOutputFile,
         ]

def test_parser_units():
    units = UnitsHandler().units_map.keys()
    for pa in parsers:
        pp = pa()
        print "testing:", str(pp)
        for key in units:
            if pp.default_units.has_key(key):
                # Check if default units go correctly into self.units.
                assert pp.default_units[key] == pp.units[key], ("default unit "
                    "not passed on correct: key={0}, default={1}, "
                    "current={2}".format(key, pp.default_units[key],
                                        pp.units[key]))
                # Check if units passed by the user are correctly passed on.
                dval = pp.default_units[key]
                val = pp.units[key] 
                print "  key, default, curent:", key, dval, val
                pp2 = pa(units={key: val*20})
                dval2 = pp2.default_units[key]
                val2 = pp2.units[key] 
                print "  key, default, curent:", key, dval2, val2
                assert dval2 == dval
                assert val2 == 20*val
            else:
                if pp.units.has_key(key):
                    val = pp.units[key]
                    print "  key, current:", key, val
                    assert val == 1.0

