def test():
    from pwtools.parse import PwInputFile
    from pwtools import common

    filename = 'files/pw.scf.in.2'

    pp = PwInputFile(filename=filename)
    pp.parse()
    common.print_dct(pp.__dict__)

    none_attrs = ['cell',
                  'cryst_const',
                 ]             
    for attr_name in pp.attr_lst:
        attr = getattr(pp, attr_name)
        if attr_name not in none_attrs:
            assert attr is not None, "FAILED: %s" %attr_name
        ##if attr is None:
        ##    print ">>> '%s'," %attr_name
