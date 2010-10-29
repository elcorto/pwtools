def test():    
    from pwtools.parse import PwOutputFile
    from pwtools import common

    filename = 'files/pw.md.out'
    infile = 'files/pw.md.in'

    common.system('gunzip %s.gz' %filename)
    pp = PwOutputFile(filename=filename, infile=infile)
    pp.parse()
    common.print_dct(pp.__dict__)
    none_attrs = ['cell',
                  'volume',
                 ]             
    for attr_name in pp.attr_lst:
        attr = getattr(pp, attr_name)
        if attr_name not in none_attrs:
            assert attr is not None, "FAILED: %s" %attr_name
        ##if attr is None:
        ##    print ">>> '%s'," %attr_name
    common.system('gzip %s' %filename)
