# Test if backwd compat mode for parse.Pw*file.get_gell_parameters() works.

def test():
    from pwtools import parse
    from pwtools import common
    
    pp = parse.PwInputFile('files/pw.md.in')
    pp.set_attr_lst(['cell_parameters'])
    pp.parse()
    # get_cell_parameters() should call get_cell() ans thus self.cell must be
    # set.
    assert pp.cell_parameters is not None
    assert pp.cell is not None
    assert (pp.cell == pp.cell_parameters).all()
    # Must also work by calling the getter directly, i.e. w/o 
    # set_attr_lst() + parse()
    pp.set_attr_lst([])
    assert pp.get_cell_parameters() is not None

    filename = 'files/pw.vc-relax.out'
    common.system('gunzip %s.gz' %filename)
    pp = parse.PwOutputFile(filename=filename)
    pp.set_attr_lst(['cell_parameters'])
    pp.parse()
    assert pp.cell_parameters is not None
    common.system('gzip %s' %filename)
