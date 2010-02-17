from pwtools.lib.parse import CMLFile
from pwtools import common

c = CMLFile('files/cml_struct.cml')
c.parse()

common.print_dct(c.__dict__)
