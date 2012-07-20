#!/usr/bin/env python

# Use from within pwtools/doc/. Writes source/modules/<module>.rst for all
# modules defined in pwtools.__init__.__all__ .
#
# Prints a list
#   modules/crys
#   modules/io
#   ...
# 
# Copy-paste that into source/index.rst

import pwtools
from pwtools import common

txt = """
.. module:: {module}

{module}
{header_line}

.. automodule:: pwtools.{module}
   :members:
   :show-inheritance:
   :special-members:
"""

for module in pwtools.__all__:
    newtxt = txt.format(module=module,
                        header_line='='*len(module))
    common.file_write("source/modules/{module}.rst".format(module=module),
                      newtxt)
    
    print '   modules/%s' %module
