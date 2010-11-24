import numpy as np
from pwtools.batch import conv_table

def test():
    xx = ['a', 'b', 'c']
    yy = [1,2,3]
    print conv_table(xx,yy)

    xx = np.array([1,2,3])*np.pi
    yy = [1,2,3]
    print conv_table(xx,yy)
