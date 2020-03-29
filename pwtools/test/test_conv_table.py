import numpy as np
from pwtools.batch import conv_table

def test_conv_table():
    xx1 = ['a', 'b', 'c']
    xx2 = np.array([1,2,3])*np.pi
    yy1 = [1,2,3]
    yy2 = [1.2,2.3,3.4]
    yy3 = [1.666,2.777,3.888]
    for xx in [xx1, xx2]:
        for mode in ['next','last']:
            st1 = conv_table(xx, yy1, mode=mode)
            st2 = conv_table(xx, [yy1], mode=mode)
            assert st1 == st2
            st1 = conv_table(xx, [yy1,yy2], mode=mode)
            st2 = conv_table(xx, np.array([yy1,yy2]), mode=mode)
            assert st1 == st2
            st1 = conv_table(xx, [yy1,yy2,yy3], mode=mode)
            st2 = conv_table(xx, np.array([yy1,yy2,yy3]), mode=mode)
            assert st1 == st2

    # API
    conv_table(xx, [yy1,yy2,yy3], mode='last', orig=True)
