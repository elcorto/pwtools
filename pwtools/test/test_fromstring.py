import numpy as np

# test that fromstring does what we expect, don't get broken code by numpy
# update :)
def test_fromstring():
    txt1 = """1
2.0 3.0      
    4

  5 6
"""  
    txt2 = """
    1 2 3
    4 5 6"""
    
    txt3 = "1 2 3 4 5             6"

    arr = np.array([1,2,3,4,5.0, 6])
    for sep in [' ', '      ']:
        for txt in [txt1, txt2, txt3]:
            assert (arr == np.fromstring(txt, sep=sep, dtype=float)).all()
