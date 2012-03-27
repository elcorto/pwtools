
def test():
    import numpy as np
    from pwtools.pydos import velocity

    coords = np.arange(2*3*6).reshape(2,3,6)
    v1 = velocity(coords, axis=-1)
    v2 = coords[...,1:] - coords[...,:-1]
    v3 = np.diff(coords, n=1, axis=-1)

    assert v1.shape == v2.shape == v3.shape == (2,3,5)
    assert (v1 == v2).all()
    assert (v1 == v3).all()
