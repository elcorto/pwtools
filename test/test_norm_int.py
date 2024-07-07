import numpy as np
from scipy.integrate import simpson as simps
from pwtools.num import norm_int

def test_norm_int():
    # simps(y, x) == 2.0
    x = np.linspace(0, np.pi, 100)
    y = np.sin(x)

    for scale in [True, False]:
        yy = norm_int(y, x, area=10.0, scale=scale)
        assert np.allclose(simps(yy,x=x), 10.0)
