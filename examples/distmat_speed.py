"""
tl;dr
-----
for serial, use scipy.spatial.distance.cdist()

measurements
------------
distsq serial: distsq == cdist (fastest C implementation one can find)

100 loops: 2.751 +- 0.0071: ((arr[:,None,:] - arr[None,...])**2.0).sum(-1)
100 loops: 0.431 +- 0.0049: squareform(pdist(arr, metric='sqeuclidean'))
100 loops: 0.276 +- 0.0014: cdist(arr, arr, metric='sqeuclidean')
100 loops: 0.275 +- 0.0011: num.distsq(arr, arr)

distsq OpenMP (2 cores), only compare fastest versions

2000 loops: 5.527 +- 0.0159: cdist(arr, arr, metric='sqeuclidean')
2000 loops: 2.963 +- 0.0402: num.distsq(arr, arr)
"""


import timeit

from scipy.spatial.distance import squareform, pdist, cdist
import numpy as np

from pwtools import num


arr = np.random.rand(1000, 3)
globs = globals()

statements = [
    "((arr[:,None,:] - arr[None,...])**2.0).sum(-1)",
    "squareform(pdist(arr, metric='sqeuclidean'))",
    "cdist(arr, arr, metric='sqeuclidean')",
    "num.distsq(arr, arr)",
]

ref_stmt = statements[0]
ref = eval(ref_stmt)
for stmt in statements[1:]:
    diff = np.abs(ref - eval(stmt)).max()
    assert diff < 1e-15, f"ref={ref_stmt} stmt={stmt} diff={diff}"

for stmt in statements:
    number = 100
    times = np.array(
        timeit.repeat(stmt, globals=globs, number=number, repeat=5)
    )
    print(f"{number} loops: {times.mean():.3f} +- {times.std():.4f}: {stmt}")
