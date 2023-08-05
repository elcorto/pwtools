import os

use_jax = False

name = "PWTOOLS_USE_JAX"
if name in os.environ:
    use_jax = bool(int(os.environ[name]))
