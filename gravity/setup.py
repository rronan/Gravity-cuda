from distutils.core import Extension, setup

import numpy as np

module = Extension("_gravity", sources=["gravity.c"], include_dirs=[np.get_include()])

setup(name="gravity", version="1.0", description="", ext_modules=[module])
