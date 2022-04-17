from distutils.core import Extension, setup

module = Extension("_gravity", sources=["gravity.c"])

setup(name="gravity", version="1.0", description="", ext_modules=[module])
