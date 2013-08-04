#! /usr/bin/env python

# System imports
from distutils.core import *
from distutils      import sysconfig

# Third-party modules
import numpy

# Obtain the numpy include directory.
numpy_include = numpy.get_include()

_mwumkl = Extension("_mwumkl",
                    ["mwu_main_wrap.cxx",
                     "../mwu_dynamic.cpp"],
                    include_dirs = [numpy_include],
                    )

# mwumkl setup
setup(name        = "mwumkl",
      description = "MWU-MKL Algorithm",
      author      = "John Moeller",
      py_modules  = ["mwumkl"],
      ext_modules = [_mwumkl]
      )
