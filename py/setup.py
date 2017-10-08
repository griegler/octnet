# Copyright (c) 2017, The OctNet authors
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the <organization> nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL OCTNET AUTHORS BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np
import platform

extra_compile_args = ['-msse', '-msse2', '-msse3', '-msse4.2']
extra_link_args = []
if 'Linux' in platform.system():
  print('Added OpenMP')
  extra_compile_args.append('-fopenmp')
  extra_link_args.append('-fopenmp')

setup(
  name="pyoctnet",
  cmdclass= {'build_ext': build_ext},
  ext_modules=[
    Extension('pyoctnet',
      ['pyoctnet.pyx',
       '../core/src/core.cpp',
       '../core/src/d2o.cpp',
       '../core/src/o2d.cpp',
       '../core/src/io.cpp',
       '../core/src/misc.cpp',
       '../core/src/gridpool.cpp',
       '../core/src/gridunpool.cpp',
       '../core/src/conv.cpp',
       '../core/src/combine.cpp',
       '../core/src/split.cpp',
       '../create/src/create.cpp',
       '../create/src/create_dense.cpp',
       '../create/src/create_mesh.cpp',
       '../create/src/create_obj.cpp',
       '../create/src/create_off.cpp',
       '../create/src/create_pc.cpp',
       '../create/src/utils.cpp',
        ],
      language='c++',
      include_dirs=[np.get_include(),
        '../core/include/',
        '../create/include/',
        '../geometry/include/',
        ],
      extra_compile_args=extra_compile_args,
      extra_link_args=extra_link_args
    )
  ]
)


