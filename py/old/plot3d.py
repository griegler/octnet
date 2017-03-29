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

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def axis_equal_3d(ax):
  extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
  sz = extents[:,1] - extents[:,0]
  centers = np.mean(extents, axis=1)
  maxsize = max(abs(sz))
  r = maxsize/2
  for ctr, dim in zip(centers, 'xyz'):
    getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)


def plot_box(ax, c, w, style='k--'):
  x1, x2 = c[0]-w/2., c[0]+w/2.
  y1, y2 = c[1]-w/2., c[1]+w/2.
  z1, z2 = c[2]-w/2., c[2]+w/2.
  ax.plot((x1,x2), (y1,y1), (z1,z1), style)
  ax.plot((x1,x1), (y1,y2), (z1,z1), style)
  ax.plot((x1,x1), (y1,y1), (z1,z2), style)
  ax.plot((x2,x1), (y2,y2), (z2,z2), style)
  ax.plot((x2,x2), (y2,y1), (z2,z2), style)
  ax.plot((x2,x2), (y2,y2), (z2,z1), style)
  ax.plot((x2,x1), (y1,y1), (z2,z2), style)
  ax.plot((x2,x2), (y1,y1), (z2,z1), style)
  ax.plot((x2,x2), (y1,y2), (z1,z1), style)
  ax.plot((x1,x1), (y1,y2), (z2,z2), style)
  ax.plot((x1,x1), (y2,y2), (z1,z2), style)
  ax.plot((x1,x2), (y2,y2), (z1,z1), style)


def plot_octree_structure(ax, grid, grid_depth, grid_height, grid_width, GD=8):
  def plot_block(block, x, y, z, depth):
    # print('plot_block %s | d=%d | %s' % ((x, y, z), depth, type(block)))
    width = GD / 2**depth
    width2 = width / 2.
    if type(block) == list:
      child_idx = 0
      for d in [0, 1]:
        for h in [0, 1]:
          for w in [0, 1]:
            plot_block(block[child_idx], x+w*width2, y+h*width2, z+d*width2, depth+1)
            child_idx += 1
    else:
      # print('  plot box at %s with width %f (depth=%f, %f)' % ((x,y,z), GD / 2**depth, depth, 2**depth))
      style = 'k--' if block == 0 else 'r-'
      plot_box(ax, (x+width2, y+width2, z+width2), width, style)

  grid_idx = 0
  for d in range(grid_depth):
    for h in range(grid_height):
      for w in range(grid_width):
        block = grid[grid_idx]
        grid_idx += 1
        plot_block(block, w*GD, h*GD, d*GD, 0)
