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

cimport cython
import numpy as np
from libcpp cimport bool

CREATE_INIT = True # workaround, so cython builds a init function

cdef extern from "../include/geometry.h":
  ctypedef struct float3:
    float x;  
    float y;  
    float z;

  bool intersection_ray_voxel(float3 p, float3 d, float3 vx, float3 vx_w, float& tmin);

cdef extern from "../include/geometry_cpu.h":
  void int_array_delete(int* array);
  void float_array_delete(float* array);
  int ray_casting_cpu(float3 vx_offset, float3 vx_width, int depth, int height, int width, float3 C, float3 ray, int** pos, float** t);
  void ray_casting_depth_map_max_cpu(const float* vxs, float3 vx_offset, float3 vx_width, int depth, int height, int width, float3 C, const float* Pi, float* im, int im_height, int im_width);
  void ray_casting_depth_map_avg_cpu(const float* vxs, float3 vx_offset, float3 vx_width, int depth, int height, int width, float3 C, const float* Pi, float* im, int im_height, int im_width);


def inters_ray_voxel(float[::1] C, float[::1] ray, float[::1] vx, float[::1] vx_w):
  cdef float tmin = 0
  cdef float3 p, d, vx_, vx_w_
  p.x, p.y, p.z = C[0], C[1], C[2]
  d.x, d.y, d.z = ray[0], ray[1], ray[2]
  vx_.x, vx_.y, vx_.z = vx[0], vx[1], vx[2]
  vx_w_.x, vx_w_.y, vx_w_.z = vx_w[0], vx_w[1], vx_w[2]
  cdef bool intersect = intersection_ray_voxel(p, d, vx_, vx_w_, tmin)
  return intersect, tmin

def ray_casting(float[:,:,:,::1] vxs, vx_offset, vx_width, C, ray):
  cdef float3 vx_offset_, vx_width_, C_, ray_
  vx_offset_.x, vx_offset_.y, vx_offset_.z = vx_offset[0], vx_offset[1], vx_offset[2]
  vx_width_.x, vx_width_.y, vx_width_.z = vx_width[0], vx_width[1], vx_width[2]
  C_.x, C_.y, C_.z = C[0], C[1], C[2]
  ray_.x, ray_.y, ray_.z = ray[0], ray[1], ray[2]
  cdef int* pos
  cdef float* ts
  cdef n = ray_casting_cpu(vx_offset_, vx_width_, vxs.shape[0], vxs.shape[1], vxs.shape[2], C_, ray_, &pos, &ts)
  
  if n > 0:
    np_pos = np.empty((n, 3), dtype=np.int)
    np_ts = np.empty((n), dtype=np.float32)
    for i in range(n):
      np_pos[i, 0] = pos[i * 3 + 0]
      np_pos[i, 1] = pos[i * 3 + 1]
      np_pos[i, 2] = pos[i * 3 + 2]
      np_ts[i] = ts[i]
    int_array_delete(pos)
    float_array_delete(ts)
    return np_pos, np_ts
  else:
    return None, None

def ray_casting_depth_map_max(float[:,:,::1] vxs, vx_offset, vx_width, float[::1] C, float[:, ::1] Pi, float[:, ::1] im):
  cdef float3 vx_offset_, vx_width_, C_,
  vx_offset_.x, vx_offset_.y, vx_offset_.z = vx_offset[0], vx_offset[1], vx_offset[2]
  vx_width_.x, vx_width_.y, vx_width_.z = vx_width[0], vx_width[1], vx_width[2]
  C_.x, C_.y, C_.z = C[0], C[1], C[2]
  ray_casting_depth_map_max_cpu(&(vxs[0,0,0]), vx_offset_, vx_width_, vxs.shape[0], vxs.shape[1], vxs.shape[2], C_, &(Pi[0,0]), &(im[0,0]), im.shape[0], im.shape[1])

def ray_casting_depth_map_avg(float[:,:,::1] vxs, vx_offset, vx_width, float[::1] C, float[:, ::1] Pi, float[:, ::1] im):
  cdef float3 vx_offset_, vx_width_, C_,
  vx_offset_.x, vx_offset_.y, vx_offset_.z = vx_offset[0], vx_offset[1], vx_offset[2]
  vx_width_.x, vx_width_.y, vx_width_.z = vx_width[0], vx_width[1], vx_width[2]
  C_.x, C_.y, C_.z = C[0], C[1], C[2]
  ray_casting_depth_map_avg_cpu(&(vxs[0,0,0]), vx_offset_, vx_width_, vxs.shape[0], vxs.shape[1], vxs.shape[2], C_, &(Pi[0,0]), &(im[0,0]), im.shape[0], im.shape[1])
  

def volume_to_pts(vol, vol_offset=(0,0,0), vx_width=(1,1,1), thr=0):
  voff_x = vol_offset[0] + vx_width[0]/2.
  voff_y = vol_offset[1] + vx_width[1]/2.
  voff_z = vol_offset[2] + vx_width[2]/2.

  vis_vxs = []
  c = []
  for z in range(vol.shape[0]):
    for y in range(vol.shape[1]):
      for x in range(vol.shape[2]):
        if np.all(vol[z, y, x] > thr):
          vis_vxs.append((x+voff_x, y+voff_y, z+voff_z))
          c.append(vol[z, y, x])
  vis_vxs = np.array(vis_vxs)
  return vis_vxs, c
