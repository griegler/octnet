// Copyright (c) 2017, The OctNet authors
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//     * Neither the name of the <organization> nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL OCTNET AUTHORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef GEOMETRY_CPU_H
#define GEOMETRY_CPU_H

#include "geometry.h"

void int_array_delete(int* array) {
  delete[] array;
}
void float_array_delete(float* array) {
  delete[] array;
}

// general ray casting method that returns as parameters the positions pos of the 
// voxels that are hit by the viewing rays along with the distances ts.
// return value is the number of voxels hit by the viewing ray
int ray_casting_cpu(float3 vx_offset, float3 vx_width, int depth, int height, int width, float3 C, float3 ray, int** pos, float** ts);

// given that vxs contain probabilities, e.g. from patch compare, compute for 
// each image pixel the depth value of the voxel that maximizes the probability 
// along the viewing ray.
void ray_casting_depth_map_max_cpu(const float* vxs, float3 vx_offset, float3 vx_width, int depth, int height, int width, float3 C, const float* Pi, float* im, int im_height, int im_width);

// given that vxs contain probabilities, e.g. from patch compare, compute for 
// each image pixel the depth value as mean depth weighted by the probability
// in the voxel along the viewing ray.
void ray_casting_depth_map_avg_cpu(const float* vxs, float3 vx_offset, float3 vx_width, int depth, int height, int width, float3 C, const float* Pi, float* im, int im_height, int im_width);



#endif
