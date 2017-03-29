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

#include <iostream>
#include <vector>
#include <cstdio>

#if defined(_OPENMP)
#include <omp.h>
#endif

#include "geometry_cpu.h"


int ray_casting_cpu(float3 vx_offset, float3 vx_width, int depth, int height, int width, float3 C, float3 ray, int** pos, float** ts) {
  // // test if ray hits volume in general and where
  // float3 vol_c;
  // vol_c.x = vx_offset.x + width/2.f;
  // vol_c.y = vx_offset.y + height/2.f;
  // vol_c.z = vx_offset.z + depth/2.f;
  // float3 vol_w;
  // vol_w.x = width;
  // vol_w.y = height;
  // vol_w.z = depth;

  // float thit;
  // bool intersection = intersection_ray_voxel(C, ray, vol_c, vol_w, thit);
  // if(!intersection) {
  //   return 0;
  // }

  // //get first voxel
  // float3 fst_vx;
  // for(int idx = 0; idx < 3; ++idx) {
  //   fst_vx[idx] = C[idx] + thit * ray[idx];
  // }
  // std::cout << fst_vx.x << ", " << fst_vx.y << ", " << fst_vx.z << std::endl;
  
  // this is the brute-force approach, but hey...
  std::vector<int> ws;
  std::vector<int> hs;
  std::vector<int> ds;
  std::vector<float> thits;
  for(int d = 0; d < depth; ++d) {
    for(int h = 0; h < height; ++h) {
      for(int w = 0; w < width; ++w) {
        float3 vx;
        vx.x = w * vx_width.x + vx_width.x/2.f + vx_offset.x;
        vx.y = h * vx_width.y + vx_width.y/2.f + vx_offset.y;
        vx.z = d * vx_width.z + vx_width.z/2.f + vx_offset.z;

        float thit;
        bool intersection = intersection_ray_voxel(C, ray, vx, vx_width, thit);
        // std::cout << "hit=" << intersection << " at " << vx.x << ", " << vx.y << ", " << vx.z << " | " << vx_width.x << ", " << vx_width.y << ", " << vx_width.z << std::endl;
        if(intersection) {
          ws.push_back(w);
          hs.push_back(h);
          ds.push_back(d);
          thits.push_back(thit);
        }
      }
    }
  }
  
  //copy to parameter arrays
  int n = ws.size();
  if(n > 0) {
    (*pos) = new int[n * 3];
    (*ts) = new float[n];
    for(int idx = 0; idx < n; ++idx) {
      (*pos)[idx * 3 + 0] = ws[idx];
      (*pos)[idx * 3 + 1] = hs[idx];
      (*pos)[idx * 3 + 2] = ds[idx];
      (*ts)[idx] = thits[idx];
    }
    return n;
  }

  return 0;
}


void ray_casting_depth_map_max_cpu(const float* vxs, float3 vx_offset, float3 vx_width, int depth, int height, int width, float3 C, const float* Pi, float* im, int im_height, int im_width) {
  #pragma omp parallel for
  for(int v = 0; v < im_height; ++v) {
    for(int u = 0; u < im_width; ++u) {
      float3 ray;
      ray.x = Pi[0] * u + Pi[1] * v + Pi[2];
      ray.y = Pi[4] * u + Pi[5] * v + Pi[6];
      ray.z = Pi[8] * u + Pi[9] * v + Pi[10];
      float ray_norm = sqrt( ray.x*ray.x + ray.y*ray.y + ray.z*ray.z );
      ray.x /= ray_norm;
      ray.y /= ray_norm;
      ray.z /= ray_norm;

      int* vx_pos = 0;
      float* ts = 0;
      int vx_n = ray_casting_cpu(vx_offset, vx_width, depth, height, width, C, ray, &vx_pos, &ts);

      float max_prob = 0;
      float max_t = -1;
      for(int vx_idx = 0; vx_idx < vx_n; ++vx_idx) {
        int vx_x = vx_pos[vx_idx * 3 + 0];
        int vx_y = vx_pos[vx_idx * 3 + 1];
        int vx_z = vx_pos[vx_idx * 3 + 2];
        float vx_prob = vxs[(vx_z * height + vx_y) * width + vx_x];
        if(vx_prob > max_prob) {
          max_prob = vx_prob;
          max_t = ts[vx_idx];
        }
      }

      im[v * im_width + u] = max_t;

      delete[] vx_pos;
      delete[] ts;
    }
  }
}

void ray_casting_depth_map_avg_cpu(const float* vxs, float3 vx_offset, float3 vx_width, int depth, int height, int width, float3 C, const float* Pi, float* im, int im_height, int im_width) {
  #pragma omp parallel for
  for(int v = 0; v < im_height; ++v) {
    for(int u = 0; u < im_width; ++u) {
      float3 ray;
      ray.x = Pi[0] * u + Pi[1] * v + Pi[2];
      ray.y = Pi[4] * u + Pi[5] * v + Pi[6];
      ray.z = Pi[8] * u + Pi[9] * v + Pi[10];
      float ray_norm = sqrt( ray.x*ray.x + ray.y*ray.y + ray.z*ray.z );
      ray.x /= ray_norm;
      ray.y /= ray_norm;
      ray.z /= ray_norm;

      int* vx_pos = 0;
      float* ts = 0;
      int vx_n = ray_casting_cpu(vx_offset, vx_width, depth, height, width, C, ray, &vx_pos, &ts);

      float prob_sum = 0;
      float t_sum = 0;
      for(int vx_idx = 0; vx_idx < vx_n; ++vx_idx) {
        int vx_x = vx_pos[vx_idx * 3 + 0];
        int vx_y = vx_pos[vx_idx * 3 + 1];
        int vx_z = vx_pos[vx_idx * 3 + 2];
        float vx_prob = vxs[(vx_z * height + vx_y) * width + vx_x];
        if(vx_prob > 0) {
          prob_sum += vx_prob;
          t_sum += ts[vx_idx];
        }
      }

      if(prob_sum > 0) {
        im[v * im_width + u] = t_sum / prob_sum;
      }
      else {
        im[v * im_width + u] = -1;
      }

      delete[] vx_pos;
      delete[] ts;
    }
  }
}
