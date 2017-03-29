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

#include "octnet/cpu/volumetric_upsampling.h"

#if defined(_OPENMP)
#include <omp.h>
#endif


void volumetric_nn_upsampling_cdhw_cpu(const ot_data_t* in, int n, int in_depth, int in_height, int in_width, int feature_size, int upsampling_factor, ot_data_t* out) {
  int out_depth = upsampling_factor * in_depth;
  int out_height = upsampling_factor * in_height;
  int out_width = upsampling_factor * in_width;

  #pragma omp parallel for
  for(int vx_idx = 0; vx_idx < n * out_depth * out_height * out_width; ++vx_idx) {
    int n = vx_idx / (out_depth * out_height * out_width);
    int ow = vx_idx % out_width;
    int oh = ((vx_idx - ow) / out_width) % out_height;
    int od = (((((vx_idx - ow) / out_width) - oh) / out_height) % out_depth);

    int id = od / upsampling_factor;
    int ih = oh / upsampling_factor;
    int iw = ow / upsampling_factor;
    for(int f = 0; f < feature_size; ++f) {
      int in_idx = (((n * feature_size + f) * in_depth + id) * in_height + ih) * in_width + iw;
      int out_idx = (((n * feature_size + f) * out_depth + od) * out_height + oh) * out_width + ow;
      out[out_idx] = in[in_idx];
    }
  }
}

void volumetric_nn_upsampling_cdhw_bwd_cpu(const ot_data_t* grad_out, int n, int in_depth, int in_height, int in_width, int feature_size, int upsampling_factor, ot_data_t* grad_in) {
  int out_depth = upsampling_factor * in_depth;
  int out_height = upsampling_factor * in_height;
  int out_width = upsampling_factor * in_width;

  #pragma omp parallel for
  for(int vx_idx = 0; vx_idx < n * in_depth * in_height * in_width; ++vx_idx) {
    int n = vx_idx / (in_depth * in_height * in_width);
    int iw = vx_idx % in_width;
    int ih = ((vx_idx - iw) / in_width) % in_height;
    int id = (((((vx_idx - iw) / in_width) - ih) / in_height) % in_depth);

    for(int f = 0; f < feature_size; ++f) {
      int in_idx = (((n * feature_size + f) * in_depth + id) * in_height + ih) * in_width + iw;
      grad_in[in_idx] = 0;
      for(int d = 0; d < upsampling_factor; ++ d) {
        for(int h = 0; h < upsampling_factor; ++ h) {
          for(int w = 0; w < upsampling_factor; ++ w) {
            int od = id * upsampling_factor + d;
            int oh = ih * upsampling_factor + h;
            int ow = iw * upsampling_factor + w;
            int out_idx = (((n * feature_size + f) * out_depth + od) * out_height + oh) * out_width + ow;
            grad_in[in_idx] += grad_out[out_idx];
          }
        }
      }
    }
  }
}
