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

#include "octnet/cpu/activations.h"
#include "octnet/cpu/cpu.h"

#if defined(_OPENMP)
#include <omp.h>
#endif

extern "C"
void octree_relu_cpu(const octree* grid_in, bool inplace, octree* grid_out) {
  if(!inplace) {
    octree_resize_as_cpu(grid_in, grid_out);
    octree_cpy_scalars(grid_in, grid_out);
    octree_cpy_trees_cpu_cpu(grid_in, grid_out);
    octree_cpy_prefix_leafs_cpu_cpu(grid_in, grid_out);
  }

  ot_size_t feature_size = grid_in->feature_size;
  #pragma omp parallel for
  for(int vx_idx = 0; vx_idx < grid_in->n_leafs; ++vx_idx) {
    for(int f = 0; f < feature_size; ++f) {
      ot_data_t in_val = grid_in->data[vx_idx * feature_size + f];
      grid_out->data[vx_idx * feature_size + f] = in_val <= 0 ? 0 : in_val;
    }
  }
}


extern "C"
void octree_relu_bwd_cpu(const octree* grid_in, const octree* grad_out, bool inplace, octree* grad_in) {
  if(!inplace) {
    octree_resize_as_cpu(grad_out, grad_in);
    octree_cpy_scalars(grad_out, grad_in);
    octree_cpy_trees_cpu_cpu(grad_out, grad_in);
    octree_cpy_prefix_leafs_cpu_cpu(grad_out, grad_in);
  }  
  
  ot_size_t feature_size = grid_in->feature_size;
  #pragma omp parallel for
  for(int vx_idx = 0; vx_idx < grad_out->n_leafs; ++vx_idx) {
    for(int f = 0; f < feature_size; ++f) {
      ot_data_t in_val = grid_in->data[vx_idx * feature_size + f];
      ot_data_t grad_val = grad_out->data[vx_idx * feature_size + f];
      grad_in->data[vx_idx * feature_size + f] = in_val <= 0 ? 0 : grad_val;
    }
  }
}



extern "C"
void octree_sigmoid_cpu(const octree* in, bool inplace, octree* out) {
  if(!inplace) {
    octree_resize_as_cpu(in, out);
    octree_cpy_scalars(in, out);
    octree_cpy_trees_cpu_cpu(in, out);
    octree_cpy_prefix_leafs_cpu_cpu(in, out);
  }

  ot_size_t feature_size = in->feature_size;
  #pragma omp parallel for
  for(int vx_idx = 0; vx_idx < in->n_leafs; ++vx_idx) {
    for(int f = 0; f < feature_size; ++f) {
      ot_data_t in_val = in->data[vx_idx * feature_size + f];
      out->data[vx_idx * feature_size + f] = 1. / (1. + expf(-in_val));
    }
  }
}

extern "C"
void octree_sigmoid_bwd_cpu(const octree* in, const octree* out, const octree* grad_out, bool inplace, octree* grad_in) {
  if(!inplace) {
    octree_resize_as_cpu(in, grad_in);
    octree_cpy_scalars(in, grad_in);
    octree_cpy_trees_cpu_cpu(in, grad_in);
    octree_cpy_prefix_leafs_cpu_cpu(in, grad_in);
  }

  ot_size_t feature_size = in->feature_size;
  #pragma omp parallel for
  for(int vx_idx = 0; vx_idx < in->n_leafs; ++vx_idx) {
    for(int f = 0; f < feature_size; ++f) {
      ot_data_t out_val = out->data[vx_idx * feature_size + f];
      ot_data_t grad_val = grad_out->data[vx_idx * feature_size + f];
      grad_in->data[vx_idx * feature_size + f] = grad_val * (1. - out_val) * out_val;
    }
  }
}



extern "C"
void octree_logsoftmax_cpu(const octree* in, octree* out) {
  octree_resize_as_cpu(in, out);
  octree_cpy_scalars(in, out);
  octree_cpy_trees_cpu_cpu(in, out);
  octree_cpy_prefix_leafs_cpu_cpu(in, out);

  ot_size_t feature_size = in->feature_size;
  #pragma omp parallel for
  for(int vx_idx = 0; vx_idx < in->n_leafs; ++vx_idx) {
    ot_data_t max_val = -1e9;
    for(int f = 0; f < feature_size; ++f) {
      ot_data_t val = in->data[vx_idx * feature_size + f];
      max_val = FMAX(max_val, val);
    }

    ot_data_t logsum = 0;
    for(int f = 0; f < feature_size; ++f) {
      ot_data_t val = in->data[vx_idx * feature_size + f];
      logsum += expf(val - max_val);
    }
    logsum = max_val + logf(logsum);

    for(int f = 0; f < feature_size; ++f) {
      ot_data_t val = in->data[vx_idx * feature_size + f];
      out->data[vx_idx * feature_size + f] = val - logsum;
    }
  }
}

extern "C"
void octree_logsoftmax_bwd_cpu(const octree* in, const octree* out, const octree* grad_out, octree* grad_in) {
  octree_resize_as_cpu(in, grad_in);
  octree_cpy_scalars(in, grad_in);
  octree_cpy_trees_cpu_cpu(in, grad_in);
  octree_cpy_prefix_leafs_cpu_cpu(in, grad_in);
  
  ot_size_t feature_size = in->feature_size;
  #pragma omp parallel for
  for(int vx_idx = 0; vx_idx < in->n_leafs; ++vx_idx) {
    ot_data_t sum = 0;
    for(int f = 0; f < feature_size; ++f) {
      sum += grad_out->data[vx_idx * feature_size + f];
    }

    for(int f = 0; f < feature_size; ++f) {
      const int didx = vx_idx * feature_size + f;
      grad_in->data[didx] = grad_out->data[didx] - expf(out->data[didx]) * sum;
    }
  }
}
