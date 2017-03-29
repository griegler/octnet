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

#include "octnet/gpu/activations.h"
#include "octnet/gpu/gpu.h"

#include <cstdio>
#include <cstdlib>


__global__ void kernel_relu(ot_data_t* out, int n_data, const ot_data_t* in) {
  CUDA_KERNEL_LOOP(data_idx, n_data) {
    ot_data_t in_val = in[data_idx];
    out[data_idx] = in_val <= 0 ? 0 : in_val;
  }
}

void octree_relu_gpu(const octree* in, bool inplace, octree* out) {
  if(DEBUG) { printf("[DEBUG] octree_relu_gpu\n"); }

  if(!inplace) {
    octree_resize_as_gpu(in, out);
    octree_cpy_scalars(in, out);
    octree_cpy_trees_gpu_gpu(in, out);
    octree_cpy_prefix_leafs_gpu_gpu(in, out);
  }

  int n_data = in->n_leafs * in->feature_size;
  kernel_relu<<<GET_BLOCKS(n_data), CUDA_NUM_THREADS>>>(
      out->data, n_data, in->data
  );
  CUDA_POST_KERNEL_CHECK;
}



__global__ void kernel_relu_bwd(ot_data_t* grad_in, int n_data, const ot_data_t* in, const ot_data_t* grad_out) {
  CUDA_KERNEL_LOOP(data_idx, n_data) {
    ot_data_t in_val = in[data_idx];
    ot_data_t grad_val = grad_out[data_idx];
    grad_in[data_idx] = in_val <= 0 ? 0 : grad_val;
  }
}

void octree_relu_bwd_gpu(const octree* in, const octree* grad_out, bool inplace, octree* grad_in) {
  if(DEBUG) { printf("[DEBUG] octree_relu_bwd_gpu\n"); }

  if(!inplace) {
    octree_resize_as_gpu(grad_out, grad_in);
    octree_cpy_scalars(grad_out, grad_in);
    octree_cpy_trees_gpu_gpu(grad_out, grad_in);
    octree_cpy_prefix_leafs_gpu_gpu(grad_out, grad_in);
  }

  int n_data = in->n_leafs * in->feature_size;
  kernel_relu_bwd<<<GET_BLOCKS(n_data), CUDA_NUM_THREADS>>>(
      grad_in->data, n_data, in->data, grad_out->data
  );
  CUDA_POST_KERNEL_CHECK;
}







__global__ void kernel_sigmoid(ot_data_t* out, int n_data, const ot_data_t* in) {
  CUDA_KERNEL_LOOP(data_idx, n_data) {
    ot_data_t in_val = in[data_idx];
    out[data_idx] = 1. / (1. + expf(-in_val));
  }
}

void octree_sigmoid_gpu(const octree* in, bool inplace, octree* out) {
  if(DEBUG) { printf("[DEBUG] octree_sigmoid_gpu\n"); }

  if(!inplace) {
    octree_resize_as_gpu(in, out);
    octree_cpy_scalars(in, out);
    octree_cpy_trees_gpu_gpu(in, out);
    octree_cpy_prefix_leafs_gpu_gpu(in, out);
  }

  int n_data = in->n_leafs * in->feature_size;
  kernel_sigmoid<<<GET_BLOCKS(n_data), CUDA_NUM_THREADS>>>(
      out->data, n_data, in->data
  );
  CUDA_POST_KERNEL_CHECK;
}



__global__ void kernel_sigmoid_bwd(ot_data_t* grad_in, int n_data, const ot_data_t* in, const ot_data_t* out, const ot_data_t* grad_out) {
  CUDA_KERNEL_LOOP(data_idx, n_data) {
    ot_data_t out_val = out[data_idx];
    ot_data_t grad_val = grad_out[data_idx];
    grad_in[data_idx] = grad_val * (1. - out_val) * out_val;
  }
}

void octree_sigmoid_bwd_gpu(const octree* in, const octree* out, const octree* grad_out, bool inplace, octree* grad_in) {
  if(DEBUG) { printf("[DEBUG] octree_sigmoid_bwd_gpu\n"); }

  if(!inplace) {
    octree_resize_as_gpu(grad_out, grad_in);
    octree_cpy_scalars(grad_out, grad_in);
    octree_cpy_trees_gpu_gpu(grad_out, grad_in);
    octree_cpy_prefix_leafs_gpu_gpu(grad_out, grad_in);
  }

  int n_data = in->n_leafs * in->feature_size;
  kernel_sigmoid_bwd<<<GET_BLOCKS(n_data), CUDA_NUM_THREADS>>>(
      grad_in->data, n_data, in->data, out->data, grad_out->data
  );
  CUDA_POST_KERNEL_CHECK;
}





__global__ void kernel_logsoftmax(ot_data_t* out, int n_leafs, const ot_data_t* in, int feature_size) {
  CUDA_KERNEL_LOOP(vx_idx, n_leafs) {
    ot_data_t max_val = -1e9;
    for(int f = 0; f < feature_size; ++f) {
      ot_data_t val = in[vx_idx * feature_size + f];
      max_val = FMAX(max_val, val);
    }

    ot_data_t logsum = 0;
    for(int f = 0; f < feature_size; ++f) {
      ot_data_t val = in[vx_idx * feature_size + f];
      logsum += expf(val - max_val);
    }
    logsum = max_val + logf(logsum);

    for(int f = 0; f < feature_size; ++f) {
      ot_data_t val = in[vx_idx * feature_size + f];
      out[vx_idx * feature_size + f] = val - logsum;
    }
  }
}

extern "C"
void octree_logsoftmax_gpu(const octree* in, octree* out) {
  octree_resize_as_gpu(in, out);
  octree_cpy_scalars(in, out);
  octree_cpy_trees_gpu_gpu(in, out);
  octree_cpy_prefix_leafs_gpu_gpu(in, out);

  kernel_logsoftmax<<<GET_BLOCKS(in->n_leafs), CUDA_NUM_THREADS>>>(
      out->data, in->n_leafs, in->data, in->feature_size
  );
  CUDA_POST_KERNEL_CHECK;
}


__global__ void kernel_logsoftmax_bwd(ot_data_t* grad_in, int n_leafs, const ot_data_t* out, const ot_data_t* grad_out, int feature_size) {
  CUDA_KERNEL_LOOP(vx_idx, n_leafs) {
    ot_data_t sum = 0;
    for(int f = 0; f < feature_size; ++f) {
      sum += grad_out[vx_idx * feature_size + f];
    }

    for(int f = 0; f < feature_size; ++f) {
      const int didx = vx_idx * feature_size + f;
      grad_in[didx] = grad_out[didx] - expf(out[didx]) * sum;
    }
  }
}

extern "C"
void octree_logsoftmax_bwd_gpu(const octree* in, const octree* out, const octree* grad_out, octree* grad_in) {
  octree_resize_as_gpu(in, grad_in);
  octree_cpy_scalars(in, grad_in);
  octree_cpy_trees_gpu_gpu(in, grad_in);
  octree_cpy_prefix_leafs_gpu_gpu(in, grad_in);

  kernel_logsoftmax_bwd<<<GET_BLOCKS(in->n_leafs), CUDA_NUM_THREADS>>>(
      grad_in->data, in->n_leafs, out->data, grad_out->data, in->feature_size
  );
  CUDA_POST_KERNEL_CHECK;
}
