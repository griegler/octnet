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

#include "octnet/gpu/math.h"
#include "octnet/gpu/gpu.h"

#include <iostream>
#include <cstdio>
#include <cstdlib>

#include <thrust/execution_policy.h>
#include <thrust/extrema.h>



__global__ void kernel_add(ot_data_t* out, int n_data, const ot_data_t* in1, ot_data_t fac1, const ot_data_t* in2, ot_data_t fac2) {
  CUDA_KERNEL_LOOP(data_idx, n_data) {
    out[data_idx] = fac1 * in1[data_idx] + fac2 * in2[data_idx];
  }
}


extern "C"
void octree_add_gpu(const octree* in1, ot_data_t fac1, const octree* in2, ot_data_t fac2, bool check, octree* out) {
  if(DEBUG) { 
    printf("[DEBUG] octree_add_gpu\n");
    printCudaMemUsage();
  }

  if(check && (in1->feature_size != in2->feature_size || !octree_equal_trees_gpu(in1, in2))) {
    printf("[ERROR] add - tree structure of inputs does not match\n");
    exit(-1);
  }

  //check if inplace
  if(out != in1 && out != in2) {
    octree_resize_as_gpu(in1, out);
    octree_cpy_scalars(in1, out);
    octree_cpy_trees_gpu_gpu(in1, out);
    octree_cpy_prefix_leafs_gpu_gpu(in1, out);
  }

  int n_data = in1->n_leafs * in1->feature_size;
  kernel_add<<<GET_BLOCKS(n_data), CUDA_NUM_THREADS>>>(
      out->data, n_data, in1->data, fac1, in2->data, fac2
  );
  CUDA_POST_KERNEL_CHECK;
}



__global__ void kernel_scalar_mul(ot_data_t* data, int N, const ot_data_t scalar) {
  CUDA_KERNEL_LOOP(idx, N) {
    data[idx] *= scalar;
  }
}

extern "C"
void octree_scalar_mul_gpu(octree* grid, const ot_data_t scalar) {
  int n = grid->n_leafs * grid->feature_size;
  kernel_scalar_mul<<<GET_BLOCKS(n), CUDA_NUM_THREADS>>>(
      grid->data, n, scalar
  );
  CUDA_POST_KERNEL_CHECK;
}


__global__ void kernel_scalar_add(ot_data_t* data, int N, const ot_data_t scalar) {
  CUDA_KERNEL_LOOP(idx, N) {
    data[idx] += scalar;
  }
}

extern "C"
void octree_scalar_add_gpu(octree* grid, const ot_data_t scalar) {
  int n = grid->n_leafs * grid->feature_size;
  kernel_scalar_add<<<GET_BLOCKS(n), CUDA_NUM_THREADS>>>(
      grid->data, n, scalar
  );
  CUDA_POST_KERNEL_CHECK;
}


__global__ void kernel_sign(ot_data_t* data, int N) {
  CUDA_KERNEL_LOOP(idx, N) {
    float val = data[idx];
    if(val < 0) {
      data[idx] = -1;
    }
    else if(val > 0) {
      data[idx] = 1;
    }
    else {
      data[idx] = 0;
    }
  }
}
extern "C"
void octree_sign_gpu(octree* grid) {
  int n = grid->n_leafs * grid->feature_size;
  kernel_sign<<<GET_BLOCKS(n), CUDA_NUM_THREADS>>>(
      grid->data, n
  );
  CUDA_POST_KERNEL_CHECK;
}


__global__ void kernel_abs(ot_data_t* data, int N) {
  CUDA_KERNEL_LOOP(idx, N) {
    float val = data[idx];
    data[idx] = fabs(val);
  }
}
extern "C"
void octree_abs_gpu(octree* grid) {
  int n = grid->n_leafs * grid->feature_size;
  kernel_abs<<<GET_BLOCKS(n), CUDA_NUM_THREADS>>>(
      grid->data, n
  );
  CUDA_POST_KERNEL_CHECK;
}


__global__ void kernel_log(ot_data_t* data, int N) {
  CUDA_KERNEL_LOOP(idx, N) {
    float val = data[idx];
    data[idx] = log(val);
  }
}
extern "C"
void octree_log_gpu(octree* grid) {
  int n = grid->n_leafs * grid->feature_size;
  kernel_log<<<GET_BLOCKS(n), CUDA_NUM_THREADS>>>(
      grid->data, n
  );
  CUDA_POST_KERNEL_CHECK;
}




extern "C"
ot_data_t octree_min_gpu(const octree* grid_in) {
  int n = grid_in->n_leafs * grid_in->feature_size;
  float* min_d = thrust::min_element(thrust::device, grid_in->data, grid_in->data + n);
  float min_h;
  device_to_host(min_d, &min_h, 1);
  return min_h;
}

extern "C"
ot_data_t octree_max_gpu(const octree* grid_in) {
  int n = grid_in->n_leafs * grid_in->feature_size;
  float* max_d = thrust::max_element(thrust::device, grid_in->data, grid_in->data + n);
  float max_h;
  device_to_host(max_d, &max_h, 1);
  return max_h;
}
