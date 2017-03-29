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

#include "octnet/gpu/conv.h"
#include "octnet/gpu/gpu.h"
#include "octnet/gpu/oc2col.h"
#include "octnet/gpu/col2oc.h"
#include "octnet/gpu/buffer.h"

#include <thrust/fill.h>
#include <thrust/execution_policy.h>


void print_matrix_gpu(const ot_data_t* data_d, int rows, int cols) {
  ot_data_t* data_h = device_to_host_malloc(data_d, rows*cols);
  int idx = 0;
  printf("[");
  for(int row = 0; row < rows; ++row) {
    if(row > 0) printf(" ");
    printf("[ ");
    for(int col = 0; col < cols; ++col) {
      printf("%f", data_h[idx]);
      idx++;
      if(col < cols-1) {
        printf(", ");
      }
    }
    if(row < rows - 1) {
      printf(" ], \n");
    }
    else {
      printf(" ]] \n");
    }
  }
  delete[] data_h;
}


__global__ void kernel_conv_mm_add_bias(ot_data_t* out, int n_leafs, int channels_out, const ot_data_t* bias) {
  CUDA_KERNEL_LOOP(leaf_idx, n_leafs) {
    for(int f = 0; f < channels_out; ++f) {
      out[leaf_idx * channels_out + f] += bias[f];
    }
  }
}

void octree_conv_mm_block_gpu(cublasHandle_t cublas_handle, const octree* in, const ot_data_t* weights, int leafs_offset, int n_leafs, octree* out) {
  ot_data_t_buffer_gpu& col_buffer = ot_data_t_buffer_gpu::i();
  col_buffer.resize(long(n_leafs) * K333 * in->feature_size);

  oc2col_gpu(in, col_buffer.data(), col_buffer.capacity(), leafs_offset, n_leafs);

  float alpha = 1;
  float beta = 0;
  int m = out->feature_size;
  int n = n_leafs;
  int k = in->feature_size * K333;
  CUBLAS_CHECK(
    cublasSgemm(
      cublas_handle,
      CUBLAS_OP_T, CUBLAS_OP_N,
      m, n, k,
      &alpha,
      weights, k,
      col_buffer.data(), k,
      &beta,
      out->data + leafs_offset * out->feature_size, m
    )
  );
  
  // printf("weights=\n");
  // print_matrix_gpu(weights, out->feature_size, K333*in->feature_size);
  // printf("out=\n");
  // print_matrix_gpu(out->data, out->n_leafs, out->feature_size);
  // printf("col_buffer=\n");
  // print_matrix_gpu(col_buffer.data(), n_leafs, K333*in->feature_size);
}

void octree_conv_mm_gpu(cublasHandle_t cublas_handle, const octree* in, const ot_data_t* weights, const ot_data_t* bias, int channels_out, int n_grids, octree* out) {
  if(DEBUG) { printf("[DEBUG] octree_conv_mm_gpu\n"); }

  octree_resize_gpu(in->n, in->grid_depth, in->grid_height, in->grid_width, channels_out, in->n_leafs, out);
  octree_cpy_scalars(in, out);
  out->feature_size = channels_out;
  octree_cpy_trees_gpu_gpu(in, out);
  octree_cpy_prefix_leafs_gpu_gpu(in, out);

  if(n_grids < 0) {
    int leafs_offset = 0;
    int n_leafs = in->n_leafs;
    octree_conv_mm_block_gpu(cublas_handle, in, weights, leafs_offset, n_leafs, out);
  }
  else {
    if(n_grids == 0) {
      n_grids = in->grid_depth * in->grid_height * in->grid_width;
    }
    int n_blocks = octree_num_blocks(in);
    ot_size_t* prefix_leafs_cpu = device_to_host_malloc<ot_size_t>(in->prefix_leafs, n_blocks);
    int grid_idx = 0;
    while(grid_idx < n_blocks) {
      int leafs_offset = prefix_leafs_cpu[grid_idx]; 
      int n_leafs;
      if(grid_idx + n_grids < n_blocks) {
        n_leafs = prefix_leafs_cpu[grid_idx + n_grids] -  prefix_leafs_cpu[grid_idx];
      }
      else {
        n_leafs = in->n_leafs -  prefix_leafs_cpu[grid_idx];
      }
      octree_conv_mm_block_gpu(cublas_handle, in, weights, leafs_offset, n_leafs, out);
      grid_idx += n_grids;
    }
    delete[] prefix_leafs_cpu;
  }

  // add bias
  kernel_conv_mm_add_bias<<<GET_BLOCKS(out->n_leafs), CUDA_NUM_THREADS>>>(
     out->data, out->n_leafs, out->feature_size, bias
  );
  CUDA_POST_KERNEL_CHECK;
}

void octree_conv_mm_bwd_block_gpu(cublasHandle_t cublas_handle, const octree* grad_out, const ot_data_t* weights, int leafs_offset, int n_leafs, bool atomic, octree* grad_in) {
  ot_data_t_buffer_gpu& col_buffer = ot_data_t_buffer_gpu::i();
  col_buffer.resize(long(n_leafs) * K333 * grad_in->feature_size);

  float alpha = 1;
  float beta = 0;
  int m = grad_in->feature_size * K333;
  int n = n_leafs;
  int k = grad_out->feature_size;
  CUBLAS_CHECK(
    cublasSgemm(
      cublas_handle,
      CUBLAS_OP_N, CUBLAS_OP_N,
      m, n, k,
      &alpha,
      weights, m,
      grad_out->data + leafs_offset * grad_out->feature_size, k,
      &beta,
      col_buffer.data(), m
    )
  );

  if(atomic) {
    col2oc_atomic_gpu(col_buffer.data(), grad_in, leafs_offset, n_leafs);
  }
  else {
    col2oc_gpu(col_buffer.data(), grad_in, leafs_offset, n_leafs);
  }
  

  // printf("weights=\n");
  // print_matrix_gpu(weights, grad_out->feature_size, K333*grad_in->feature_size);
  // printf("grad_out=\n");
  // print_matrix_gpu(grad_out->data, grad_in->n_leafs, grad_out->feature_size);
  // printf("col_buffer=\n");
  // print_matrix_gpu(col_buffer.data(), n_leafs, K333*grad_in->feature_size);
}

void octree_conv_mm_bwd_gpu(cublasHandle_t cublas_handle, const octree* grad_out, const ot_data_t* weights, int channels_in, int n_grids, octree* grad_in) {
  if(DEBUG) { printf("[DEBUG] octree_conv_mm_bwd_gpu\n"); }

  octree_resize_gpu(grad_out->n, grad_out->grid_depth, grad_out->grid_height, grad_out->grid_width, channels_in, grad_out->n_leafs, grad_in);
  octree_cpy_scalars(grad_out, grad_in);
  grad_in->feature_size = channels_in;
  octree_cpy_trees_gpu_gpu(grad_out, grad_in);
  octree_cpy_prefix_leafs_gpu_gpu(grad_out, grad_in);

  bool atomic = false;
  if(n_grids < 0) {
    int leafs_offset = 0;
    int n_leafs = grad_out->n_leafs;
    octree_conv_mm_bwd_block_gpu(cublas_handle, grad_out, weights, leafs_offset, n_leafs, atomic, grad_in);
  }
  else {
    if(n_grids == 0) {
      n_grids = grad_out->grid_depth * grad_out->grid_height * grad_out->grid_width;
    }
    else {
      atomic = true;
      octree_fill_data_gpu(grad_in, 0);
    }
    int n_blocks = octree_num_blocks(grad_out);
    ot_size_t* prefix_leafs_cpu = device_to_host_malloc<ot_size_t>(grad_out->prefix_leafs, n_blocks);
    int grid_idx = 0;
    while(grid_idx < n_blocks) {
      int leafs_offset = prefix_leafs_cpu[grid_idx]; 
      int n_leafs;
      if(grid_idx + n_grids < n_blocks) {
        n_leafs = prefix_leafs_cpu[grid_idx + n_grids] -  prefix_leafs_cpu[grid_idx];
      }
      else {
        n_leafs = grad_out->n_leafs -  prefix_leafs_cpu[grid_idx];
      }
      octree_conv_mm_bwd_block_gpu(cublas_handle, grad_out, weights, leafs_offset, n_leafs, atomic, grad_in);
      grid_idx += n_grids;
    }
    delete[] prefix_leafs_cpu;
  }
  
  // printf("grad_in=\n");
  // print_matrix_gpu(grad_in->data, grad_in->n_leafs, grad_in->feature_size);
}



void octree_conv_mm_wbwd_block_gpu(cublasHandle_t cublas_handle, const octree* in, const octree* grad_out, const float scale, int leafs_offset, int n_leafs, ot_data_t* grad_weights, ot_data_t* grad_bias) {

  ot_data_t_buffer_gpu& col_buffer = ot_data_t_buffer_gpu::i();
  col_buffer.resize(long(n_leafs) * K333 * in->feature_size);

  oc2col_gpu(in, col_buffer.data(), col_buffer.capacity(), leafs_offset, n_leafs);

  float alpha = scale;
  float beta = 1;
  int m = in->feature_size * K333;
  int n = grad_out->feature_size;
  int k = n_leafs;
  CUBLAS_CHECK(
    cublasSgemm(
      cublas_handle,
      CUBLAS_OP_N, CUBLAS_OP_T,
      m, n, k,
      &alpha,
      col_buffer.data(), m,
      grad_out->data + leafs_offset * grad_out->feature_size, n,
      &beta,
      grad_weights, m
    )
  );

  thrust::fill(thrust::device, col_buffer.data(), col_buffer.data() + n_leafs, 1.f);

  alpha = scale;
  beta = 1;
  m = grad_out->feature_size;
  n = n_leafs;
  CUBLAS_CHECK(
    cublasSgemv(
      cublas_handle,
      CUBLAS_OP_N,
      m, n,
      &alpha,
      grad_out->data + leafs_offset * grad_out->feature_size, m,
      col_buffer.data(), 1,
      &beta,
      grad_bias, 1
    )
  );
}

void octree_conv_mm_wbwd_gpu(cublasHandle_t cublas_handle, const octree* in, const octree* grad_out, const float scale, int n_grids, ot_data_t* grad_weights, ot_data_t* grad_bias) {
  if(DEBUG) { printf("[DEBUG] octree_conv_mm_wbwd_gpu\n"); }

  if(n_grids < 0) {
    int leafs_offset = 0;
    int n_leafs = in->n_leafs;
    octree_conv_mm_wbwd_block_gpu(cublas_handle, in, grad_out, scale, leafs_offset, n_leafs, grad_weights, grad_bias);
  }
  else {
    if(n_grids == 0) {
      n_grids = in->grid_depth * in->grid_height * in->grid_width;
    }
    int n_blocks = octree_num_blocks(in);
    ot_size_t* prefix_leafs_cpu = device_to_host_malloc<ot_size_t>(in->prefix_leafs, n_blocks);
    int grid_idx = 0;
    while(grid_idx < n_blocks) {
      int leafs_offset = prefix_leafs_cpu[grid_idx]; 
      int n_leafs;
      if(grid_idx + n_grids < n_blocks) {
        n_leafs = prefix_leafs_cpu[grid_idx + n_grids] -  prefix_leafs_cpu[grid_idx];
      }
      else {
        n_leafs = in->n_leafs -  prefix_leafs_cpu[grid_idx];
      }
      octree_conv_mm_wbwd_block_gpu(cublas_handle, in, grad_out, scale, leafs_offset, n_leafs, grad_weights, grad_bias);
      grid_idx += n_grids;
    }
    delete[] prefix_leafs_cpu;
  }
}
