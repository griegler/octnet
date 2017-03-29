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

#include "octnet/gpu/dense.h"
#include "octnet/gpu/gpu.h"


template <int dense_format, int reduce_fcn>
__global__ void kernel_dense_to_octree(octree out, int n_leafs, const int dense_depth, const int dense_height, const int dense_width, const octree in, const ot_data_t* dense) {

  const int vx_depth_off = (dense_depth - in.grid_depth * 8) / 2;
  const int vx_height_off = (dense_height - in.grid_height * 8) / 2;
  const int vx_width_off = (dense_width - in.grid_width * 8) / 2;

  CUDA_KERNEL_LOOP(leaf_idx, n_leafs) {
    const int grid_idx = out.data[leaf_idx * out.feature_size];
    const ot_tree_t* tree = octree_get_tree(&in, grid_idx);

    // const int cum_n_leafs = n_leafs_upto(&in, grid_idx);
    const int cum_n_leafs = in.prefix_leafs[grid_idx];
    const int data_idx = leaf_idx - cum_n_leafs;
    const int bit_idx = data_idx_to_bit_idx(tree, data_idx);

    int n,d,h,w;
    const int depth = octree_ind_to_dense_ind(&in, grid_idx, bit_idx, &n, &d,&h,&w);
    d += vx_depth_off;
    h += vx_height_off;
    w += vx_width_off;
    const int size = width_from_depth(depth);

    // ot_data_t* out_data = out.data_ptrs[grid_idx] + data_idx * out.feature_size;
    ot_data_t* out_data = octree_get_data(&out, grid_idx) + data_idx * out.feature_size;

    if(depth < 3) {
      dense_to_octree_fcn<reduce_fcn, dense_format>(dense, n,dense_depth,dense_height,dense_width, out.feature_size, d,d+size, h,h+size, w,w+size, out_data);  
    }
    else {
      if(dense_format == DENSE_FORMAT_DHWC) {
        for(int f = 0; f < out.feature_size; ++f) {
          out_data[f] = dense[(((n * dense_depth + d) * dense_height + h) * dense_width + w) * out.feature_size + f];
        }
      }
      else if(dense_format == DENSE_FORMAT_CDHW) {
        for(int f = 0; f < out.feature_size; ++f) {
          out_data[f] = dense[(((n * out.feature_size + f) * dense_depth + d) * dense_height + h) * dense_width + w];
        }
      }
    }
  }
}




void dhwc_to_octree_sum_gpu(const octree* grid_d_in, const int dense_depth, const int dense_height, const int dense_width, const ot_data_t* data, int out_feature_size, octree* grid_d) {
  if(DEBUG) { printf("[DEBUG] dhwc_to_octree_sum_gpu\n"); }

  int n_blocks = octree_num_blocks(grid_d_in);
 
  octree_resize_gpu(grid_d_in->n, grid_d_in->grid_depth, grid_d_in->grid_height, grid_d_in->grid_width, out_feature_size, grid_d_in->n_leafs, grid_d);
  octree_cpy_scalars(grid_d_in, grid_d);
  grid_d->feature_size = out_feature_size;
  octree_cpy_trees_gpu_gpu(grid_d_in, grid_d);
  octree_cpy_prefix_leafs_gpu_gpu(grid_d_in, grid_d);
  
  octree_leaf_idx_to_grid_idx_gpu(grid_d, grid_d->feature_size, grid_d->data_capacity, grid_d->data);
  kernel_dense_to_octree<DENSE_FORMAT_DHWC, REDUCE_SUM><<<GET_BLOCKS(grid_d->n_leafs), CUDA_NUM_THREADS>>>(
      *grid_d, grid_d->n_leafs, dense_depth, dense_height, dense_width, *grid_d_in, data
  );
  CUDA_POST_KERNEL_CHECK;
}
void cdhw_to_octree_sum_gpu(const octree* grid_d_in, const int dense_depth, const int dense_height, const int dense_width, const ot_data_t* data, int out_feature_size, octree* grid_d) {
  if(DEBUG) { printf("[DEBUG] cdhw_to_octree_sum_gpu\n"); }

  int n_blocks = octree_num_blocks(grid_d_in);
 
  octree_resize_gpu(grid_d_in->n, grid_d_in->grid_depth, grid_d_in->grid_height, grid_d_in->grid_width, out_feature_size, grid_d_in->n_leafs, grid_d);
  octree_cpy_scalars(grid_d_in, grid_d);
  grid_d->feature_size = out_feature_size;
  octree_cpy_trees_gpu_gpu(grid_d_in, grid_d);
  octree_cpy_prefix_leafs_gpu_gpu(grid_d_in, grid_d);
  
  octree_leaf_idx_to_grid_idx_gpu(grid_d, grid_d->feature_size, grid_d->data_capacity, grid_d->data);
  kernel_dense_to_octree<DENSE_FORMAT_CDHW, REDUCE_SUM><<<GET_BLOCKS(grid_d->n_leafs), CUDA_NUM_THREADS>>>(
      *grid_d, grid_d->n_leafs, dense_depth, dense_height, dense_width, *grid_d_in, data
  );
  CUDA_POST_KERNEL_CHECK;
}



void dhwc_to_octree_avg_gpu(const octree* grid_d_in, const int dense_depth, const int dense_height, const int dense_width, const ot_data_t* data, int out_feature_size, octree* grid_d) {
  if(DEBUG) { printf("[DEBUG] dhwc_to_octree_avg_gpu\n"); }
  int n_blocks = octree_num_blocks(grid_d_in);

  octree_resize_gpu(grid_d_in->n, grid_d_in->grid_depth, grid_d_in->grid_height, grid_d_in->grid_width, out_feature_size, grid_d_in->n_leafs, grid_d);
  octree_cpy_scalars(grid_d_in, grid_d);
  grid_d->feature_size = out_feature_size;
  octree_cpy_trees_gpu_gpu(grid_d_in, grid_d);
  octree_cpy_prefix_leafs_gpu_gpu(grid_d_in, grid_d);
  
  octree_leaf_idx_to_grid_idx_gpu(grid_d, grid_d->feature_size, grid_d->data_capacity, grid_d->data);
  kernel_dense_to_octree<DENSE_FORMAT_DHWC, REDUCE_AVG><<<GET_BLOCKS(grid_d->n_leafs), CUDA_NUM_THREADS>>>(
      *grid_d, grid_d->n_leafs, dense_depth, dense_height, dense_width, *grid_d_in, data
  );
  CUDA_POST_KERNEL_CHECK;
}
void cdhw_to_octree_avg_gpu(const octree* grid_d_in, const int dense_depth, const int dense_height, const int dense_width, const ot_data_t* data, int out_feature_size, octree* grid_d) {
  if(DEBUG) { printf("[DEBUG] cdhw_to_octree_avg_gpu\n"); }
  int n_blocks = octree_num_blocks(grid_d_in);

  octree_resize_gpu(grid_d_in->n, grid_d_in->grid_depth, grid_d_in->grid_height, grid_d_in->grid_width, out_feature_size, grid_d_in->n_leafs, grid_d);
  octree_cpy_scalars(grid_d_in, grid_d);
  grid_d->feature_size = out_feature_size;
  octree_cpy_trees_gpu_gpu(grid_d_in, grid_d);
  octree_cpy_prefix_leafs_gpu_gpu(grid_d_in, grid_d);
  
  octree_leaf_idx_to_grid_idx_gpu(grid_d, grid_d->feature_size, grid_d->data_capacity, grid_d->data);
  kernel_dense_to_octree<DENSE_FORMAT_CDHW, REDUCE_AVG><<<GET_BLOCKS(grid_d->n_leafs), CUDA_NUM_THREADS>>>(
      *grid_d, grid_d->n_leafs, dense_depth, dense_height, dense_width, *grid_d_in, data
  );
  CUDA_POST_KERNEL_CHECK;
}



void dhwc_to_octree_max_gpu(const octree* grid_d_in, const int dense_depth, const int dense_height, const int dense_width, const ot_data_t* data, int out_feature_size, octree* grid_d) {
  if(DEBUG) { printf("[DEBUG] dhwc_to_octree_max_gpu\n"); }
  int n_blocks = octree_num_blocks(grid_d_in);

  octree_resize_gpu(grid_d_in->n, grid_d_in->grid_depth, grid_d_in->grid_height, grid_d_in->grid_width, out_feature_size, grid_d_in->n_leafs, grid_d);
  octree_cpy_scalars(grid_d_in, grid_d);
  grid_d->feature_size = out_feature_size;
  octree_cpy_trees_gpu_gpu(grid_d_in, grid_d);
  octree_cpy_prefix_leafs_gpu_gpu(grid_d_in, grid_d);
  
  kernel_dense_to_octree<DENSE_FORMAT_DHWC, REDUCE_MAX><<<GET_BLOCKS(grid_d->n_leafs), CUDA_NUM_THREADS>>>(
      *grid_d, grid_d->n_leafs, dense_depth, dense_height, dense_width, *grid_d_in, data
  );
  CUDA_POST_KERNEL_CHECK;
}
void cdhw_to_octree_max_gpu(const octree* grid_d_in, const int dense_depth, const int dense_height, const int dense_width, const ot_data_t* data, int out_feature_size, octree* grid_d) {
  if(DEBUG) { printf("[DEBUG] cdhw_to_octree_max_gpu\n"); }
  int n_blocks = octree_num_blocks(grid_d_in);

  octree_resize_gpu(grid_d_in->n, grid_d_in->grid_depth, grid_d_in->grid_height, grid_d_in->grid_width, out_feature_size, grid_d_in->n_leafs, grid_d);
  octree_cpy_scalars(grid_d_in, grid_d);
  grid_d->feature_size = out_feature_size;
  octree_cpy_trees_gpu_gpu(grid_d_in, grid_d);
  octree_cpy_prefix_leafs_gpu_gpu(grid_d_in, grid_d);
  
  kernel_dense_to_octree<DENSE_FORMAT_CDHW, REDUCE_MAX><<<GET_BLOCKS(grid_d->n_leafs), CUDA_NUM_THREADS>>>(
      *grid_d, grid_d->n_leafs, dense_depth, dense_height, dense_width, *grid_d_in, data
  );
  CUDA_POST_KERNEL_CHECK;
}



void dhwc_to_octree_sum_bwd_gpu(const octree* grad_out_grid_d, const int dense_depth, const int dense_height, const int dense_width, ot_data_t* grad_in_data) {
  if(DEBUG) { printf("[DEBUG] dhwc_to_octree_sum_bwd_gpu\n"); }
  int n_voxels = grad_out_grid_d->n * dense_depth * dense_height * dense_width;
  kernel_octree_to_dense<DENSE_FORMAT_DHWC, false><<<GET_BLOCKS(n_voxels), CUDA_NUM_THREADS>>>(
      grad_in_data, n_voxels, dense_depth, dense_height, dense_width, *grad_out_grid_d
  );
  CUDA_POST_KERNEL_CHECK;
}
void cdhw_to_octree_sum_bwd_gpu(const octree* grad_out_grid_d, const int dense_depth, const int dense_height, const int dense_width, ot_data_t* grad_in_data) {
  if(DEBUG) { printf("[DEBUG] cdhw_to_octree_sum_bwd_gpu\n"); }
  int n_voxels = grad_out_grid_d->n * dense_depth * dense_height * dense_width;
  kernel_octree_to_dense<DENSE_FORMAT_CDHW, false><<<GET_BLOCKS(n_voxels), CUDA_NUM_THREADS>>>(
      grad_in_data, n_voxels, dense_depth, dense_height, dense_width, *grad_out_grid_d
  );
  CUDA_POST_KERNEL_CHECK;
}



void dhwc_to_octree_avg_bwd_gpu(const octree* grad_out_grid_d, const int dense_depth, const int dense_height, const int dense_width, ot_data_t* grad_in_data) {
  if(DEBUG) { printf("[DEBUG] dhwc_to_octree_avg_bwd_gpu\n"); }
  int n_voxels = grad_out_grid_d->n * dense_depth * dense_height * dense_width;
  kernel_octree_to_dense<DENSE_FORMAT_DHWC, true><<<GET_BLOCKS(n_voxels), CUDA_NUM_THREADS>>>(
      grad_in_data, n_voxels, dense_depth, dense_height, dense_width, *grad_out_grid_d
  );
  CUDA_POST_KERNEL_CHECK;
}
void cdhw_to_octree_avg_bwd_gpu(const octree* grad_out_grid_d, const int dense_depth, const int dense_height, const int dense_width, ot_data_t* grad_in_data) {
  if(DEBUG) { printf("[DEBUG] cdhw_to_octree_avg_bwd_gpu\n"); }
  int n_voxels = grad_out_grid_d->n * dense_depth * dense_height * dense_width;
  kernel_octree_to_dense<DENSE_FORMAT_CDHW, true><<<GET_BLOCKS(n_voxels), CUDA_NUM_THREADS>>>(
      grad_in_data, n_voxels, dense_depth, dense_height, dense_width, *grad_out_grid_d
  );
  CUDA_POST_KERNEL_CHECK;
}



