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

#include "octnet/gpu/gpu.h"
#include "octnet/cpu/cpu.h"

#include <thrust/execution_policy.h>
#include <thrust/transform_reduce.h>
#include <thrust/transform_scan.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>
#include <thrust/equal.h>

//-------------------------------------------------------------------------------
// helper functions
//-------------------------------------------------------------------------------




extern "C"
octree* octree_new_gpu() {
  octree* grid = new octree;
  grid->n = 0;
  grid->grid_depth = 0;
  grid->grid_height = 0;
  grid->grid_width = 0;
  grid->feature_size = 0;
  grid->n_leafs = 0;

  grid->trees = 0;
  grid->prefix_leafs = 0;
  grid->data = 0;

  grid->grid_capacity = 0;
  grid->data_capacity = 0;

  return grid;
}

extern "C"
void octree_free_gpu(octree* grid_d) {
  device_free(grid_d->trees);
  device_free(grid_d->prefix_leafs);
  device_free(grid_d->data);
  delete grid_d;
}


void octree_resize_gpu(int n, int grid_depth, int grid_height, int grid_width, int feature_size, int n_leafs, octree* dst) {
  if(DEBUG) { printf("[DEBUG] octree_resize_gpu\n"); }

  dst->n = n;
  dst->grid_depth = grid_depth;
  dst->grid_height = grid_height;
  dst->grid_width = grid_width;
  dst->feature_size = feature_size;
  dst->n_leafs = n_leafs;

  int grid_capacity = octree_num_blocks(dst);
  if(dst->grid_capacity < grid_capacity) {
    dst->grid_capacity = grid_capacity;

    if(dst->trees != 0) {
      device_free(dst->trees);
    }
    dst->trees = device_malloc<ot_tree_t>(grid_capacity * N_TREE_INTS);

    if(dst->prefix_leafs != 0) {
      device_free(dst->prefix_leafs);
    }
    dst->prefix_leafs = device_malloc<ot_size_t>(grid_capacity);
  }

  int data_capacity = n_leafs * feature_size;
  if(dst->data_capacity < data_capacity) {
    dst->data_capacity = data_capacity;

    if(dst->data != 0) {
      device_free(dst->data);
    }
    dst->data = device_malloc<ot_data_t>(data_capacity);
  }
}


void octree_resize_as_gpu(const octree* src, octree* dst) {
  octree_resize_gpu(src->n, src->grid_depth, src->grid_height, src->grid_width, src->feature_size, src->n_leafs, dst);
}


__global__ void kernel_octree_clr_trees(ot_tree_t* trees, const int n_tree_ints) {
  CUDA_KERNEL_LOOP(idx, n_tree_ints) {
    trees[idx] = 0;
  }
}

extern "C"
void octree_clr_trees_gpu(octree* grid_d) {
  // cudaMemset(grid_d->trees, 0, octree_num_blocks(grid_d) * N_TREE_INTS * sizeof(ot_tree_t));
  int n_tree_ints = octree_num_blocks(grid_d) * N_TREE_INTS;
  kernel_octree_clr_trees<<<GET_BLOCKS(n_tree_ints), CUDA_NUM_THREADS>>>(
    grid_d->trees, n_tree_ints
  );
  CUDA_POST_KERNEL_CHECK; 
}

extern "C"
void octree_fill_data_gpu(octree* grid_d, ot_data_t fill_value) {
  int n = grid_d->feature_size * grid_d->n_leafs;
  thrust::fill_n(thrust::device, grid_d->data, n, fill_value);
}


// template <int grid_idx_offset>
template <typename OUT_TYPE>
struct thrust_tree_num_leafs : public thrust::unary_function<int, OUT_TYPE> {
  const octree grid;
  const OUT_TYPE mul;
  // const int n_blocks;
  thrust_tree_num_leafs(const octree grid_, const OUT_TYPE mul_) : 
      grid(grid_), mul(mul_) {
  // thrust_tree_num_leafs(const octree grid_, const int n_block_) : 
      // grid(grid_), n_blocks(n_block_) {
  }

  __host__ __device__ OUT_TYPE operator()(const int grid_idx) {
    // printf("  ... grid_idx %d -  %ld\n", grid_idx, mul * tree_n_leafs( octree_get_tree(&grid, grid_idx) ));
    return mul * tree_n_leafs( octree_get_tree(&grid, grid_idx) );
  }
};

extern "C"
void octree_upd_n_leafs_gpu(octree* grid_d) {
  int n_blocks = octree_num_blocks(grid_d);
  thrust::counting_iterator<int> grid_idx_iter(0);

  grid_d->n_leafs = thrust::transform_reduce(
      thrust::device,
      grid_idx_iter, grid_idx_iter + n_blocks, 
      thrust_tree_num_leafs<ot_size_t>(*grid_d, 1), 0, thrust::plus<ot_size_t>()
  );
}

extern "C"
void octree_upd_prefix_leafs_gpu(octree* grid_d) {
  int n_blocks = octree_num_blocks(grid_d);
  thrust::counting_iterator<int> grid_idx_iter(0);

  thrust::transform_exclusive_scan(thrust::device,
                                   grid_idx_iter, grid_idx_iter + n_blocks,
                                   grid_d->prefix_leafs,
                                   thrust_tree_num_leafs<ot_size_t>(*grid_d, 1),
                                   0, thrust::plus<ot_size_t>());
}


void octree_cpy_trees_cpu_gpu(const octree* src_h, octree* dst_d) {
  if(DEBUG) { printf("[DEBUG] octree_cpy_trees_cpu_gpu\n"); }
  host_to_device(src_h->trees, dst_d->trees, octree_num_blocks(src_h) * N_TREE_INTS);
}
void octree_cpy_prefix_leafs_cpu_gpu(const octree* src_h, octree* dst_d) {
  if(DEBUG) { printf("[DEBUG] octree_cpy_prefix_leafs_cpu_gpu\n"); }
  host_to_device(src_h->prefix_leafs, dst_d->prefix_leafs, octree_num_blocks(src_h));
}
void octree_cpy_data_cpu_gpu(const octree* src_h, octree* dst_d) {
  if(DEBUG) { printf("[DEBUG] octree_cpy_data_cpu_gpu\n"); }
  host_to_device(src_h->data, dst_d->data, src_h->n_leafs * src_h->feature_size);
}

void octree_cpy_trees_gpu_cpu(const octree* src_d, octree* dst_h) {
  if(DEBUG) { printf("[DEBUG] octree_cpy_trees_gpu_cpu\n"); }
  device_to_host(src_d->trees, dst_h->trees, octree_num_blocks(src_d) * N_TREE_INTS);
}
void octree_cpy_prefix_leafs_gpu_cpu(const octree* src_d, octree* dst_h) {
  if(DEBUG) { printf("[DEBUG] octree_cpy_prefix_leafs_gpu_cpu\n"); }
  device_to_host(src_d->prefix_leafs, dst_h->prefix_leafs, octree_num_blocks(src_d));
}
void octree_cpy_data_gpu_cpu(const octree* src_d, octree* dst_h) {
  if(DEBUG) { printf("[DEBUG] octree_cpy_data_gpu_cpu\n"); }
  device_to_host(src_d->data, dst_h->data, src_d->n_leafs * src_d->feature_size);
}

void octree_cpy_trees_gpu_gpu(const octree* src_d, octree* dst_d) {
  if(DEBUG) { printf("[DEBUG] octree_cpy_trees_gpu_gpu\n"); }
  device_to_device(src_d->trees, dst_d->trees, octree_num_blocks(src_d) * N_TREE_INTS);
}
void octree_cpy_prefix_leafs_gpu_gpu(const octree* src_d, octree* dst_d) {
  if(DEBUG) { printf("[DEBUG] octree_cpy_prefix_leafs_gpu_gpu\n"); }
  device_to_device(src_d->prefix_leafs, dst_d->prefix_leafs, octree_num_blocks(src_d));
}
void octree_cpy_data_gpu_gpu(const octree* src_d, octree* dst_d) {
  if(DEBUG) { printf("[DEBUG] octree_cpy_data_gpu_gpu\n"); }
  device_to_device(src_d->data, dst_d->data, src_d->n_leafs * src_d->feature_size);
}


__global__ void kernel_cpy_sup_to_sub(octree sub, int n_leafs, const octree sup) {
  CUDA_KERNEL_LOOP(sub_leaf_idx, n_leafs) {
    int grid_idx = sub.data[sub_leaf_idx * sub.feature_size];
    int data_idx = sub_leaf_idx - sub.prefix_leafs[grid_idx];
    int sub_bit_idx = data_idx_to_bit_idx(octree_get_tree(&sub, grid_idx), data_idx);

    const ot_tree_t* sup_tree = octree_get_tree(&sup, grid_idx);
    int sup_bit_idx = tree_bit_idx_leaf(sup_tree, sub_bit_idx);
    int sup_data_idx = tree_data_idx(sup_tree, sup_bit_idx, sup.feature_size);

    octree_cpy_leaf(octree_get_data(&sup, grid_idx) + sup_data_idx, sup.feature_size, sub.data + sub_leaf_idx * sub.feature_size); 
  }
}

extern "C"
void octree_cpy_sup_to_sub_gpu(const octree* sup, octree* sub) {
  octree_leaf_idx_to_grid_idx_gpu(sub, sub->feature_size, sub->data_capacity, sub->data);
  kernel_cpy_sup_to_sub<<<GET_BLOCKS(sub->n_leafs), CUDA_NUM_THREADS>>>(
    *sub, sub->n_leafs, *sup  
  );
  CUDA_POST_KERNEL_CHECK; 
}

__global__ void kernel_cpy_sub_to_sup(octree sup, int n_leafs, const octree sub) {
  CUDA_KERNEL_LOOP(sub_leaf_idx, n_leafs) {
    int grid_idx = leaf_idx_to_grid_idx(&sub, sub_leaf_idx);
    int data_idx = sub_leaf_idx - sub.prefix_leafs[grid_idx];
    int sub_bit_idx = data_idx_to_bit_idx(octree_get_tree(&sub, grid_idx), data_idx);

    const ot_tree_t* sup_tree = octree_get_tree(&sup, grid_idx);
    int sup_bit_idx = tree_bit_idx_leaf(sup_tree, sub_bit_idx);
    int sup_data_idx = tree_data_idx(sup_tree, sup_bit_idx, sup.feature_size);
    ot_data_t* sup_data = octree_get_data(&sup, grid_idx);

    for(int f = 0; f < sup.feature_size; ++f) {
      atomicAdd(sup_data + (sup_data_idx + f), sub.data[sub_leaf_idx * sub.feature_size + f]);
    }
  }
}

extern "C"
void octree_cpy_sub_to_sup_sum_gpu(const octree* sub, octree* sup) {
  octree_fill_data_gpu(sup, 0); 
  kernel_cpy_sub_to_sup<<<GET_BLOCKS(sub->n_leafs), CUDA_NUM_THREADS>>>(
    *sup, sub->n_leafs, *sub  
  );
  CUDA_POST_KERNEL_CHECK; 
}

extern "C"
void octree_copy_gpu(const octree* src, octree* dst) {
  octree_resize_as_gpu(src, dst);
  octree_cpy_trees_gpu_gpu(src, dst);
  octree_cpy_prefix_leafs_gpu_gpu(src, dst);
  octree_cpy_data_gpu_gpu(src, dst);
}



void octree_to_gpu(const octree* grid_h, octree* grid_d) {
  int n_blocks = octree_num_blocks(grid_h);

  octree_resize_as_gpu(grid_h, grid_d);
  if(n_blocks > 0) {
    octree_cpy_trees_cpu_gpu(grid_h, grid_d);
    octree_cpy_prefix_leafs_cpu_gpu(grid_h, grid_d);
    octree_cpy_data_cpu_gpu(grid_h, grid_d);
  }
}

void octree_to_cpu(const octree* grid_d, octree* grid_h) {
  int n_blocks = octree_num_blocks(grid_d);

  octree_resize_as_cpu(grid_d, grid_h);
  if(n_blocks > 0) {
    octree_cpy_trees_gpu_cpu(grid_d, grid_h);
    octree_cpy_prefix_leafs_gpu_cpu(grid_d, grid_h);
    octree_cpy_data_gpu_cpu(grid_d, grid_h);
  }
}




template <typename T>
__global__ void kernel_octree_leaf_idx_to_grid_idx(T* inds, int n_blocks, const octree in, const int stride, const int inds_length) {
  CUDA_KERNEL_LOOP(grid_idx, n_blocks) {
    const ot_tree_t* tree = octree_get_tree(&in, grid_idx);
    // int cum_n_leafs = n_leafs_upto(&in, grid_idx);
    int cum_n_leafs = in.prefix_leafs[grid_idx];
    int n_leafs = tree_n_leafs(tree);
    for(int leaf_idx = 0; leaf_idx < n_leafs; ++leaf_idx) {
      int inds_idx = cum_n_leafs * stride + leaf_idx * stride; 
      // if(leaf_idx >= inds_length) printf("[ERROR] in kernel_octree_leaf_idx_to_grid_idx, %d, %d\n", leaf_idx, inds_length);
      inds[inds_idx] = grid_idx;
    }
  }
}

template <typename T>
void octree_leaf_idx_to_grid_idx_gpu(const octree* in, const int stride, const int inds_length, T* inds) {
  if(DEBUG > 1) { printf("[DEBUG] octree_leaf_idx_to_grid_idx_gpu stride=%d, n_blocks=%d\n", stride, octree_num_blocks(in)); }

  const int n_blocks = octree_num_blocks(in);
  kernel_octree_leaf_idx_to_grid_idx<<<GET_BLOCKS_T(n_blocks, 1024), 1024>>>(inds, n_blocks, *in, stride, inds_length);
  CUDA_POST_KERNEL_CHECK;
}

template void octree_leaf_idx_to_grid_idx_gpu<ot_data_t>(const octree* in, const int stride, const int inds_length, ot_data_t* inds);
template void octree_leaf_idx_to_grid_idx_gpu<ot_size_t>(const octree* in, const int stride, const int inds_length, ot_size_t* inds);



bool octree_equal_trees_gpu(const octree* in1, const octree* in2) {
  if(DEBUG) { printf("[DEBUG] octree_equal_trees_gpu\n"); }

  if(in1->n_leafs != in2->n_leafs) {
    return false;
  }

  int n_blocks1 = octree_num_blocks(in1);
  int n_blocks2 = octree_num_blocks(in2);
  if(n_blocks1 != n_blocks2) {
    return false;
  }


  thrust::equal(thrust::device, in2->trees, in2->trees + (N_TREE_INTS * n_blocks1), in2->trees);
  thrust::equal(thrust::device, in1->trees, in1->trees + (N_TREE_INTS * n_blocks1), in1->trees);

  bool eq = thrust::equal(thrust::device, in1->trees, in1->trees + (N_TREE_INTS * n_blocks1), in2->trees);

  return eq;
}

bool octree_equal_prefix_leafs_gpu(const octree* in1, const octree* in2) {
  int n_blocks1 = octree_num_blocks(in1);
  int n_blocks2 = octree_num_blocks(in2);
  if(n_blocks1 != n_blocks2) {
    return false;
  }

  return thrust::equal(thrust::device, in1->prefix_leafs, in1->prefix_leafs + n_blocks1, in2->prefix_leafs);
}

bool octree_equal_data_gpu(const octree* in1, const octree* in2) {
  if(in1->feature_size * in1->n_leafs != in2->feature_size * in2->n_leafs) {
    return false;
  }
  
  return thrust::equal(thrust::device, in1->data, in1->data + (in1->feature_size * in1->n_leafs), in2->data);
}

bool octree_equal_gpu(const octree* in1, const octree* in2) {
  if(!octree_equal_shape(in1, in2)) { return false; }
  if(in1->n_leafs != in2->n_leafs) { return false; }

  if(!octree_equal_trees_gpu(in1, in2)) {
    return false;
  }
  
  if(!octree_equal_prefix_leafs_gpu(in1, in2)) {
    return false;
  }

  if(!octree_equal_data_gpu(in1, in2)) {
    return false;
  }

  return true;
}
