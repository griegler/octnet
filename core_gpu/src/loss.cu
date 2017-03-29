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

#include "octnet/gpu/loss.h"
#include "octnet/gpu/gpu.h"

#include <cstdio>
#include <cstdlib>

#include <thrust/execution_policy.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>

struct octree_plus_float2_bfcn : public thrust::binary_function<float2, float2, float2> {
  __host__ __device__ 
  float2 operator()(float2 in1, float2 in2) {
    float2 ret;
    ret.x = in1.x + in2.x;
    ret.y = in1.y + in2.y;
    return ret;
  }
};





struct octree_mse_loss_from_leaf_idx : public thrust::unary_function<int, ot_data_t> {
  const octree input;
  const octree target;
  
  octree_mse_loss_from_leaf_idx(const octree input_, const octree target_) : 
      input(input_), target(target_) {
  }

  __host__ __device__ ot_data_t operator()(const int leaf_idx) {
    const int grid_idx = leaf_idx_to_grid_idx(&input, leaf_idx);
    const ot_tree_t* tree = octree_get_tree(&input, grid_idx);

    // const int cum_n_leafs = n_leafs_upto(&input, grid_idx);
    const int cum_n_leafs = input.prefix_leafs[grid_idx];
    const int data_idx = leaf_idx - cum_n_leafs;
    const int bit_idx = data_idx_to_bit_idx(tree, data_idx);

    // const ot_data_t* in = input.data_ptrs[grid_idx] + data_idx * input.feature_size;
    const ot_data_t* in = octree_get_data(&input, grid_idx) + data_idx * input.feature_size;
    // const ot_data_t* ta = target.data_ptrs[grid_idx] + data_idx * target.feature_size;
    const ot_data_t* ta = octree_get_data(&target, grid_idx) + data_idx * target.feature_size;

    const int depth = depth_from_bit_idx(bit_idx);
    const ot_data_t vol = depth == 0 ? 512 : (depth == 1 ? 64 : (depth == 2 ? 8 : 1));

    ot_data_t sum = 0;
    for(int f = 0; f < input.feature_size; ++f) {
      const ot_data_t z = in[f] - ta[f];
      sum += vol * z * z;
    }

    return sum;
  }
};

extern "C"
ot_data_t octree_mse_loss_gpu(const octree* input, const octree* target, bool size_average, bool check) {
  if(!octree_equal_shape(input, target)) {
    printf("[ERROR] mse_loss - shape of inputs do not match\n");
    exit(-1);
  }
  if(check && !octree_equal_trees_gpu(input, target)) {
    printf("[ERROR] mse_loss - tree structure of inputs do not match\n");
    exit(-1);
  }

  thrust::counting_iterator<int> iter(0);
  ot_data_t output = thrust::transform_reduce(
    thrust::device, iter, iter + input->n_leafs, 
    octree_mse_loss_from_leaf_idx(*input, *target), ot_data_t(0), thrust::plus<ot_data_t>());

  if(size_average) {
    output = output / (octree_num_blocks(input) * input->feature_size * 8 * 8 * 8);
  }
  return output;
}


__global__ void kernel_mse_loss_bwd(octree grad, int n_leafs, const octree input, const octree target, const ot_data_t norm) {
  
  CUDA_KERNEL_LOOP(leaf_idx, n_leafs) {
    const int grid_idx = grad.data[leaf_idx * grad.feature_size];
    const ot_tree_t* tree = octree_get_tree(&grad, grid_idx);

    // const int cum_n_leafs = n_leafs_upto(&grad, grid_idx);
    const int cum_n_leafs = grad.prefix_leafs[grid_idx];
    const int data_idx = leaf_idx - cum_n_leafs;
    const int bit_idx = data_idx_to_bit_idx(tree, data_idx);

    // ot_data_t* grad_data = grad.data_ptrs[grid_idx] + data_idx * grad.feature_size;
    ot_data_t* grad_data = octree_get_data(&grad, grid_idx) + data_idx * grad.feature_size;
    // ot_data_t* input_data = input.data_ptrs[grid_idx] + data_idx * input.feature_size;
    ot_data_t* input_data = octree_get_data(&input, grid_idx) + data_idx * input.feature_size;
    // ot_data_t* target_data = target.data_ptrs[grid_idx] + data_idx * target.feature_size;
    ot_data_t* target_data = octree_get_data(&target, grid_idx) + data_idx * target.feature_size;

    const int depth = depth_from_bit_idx(bit_idx);
    const ot_data_t vol = depth == 0 ? 512 : (depth == 1 ? 64 : (depth == 2 ? 8 : 1));

    for(int f = 0; f < input.feature_size; ++f) {
      const ot_data_t z = input_data[f] - target_data[f];
      grad_data[f] = vol * norm * z;
    }
  }
}


extern "C"
void octree_mse_loss_bwd_gpu(const octree* input, const octree* target, bool size_average, bool check, octree* grad) {
  if(!octree_equal_shape(input, target)) {
    printf("[ERROR] mse_loss_bwd - shape of inputs do not match\n");
    exit(-1);
  }
  if(check && !octree_equal_trees_gpu(input, target)) {
    printf("[ERROR] mse_loss_bwd - tree structure of inputs do not match\n");
    exit(-1);
  }

  octree_cpy_scalars(input, grad);
  octree_resize_as_gpu(input, grad);
  octree_cpy_trees_gpu_gpu(input, grad);
  octree_cpy_prefix_leafs_gpu_gpu(input, grad);

  ot_data_t norm = 2.0;
  if(size_average) {
    norm = norm / (octree_num_blocks(input) * input->feature_size * 8 * 8 * 8);
  }
  
  octree_leaf_idx_to_grid_idx_gpu(grad, grad->feature_size, grad->data_capacity, grad->data);
  kernel_mse_loss_bwd<<<GET_BLOCKS(grad->n_leafs), CUDA_NUM_THREADS>>>(
      *grad, grad->n_leafs, *input, *target, norm
  );
  CUDA_POST_KERNEL_CHECK; 
}


struct octree_mse_ds_loss_from_leaf_idx : public thrust::unary_function<int, ot_data_t> {
  const octree input;
  const octree target;
  
  octree_mse_ds_loss_from_leaf_idx(const octree input_, const octree target_) : 
      input(input_), target(target_) {
  }

  __host__ __device__ ot_data_t operator()(const int leaf_idx) {
    const int in_grid_idx = leaf_idx_to_grid_idx(&input, leaf_idx);
    const ot_tree_t* in_tree = octree_get_tree(&input, in_grid_idx);

    int in_data_idx = leaf_idx - input.prefix_leafs[in_grid_idx];
    int in_bit_idx = data_idx_to_bit_idx(in_tree, in_data_idx);

    int n,ds,hs,ws;
    int depth = octree_ind_to_dense_ind(&input, in_grid_idx, in_bit_idx, &n, &ds,&hs,&ws);
    int width = width_from_depth(depth);
    
    ot_data_t grid_out = 0;
    for(int d = ds; d < (ds+width); ++d) {
      for(int h = hs; h < (hs+width); ++h) {
        for(int w = ws; w < (ws+width); ++w) {
          int gd = d / 8;
          int gh = h / 8;
          int gw = w / 8;
          int bd = d % 8;
          int bh = h % 8;
          int bw = w % 8;
          int ta_grid_idx = octree_grid_idx(&target, n, gd,gh,gw);
          const ot_tree_t* ta_tree = octree_get_tree(&target, ta_grid_idx);
          int ta_bit_idx = tree_bit_idx(ta_tree, bd,bh,bw);
          int ta_data_idx = tree_data_idx(ta_tree, ta_bit_idx, target.feature_size);
          const ot_data_t* ta_data = octree_get_data(&target, ta_grid_idx);
          for(int f = 0; f < input.feature_size; ++f) {
            ot_data_t x = input.data[leaf_idx * input.feature_size + f];
            ot_data_t y = ta_data[ta_data_idx + f];
            grid_out += (x-y) * (x-y);
          }
        }
      }
    } 

    return grid_out;
  }
};

extern "C"
ot_data_t octree_mse_ds_loss_gpu(const octree* input, const octree* target, bool size_average) {
  if(!octree_equal_shape(input, target)) {
    printf("[ERROR] mse_ds_loss - shape of inputs do not match\n");
    exit(-1);
  }

  thrust::counting_iterator<int> iter(0);
  ot_data_t output = thrust::transform_reduce(
    thrust::device, iter, iter + input->n_leafs, 
    octree_mse_ds_loss_from_leaf_idx(*input, *target), 
    ot_data_t(0), thrust::plus<ot_data_t>());
  
  if(size_average) {
    output = output / (octree_num_blocks(input) * input->feature_size * 8 * 8 * 8);
  }
  return output;
}


__global__ void kernel_mse_ds_loss_bwd(octree grad, int n_leafs, const octree input, const octree target, const ot_data_t norm) {
  
  CUDA_KERNEL_LOOP(leaf_idx, n_leafs) {
    const int in_grid_idx = grad.data[leaf_idx * grad.feature_size];
    const ot_tree_t* in_tree = octree_get_tree(&input, in_grid_idx);

    int in_data_idx = leaf_idx - input.prefix_leafs[in_grid_idx];
    int in_bit_idx = data_idx_to_bit_idx(in_tree, in_data_idx);

    int n,ds,hs,ws;
    int depth = octree_ind_to_dense_ind(&input, in_grid_idx, in_bit_idx, &n, &ds,&hs,&ws);
    int width = width_from_depth(depth);
    
    for(int f = 0; f < grad.feature_size; ++f) {
      grad.data[leaf_idx * grad.feature_size + f] = 0;
    }

    for(int d = ds; d < (ds+width); ++d) {
      for(int h = hs; h < (hs+width); ++h) {
        for(int w = ws; w < (ws+width); ++w) {
          int gd = d / 8;
          int gh = h / 8;
          int gw = w / 8;
          int bd = d % 8;
          int bh = h % 8;
          int bw = w % 8;
          int ta_grid_idx = octree_grid_idx(&target, n, gd,gh,gw);
          const ot_tree_t* ta_tree = octree_get_tree(&target, ta_grid_idx);
          int ta_bit_idx = tree_bit_idx(ta_tree, bd,bh,bw);
          int ta_data_idx = tree_data_idx(ta_tree, ta_bit_idx, target.feature_size);
          const ot_data_t* ta_data = octree_get_data(&target, ta_grid_idx);
          for(int f = 0; f < input.feature_size; ++f) {
            ot_data_t x = input.data[leaf_idx * input.feature_size + f];
            ot_data_t y = ta_data[ta_data_idx + f];
            grad.data[leaf_idx * grad.feature_size + f] += norm * (x-y);
          }
        }
      }
    }
  }
}


extern "C"
void octree_mse_loss_ds_bwd_gpu(const octree* input, const octree* target, bool size_average, octree* grad) {
  if(!octree_equal_shape(input, target)) {
    printf("[ERROR] mse_ds_loss_bwd - shape of inputs do not match\n");
    exit(-1);
  }

  octree_cpy_scalars(input, grad);
  octree_resize_as_gpu(input, grad);
  octree_cpy_trees_gpu_gpu(input, grad);
  octree_cpy_prefix_leafs_gpu_gpu(input, grad);

  ot_data_t norm = 2.0;
  if(size_average) {
    norm = norm / (octree_num_blocks(input) * input->feature_size * 8 * 8 * 8);
  }

  octree_leaf_idx_to_grid_idx_gpu(grad, grad->feature_size, grad->data_capacity, grad->data);
  kernel_mse_ds_loss_bwd<<<GET_BLOCKS(grad->n_leafs), CUDA_NUM_THREADS>>>(
      *grad, grad->n_leafs, *input, *target, norm
  );
  CUDA_POST_KERNEL_CHECK;
}





struct octree_nll_loss_from_leaf_idx : public thrust::unary_function<int, float2> {
  const octree input;
  const octree target;
  const ot_data_t* weights;
  const int class_base;
  
  octree_nll_loss_from_leaf_idx(const octree input_, const octree target_, const ot_data_t* weights_, int class_base_) : 
      input(input_), target(target_), weights(weights_), class_base(class_base_) {
  }

  __host__ __device__ 
  float2 operator()(const int leaf_idx) {
    const int grid_idx = leaf_idx_to_grid_idx(&input, leaf_idx);
    const ot_tree_t* tree = octree_get_tree(&input, grid_idx);

    // const int cum_n_leafs = n_leafs_upto(&input, grid_idx);
    const int cum_n_leafs = input.prefix_leafs[grid_idx];
    const int data_idx = leaf_idx - cum_n_leafs;
    const int bit_idx = data_idx_to_bit_idx(tree, data_idx);

    // const ot_data_t* in = input.data_ptrs[grid_idx] + data_idx * input.feature_size;
    const ot_data_t* in = octree_get_data(&input, grid_idx) + data_idx * input.feature_size;
    // const ot_data_t* ta = target.data_ptrs[grid_idx] + data_idx * target.feature_size;
    const ot_data_t* ta = octree_get_data(&target, grid_idx) + data_idx * target.feature_size;

    const int depth = depth_from_bit_idx(bit_idx);
    const ot_data_t vol = depth == 0 ? 512 : (depth == 1 ? 64 : (depth == 2 ? 8 : 1));

    int cur_target = round(ta[0]) - class_base;
    assert(cur_target >= 0 && cur_target < input.feature_size);
    ot_data_t weight = vol * weights[cur_target];

    // if(cur_target >= input.feature_size || cur_target <= 0)
    //   printf("  ERROR cur_target=%d, weight=%f\n", cur_target, weight);

    float2 ret_val;
    ret_val.x = -weight * in[cur_target];
    ret_val.y = weight;

    return ret_val;
  }
};


extern "C"
void octree_nll_loss_gpu(const octree* input, const octree* target, const ot_data_t* weights, int class_base, bool size_average, bool check, ot_data_t* output, ot_data_t* total_weight) {
  if(input->n != target->n || input->grid_depth != target->grid_depth || input->grid_height != target->grid_height || input->grid_width != target->grid_width || 1 != target->feature_size) {
    printf("[ERROR] nll_loss - shape of inputs do not match\n");
    exit(-1);
  }
  if(check && !octree_equal_trees_gpu(input, target)) {
    printf("[ERROR] nll_loss - tree structure of inputs do not match\n");
    exit(-1);
  }

  float2 init;
  init.x = 0;
  init.y = 0;
  thrust::counting_iterator<int> iter(0);
  float2 res = thrust::transform_reduce(
    thrust::device, iter, iter + input->n_leafs, 
    octree_nll_loss_from_leaf_idx(*input, *target, weights, class_base), init, 
    octree_plus_float2_bfcn());

  output[0] = res.x;
  total_weight[0] = res.y;

  if(size_average && total_weight[0] != 0) {
    output[0] /= total_weight[0];
  }
}


__global__ void kernel_nll_loss_bwd(octree grad, int n_leafs, const octree input, const octree target, const ot_data_t* weights, const ot_data_t norm, const int class_base) {
  
  CUDA_KERNEL_LOOP(leaf_idx, n_leafs) {
    const int grid_idx = grad.data[leaf_idx * grad.feature_size];
    const ot_tree_t* tree = octree_get_tree(&grad, grid_idx);

    // const int cum_n_leafs = n_leafs_upto(&grad, grid_idx);
    const int cum_n_leafs = grad.prefix_leafs[grid_idx];
    const int data_idx = leaf_idx - cum_n_leafs;
    const int bit_idx = data_idx_to_bit_idx(tree, data_idx);

    // ot_data_t* grad_data = grad.data_ptrs[grid_idx] + data_idx * grad.feature_size;
    ot_data_t* grad_data = octree_get_data(&grad, grid_idx) + data_idx * grad.feature_size;
    // ot_data_t* target_data = target.data_ptrs[grid_idx] + data_idx * target.feature_size;
    ot_data_t* target_data = octree_get_data(&target, grid_idx) + data_idx * target.feature_size;

    const int depth = depth_from_bit_idx(bit_idx);
    const ot_data_t vol = depth == 0 ? 512 : (depth == 1 ? 64 : (depth == 2 ? 8 : 1));
    
    int cur_target = round(target_data[0]) - class_base;
    assert(cur_target >= 0 && cur_target < input.feature_size);

    for(int f = 0; f < grad.feature_size; ++f) {
      grad_data[f] = 0;
    }
    grad_data[cur_target] = vol * -weights[cur_target] * norm;
  }
}

extern "C"
void octree_nll_loss_bwd_gpu(const octree* input, const octree* target, const ot_data_t* weights, const ot_data_t total_weight, int class_base, bool size_average, bool check, octree* grad) {
  if(input->n != target->n || input->grid_depth != target->grid_depth || input->grid_height != target->grid_height || input->grid_width != target->grid_width || 1 != target->feature_size) {
    printf("[ERROR] nll_loss_bwd - shape of inputs do not match\n");
    exit(-1);
  }
  if(check && !octree_equal_trees_gpu(input, target)) {
    printf("[ERROR] nll_loss_bwd - tree structure of inputs do not match\n");
    exit(-1);
  }

  octree_cpy_scalars(input, grad);
  octree_resize_as_gpu(input, grad);
  octree_cpy_trees_gpu_gpu(input, grad);
  octree_cpy_prefix_leafs_gpu_gpu(input, grad);

  ot_data_t norm = 1.0;
  if(size_average) {
    norm /= total_weight;
  }
   
  octree_leaf_idx_to_grid_idx_gpu(grad, grad->feature_size, grad->data_capacity, grad->data);
  kernel_nll_loss_bwd<<<GET_BLOCKS(grad->n_leafs), CUDA_NUM_THREADS>>>(
      *grad, grad->n_leafs, *input, *target, weights, norm, class_base
  );
  CUDA_POST_KERNEL_CHECK; 
}





#define EPS 1e-12

struct octree_bce_loss_from_leaf_idx : public thrust::unary_function<int, ot_data_t> {
  const octree input;
  const octree target;
  
  octree_bce_loss_from_leaf_idx(const octree input_, const octree target_) : 
      input(input_), target(target_) {
  }

  __host__ __device__ ot_data_t operator()(const int leaf_idx) {
    const int grid_idx = leaf_idx_to_grid_idx(&input, leaf_idx);
    const ot_tree_t* tree = octree_get_tree(&input, grid_idx);

    // const int cum_n_leafs = n_leafs_upto(&input, grid_idx);
    const int cum_n_leafs = input.prefix_leafs[grid_idx];
    const int data_idx = leaf_idx - cum_n_leafs;
    const int bit_idx = data_idx_to_bit_idx(tree, data_idx);

    // const ot_data_t* in = input.data_ptrs[grid_idx] + data_idx * input.feature_size;
    const ot_data_t* in = octree_get_data(&input, grid_idx) + data_idx * input.feature_size;
    // const ot_data_t* ta = target.data_ptrs[grid_idx] + data_idx * target.feature_size;
    const ot_data_t* ta = octree_get_data(&target, grid_idx) + data_idx * target.feature_size;

    const int depth = depth_from_bit_idx(bit_idx);
    const ot_data_t vol = depth == 0 ? 512 : (depth == 1 ? 64 : (depth == 2 ? 8 : 1));

    ot_data_t sum = 0;
    for(int f = 0; f < input.feature_size; ++f) {
      const ot_data_t x = in[f];
      const ot_data_t y = ta[f];
      sum -= vol *(log(x + EPS) * y + log(1. - x + EPS) * (1. - y)); 
    }

    return sum;
  }
};

extern "C"
void octree_bce_loss_gpu(const octree* input, const octree* target, bool size_average, bool check, ot_data_t* output, ot_data_t* total_weight) {
  if(!octree_equal_shape(input, target)) {
    printf("[ERROR] bce_loss - shape of inputs do not match\n");
    exit(-1);
  }
  if(check && !octree_equal_trees_gpu(input, target)) {
    printf("[ERROR] bce_loss - tree structure of inputs do not match\n");
    exit(-1);
  }

  thrust::counting_iterator<int> iter(0);
  *output = thrust::transform_reduce(
    thrust::device, iter, iter + input->n_leafs, 
    octree_bce_loss_from_leaf_idx(*input, *target), ot_data_t(0), thrust::plus<ot_data_t>());

  if(size_average) {
    *total_weight = octree_num_blocks(input) * input->feature_size * 8 * 8 * 8;
    *output = (*output) / (*total_weight) ;
  }
  else {
    *total_weight = 1;
  }
}


__global__ void kernel_bce_loss_bwd(octree grad, int n_leafs, const octree input, const octree target, const ot_data_t norm) {
  
  CUDA_KERNEL_LOOP(leaf_idx, n_leafs) {
    const int grid_idx = grad.data[leaf_idx * grad.feature_size];
    const ot_tree_t* tree = octree_get_tree(&grad, grid_idx);

    // const int cum_n_leafs = n_leafs_upto(&grad, grid_idx);
    const int cum_n_leafs = grad.prefix_leafs[grid_idx];
    const int data_idx = leaf_idx - cum_n_leafs;
    const int bit_idx = data_idx_to_bit_idx(tree, data_idx);

    // ot_data_t* grad_data = grad.data_ptrs[grid_idx] + data_idx * grad.feature_size;
    ot_data_t* grad_data = octree_get_data(&grad, grid_idx) + data_idx * grad.feature_size;
    // ot_data_t* input_data = input.data_ptrs[grid_idx] + data_idx * input.feature_size;
    ot_data_t* input_data = octree_get_data(&input, grid_idx) + data_idx * input.feature_size;
    // ot_data_t* target_data = target.data_ptrs[grid_idx] + data_idx * target.feature_size;
    ot_data_t* target_data = octree_get_data(&target, grid_idx) + data_idx * target.feature_size;

    const int depth = depth_from_bit_idx(bit_idx);
    const ot_data_t vol = depth == 0 ? 512 : (depth == 1 ? 64 : (depth == 2 ? 8 : 1));

    for(int f = 0; f < input.feature_size; ++f) {
      const ot_data_t x = input_data[f];
      const ot_data_t y = target_data[f];
      grad_data[f] = - vol * norm * (y - x) / ((1. - x + EPS) * (x + EPS));
    }
  }
}


extern "C"
void octree_bce_loss_bwd_gpu(const octree* input, const octree* target, bool size_average, bool check, octree* grad) {
  if(!octree_equal_shape(input, target)) {
    printf("[ERROR] bce_loss_bwd - shape of inputs do not match\n");
    exit(-1);
  }
  if(check && !octree_equal_trees_gpu(input, target)) {
    printf("[ERROR] bce_loss_bwd - tree structure of inputs do not match\n");
    exit(-1);
  }

  octree_cpy_scalars(input, grad);
  octree_resize_as_gpu(input, grad);
  octree_cpy_trees_gpu_gpu(input, grad);
  octree_cpy_prefix_leafs_gpu_gpu(input, grad);

  ot_data_t norm = 1.0;
  if(size_average) {
    norm = norm / (octree_num_blocks(input) * input->feature_size * 8 * 8 * 8);
  }
  
  octree_leaf_idx_to_grid_idx_gpu(grad, grad->feature_size, grad->data_capacity, grad->data);
  kernel_bce_loss_bwd<<<GET_BLOCKS(grad->n_leafs), CUDA_NUM_THREADS>>>(
      *grad, grad->n_leafs, *input, *target, norm
  );
  CUDA_POST_KERNEL_CHECK; 
}




struct octree_bce_dense_loss_from_leaf_idx : public thrust::unary_function<int, ot_data_t> {
  const octree input;
  const ot_data_t* target;
  
  octree_bce_dense_loss_from_leaf_idx(const octree input_, const ot_data_t* target_) : 
      input(input_), target(target_) {
  }

  __host__ __device__ ot_data_t operator()(const int leaf_idx) {
    const int grid_idx = leaf_idx_to_grid_idx(&input, leaf_idx);
    const ot_tree_t* tree = octree_get_tree(&input, grid_idx);

    // const int cum_n_leafs = n_leafs_upto(&input, grid_idx);
    const int cum_n_leafs = input.prefix_leafs[grid_idx];
    const int data_idx = leaf_idx - cum_n_leafs;
    const int bit_idx = data_idx_to_bit_idx(tree, data_idx);

    // const ot_data_t* estimate = input.data_ptrs[grid_idx] + data_idx * input.feature_size;
    const ot_data_t* estimate = octree_get_data(&input, grid_idx) + data_idx * input.feature_size;

    int n,ds,hs,ws;
    const int depth = octree_ind_to_dense_ind(&input, grid_idx, bit_idx, &n, &ds,&hs,&ws);
    const int size = width_from_depth(depth);

    const int dense_depth = 8 * input.grid_depth;
    const int dense_height = 8 * input.grid_height;
    const int dense_width = 8 * input.grid_width;

    // printf("leaf_idx=%d, n=%d, ds=%d,hs=%d,ws=%d, size=%d\n", leaf_idx, n, ds,hs,ws, size);

    ot_data_t out = 0;
    for(int d = ds; d < (ds+size); ++d) {
      for(int h = hs; h < (hs+size); ++h) {
        for(int w = ws; w < (ws+size); ++w) {
          for(int f = 0; f < input.feature_size; ++f) {
            ot_data_t x = estimate[f];
            ot_data_t y = target[(((n * input.feature_size + f) * dense_depth + d) * dense_height + h) * dense_width + w];
            // printf("  x=%f, y=%f\n", x, y); 
            out += (log(x + EPS) * y + log(1. - x + EPS) * (1. - y));
          }
        }
      }
    }

    return -out;
  }
};

extern "C"
void octree_bce_dense_loss_gpu(const octree* input, const ot_data_t* target, bool size_average, ot_data_t* output, ot_data_t* total_weight) {
  thrust::counting_iterator<int> iter(0);
  *output = thrust::transform_reduce(
    thrust::device, iter, iter + input->n_leafs, 
    octree_bce_dense_loss_from_leaf_idx(*input, target), ot_data_t(0), thrust::plus<ot_data_t>());

  if(size_average) {
    *total_weight = octree_num_blocks(input) * input->feature_size * 8 * 8 * 8;
    *output = (*output) / (*total_weight) ;
  }
  else {
    *total_weight = 1;
  }
}


__global__ void kernel_bce_dense_loss_bwd(octree grad, int n_leafs, const octree input, const ot_data_t* target, const ot_data_t norm) {
  
  CUDA_KERNEL_LOOP(leaf_idx, n_leafs) {
    const int grid_idx = grad.data[leaf_idx * grad.feature_size];
    const ot_tree_t* tree = octree_get_tree(&grad, grid_idx);

    // const int cum_n_leafs = n_leafs_upto(&grad, grid_idx);
    const int cum_n_leafs = grad.prefix_leafs[grid_idx];
    const int data_idx = leaf_idx - cum_n_leafs;
    const int bit_idx = data_idx_to_bit_idx(tree, data_idx);

    // ot_data_t* grad_data = grad.data_ptrs[grid_idx] + data_idx * grad.feature_size;
    ot_data_t* grad_data = octree_get_data(&grad, grid_idx) + data_idx * grad.feature_size;
    // ot_data_t* estimate = input.data_ptrs[grid_idx] + data_idx * input.feature_size;
    ot_data_t* estimate = octree_get_data(&input, grid_idx) + data_idx * input.feature_size;
    
    int n,ds,hs,ws;
    const int depth = octree_ind_to_dense_ind(&input, grid_idx, bit_idx, &n, &ds,&hs,&ws);
    const int size = width_from_depth(depth);

    const int dense_depth = 8 * input.grid_depth;
    const int dense_height = 8 * input.grid_height;
    const int dense_width = 8 * input.grid_width;

    for(int f = 0; f < input.feature_size; ++f) {
      grad_data[f] = 0;
    }
    
    for(int d = ds; d < (ds+size); ++d) {
      for(int h = hs; h < (hs+size); ++h) {
        for(int w = ws; w < (ws+size); ++w) {
          for(int f = 0; f < input.feature_size; ++f) {
            ot_data_t x = estimate[f];
            ot_data_t y = target[(((n * input.feature_size + f) * dense_depth + d) * dense_height + h) * dense_width + w];
            grad_data[f] -= norm * (y - x) / ((1. - x + EPS) * (x + EPS));
          }
        }
      }
    }
  }
}


extern "C"
void octree_bce_dense_loss_bwd_gpu(const octree* input, const ot_data_t* target, bool size_average, octree* grad) {
  octree_cpy_scalars(input, grad);
  octree_resize_as_gpu(input, grad);
  octree_cpy_trees_gpu_gpu(input, grad);
  octree_cpy_prefix_leafs_gpu_gpu(input, grad);

  ot_data_t norm = 1.0;
  if(size_average) {
    norm = norm / (octree_num_blocks(input) * input->feature_size * 8 * 8 * 8);
  }

  octree_leaf_idx_to_grid_idx_gpu(grad, grad->feature_size, grad->data_capacity, grad->data);
  kernel_bce_dense_loss_bwd<<<GET_BLOCKS(grad->n_leafs), CUDA_NUM_THREADS>>>(
      *grad, grad->n_leafs, *input, target, norm
  );
  CUDA_POST_KERNEL_CHECK;
}





struct octree_bce_ds_loss_from_leaf_idx : public thrust::unary_function<int, float2> {
  const octree input;
  const octree target;
  const octree weights;
  bool use_weights;
  
  octree_bce_ds_loss_from_leaf_idx(const octree input_, const octree target_, const octree weights_, bool use_weights_) : 
      input(input_), target(target_), weights(weights_), use_weights(use_weights_) {
  }

  __host__ __device__ float2 operator()(const int leaf_idx) {
    const int in_grid_idx = leaf_idx_to_grid_idx(&input, leaf_idx);
    const ot_tree_t* in_tree = octree_get_tree(&input, in_grid_idx);

    int in_data_idx = leaf_idx - input.prefix_leafs[in_grid_idx];
    int in_bit_idx = data_idx_to_bit_idx(in_tree, in_data_idx);

    int n,ds,hs,ws;
    int depth = octree_ind_to_dense_ind(&input, in_grid_idx, in_bit_idx, &n, &ds,&hs,&ws);
    int width = width_from_depth(depth);
    
    ot_data_t grid_out = 0;
    ot_data_t grid_weight = 0;
    for(int d = ds; d < (ds+width); ++d) {
      for(int h = hs; h < (hs+width); ++h) {
        for(int w = ws; w < (ws+width); ++w) {
          int gd = d / 8;
          int gh = h / 8;
          int gw = w / 8;
          int bd = d % 8;
          int bh = h % 8;
          int bw = w % 8;
          int ta_grid_idx = octree_grid_idx(&target, n, gd,gh,gw);
          const ot_tree_t* ta_tree = octree_get_tree(&target, ta_grid_idx);
          int ta_bit_idx = tree_bit_idx(ta_tree, bd,bh,bw);
          int ta_data_idx = tree_data_idx(ta_tree, ta_bit_idx, target.feature_size);
          const ot_data_t* ta_data = octree_get_data(&target, ta_grid_idx);
          const ot_data_t* we_data = use_weights ? octree_get_data(&weights, ta_grid_idx) : 0;
          for(int f = 0; f < input.feature_size; ++f) {
            ot_data_t x = input.data[leaf_idx * input.feature_size + f];
            ot_data_t y = ta_data[ta_data_idx + f];
            ot_data_t w = use_weights ? we_data[ta_data_idx + f] : 1;
            grid_out += w * (log(x + EPS) * y + log(1. - x + EPS) * (1. - y));
            grid_weight += w;
          }
        }
      }
    } 

    float2 ret;
    ret.x = -grid_out;
    ret.y = grid_weight;
    return ret;
  }
};

extern "C"
void octree_bce_ds_loss_gpu(const octree* input, const octree* target, const octree* weights, bool size_average, ot_data_t* output, ot_data_t* total_weight) {
  if(!octree_equal_shape(input, target)) {
    printf("[ERROR] bce_ds_loss - shape of inputs do not match\n");
    exit(-1);
  }
  if(!octree_equal_shape(input, weights)) {
    printf("[ERROR] bce_ds_loss - shape of inputs do not match\n");
    exit(-1);
  }


  float2 init;
  init.x = 0;
  init.y = 0;
  thrust::counting_iterator<int> iter(0);
  bool use_weights = weights != 0;
  float2 ret = thrust::transform_reduce(
    thrust::device, iter, iter + input->n_leafs, 
    octree_bce_ds_loss_from_leaf_idx(*input, *target, use_weights ? *weights : *target, use_weights), 
    init, octree_plus_float2_bfcn());
  
  if(ret.y > 0) {
    if(size_average) {
      *total_weight = ret.y;
      *output = ret.x / ret.y ;
    }
    else {
      *output = ret.x;
      *total_weight = 1;
    }
  }
  else {
    *output = 0;
    *total_weight = 1;
  }
}


__global__ void kernel_bce_ds_loss_bwd(octree grad, int n_leafs, const octree input, const octree target, const octree weights, bool use_weights, const ot_data_t norm) {
  
  CUDA_KERNEL_LOOP(leaf_idx, n_leafs) {
    const int in_grid_idx = grad.data[leaf_idx * grad.feature_size];
    const ot_tree_t* in_tree = octree_get_tree(&input, in_grid_idx);

    int in_data_idx = leaf_idx - input.prefix_leafs[in_grid_idx];
    int in_bit_idx = data_idx_to_bit_idx(in_tree, in_data_idx);

    int n,ds,hs,ws;
    int depth = octree_ind_to_dense_ind(&input, in_grid_idx, in_bit_idx, &n, &ds,&hs,&ws);
    int width = width_from_depth(depth);
    
    for(int f = 0; f < grad.feature_size; ++f) {
      grad.data[leaf_idx * grad.feature_size + f] = 0;
    }

    for(int d = ds; d < (ds+width); ++d) {
      for(int h = hs; h < (hs+width); ++h) {
        for(int w = ws; w < (ws+width); ++w) {
          int gd = d / 8;
          int gh = h / 8;
          int gw = w / 8;
          int bd = d % 8;
          int bh = h % 8;
          int bw = w % 8;
          int ta_grid_idx = octree_grid_idx(&target, n, gd,gh,gw);
          const ot_tree_t* ta_tree = octree_get_tree(&target, ta_grid_idx);
          int ta_bit_idx = tree_bit_idx(ta_tree, bd,bh,bw);
          int ta_data_idx = tree_data_idx(ta_tree, ta_bit_idx, target.feature_size);
          const ot_data_t* ta_data = octree_get_data(&target, ta_grid_idx);
          const ot_data_t* we_data = use_weights ? octree_get_data(&weights, ta_grid_idx) : 0;
          for(int f = 0; f < input.feature_size; ++f) {
            ot_data_t x = input.data[leaf_idx * input.feature_size + f];
            ot_data_t y = ta_data[ta_data_idx + f];
            ot_data_t w = use_weights ? we_data[ta_data_idx + f] : 1;
            grad.data[leaf_idx * grad.feature_size + f] -= norm * w * (y - x) / ((1. - x + EPS) * (x + EPS));
          }
        }
      }
    }
  }
}


extern "C"
void octree_bce_ds_loss_bwd_gpu(const octree* input, const octree* target, const octree* weights, bool size_average, ot_data_t total_weight, octree* grad) {
  if(!octree_equal_shape(input, target)) {
    printf("[ERROR] bce_ds_loss_bwd - shape of inputs do not match\n");
    exit(-1);
  }

  octree_cpy_scalars(input, grad);
  octree_resize_as_gpu(input, grad);
  octree_cpy_trees_gpu_gpu(input, grad);
  octree_cpy_prefix_leafs_gpu_gpu(input, grad);

  ot_data_t norm = 1.0;
  if(size_average && total_weight > 0) {
    norm = norm / total_weight;
  } 

  bool use_weights = weights != 0;
  octree_leaf_idx_to_grid_idx_gpu(grad, grad->feature_size, grad->data_capacity, grad->data);
  kernel_bce_ds_loss_bwd<<<GET_BLOCKS(grad->n_leafs), CUDA_NUM_THREADS>>>(
      *grad, grad->n_leafs, *input, *target, use_weights ? *weights : *target, use_weights, norm
  );
  CUDA_POST_KERNEL_CHECK;
}



struct dense_bce_loss_fcn : public thrust::unary_function<int, float2> {
  const ot_data_t* input;
  const ot_data_t* target;
  const ot_data_t* weights;
  
  dense_bce_loss_fcn(const ot_data_t* input_, const ot_data_t* target_, const ot_data_t* weights_) : 
      input(input_), target(target_), weights(weights_) {
  }

  __host__ __device__ float2 operator()(const int idx) { 
    ot_data_t w = weights != 0 ? weights[idx] : 1;
    ot_data_t x = input[idx];
    ot_data_t y = target[idx];
    float2 ret;
    ret.x = w * -( (log(x + EPS) * y + log(1. - x + EPS) * (1. - y)) );
    ret.y = w;
    return ret;
  }
};

extern "C"
void dense_bce_loss_gpu(const ot_data_t* input, const ot_data_t* target, const ot_data_t* weights, ot_size_t N, ot_data_t* output, ot_data_t* total_weight) {

  float2 init;
  init.x = 0; 
  init.y = 0;
  thrust::counting_iterator<int> iter(0);
  float2 result = thrust::transform_reduce(
    thrust::device, iter, iter + N, 
    dense_bce_loss_fcn(input, target, weights), init, octree_plus_float2_bfcn());

  if(result.y > 0) {
    *output = result.x / result.y;
    *total_weight = result.y;
  }
  else {
    *output = 0;
    *total_weight = 1;
  }
}


__global__ void kernel_dense_bce_loss_bwd(ot_data_t* grad, ot_size_t N, const ot_data_t* input, const ot_data_t* target, const ot_data_t* weights, ot_data_t norm) {
  
  CUDA_KERNEL_LOOP(idx, N) {
    ot_data_t x = input[idx];
    ot_data_t y = target[idx];
    ot_data_t w = weights != 0 ? weights[idx] : 1;
    w *= norm;
    grad[idx] = w * -(y - x) / ((1. - x + EPS) * (x + EPS));
  }
}


extern "C"
void dense_bce_loss_bwd_gpu(const ot_data_t* input, const ot_data_t* target, const ot_data_t* weights, ot_size_t N, ot_data_t total_weight, ot_data_t* grad) {
  ot_data_t norm = total_weight > 0 ? 1.0 / total_weight : 0;

  kernel_dense_bce_loss_bwd<<<GET_BLOCKS(N), CUDA_NUM_THREADS>>>(
      grad, N, input, target, weights, norm
  );
  CUDA_POST_KERNEL_CHECK;
}


#undef EPS
