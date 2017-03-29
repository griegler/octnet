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

#include "octnet/cpu/loss.h"
#include "octnet/cpu/cpu.h"

#include <cstdio>
#include <cstdlib>

#if defined(_OPENMP)
#include <omp.h>
#endif


extern "C"
ot_data_t octree_mse_loss_cpu(const octree* input, const octree* target, bool size_average, bool check) {
  if(check && (input->feature_size != target->feature_size || !octree_equal_trees_cpu(input, target))) {
    printf("[ERROR] mse_loss - tree structure of inputs do not match\n");
    exit(-1);
  }

  const int n_blocks = octree_num_blocks(input);
  const int feature_size = input->feature_size;

  ot_data_t output = 0;

  #pragma omp parallel for
  for(int grid_idx = 0; grid_idx < n_blocks; ++grid_idx) {
    const ot_tree_t* tree = octree_get_tree(input, grid_idx);
    // const ot_data_t* data_in = input->data_ptrs[grid_idx];
    const ot_data_t* data_in = octree_get_data(input, grid_idx);
    // const ot_data_t* data_ta = target->data_ptrs[grid_idx];
    const ot_data_t* data_ta = octree_get_data(target, grid_idx);

    ot_data_t grid_out = 0;

    if(!tree_isset_bit(tree, 0)) {
      for(int f = 0; f < feature_size; ++f) {
        ot_data_t z = data_in[f] - data_ta[f];
        grid_out += 8*8*8 * z*z;
      }
    }
    else {

      for(int bit_idx_l1 = 1; bit_idx_l1 < 9; ++bit_idx_l1) {
        if(!tree_isset_bit(tree, bit_idx_l1)) {
          int data_idx = tree_data_idx(tree, bit_idx_l1, feature_size);
          for(int f = 0; f < feature_size; ++f) {
            ot_data_t z = data_in[data_idx + f] - data_ta[data_idx + f];
            grid_out += 4*4*4 * z*z;
          }
        }
        else {

          for(int add_bit_idx_l2 = 0; add_bit_idx_l2 < 8; ++add_bit_idx_l2) {
            int bit_idx_l2 = tree_child_bit_idx(bit_idx_l1) + add_bit_idx_l2;
            if(!tree_isset_bit(tree, bit_idx_l2)) {
              int data_idx = tree_data_idx(tree, bit_idx_l2, feature_size);
              for(int f = 0; f < feature_size; ++f) {
                ot_data_t z = data_in[data_idx + f] - data_ta[data_idx + f];
                grid_out += 2*2*2 * z*z;
              }
            }
            else {

              for(int add_bit_idx_l3 = 0; add_bit_idx_l3 < 8; ++add_bit_idx_l3) {
                int bit_idx_l3 = tree_child_bit_idx(bit_idx_l2) + add_bit_idx_l3;
                int data_idx = tree_data_idx(tree, bit_idx_l3, feature_size);
                for(int f = 0; f < feature_size; ++f) {
                  ot_data_t z = data_in[data_idx + f] - data_ta[data_idx + f];
                  grid_out += z*z;
                }
              }

            }
          }

        }
      }
    }

    #pragma omp atomic
    output += grid_out;
  }

  if(size_average) {
    output = output / (n_blocks * feature_size * 8 * 8 * 8);
  }
  return output;
}



extern "C"
void octree_mse_loss_bwd_cpu(const octree* input, const octree* target, bool size_average, bool check, octree* grad) {
  if(check && (input->feature_size != target->feature_size || !octree_equal_trees_cpu(input, target))) {
    printf("[ERROR] mse_loss_bwd - tree structure of inputs do not match\n");
    exit(-1);
  }

  octree_cpy_scalars(input, grad);
  octree_resize_as_cpu(input, grad);
  octree_cpy_trees_cpu_cpu(input, grad);
  octree_cpy_prefix_leafs_cpu_cpu(input, grad);

  const int n_blocks = octree_num_blocks(input);
  const int feature_size = input->feature_size;

  ot_data_t norm = 2.0;
  if(size_average) {
    norm = norm / (n_blocks * feature_size * 8 * 8 * 8);
  }

  #pragma omp parallel for
  for(int grid_idx = 0; grid_idx < n_blocks; ++grid_idx) {
    const ot_tree_t* tree = octree_get_tree(input, grid_idx);
    // const ot_data_t* data_in = input->data_ptrs[grid_idx];
    const ot_data_t* data_in = octree_get_data(input, grid_idx);
    // const ot_data_t* data_ta = target->data_ptrs[grid_idx];
    const ot_data_t* data_ta = octree_get_data(target, grid_idx);
    // ot_data_t* data_grad = grad->data_ptrs[grid_idx];
    ot_data_t* data_grad = octree_get_data(grad, grid_idx);

    if(!tree_isset_bit(tree, 0)) {
      for(int f = 0; f < feature_size; ++f) {
        ot_data_t z = data_in[f] - data_ta[f];
        data_grad[f] = 8*8*8 * norm * z;
      }
    }
    else {

      for(int bit_idx_l1 = 1; bit_idx_l1 < 9; ++bit_idx_l1) {
        if(!tree_isset_bit(tree, bit_idx_l1)) {
          int data_idx = tree_data_idx(tree, bit_idx_l1, feature_size);
          for(int f = 0; f < feature_size; ++f) {
            ot_data_t z = data_in[data_idx + f] - data_ta[data_idx + f];
            data_grad[data_idx + f] = 4*4*4 * norm * z;
          }
        }
        else {

          for(int add_bit_idx_l2 = 0; add_bit_idx_l2 < 8; ++add_bit_idx_l2) {
            int bit_idx_l2 = tree_child_bit_idx(bit_idx_l1) + add_bit_idx_l2;
            if(!tree_isset_bit(tree, bit_idx_l2)) {
              int data_idx = tree_data_idx(tree, bit_idx_l2, feature_size);
              for(int f = 0; f < feature_size; ++f) {
                ot_data_t z = data_in[data_idx + f] - data_ta[data_idx + f];
                data_grad[data_idx + f] = 2*2*2 * norm * z;
              }
            }
            else {

              for(int add_bit_idx_l3 = 0; add_bit_idx_l3 < 8; ++add_bit_idx_l3) {
                int bit_idx_l3 = tree_child_bit_idx(bit_idx_l2) + add_bit_idx_l3;
                int data_idx = tree_data_idx(tree, bit_idx_l3, feature_size);
                for(int f = 0; f < feature_size; ++f) {
                  ot_data_t z = data_in[data_idx + f] - data_ta[data_idx + f];
                  data_grad[data_idx + f] = norm * z;
                }
              }

            }
          }

        }
      }
    }
  }
}


inline void octree_nll_loss_vx(const ot_data_t* input, const ot_data_t* target, const ot_data_t* weights, const int feature_size, const int class_base, const int data_idx, const int vxs, ot_data_t* out, ot_data_t* total_weight) {
  int cur_target = round(target[data_idx]) - class_base;
  if(cur_target < 0 || cur_target >= feature_size) { printf("ERROR: target out of range %d\n", cur_target); exit(-1); }
  ot_data_t weight = vxs*vxs*vxs * weights[cur_target];

  out[0] -= weight * input[data_idx * feature_size + cur_target];
  total_weight[0] += weight;
}

extern "C"
void octree_nll_loss_cpu(const octree* input, const octree* target, const ot_data_t* weights, int class_base, bool size_average, bool check, ot_data_t* output, ot_data_t* total_weight) {
  if(check && (1 != target->feature_size || !octree_equal_trees_cpu(input, target))) {
    printf("[ERROR] nll_loss - tree structure of inputs do not match\n");
    exit(-1);
  }

  const int n_blocks = octree_num_blocks(input);
  const int feature_size = input->feature_size;

  output[0] = 0;
  total_weight[0] = 0;

  #pragma omp parallel for
  for(int grid_idx = 0; grid_idx < n_blocks; ++grid_idx) {
    const ot_tree_t* tree = octree_get_tree(input, grid_idx);
    const ot_data_t* data_in = octree_get_data(input, grid_idx);
    const ot_data_t* data_ta = octree_get_data(target, grid_idx);

    ot_data_t grid_out = 0;
    ot_data_t grid_weight = 0;

    if(!tree_isset_bit(tree, 0)) {
      octree_nll_loss_vx(data_in, data_ta, weights, feature_size, class_base, 0, 8, &grid_out, &grid_weight);
    }
    else {

      for(int bit_idx_l1 = 1; bit_idx_l1 < 9; ++bit_idx_l1) {
        if(!tree_isset_bit(tree, bit_idx_l1)) {
          int data_idx = tree_data_idx(tree, bit_idx_l1, 1);
          octree_nll_loss_vx(data_in, data_ta, weights, feature_size, class_base, data_idx, 4, &grid_out, &grid_weight);
        }
        else {

          for(int add_bit_idx_l2 = 0; add_bit_idx_l2 < 8; ++add_bit_idx_l2) {
            int bit_idx_l2 = tree_child_bit_idx(bit_idx_l1) + add_bit_idx_l2;
            if(!tree_isset_bit(tree, bit_idx_l2)) {
              int data_idx = tree_data_idx(tree, bit_idx_l2, 1);
              octree_nll_loss_vx(data_in, data_ta, weights, feature_size, class_base, data_idx, 2, &grid_out, &grid_weight);
            }
            else {

              for(int add_bit_idx_l3 = 0; add_bit_idx_l3 < 8; ++add_bit_idx_l3) {
                int bit_idx_l3 = tree_child_bit_idx(bit_idx_l2) + add_bit_idx_l3;
                int data_idx = tree_data_idx(tree, bit_idx_l3, 1);
                octree_nll_loss_vx(data_in, data_ta, weights, feature_size, class_base, data_idx, 1, &grid_out, &grid_weight);
              }

            }
          }

        }
      }
    }

    #pragma omp atomic
    output[0] += grid_out;
    #pragma omp atomic
    total_weight[0] += grid_weight;
  }

  if(size_average && total_weight[0] != 0) {
    output[0] /= total_weight[0];
  }
}


extern "C"
void octree_nll_loss_bwd_cpu(const octree* input, const octree* target, const ot_data_t* weights, const ot_data_t total_weight, int class_base, bool size_average, bool check, octree* grad) {
  if(check && (1 != target->feature_size || !octree_equal_trees_cpu(input, target))) {
    printf("[ERROR] nll_loss_bwd - tree structure of inputs do not match\n");
    exit(-1);
  }

  octree_cpy_scalars(input, grad);
  octree_resize_as_cpu(input, grad);
  octree_cpy_trees_cpu_cpu(input, grad);
  octree_cpy_prefix_leafs_cpu_cpu(input, grad);

  octree_fill_data_cpu(grad, 0);

  const int n_blocks = octree_num_blocks(input);
  const int feature_size = input->feature_size;

  ot_data_t norm = 1.0;
  if(size_average) {
    norm /= total_weight;
  }

  #pragma omp parallel for
  for(int grid_idx = 0; grid_idx < n_blocks; ++grid_idx) {
    const ot_tree_t* tree = octree_get_tree(input, grid_idx);
    // const ot_data_t* data_in = input->data_ptrs[grid_idx];
    // const ot_data_t* data_in = octree_get_data(input, grid_idx);
    // const ot_data_t* data_ta = target->data_ptrs[grid_idx];
    const ot_data_t* data_ta = octree_get_data(target, grid_idx);
    // ot_data_t* data_grad = grad->data_ptrs[grid_idx];
    ot_data_t* data_grad = octree_get_data(grad, grid_idx);

    if(!tree_isset_bit(tree, 0)) {
      int data_idx = 0;

      int cur_target = round(data_ta[data_idx]) - class_base;
      if(cur_target < 0 || cur_target >= feature_size) { printf("ERROR: target out of range %d\n", cur_target); exit(-1); }
      data_grad[data_idx * feature_size + cur_target] = 8*8*8 * -weights[cur_target] * norm;
    }
    else {

      for(int bit_idx_l1 = 1; bit_idx_l1 < 9; ++bit_idx_l1) {
        if(!tree_isset_bit(tree, bit_idx_l1)) {
          int data_idx = tree_data_idx(tree, bit_idx_l1, 1);

          int cur_target = round(data_ta[data_idx]) - class_base;
          if(cur_target < 0 || cur_target >= feature_size) { printf("ERROR: target out of range %d\n", cur_target); exit(-1); }
          data_grad[data_idx * feature_size + cur_target] = 4*4*4 * -weights[cur_target] * norm;
        }
        else {

          for(int add_bit_idx_l2 = 0; add_bit_idx_l2 < 8; ++add_bit_idx_l2) {
            int bit_idx_l2 = tree_child_bit_idx(bit_idx_l1) + add_bit_idx_l2;
            if(!tree_isset_bit(tree, bit_idx_l2)) {
              int data_idx = tree_data_idx(tree, bit_idx_l2, 1);

              int cur_target = round(data_ta[data_idx]) - class_base;
              if(cur_target < 0 || cur_target >= feature_size) { printf("ERROR: target out of range %d\n", cur_target); exit(-1); }
              data_grad[data_idx * feature_size + cur_target] = 2*2*2 * -weights[cur_target] * norm;
            }
            else {

              for(int add_bit_idx_l3 = 0; add_bit_idx_l3 < 8; ++add_bit_idx_l3) {
                int bit_idx_l3 = tree_child_bit_idx(bit_idx_l2) + add_bit_idx_l3;
                int data_idx = tree_data_idx(tree, bit_idx_l3, 1);

                int cur_target = round(data_ta[data_idx]) - class_base;
                if(cur_target < 0 || cur_target >= feature_size) { printf("ERROR: target out of range %d\n", cur_target); exit(-1); }
                data_grad[data_idx * feature_size + cur_target] = -weights[cur_target] * norm;
              }

            }
          }

        }
      }
    }

  }
}



#define EPS 1e-12

extern "C"
void octree_bce_loss_cpu(const octree* input, const octree* target, bool size_average, bool check, ot_data_t* output, ot_data_t* total_weight) {
  if(!octree_equal_shape(input, target) || (check && !octree_equal_trees_cpu(input, target))) {
    printf("[ERROR] bce_loss - tree structure of inputs do not match\n");
    exit(-1);
  }

  const int n_blocks = octree_num_blocks(input);
  const int feature_size = input->feature_size;
  
  *output = 0;

  #pragma omp parallel for
  for(int leaf_idx = 0; leaf_idx < input->n_leafs; ++leaf_idx) {
    int grid_idx = leaf_idx_to_grid_idx(input, leaf_idx);
    const ot_tree_t* tree = octree_get_tree(input, grid_idx);

    int cum_n_leafs = input->prefix_leafs[grid_idx];
    int data_idx = leaf_idx - cum_n_leafs;
    int bit_idx = data_idx_to_bit_idx(tree, data_idx);

    int n,d,h,w;
    int depth = octree_ind_to_dense_ind(input, grid_idx, bit_idx, &n, &d,&h,&w);
    int width = width_from_depth(depth);
    
    ot_data_t grid_out = 0;
    for(int f = 0; f < feature_size; ++f) {
      ot_data_t x = input->data[leaf_idx * input->feature_size + f]; 
      ot_data_t y = target->data[leaf_idx * input->feature_size + f];
      grid_out += width*width*width * (log(x + EPS) * y + log(1. - x + EPS) * (1. - y));
    }

    #pragma omp atomic
    *output -= grid_out;
  }

  if(size_average) {
    *total_weight = octree_num_blocks(input) * input->feature_size * 8 * 8 * 8;
    *output = (*output) / (*total_weight) ;
  }
  else {
    *total_weight = 1;
  }
}



extern "C"
void octree_bce_loss_bwd_cpu(const octree* input, const octree* target, bool size_average, bool check, octree* grad) {
  if(!octree_equal_shape(input, target) || (check && !octree_equal_trees_cpu(input, target))) {
    printf("[ERROR] bce_loss_bwd - tree structure of inputs do not match\n");
    exit(-1);
  }

  octree_cpy_scalars(input, grad);
  octree_resize_as_cpu(input, grad);
  octree_cpy_trees_cpu_cpu(input, grad);
  octree_cpy_prefix_leafs_cpu_cpu(input, grad);

  const int n_blocks = octree_num_blocks(input);
  const int feature_size = input->feature_size;

  ot_data_t norm = 1.0;
  if(size_average) {
    norm = norm / (n_blocks * feature_size * 8 * 8 * 8);
  }
  
  #pragma omp parallel for
  for(int leaf_idx = 0; leaf_idx < input->n_leafs; ++leaf_idx) {
    int grid_idx = leaf_idx_to_grid_idx(input, leaf_idx);
    const ot_tree_t* tree = octree_get_tree(input, grid_idx);

    int cum_n_leafs = input->prefix_leafs[grid_idx];
    int data_idx = leaf_idx - cum_n_leafs;
    int bit_idx = data_idx_to_bit_idx(tree, data_idx);

    int n,d,h,w;
    int depth = octree_ind_to_dense_ind(input, grid_idx, bit_idx, &n, &d,&h,&w);
    int width = width_from_depth(depth);
    
    for(int f = 0; f < feature_size; ++f) {
      ot_data_t x = input->data[leaf_idx * input->feature_size + f]; 
      ot_data_t y = target->data[leaf_idx * input->feature_size + f];
      grad->data[leaf_idx * grad->feature_size + f] = - width*width*width * norm * (y - x) / ((1. - x + EPS) * (x + EPS));
    }
  }
}



extern "C"
void octree_bce_dense_loss_cpu(const octree* input, const ot_data_t* target, bool size_average, ot_data_t* output, ot_data_t* total_weight) {
  const int n_blocks = octree_num_blocks(input);

  const int dense_depth = 8 * input->grid_depth;
  const int dense_height = 8 * input->grid_height;
  const int dense_width = 8 * input->grid_width;
  const int feature_size = input->feature_size;
  
  *output = 0;

  #pragma omp parallel for
  for(int leaf_idx = 0; leaf_idx < input->n_leafs; ++leaf_idx) {
    int grid_idx = leaf_idx_to_grid_idx(input, leaf_idx);
    const ot_tree_t* tree = octree_get_tree(input, grid_idx);

    int cum_n_leafs = input->prefix_leafs[grid_idx];
    int data_idx = leaf_idx - cum_n_leafs;
    int bit_idx = data_idx_to_bit_idx(tree, data_idx);

    int n,ds,hs,ws;
    int depth = octree_ind_to_dense_ind(input, grid_idx, bit_idx, &n, &ds,&hs,&ws);
    int width = width_from_depth(depth);

    ot_data_t grid_out = 0;
    for(int d = ds; d < (ds+width); ++d) {
      for(int h = hs; h < (hs+width); ++h) {
        for(int w = ws; w < (ws+width); ++w) {
          for(int f = 0; f < feature_size; ++f) {
            ot_data_t x = input->data[leaf_idx * input->feature_size + f];
            ot_data_t y = target[(((n * feature_size + f) * dense_depth + d) * dense_height + h) * dense_width + w];
            grid_out += (log(x + EPS) * y + log(1. - x + EPS) * (1. - y));
          }
        }
      }
    }
    
    #pragma omp atomic
    *output -= grid_out;
  }

  if(size_average) {
    *total_weight = octree_num_blocks(input) * input->feature_size * 8 * 8 * 8;
    *output = (*output) / (*total_weight) ;
  }
  else {
    *total_weight = 1;
  }
}


extern "C"
void octree_bce_dense_loss_bwd_cpu(const octree* input, const ot_data_t* target, bool size_average, octree* grad) {
  octree_cpy_scalars(input, grad);
  octree_resize_as_cpu(input, grad);
  octree_cpy_trees_cpu_cpu(input, grad);
  octree_cpy_prefix_leafs_cpu_cpu(input, grad);

  const int n_blocks = octree_num_blocks(input);
  
  const int dense_depth = 8 * input->grid_depth;
  const int dense_height = 8 * input->grid_height;
  const int dense_width = 8 * input->grid_width;
  const int feature_size = input->feature_size;

  ot_data_t norm = 1.0;
  if(size_average) {
    norm = norm / (n_blocks * feature_size * 8 * 8 * 8);
  }

  #pragma omp parallel for
  for(int leaf_idx = 0; leaf_idx < input->n_leafs; ++leaf_idx) {
    int grid_idx = leaf_idx_to_grid_idx(input, leaf_idx);
    const ot_tree_t* tree = octree_get_tree(input, grid_idx);

    int cum_n_leafs = input->prefix_leafs[grid_idx];
    int data_idx = leaf_idx - cum_n_leafs;
    int bit_idx = data_idx_to_bit_idx(tree, data_idx);

    int n,ds,hs,ws;
    int depth = octree_ind_to_dense_ind(input, grid_idx, bit_idx, &n, &ds,&hs,&ws);
    int width = width_from_depth(depth);

    for(int f = 0; f < feature_size; ++f) {
      grad->data[leaf_idx * grad->feature_size + f] = 0;
    }
    
    for(int d = ds; d < (ds+width); ++d) {
      for(int h = hs; h < (hs+width); ++h) {
        for(int w = ws; w < (ws+width); ++w) {
          for(int f = 0; f < feature_size; ++f) {
            ot_data_t x = input->data[leaf_idx * input->feature_size + f];
            ot_data_t y = target[(((n * feature_size + f) * dense_depth + d) * dense_height + h) * dense_width + w];
            grad->data[leaf_idx * grad->feature_size + f] -= norm * (y - x) / ((1. - x + EPS) * (x + EPS));
          }
        }
      }
    }
  }
}


extern "C"
void octree_bce_ds_loss_cpu(const octree* input, const octree* target, const octree* weights, bool size_average, ot_data_t* output, ot_data_t* total_weight) {
  if(!octree_equal_shape(input, target)) {
    printf("[ERROR] bce_ds_loss - shape of inputs do not match\n");
    exit(-1);
  }

  const int n_blocks = octree_num_blocks(input);
  const int feature_size = input->feature_size;
  
  *output = 0;
  *total_weight = 0;

  #pragma omp parallel for
  for(int leaf_idx = 0; leaf_idx < input->n_leafs; ++leaf_idx) {
    int in_grid_idx = leaf_idx_to_grid_idx(input, leaf_idx);
    const ot_tree_t* in_tree = octree_get_tree(input, in_grid_idx);

    int in_data_idx = leaf_idx - input->prefix_leafs[in_grid_idx];
    int in_bit_idx = data_idx_to_bit_idx(in_tree, in_data_idx);

    int n,ds,hs,ws;
    int depth = octree_ind_to_dense_ind(input, in_grid_idx, in_bit_idx, &n, &ds,&hs,&ws);
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
          int ta_grid_idx = octree_grid_idx(target, n, gd,gh,gw);
          const ot_tree_t* ta_tree = octree_get_tree(target, ta_grid_idx);
          int ta_bit_idx = tree_bit_idx(ta_tree, bd,bh,bw);
          int ta_data_idx = tree_data_idx(ta_tree, ta_bit_idx, target->feature_size);
          const ot_data_t* ta_data = octree_get_data(target, ta_grid_idx);
          const ot_data_t* we_data = weights != 0 ? octree_get_data(weights, ta_grid_idx) : 0;
          for(int f = 0; f < feature_size; ++f) {
            ot_data_t x = input->data[leaf_idx * input->feature_size + f];
            ot_data_t y = ta_data[ta_data_idx + f];
            ot_data_t w = we_data != 0 ? we_data[ta_data_idx + f] : 1;
            grid_out += w * (log(x + EPS) * y + log(1. - x + EPS) * (1. - y));
            grid_weight += w;
          }
        }
      }
    }

    #pragma omp atomic
    *output -= grid_out;
    #pragma omp atomic
    *total_weight += grid_weight;
  }

  if(size_average) {
    *output = (*output) / (*total_weight) ;
  }
  else {
    *total_weight = 1;
  }
}

extern "C"
void octree_bce_ds_loss_bwd_cpu(const octree* input, const octree* target, const octree* weights, bool size_average, ot_data_t total_weight, octree* grad) {
  if(!octree_equal_shape(input, target)) {
    printf("[ERROR] bce_ds_loss_bwd - shape of inputs do not match\n");
    exit(-1);
  }

  octree_cpy_scalars(input, grad);
  octree_resize_as_cpu(input, grad);
  octree_cpy_trees_cpu_cpu(input, grad);
  octree_cpy_prefix_leafs_cpu_cpu(input, grad);

  const int n_blocks = octree_num_blocks(input);
  const int feature_size = input->feature_size;

  ot_data_t norm = 1.0;
  if(size_average && total_weight > 0) {
    norm = norm / total_weight;
  }
  
  #pragma omp parallel for
  for(int leaf_idx = 0; leaf_idx < input->n_leafs; ++leaf_idx) {
    int in_grid_idx = leaf_idx_to_grid_idx(input, leaf_idx);
    const ot_tree_t* in_tree = octree_get_tree(input, in_grid_idx);

    int in_data_idx = leaf_idx - input->prefix_leafs[in_grid_idx];
    int in_bit_idx = data_idx_to_bit_idx(in_tree, in_data_idx);

    int n,ds,hs,ws;
    int depth = octree_ind_to_dense_ind(input, in_grid_idx, in_bit_idx, &n, &ds,&hs,&ws);
    int width = width_from_depth(depth);
    
    for(int f = 0; f < feature_size; ++f) {
      grad->data[leaf_idx * grad->feature_size + f] = 0;
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
          int ta_grid_idx = octree_grid_idx(target, n, gd,gh,gw);
          const ot_tree_t* ta_tree = octree_get_tree(target, ta_grid_idx);
          int ta_bit_idx = tree_bit_idx(ta_tree, bd,bh,bw);
          int ta_data_idx = tree_data_idx(ta_tree, ta_bit_idx, target->feature_size);
          const ot_data_t* ta_data = octree_get_data(target, ta_grid_idx);
          const ot_data_t* we_data = weights != 0 ? octree_get_data(weights, ta_grid_idx) : 0;
          for(int f = 0; f < feature_size; ++f) {
            ot_data_t x = input->data[leaf_idx * input->feature_size + f];
            ot_data_t y = ta_data[ta_data_idx + f];
            ot_data_t w = we_data != 0 ? we_data[ta_data_idx + f] : 1;
            grad->data[leaf_idx * grad->feature_size + f] -= norm * w * (y - x) / ((1. - x + EPS) * (x + EPS));
          }
        }
      }
    }
  }
}
