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

#include "octnet/cpu/pool.h"
#include "octnet/cpu/cpu.h"

#include <cstdio>
#include <cstdlib>

template <int level>
void octree_pool2x2x2_struct_cpu(const octree* in, octree* out) {
  const int n_blocks = octree_num_blocks(in);

  #pragma omp parallel for
  for(int grid_idx = 0; grid_idx < n_blocks; ++grid_idx) {
    ot_tree_t* in_tree = octree_get_tree(in, grid_idx);
    ot_tree_t* out_tree = octree_get_tree(out, grid_idx);
  
    if(level == 0) {
      if(tree_isset_bit(in_tree, 0) && tree_cnt1(in_tree, 1, 9) == 0) {
        tree_unset_bit(out_tree, 0);
      }
    }

    if(level == 1) {
      if(tree_isset_bit(in_tree, 0)) {
        for(int bit_idx_l1 = 1; bit_idx_l1 < 9; ++bit_idx_l1) {
          int bit_idx_l2 = tree_child_bit_idx(bit_idx_l1);
          if(tree_isset_bit(in_tree, bit_idx_l1) && tree_cnt1(in_tree, bit_idx_l2, bit_idx_l2+8) == 0) {
            tree_unset_bit(out_tree, bit_idx_l1);
          }
        }
      }
    }

    if(level == 2) {
      if(tree_isset_bit(in_tree, 0)) {
        for(int bit_idx_l1 = 1; bit_idx_l1 < 9; ++bit_idx_l1) {
          if(tree_isset_bit(in_tree, bit_idx_l1)) {
            int bit_idx_l2 = tree_child_bit_idx(bit_idx_l1);
            for(int idx = 0; idx < 8; ++idx) {
              tree_unset_bit(out_tree, bit_idx_l2);
              bit_idx_l2++;
            }
          }
        }
      }
    }

  }
}


template <int pool_fcn>
void octree_pool2x2x2_data_cpu(const octree* in, octree* out) {
  const int n_blocks = octree_num_blocks(in);
  const ot_size_t feature_size = in->feature_size;

  #pragma omp parallel for
  for(int grid_idx = 0; grid_idx < n_blocks; ++grid_idx) {
    ot_tree_t* out_tree = octree_get_tree(out, grid_idx);
    // ot_data_t* out_data = out->data_ptrs[grid_idx];
    ot_data_t* out_data = octree_get_data(out, grid_idx);

    ot_tree_t* in_tree = octree_get_tree(in, grid_idx);
    // ot_data_t* in_data = in->data_ptrs[grid_idx];
    ot_data_t* in_data = octree_get_data(in, grid_idx);

    if(tree_isset_bit(in_tree, 0)) {
      if(!tree_isset_bit(out_tree, 0)) {
        octree_pool2x2x2<pool_fcn>(in_data, feature_size, out_data);
      }
      else {

        for(int bit_idx_l1 = 1; bit_idx_l1 < 9; ++bit_idx_l1) {
          int out_data_idx_l1 = tree_data_idx(out_tree, bit_idx_l1, feature_size);
          if(tree_isset_bit(in_tree, bit_idx_l1)) {
            if(!tree_isset_bit(out_tree, bit_idx_l1)) {
              int in_data_idx = tree_data_idx(in_tree, tree_child_bit_idx(bit_idx_l1), feature_size);
              octree_pool2x2x2<pool_fcn>(in_data + in_data_idx, feature_size, out_data + out_data_idx_l1);
            }
            else {

              for(int idx_l2 = 0; idx_l2 < 8; ++idx_l2) {
                int bit_idx_l2 = tree_child_bit_idx(bit_idx_l1) + idx_l2;
                int out_data_idx_l2 = tree_data_idx(out_tree, bit_idx_l2, feature_size);
                if(tree_isset_bit(in_tree, bit_idx_l2)) {
                  if(!tree_isset_bit(out_tree, bit_idx_l2)) {
                    int in_data_idx = tree_data_idx(in_tree, tree_child_bit_idx(bit_idx_l2), feature_size);
                    octree_pool2x2x2<pool_fcn>(in_data + in_data_idx, feature_size, out_data + out_data_idx_l2);
                  }
                  else {
                    
                    int bit_idx_l3 = tree_child_bit_idx(bit_idx_l2);
                    int out_data_idx_l3 = tree_data_idx(out_tree, bit_idx_l3, feature_size);
                    int in_data_idx_l3 = tree_data_idx(in_tree, bit_idx_l3, feature_size);
                    octree_cpy_leaf(in_data + in_data_idx_l3, 8*feature_size, out_data + out_data_idx_l3);

                  }
                }
                else {
                  int in_data_idx = tree_data_idx(in_tree, bit_idx_l2, feature_size);
                  octree_cpy_leaf(in_data + in_data_idx, feature_size, out_data + out_data_idx_l2);
                }

              }

            }
          }
          else {
            int in_data_idx = tree_data_idx(in_tree, bit_idx_l1, feature_size);
            octree_cpy_leaf(in_data + in_data_idx, feature_size, out_data + out_data_idx_l1);
          }
        }

      }
    }
    else {
      octree_cpy_leaf(in_data, feature_size, out_data);
    }

  } 

}


template <int pool_fcn>
void octree_pool2x2x2_cpu(const octree* in, bool level_0, bool level_1, bool level_2, octree* out) {
  octree_resize_cpu(in->n, in->grid_depth, in->grid_height, in->grid_width, in->feature_size, 0, out);
  octree_cpy_trees_cpu_cpu(in, out);

  if(level_0) {
    octree_pool2x2x2_struct_cpu<0>(in, out);
  }
  if(level_1) {
    octree_pool2x2x2_struct_cpu<1>(in, out);
  }
  if(level_2) {
    octree_pool2x2x2_struct_cpu<2>(in, out);
  }

  octree_upd_n_leafs_cpu(out);
  octree_resize_as_cpu(out, out);
  octree_upd_prefix_leafs_cpu(out);

  octree_pool2x2x2_data_cpu<pool_fcn>(in, out);
}



void octree_pool2x2x2_avg_cpu(const octree* in, bool level_0, bool level_1, bool level_2, octree* out) {
  octree_pool2x2x2_cpu<REDUCE_AVG>(in, level_0, level_1, level_2, out);
}

void octree_pool2x2x2_max_cpu(const octree* in, bool level_0, bool level_1, bool level_2, octree* out) {
  octree_pool2x2x2_cpu<REDUCE_MAX>(in, level_0, level_1, level_2, out);
}




template <int pool_fcn>
void octree_pool2x2x2_bwd_cpu(const octree* in, const octree* grad_out, octree* grad_in) {
  octree_resize_as_cpu(in, grad_in);
  octree_cpy_trees_cpu_cpu(in, grad_in);
  octree_cpy_prefix_leafs_cpu_cpu(in, grad_in);

  int n_blocks = octree_num_blocks(in);
  ot_size_t feature_size = in->feature_size;

  #pragma omp parallel for
  for(int grid_idx = 0; grid_idx < n_blocks; ++grid_idx) {
    ot_tree_t* out_tree = octree_get_tree(grad_out, grid_idx);
    // ot_data_t* grad_out_data = grad_out->data_ptrs[grid_idx];
    ot_data_t* grad_out_data = octree_get_data(grad_out, grid_idx);

    ot_tree_t* in_tree = octree_get_tree(in, grid_idx);
    // ot_data_t* in_data = in->data_ptrs[grid_idx];
    ot_data_t* in_data = octree_get_data(in, grid_idx);
    
    // ot_data_t* grad_in_data = grad_in->data_ptrs[grid_idx];
    ot_data_t* grad_in_data = octree_get_data(grad_in, grid_idx);

    if(tree_isset_bit(in_tree, 0)) {
      if(!tree_isset_bit(out_tree, 0)) {
        octree_pool2x2x2_bwd<pool_fcn>(in_data, grad_out_data, feature_size, grad_in_data);
      }
      else {

        for(int bit_idx_l1 = 1; bit_idx_l1 < 9; ++bit_idx_l1) {
          int out_data_idx_l1 = tree_data_idx(out_tree, bit_idx_l1, feature_size);
          if(tree_isset_bit(in_tree, bit_idx_l1)) {
            if(!tree_isset_bit(out_tree, bit_idx_l1)) {
              int in_data_idx = tree_data_idx(in_tree, tree_child_bit_idx(bit_idx_l1), feature_size);
              octree_pool2x2x2_bwd<pool_fcn>(in_data + in_data_idx, grad_out_data + out_data_idx_l1, feature_size, grad_in_data + in_data_idx);
            }
            else {

              for(int idx_l2 = 0; idx_l2 < 8; ++idx_l2) {
                int bit_idx_l2 = tree_child_bit_idx(bit_idx_l1) + idx_l2;
                int out_data_idx_l2 = tree_data_idx(out_tree, bit_idx_l2, feature_size);
                if(tree_isset_bit(in_tree, bit_idx_l2)) {
                  if(!tree_isset_bit(out_tree, bit_idx_l2)) {
                    int in_data_idx = tree_data_idx(in_tree, tree_child_bit_idx(bit_idx_l2), feature_size);
                    octree_pool2x2x2_bwd<pool_fcn>(in_data + in_data_idx, grad_out_data + out_data_idx_l2, feature_size, grad_in_data + in_data_idx);
                  }
                  else {
                    
                    int bit_idx_l3 = tree_child_bit_idx(bit_idx_l2);
                    int out_data_idx_l3 = tree_data_idx(out_tree, bit_idx_l3, feature_size);
                    int in_data_idx_l3 = tree_data_idx(in_tree, bit_idx_l3, feature_size);
                    octree_cpy_leaf(grad_out_data + out_data_idx_l3, 8*feature_size, grad_in_data + in_data_idx_l3);

                  }
                }
                else {
                  int in_data_idx = tree_data_idx(in_tree, bit_idx_l2, feature_size);
                  octree_cpy_leaf(grad_out_data + out_data_idx_l2, feature_size, grad_in_data + in_data_idx);
                }

              }

            }
          }
          else {
            int in_data_idx = tree_data_idx(in_tree, bit_idx_l1, feature_size);
            octree_cpy_leaf(grad_out_data + out_data_idx_l1, feature_size, grad_in_data + in_data_idx);
          }
        }

      }
    }
    else {
      octree_cpy_leaf(grad_out_data, feature_size, grad_in_data);
    }

  } 
}

void octree_pool2x2x2_avg_bwd_cpu(const octree* in, const octree* grad_out, octree* grad_in) {
  octree_pool2x2x2_bwd_cpu<REDUCE_AVG>(in, grad_out, grad_in);
}

void octree_pool2x2x2_max_bwd_cpu(const octree* in, const octree* grad_out, octree* grad_in) {
  octree_pool2x2x2_bwd_cpu<REDUCE_MAX>(in, grad_out, grad_in);
}
