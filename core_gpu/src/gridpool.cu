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

#include "octnet/gpu/pool.h"
#include "octnet/gpu/gpu.h"

#include <cstdlib>

__global__ void kernel_gridpool2x2x2_struct(octree out, int n_blocks, ot_size_t feature_size, const octree in) {
  CUDA_KERNEL_LOOP(out_grid_idx, n_blocks) {
    ot_tree_t* otree = octree_get_tree(&out, out_grid_idx);

    int gn,ogd,ogh,ogw;
    octree_split_grid_idx(&out, out_grid_idx, &gn, &ogd, &ogh, &ogw); 

    // first bit is always set, because out block consists of 8 in blocks
    tree_set_bit(otree, 0); 

    int obit_idx_l1 = 1;
    for(int dgd = 0; dgd < 2; ++dgd) {
      for(int hgh = 0; hgh < 2; ++hgh) {
        for(int wgw = 0; wgw < 2; ++wgw) {
          int igd = 2*ogd + dgd;
          int igh = 2*ogh + hgh;
          int igw = 2*ogw + wgw;
          int in_grid_idx = octree_grid_idx(&in, gn, igd, igh, igw);
          ot_tree_t* itree = octree_get_tree(&in, in_grid_idx);

          //check if first bit in in blocks is set
          if(tree_isset_bit(itree, 0)) {
            tree_set_bit(otree, obit_idx_l1);

            int obit_idx_l2 = tree_child_bit_idx(obit_idx_l1);
            for(int ibit_idx_l1 = 1; ibit_idx_l1 < 9; ++ibit_idx_l1) {
              //check if l1 bits are set in in blocks
              if(tree_isset_bit(itree, ibit_idx_l1)) {
                tree_set_bit(otree, obit_idx_l2);
              }
              obit_idx_l2++;
            }
          }
          obit_idx_l1++;
        }
      }
    }
  }
}

template <int pool_fcn>
__global__ void kernel_gridpool2x2x2_data(octree out, int n_leafs, const octree in) {
  
  CUDA_KERNEL_LOOP(leaf_idx, n_leafs) {
    const int out_grid_idx = out.data[leaf_idx * out.feature_size];
    const ot_tree_t* out_tree = octree_get_tree(&out, out_grid_idx);

    // const int cum_n_leafs = n_leafs_upto(&out, out_grid_idx);
    const int cum_n_leafs = out.prefix_leafs[out_grid_idx];
    const int out_data_idx = leaf_idx - cum_n_leafs;
    const int out_bit_idx = data_idx_to_bit_idx(out_tree, out_data_idx);
    // ot_data_t* out_data = out.data_ptrs[out_grid_idx] + out_data_idx * out.feature_size;
    ot_data_t* out_data = octree_get_data(&out, out_grid_idx) + out_data_idx * out.feature_size;

    const int depth = depth_from_bit_idx(out_bit_idx);

    int gn,gd,gh,gw;
    octree_split_grid_idx(&out, out_grid_idx, &gn, &gd, &gh, &gw); 
    int bd = 0;
    int bh = 0;
    int bw = 0;
    if(depth == 1) {
      bdhw_from_idx_l1(out_bit_idx, &bd,&bh,&bw);
    }
    else if(depth == 2) {
      bdhw_from_idx_l2(out_bit_idx, &bd,&bh,&bw);
    }
    else if(depth == 3) {
      bdhw_from_idx_l3(out_bit_idx, &bd,&bh,&bw);
    }
    const int in_gd = (gd * 2) + (bd > 3);
    const int in_gh = (gh * 2) + (bh > 3);
    const int in_gw = (gw * 2) + (bw > 3);
    
    const int in_grid_idx = octree_grid_idx(&in, gn,in_gd,in_gh,in_gw);
    // printf("  in_grid_idx %d <= %d,%d,%d, %d,%d,%d, %d,%d,%d\n", in_grid_idx, in_gd,in_gh,in_gw, gd,gh,gw, bd,bh,bw);
    const ot_tree_t* in_tree = octree_get_tree(&in, in_grid_idx);
    int in_bit_idx = 0;
    if(depth == 2) {
      in_bit_idx = (out_bit_idx - tree_child_bit_idx(tree_parent_bit_idx(out_bit_idx))) + 1;
    }
    else if(depth == 3) {
      in_bit_idx = (out_bit_idx - tree_child_bit_idx(tree_parent_bit_idx(out_bit_idx))) + (tree_parent_bit_idx(out_bit_idx) - tree_child_bit_idx(tree_parent_bit_idx(tree_parent_bit_idx(out_bit_idx)))) * 8 + 9;
    }
    // printf("  leaf_idx %d, out_grid_idx %d, out_bit_idx %d, in_grid_idx %d (%d,%d,%d), in_bit_idx %d (%d,%d,%d)\n", leaf_idx, out_grid_idx, out_bit_idx, in_grid_idx, gd,gh,gw, in_bit_idx, bd,bh,bw);

    if(tree_isset_bit(in_tree, in_bit_idx)) {
      in_bit_idx = tree_child_bit_idx(in_bit_idx);
      // const ot_data_t* in_data = in.data_ptrs[in_grid_idx] + tree_data_idx(in_tree, in_bit_idx, in.feature_size);
      const ot_data_t* in_data = octree_get_data(&in, in_grid_idx) + tree_data_idx(in_tree, in_bit_idx, in.feature_size);
      octree_pool2x2x2<pool_fcn>(in_data, in.feature_size, out_data);
    }
    else {
      // const ot_data_t* in_data = in.data_ptrs[in_grid_idx] + tree_data_idx(in_tree, in_bit_idx, in.feature_size);
      const ot_data_t* in_data = octree_get_data(&in, in_grid_idx) + tree_data_idx(in_tree, in_bit_idx, in.feature_size);
      octree_cpy_leaf(in_data, in.feature_size, out_data);
    }
  }

}

template <int pool_fcn>
void octree_gridpool2x2x2_gpu(const octree* in, octree* out) {
  if(in->grid_depth % 2 != 0 || in->grid_height % 2 != 0 || in->grid_width % 2 != 0) {
    printf("[ERROR] octree_gridpool2x2x2_gpu grid dimension should be a multiply of 2 (are %d,%d,%d)\n", in->grid_depth, in->grid_height, in->grid_width);
    exit(-1);
  }
  if(in->grid_depth / 2 == 0 || in->grid_height / 2 == 0 || in->grid_width / 2 == 0) {
    printf("[ERROR] octree_gridpool2x2x2_gpu grid dimension have to be at least 2x2x2\n");
    exit(-1);
  }

  //copy scalars
  out->n = in->n;
  out->grid_depth = in->grid_depth / 2;
  out->grid_height = in->grid_height / 2;
  out->grid_width = in->grid_width / 2;
  out->feature_size = in->feature_size;

  int n_blocks = octree_num_blocks(out);
  int feature_size = in->feature_size;

  //compute out structure
  octree_resize_as_gpu(out, out);
  octree_clr_trees_gpu(out);

  kernel_gridpool2x2x2_struct<<<GET_BLOCKS(n_blocks), CUDA_NUM_THREADS>>>(
      *out, n_blocks, feature_size, *in
  );
  CUDA_POST_KERNEL_CHECK; 


  //pool/copy data
  octree_upd_n_leafs_gpu(out);
  octree_resize_as_gpu(out, out);
  octree_upd_prefix_leafs_gpu(out);

  octree_leaf_idx_to_grid_idx_gpu(out, out->feature_size, out->data_capacity, out->data);

  kernel_gridpool2x2x2_data<pool_fcn><<<GET_BLOCKS(out->n_leafs), CUDA_NUM_THREADS>>>(
      *out, out->n_leafs, *in
  );
  CUDA_POST_KERNEL_CHECK; 

}


template <int pool_fcn>
__global__ void kernel_gridpool2x2x2_bwd(octree grad_in, int n_leafs, const octree grad_out, const octree in) {
  
  CUDA_KERNEL_LOOP(leaf_idx, n_leafs) {
    // const int out_grid_idx = grad_in.trees[leaf_idx * N_TREE_INTS];
    const int out_grid_idx = leaf_idx_to_grid_idx(&grad_out, leaf_idx);
    const ot_tree_t* out_tree = octree_get_tree(&grad_out, out_grid_idx);

    // const int cum_n_leafs = n_leafs_upto(&grad_out, out_grid_idx);
    const int cum_n_leafs = grad_out.prefix_leafs[out_grid_idx];
    const int out_data_idx = leaf_idx - cum_n_leafs;
    const int out_bit_idx = data_idx_to_bit_idx(out_tree, out_data_idx);
    // const ot_data_t* grad_out_data = grad_out.data_ptrs[out_grid_idx] + out_data_idx * grad_out.feature_size;
    const ot_data_t* grad_out_data = octree_get_data(&grad_out, out_grid_idx) + out_data_idx * grad_out.feature_size;

    const int depth = depth_from_bit_idx(out_bit_idx);

    int gn,gd,gh,gw;
    octree_split_grid_idx(&grad_out, out_grid_idx, &gn, &gd, &gh, &gw); 
    int bd = 0;
    int bh = 0;
    int bw = 0;
    if(depth == 1) {
      bdhw_from_idx_l1(out_bit_idx, &bd,&bh,&bw);
    }
    else if(depth == 2) {
      bdhw_from_idx_l2(out_bit_idx, &bd,&bh,&bw);
    }
    else if(depth == 3) {
      bdhw_from_idx_l3(out_bit_idx, &bd,&bh,&bw);
    }
    const int in_gd = (gd * 2) + (bd > 3);
    const int in_gh = (gh * 2) + (bh > 3);
    const int in_gw = (gw * 2) + (bw > 3);
    
    const int in_grid_idx = octree_grid_idx(&in, gn,in_gd,in_gh,in_gw);
    const ot_tree_t* in_tree = octree_get_tree(&in, in_grid_idx);
    int in_bit_idx = 0;
    if(depth == 2) {
      in_bit_idx = (out_bit_idx - tree_child_bit_idx(tree_parent_bit_idx(out_bit_idx))) + 1;
    }
    else if(depth == 3) {
      in_bit_idx = (out_bit_idx - tree_child_bit_idx(tree_parent_bit_idx(out_bit_idx))) + (tree_parent_bit_idx(out_bit_idx) - tree_child_bit_idx(tree_parent_bit_idx(tree_parent_bit_idx(out_bit_idx)))) * 8 + 9;
    }

    if(tree_isset_bit(in_tree, in_bit_idx)) {
      in_bit_idx = tree_child_bit_idx(in_bit_idx);
      const int in_data_idx = tree_data_idx(in_tree, in_bit_idx, in.feature_size);
      // const ot_data_t* in_data = in.data_ptrs[in_grid_idx] + in_data_idx;
      const ot_data_t* in_data = octree_get_data(&in, in_grid_idx) + in_data_idx;
      // ot_data_t* grad_in_data = grad_in.data_ptrs[in_grid_idx] + in_data_idx;
      ot_data_t* grad_in_data = octree_get_data(&grad_in, in_grid_idx) + in_data_idx;
      octree_pool2x2x2_bwd<pool_fcn>(in_data, grad_out_data, in.feature_size, grad_in_data);
    }
    else {
      const int in_data_idx = tree_data_idx(in_tree, in_bit_idx, in.feature_size);
      // ot_data_t* grad_in_data = grad_in.data_ptrs[in_grid_idx] + in_data_idx;
      ot_data_t* grad_in_data = octree_get_data(&grad_in, in_grid_idx) + in_data_idx;
      octree_cpy_leaf(grad_out_data, in.feature_size, grad_in_data);
    }
  }

}

template <int pool_fcn>
void octree_gridpool2x2x2_bwd_gpu(const octree* in, const octree* grad_out, octree* grad_in) {
  octree_cpy_scalars(in, grad_in);
  octree_resize_as_gpu(in, grad_in);
  octree_cpy_trees_gpu_gpu(in, grad_in);
  octree_cpy_prefix_leafs_gpu_gpu(in, grad_in);
  
  int n_blocks = octree_num_blocks(grad_out);

  // octree_leaf_idx_to_grid_idx_gpu(grad_in, N_TREE_INTS, grad_in->trees);

  kernel_gridpool2x2x2_bwd<pool_fcn><<<GET_BLOCKS(grad_out->n_leafs), CUDA_NUM_THREADS>>>(
      *grad_in, grad_out->n_leafs, *grad_out, *in
  );
  CUDA_POST_KERNEL_CHECK; 

}



void octree_gridpool2x2x2_avg_gpu(const octree* in, octree* out) {
  if(DEBUG) { printf("[DEBUG] octree_gridpool2x2x2_avg_gpu\n"); }
  octree_gridpool2x2x2_gpu<REDUCE_AVG>(in, out);
}
void octree_gridpool2x2x2_max_gpu(const octree* in, octree* out){
  if(DEBUG) { printf("[DEBUG] octree_gridpool2x2x2_max_gpu\n"); }
  octree_gridpool2x2x2_gpu<REDUCE_MAX>(in, out);
}

void octree_gridpool2x2x2_avg_bwd_gpu(const octree* in, const octree* grad_out, octree* grad_in) {
  if(DEBUG) { printf("[DEBUG] octree_gridpool2x2x2_avg_bwd_gpu\n"); }
  octree_gridpool2x2x2_bwd_gpu<REDUCE_AVG>(in, grad_out, grad_in);
}
void octree_gridpool2x2x2_max_bwd_gpu(const octree* in, const octree* grad_out, octree* grad_in) {
  if(DEBUG) { printf("[DEBUG] octree_gridpool2x2x2_max_bwd_gpu\n"); }
  octree_gridpool2x2x2_bwd_gpu<REDUCE_AVG>(in, grad_out, grad_in);
}
