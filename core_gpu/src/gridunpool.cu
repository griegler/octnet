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

#include "octnet/gpu/unpool.h"
#include "octnet/gpu/gpu.h"

#include <cstdlib>


__global__ void kernel_gridunpool2x2x2_struct(octree out, int n_blocks, ot_size_t feature_size, const octree in) {
  CUDA_KERNEL_LOOP(out_grid_idx, n_blocks) {
    int gn,ogd,ogh,ogw;
    octree_split_grid_idx(&out, out_grid_idx, &gn, &ogd, &ogh, &ogw); 

    int igd = ogd / 2;
    int igh = ogh / 2;
    int igw = ogw / 2;
    int in_grid_idx = octree_grid_idx(&in, gn, igd, igh, igw);

    const ot_tree_t* in_tree = octree_get_tree(&in, in_grid_idx);
    ot_tree_t* out_tree = octree_get_tree(&out, out_grid_idx);

    int in_bit_idx = 1 + (ogd % 2) * 4 + (ogh % 2) * 2 + (ogw % 2);
    if(tree_isset_bit(in_tree, in_bit_idx)) {
      tree_set_bit(out_tree, 0);

      in_bit_idx = tree_child_bit_idx(in_bit_idx);
      for(int out_bit_idx = 1; out_bit_idx < 9; ++out_bit_idx) {
        if(tree_isset_bit(in_tree, in_bit_idx)) {
          tree_set_bit(out_tree, out_bit_idx);
        }
        in_bit_idx++;
      }
    }
  }
}


__global__ void kernel_gridunpoolguided2x2x2(octree out, int n_leafs, const octree in) {
  
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

    int gn,ogd,ogh,ogw;
    octree_split_grid_idx(&out, out_grid_idx, &gn, &ogd, &ogh, &ogw); 

    const int igd = ogd / 2;
    const int igh = ogh / 2;
    const int igw = ogw / 2;
    const int in_grid_idx = octree_grid_idx(&in, gn, igd, igh, igw);
    const ot_tree_t* in_tree = octree_get_tree(&in, in_grid_idx);
    // const ot_data_t* in_data = in.data_ptrs[in_grid_idx];
    const ot_data_t* in_data = octree_get_data(&in, in_grid_idx);

    int in_bit_idx;
    if(depth == 0) {
      in_bit_idx = 1 + (ogd % 2) * 4 + (ogh % 2) * 2 + (ogw % 2);
      // printf(" %d,%d <= %d,%d (%d,%d,%d) (%d,%d,%d)\n", out_grid_idx,out_bit_idx, in_grid_idx,in_bit_idx, ogd,ogh,ogw, -1,-1,-1);
    }
    else if(depth == 1) {
      // bdhw_from_idx_l1(out_bit_idx, &bd,&bh,&bw);
      int bd = ((out_bit_idx - 1)/4);
      int bh = ((out_bit_idx - 1)%4/2);
      int bw = ((out_bit_idx - 1)%2);
      in_bit_idx = tree_child_bit_idx(1 + (ogd % 2) * 4 + (ogh % 2) * 2 + (ogw % 2)) + bd * 4 + bh * 2 + bw;
      // printf(" %d,%d <= %d,%d (%d,%d,%d) (%d,%d,%d)\n", out_grid_idx,out_bit_idx, in_grid_idx,in_bit_idx, ogd,ogh,ogw, bd,bh,bw);
    }
    else if(depth == 2) {
      // bdhw_from_idx_l2(out_bit_idx, &bd,&bh,&bw)
      int bd1 = ((out_bit_idx - 9) / 8/4);
      int bh1 = ((out_bit_idx - 9) / 8%4/2);
      int bw1 = ((out_bit_idx - 9) / 8%2);

      int bd2 = (((out_bit_idx - 9) % 8)/4);
      int bh2 = ((((out_bit_idx - 9) % 8)%4)/2);
      int bw2 = (((out_bit_idx - 9) % 8)%2);

      in_bit_idx = tree_child_bit_idx(tree_child_bit_idx(1 + (ogd % 2) * 4 + (ogh % 2) * 2 + (ogw % 2)) + bd1 * 4  + bh1 * 2 + bw1) + bd2 * 4 + bh2 * 2 + bw2;
      // printf(" %d,%d <= %d,%d (%d,%d,%d) (%d,%d,%d) (%d,%d,%d)\n", out_grid_idx,out_bit_idx, in_grid_idx,in_bit_idx, ogd,ogh,ogw, bd1,bh1,bw1, bd2,bh2,bw2);
    }
    else if(depth == 3) {
      // bdhw_from_idx_l3(out_bit_idx, &bd,&bh,&bw);
      int bd1 = (((tree_parent_bit_idx(out_bit_idx) - 9) / 8)/4);
      int bh1 = (((tree_parent_bit_idx(out_bit_idx) - 9) / 8)%4/2);
      int bw1 = (((tree_parent_bit_idx(out_bit_idx) - 9) / 8)%2);

      int bd2 = (((tree_parent_bit_idx(out_bit_idx) - 9) % 8)/4);
      int bh2 = (((tree_parent_bit_idx(out_bit_idx) - 9) % 8)%4/2);
      int bw2 = (((tree_parent_bit_idx(out_bit_idx) - 9) % 8)%2);

      in_bit_idx = tree_child_bit_idx(tree_child_bit_idx(1 + (ogd % 2) * 4 + (ogh % 2) * 2 + (ogw % 2)) + bd1 * 4  + bh1 * 2 + bw1) + bd2 * 4 + bh2 * 2 + bw2;
      // printf(" %d,%d <= %d,%d (%d,%d,%d) (%d,%d,%d) (%d,%d,%d)\n", out_grid_idx,out_bit_idx, in_grid_idx,in_bit_idx, ogd,ogh,ogw, bd1,bh1,bw1, bd2,bh2,bw2);
    }

    // int out_data_idx = tree_data_idx(out_tree, out_bit_idx, in.feature_size);
    int in_bit_idx2 = tree_bit_idx_leaf(in_tree, in_bit_idx);
    int in_data_idx = tree_data_idx(in_tree, in_bit_idx2, in.feature_size);
    // for(int f = 0; f < in.feature_size; ++f) { 
    //   out_data[f] = in_data[in_data_idx + f]; 
    // }
    octree_cpy_leaf(in_data + in_data_idx, in.feature_size, out_data);

  }
}


void octree_gridunpool2x2x2_gpu(const octree* in, octree* out) {
  out->n = in->n;
  out->grid_depth = in->grid_depth * 2;
  out->grid_height = in->grid_height * 2;
  out->grid_width = in->grid_width * 2;
  out->feature_size = in->feature_size;

  octree_resize_as_gpu(out, out);
  
  const int n_blocks = octree_num_blocks(out);
  const int feature_size = in->feature_size;
  octree_clr_trees_gpu(out);
  kernel_gridunpool2x2x2_struct<<<GET_BLOCKS(n_blocks), CUDA_NUM_THREADS>>>(
      *out, n_blocks, feature_size, *in
  );
  CUDA_POST_KERNEL_CHECK; 

  octree_upd_n_leafs_gpu(out);
  octree_resize_as_gpu(out, out);
  octree_upd_prefix_leafs_gpu(out);

  octree_leaf_idx_to_grid_idx_gpu(out, out->feature_size, out->data_capacity, out->data);
  kernel_gridunpoolguided2x2x2<<<GET_BLOCKS(out->n_leafs), CUDA_NUM_THREADS>>>(
      *out, out->n_leafs, *in
  );
  CUDA_POST_KERNEL_CHECK; 
}

void octree_gridunpoolguided2x2x2_gpu(const octree* in, const octree* in_struct, octree* out) {
  if(in->grid_depth != in_struct->grid_depth / 2 || in->grid_height != in_struct->grid_height / 2 || in->grid_width != in_struct->grid_width / 2) {
    printf("[ERROR] octree_gridunpoolguided2x2x2_gpu in dim does not fit in_struct dim\n");
    exit(-1);
  }

  octree_cpy_scalars(in_struct, out);
  octree_resize_as_gpu(in_struct, out);
  octree_cpy_trees_gpu_gpu(in_struct, out);
  octree_cpy_prefix_leafs_gpu_gpu(in_struct, out);
  
  octree_leaf_idx_to_grid_idx_gpu(out, out->feature_size, out->data_capacity, out->data);
  kernel_gridunpoolguided2x2x2<<<GET_BLOCKS(out->n_leafs), CUDA_NUM_THREADS>>>(
      *out, out->n_leafs, *in
  );
  CUDA_POST_KERNEL_CHECK; 
}




__global__ void kernel_gridunpoolguided2x2x2_bwd(octree grad_in, int n_leafs, const octree grad_out) {

  CUDA_KERNEL_LOOP(leaf_idx, n_leafs) {
    // const int out_grid_idx = out.data[leaf_idx * out.feature_size];
    const int out_grid_idx = leaf_idx_to_grid_idx(&grad_out, leaf_idx);
    const ot_tree_t* out_tree = octree_get_tree(&grad_out, out_grid_idx);

    // const int cum_n_leafs = n_leafs_upto(&grad_out, out_grid_idx);
    const int cum_n_leafs = grad_out.prefix_leafs[out_grid_idx];
    const int out_data_idx = leaf_idx - cum_n_leafs;
    const int out_bit_idx = data_idx_to_bit_idx(out_tree, out_data_idx);
    // const ot_data_t* out_data = grad_out.data_ptrs[out_grid_idx] + out_data_idx * grad_out.feature_size;
    const ot_data_t* out_data = octree_get_data(&grad_out, out_grid_idx) + out_data_idx * grad_out.feature_size;

    const int depth = depth_from_bit_idx(out_bit_idx);

    int gn,ogd,ogh,ogw;
    octree_split_grid_idx(&grad_out, out_grid_idx, &gn, &ogd, &ogh, &ogw); 

    const int igd = ogd / 2;
    const int igh = ogh / 2;
    const int igw = ogw / 2;
    const int in_grid_idx = octree_grid_idx(&grad_in, gn, igd, igh, igw);
    const ot_tree_t* in_tree = octree_get_tree(&grad_in, in_grid_idx);
    // ot_data_t* in_data = grad_in.data_ptrs[in_grid_idx];
    ot_data_t* in_data = octree_get_data(&grad_in, in_grid_idx);

    int in_bit_idx;
    if(depth == 0) {
      in_bit_idx = 1 + (ogd % 2) * 4 + (ogh % 2) * 2 + (ogw % 2);
      // printf(" %d,%d <= %d,%d (%d,%d,%d) (%d,%d,%d)\n", out_grid_idx,out_bit_idx, in_grid_idx,in_bit_idx, ogd,ogh,ogw, -1,-1,-1);
    }
    else if(depth == 1) {
      // bdhw_from_idx_l1(out_bit_idx, &bd,&bh,&bw);
      int bd = ((out_bit_idx - 1)/4);
      int bh = ((out_bit_idx - 1)%4/2);
      int bw = ((out_bit_idx - 1)%2);
      in_bit_idx = tree_child_bit_idx(1 + (ogd % 2) * 4 + (ogh % 2) * 2 + (ogw % 2)) + bd * 4 + bh * 2 + bw;
      // printf(" %d,%d <= %d,%d (%d,%d,%d) (%d,%d,%d)\n", out_grid_idx,out_bit_idx, in_grid_idx,in_bit_idx, ogd,ogh,ogw, bd,bh,bw);
    }
    else if(depth == 2) {
      // bdhw_from_idx_l2(out_bit_idx, &bd,&bh,&bw)
      int bd1 = ((out_bit_idx - 9) / 8/4);
      int bh1 = ((out_bit_idx - 9) / 8%4/2);
      int bw1 = ((out_bit_idx - 9) / 8%2);

      int bd2 = (((out_bit_idx - 9) % 8)/4);
      int bh2 = ((((out_bit_idx - 9) % 8)%4)/2);
      int bw2 = (((out_bit_idx - 9) % 8)%2);

      in_bit_idx = tree_child_bit_idx(tree_child_bit_idx(1 + (ogd % 2) * 4 + (ogh % 2) * 2 + (ogw % 2)) + bd1 * 4  + bh1 * 2 + bw1) + bd2 * 4 + bh2 * 2 + bw2;
      // printf(" %d,%d <= %d,%d (%d,%d,%d) (%d,%d,%d) (%d,%d,%d)\n", out_grid_idx,out_bit_idx, in_grid_idx,in_bit_idx, ogd,ogh,ogw, bd1,bh1,bw1, bd2,bh2,bw2);
    }
    else if(depth == 3) {
      // bdhw_from_idx_l3(out_bit_idx, &bd,&bh,&bw);
      int bd1 = (((tree_parent_bit_idx(out_bit_idx) - 9) / 8)/4);
      int bh1 = (((tree_parent_bit_idx(out_bit_idx) - 9) / 8)%4/2);
      int bw1 = (((tree_parent_bit_idx(out_bit_idx) - 9) / 8)%2);

      int bd2 = (((tree_parent_bit_idx(out_bit_idx) - 9) % 8)/4);
      int bh2 = (((tree_parent_bit_idx(out_bit_idx) - 9) % 8)%4/2);
      int bw2 = (((tree_parent_bit_idx(out_bit_idx) - 9) % 8)%2);

      in_bit_idx = tree_child_bit_idx(tree_child_bit_idx(1 + (ogd % 2) * 4 + (ogh % 2) * 2 + (ogw % 2)) + bd1 * 4  + bh1 * 2 + bw1) + bd2 * 4 + bh2 * 2 + bw2;
      // printf(" %d,%d <= %d,%d (%d,%d,%d) (%d,%d,%d) (%d,%d,%d)\n", out_grid_idx,out_bit_idx, in_grid_idx,in_bit_idx, ogd,ogh,ogw, bd1,bh1,bw1, bd2,bh2,bw2);
    }

    // int out_data_idx = tree_data_idx(out_tree, out_bit_idx, grad_in.feature_size);
    int in_bit_idx2 = tree_bit_idx_leaf(in_tree, in_bit_idx);
    int in_data_idx = tree_data_idx(in_tree, in_bit_idx2, grad_in.feature_size);
    for(int f = 0; f < grad_in.feature_size; ++f) { 
      // out_data[f] = in_data[in_data_idx + f]; 
      atomicAdd(in_data + (in_data_idx + f), out_data[f]);
    }

  }
}


void octree_gridunpool2x2x2_bwd_gpu(const octree* in, const octree* grad_out, octree* grad_in) {
  octree_cpy_scalars(in, grad_in);
  octree_resize_as_gpu(in, grad_in);
  octree_cpy_trees_gpu_gpu(in, grad_in);
  octree_cpy_prefix_leafs_gpu_gpu(in, grad_in);

  octree_fill_data_gpu(grad_in, 0);
  kernel_gridunpoolguided2x2x2_bwd<<<GET_BLOCKS(grad_out->n_leafs), CUDA_NUM_THREADS>>>(
      *grad_in, grad_out->n_leafs, *grad_out
  );
  CUDA_POST_KERNEL_CHECK; 
}

void octree_gridunpoolguided2x2x2_bwd_gpu(const octree* in, const octree* in_struct, const octree* grad_out, octree* grad_in) {
  octree_cpy_scalars(in, grad_in);
  octree_resize_as_gpu(in, grad_in);
  octree_cpy_trees_gpu_gpu(in, grad_in);
  octree_cpy_prefix_leafs_gpu_gpu(in, grad_in);

  octree_fill_data_gpu(grad_in, 0);
  kernel_gridunpoolguided2x2x2_bwd<<<GET_BLOCKS(grad_out->n_leafs), CUDA_NUM_THREADS>>>(
      *grad_in, grad_out->n_leafs, *grad_out
  );
  CUDA_POST_KERNEL_CHECK; 
}
