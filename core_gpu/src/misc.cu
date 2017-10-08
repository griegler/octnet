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

#include "octnet/gpu/misc.h"
#include "octnet/gpu/gpu.h"

#include <cstdio>
#include <cstdlib>

#if defined(_OPENMP)
#include <omp.h>
#endif



__global__ void kernel_mask_by_label(ot_data_t* values, int n_leafs, const ot_data_t* labels, int mask_label, int feature_size) {
  CUDA_KERNEL_LOOP(leaf_idx, n_leafs) {
    int label = labels[leaf_idx];
    if(label == mask_label) {
      for(int f = 0; f < feature_size; ++f) {
        values[leaf_idx * feature_size + f] = 0;
      }
    }
  }
}

extern "C"
void octree_mask_by_label_gpu(const octree* labels, int mask_label, bool check, octree* values) {
  if(check && (labels->feature_size != 1 || !octree_equal_trees_gpu(labels, values))) {
    printf("[ERROR] mask_by_label - tree structure of inputs do not match\n");
    exit(-1);
  }

  kernel_mask_by_label<<<GET_BLOCKS(values->n_leafs), CUDA_NUM_THREADS>>>(
      values->data, values->n_leafs, labels->data, mask_label, values->feature_size
  );
  CUDA_POST_KERNEL_CHECK;
}


__global__ void kernel_determine_gt_split(octree out, int n_leafs, const octree struc, const ot_data_t* gt, int dense_depth, int dense_height, int dense_width) {
  CUDA_KERNEL_LOOP(leaf_idx, n_leafs) {
    int grid_idx = out.data[leaf_idx * out.feature_size];
    const ot_tree_t* tree = octree_get_tree(&struc, grid_idx);

    int data_idx = leaf_idx - struc.prefix_leafs[grid_idx];
    int bit_idx = data_idx_to_bit_idx(tree, data_idx);

    int n,d,h,w;
    int depth = octree_ind_to_dense_ind(&struc, grid_idx, bit_idx, &n, &d,&h,&w);
    int width = width_from_depth(depth);

    int sum = 0;
    for(int dd = 0; dd < width; ++dd) {
      for(int hh = 0; hh < width; ++hh) {
        for(int ww = 0; ww < width; ++ww) {
          float val = gt[(((n * 1 + 0) * dense_depth + (d+dd)) * dense_height + (h+hh)) * dense_width + (w+ww)];
          sum += round(val);
        }
      }
    }
    // printf("grid_idx=%d, sum=%d, (width=%d)\n", grid_idx, sum, width);

    if(sum == 0 || sum == width*width*width) {
      out.data[leaf_idx * out.feature_size] = 0;
    }
    else {
      out.data[leaf_idx * out.feature_size] = 1;
    }
  }
}

extern "C"
void octree_determine_gt_split_gpu(const octree* struc, const ot_data_t* gt, octree* out) {
  octree_resize_gpu(struc->n, struc->grid_depth, struc->grid_height, struc->grid_width, 1, struc->n_leafs, out);
  octree_cpy_trees_gpu_gpu(struc, out);
  octree_cpy_prefix_leafs_gpu_gpu(struc, out);

  int dense_depth = struc->grid_depth * 8;
  int dense_height = struc->grid_height * 8;
  int dense_width = struc->grid_width * 8;

  octree_leaf_idx_to_grid_idx_gpu(out, out->feature_size, out->data_capacity, out->data);
  kernel_determine_gt_split<<<GET_BLOCKS(out->n_leafs), CUDA_NUM_THREADS>>>(
    *out, out->n_leafs, *struc, gt, dense_depth, dense_height, dense_width  
  );
  CUDA_POST_KERNEL_CHECK; 
}
