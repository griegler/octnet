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

#include "octnet/cpu/misc.h"
#include "octnet/cpu/cpu.h"

#include <cstdio>
#include <cstdlib>
#include <iostream>

#if defined(_OPENMP)
#include <omp.h>
#endif


extern "C"
void octree_mask_by_label_cpu(const octree* labels, int mask_label, bool check, octree* values) {
  if(check && (labels->feature_size != 1 || !octree_equal_trees_cpu(labels, values))) {
    printf("[ERROR] mask_by_label - tree structure of inputs do not match\n");
    exit(-1);
  }

  #pragma omp parallel for
  for(int vx_idx = 0; vx_idx < labels->n_leafs; ++vx_idx) {
    int label = labels->data[vx_idx];
    if(label == mask_label) {
      for(int f = 0; f < values->feature_size; ++f) {
        values->data[vx_idx * values->feature_size + f] = 0;
      }
    }
  }
}



extern "C"
void octree_determine_gt_split_cpu(const octree* struc, const ot_data_t* gt, octree* out) {
  octree_resize_cpu(struc->n, struc->grid_depth, struc->grid_height, struc->grid_width, 1, struc->n_leafs, out);
  octree_cpy_trees_cpu_cpu(struc, out);
  octree_cpy_prefix_leafs_cpu_cpu(struc, out);

  int dense_depth = struc->grid_depth * 8;
  int dense_height = struc->grid_height * 8;
  int dense_width = struc->grid_width * 8;

  for(int leaf_idx = 0; leaf_idx < struc->n_leafs; ++leaf_idx) {
    int grid_idx = leaf_idx_to_grid_idx(struc, leaf_idx);
    const ot_tree_t* tree = octree_get_tree(struc, grid_idx);

    int cum_n_leafs = struc->prefix_leafs[grid_idx];
    int data_idx = leaf_idx - cum_n_leafs;
    int bit_idx = data_idx_to_bit_idx(tree, data_idx);

    int n,d,h,w;
    int depth = octree_ind_to_dense_ind(struc, grid_idx, bit_idx, &n, &d,&h,&w);
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
      out->data[leaf_idx] = 0;
    }
    else {
      out->data[leaf_idx] = 1;
    }
  }
}

