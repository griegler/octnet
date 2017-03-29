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

#include "octnet/cpu/combine.h"
#include "octnet/cpu/cpu.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(_OPENMP)
#include <omp.h>
#endif


extern "C"
void octree_combine_n_cpu(octree** in, const int n, octree* out) {
  ot_size_t new_n = in[0]->n;
  ot_size_t new_grid_depth = in[0]->grid_depth;
  ot_size_t new_grid_height = in[0]->grid_height;
  ot_size_t new_grid_width = in[0]->grid_width;
  ot_size_t new_feature_size = in[0]->feature_size;
  ot_size_t new_n_leafs = in[0]->n_leafs;

  for(int in_idx = 1; in_idx < n; ++in_idx) {
    new_n += in[in_idx]->n;
    new_n_leafs += in[in_idx]->n_leafs;

    if(new_grid_depth != in[in_idx]->grid_depth) {
      printf("[ERROR] grid_depth of all input octrees have to be the same\n");
      exit(-1);
    }
    if(new_grid_height != in[in_idx]->grid_height) {
      printf("[ERROR] grid_height of all input octrees have to be the same\n");
      exit(-1);
    }
    if(new_grid_width != in[in_idx]->grid_width) {
      printf("[ERROR] grid_width of all input octrees have to be the same\n");
      exit(-1);
    }
    if(new_feature_size != in[in_idx]->feature_size) {
      printf("[ERROR] feature_size of all input octrees have to be the same\n");
      exit(-1);
    }  
  }

  octree_resize_cpu(new_n, new_grid_depth, new_grid_height, new_grid_width, new_feature_size, new_n_leafs, out);


  //copy trees
  int offset = 0;
  for(int in_idx = 0; in_idx < n; ++in_idx) {
    int in_n_blocks = octree_num_blocks(in[in_idx]);
    int n_tree_ints = in_n_blocks * N_TREE_INTS;
    memcpy(out->trees + offset, in[in_idx]->trees, n_tree_ints * sizeof(ot_tree_t));
    offset += n_tree_ints;
  }

  //copy data
  offset = 0;
  for(int in_idx = 0; in_idx < n; ++in_idx) {
    int n_data = in[in_idx]->n_leafs * in[in_idx]->feature_size;
    memcpy(out->data + offset, in[in_idx]->data, n_data * sizeof(ot_data_t));
    offset += n_data;
  }

  //update n leafs
  octree_upd_n_leafs_cpu(out);
  octree_upd_prefix_leafs_cpu(out);
}



extern "C"
void octree_extract_n_cpu(const octree* in, int from, int to, octree* out) {
  int out_n = to - from;

  int from_blocks = from * in->grid_depth * in->grid_height * in->grid_width;
  int to_blocks = to * in->grid_depth * in->grid_height * in->grid_width;
  int n_blocks = octree_num_blocks(in);

  // int n_leafs_from = n_leafs_upto(in, from_blocks);
  int n_leafs_from = in->prefix_leafs[from_blocks];
  // int n_leafs_to = to_blocks >= n_blocks ? in->n_leafs : n_leafs_upto(in, to_blocks);
  int n_leafs_to = to_blocks >= n_blocks ? in->n_leafs : in->prefix_leafs[to_blocks];
  int n_leafs = n_leafs_to - n_leafs_from;

  // printf("  from_blocks=%d, to_blocks=%d, n_blocks=%d\n", from_blocks, to_blocks, n_blocks);
  // printf("  n_leafs_from=%d, n_leafs_to=%d, n_leafs=%d\n", n_leafs_from, n_leafs_to, n_leafs);

  octree_resize_cpu(out_n, in->grid_depth, in->grid_height, in->grid_width, in->feature_size, n_leafs, out);
  
  // octree_cpy_trees_cpu_cpu(in, out);
  memcpy(out->trees, in->trees + from * in->grid_depth * in->grid_height * in->grid_width * N_TREE_INTS, octree_num_blocks(out) * N_TREE_INTS * sizeof(ot_tree_t));

  #pragma omp parallel for
  for(int leaf_idx = 0; leaf_idx < n_leafs; ++leaf_idx) {
    for(int idx = 0; idx < out->feature_size; ++idx) {
      out->data[leaf_idx * out->feature_size + idx] = in->data[(n_leafs_from + leaf_idx) * in->feature_size + idx];
    }
  }

  octree_upd_n_leafs_cpu(out);
  octree_upd_prefix_leafs_cpu(out);
}

extern "C"
void octree_extract_feature_cpu(const octree* in, int from, int to, octree* out) {
  int out_feature_size = to - from;
  octree_resize_cpu(in->n, in->grid_depth, in->grid_height, in->grid_width, out_feature_size, in->n_leafs, out);
  octree_cpy_trees_cpu_cpu(in, out);

  #pragma omp parallel for
  for(int leaf_idx = 0; leaf_idx < in->n_leafs; ++leaf_idx) {
    for(int idx = 0; idx < out->feature_size; ++idx) {
      out->data[leaf_idx * out->feature_size + idx] = in->data[leaf_idx * in->feature_size + from + idx];
    }
  }

  octree_upd_n_leafs_cpu(out);
  octree_upd_prefix_leafs_cpu(out);
}
