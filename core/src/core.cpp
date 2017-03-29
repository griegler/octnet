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

#include <iostream>
#include <sstream>
#include <cstring>
#include <cstdio>
#include <cstdlib>

#if defined(_OPENMP)
#include <omp.h>
#endif

#include "octnet/cpu/cpu.h"

extern "C"
bool tree_isset_bit_cpu(const ot_tree_t* num, int pos) { 
  return tree_isset_bit(num, pos); 
}

extern "C"
void tree_set_bit_cpu(ot_tree_t* num, int pos) { 
  tree_set_bit(num, pos); 
}

extern "C"
void tree_unset_bit_cpu(ot_tree_t* num, int pos) { 
  tree_unset_bit(num, pos); 
}

extern "C"
ot_tree_t* octree_get_tree_cpu(const octree* grid, ot_size_t grid_idx) { 
  return octree_get_tree(grid, grid_idx); 
}

extern "C"
int tree_n_leafs_cpu(const ot_tree_t* tree) { 
  return tree_n_leafs(tree);
}

extern "C"
int tree_data_idx_cpu(const ot_tree_t* tree, const int bit_idx, ot_size_t feature_size) {
  return tree_data_idx(tree, bit_idx, feature_size);
}

extern "C"
int octree_mem_capacity_cpu(const octree* grid) { 
  return octree_mem_capacity(grid); 
}

extern "C"
int octree_mem_using_cpu(const octree* grid) { 
  return octree_mem_using(grid);
}

extern "C"
int leaf_idx_to_grid_idx_cpu(const octree* grid, const int leaf_idx) {
  return leaf_idx_to_grid_idx(grid, leaf_idx);
}

extern "C"
int data_idx_to_bit_idx_cpu(const ot_tree_t* tree, int data_idx) {
  return data_idx_to_bit_idx(tree, data_idx);
}

extern "C"
int depth_from_bit_idx_cpu(const int bit_idx) {
  return depth_from_bit_idx(bit_idx);
}

extern "C"
void octree_split_grid_idx_cpu(const octree* in, const int grid_idx, int* n, int* d, int* h, int* w) {
  return octree_split_grid_idx(in, grid_idx, n, d, h, w);
}

extern "C"
void bdhw_from_idx_l1_cpu(const int bit_idx, int* d, int* h, int* w) {
  bdhw_from_idx_l1(bit_idx, d, h, w);
}

extern "C"
void bdhw_from_idx_l2_cpu(const int bit_idx, int* d, int* h, int* w) {
  bdhw_from_idx_l2(bit_idx, d, h, w);
}

extern "C"
void bdhw_from_idx_l3_cpu(const int bit_idx, int* d, int* h, int* w) {
  bdhw_from_idx_l3(bit_idx, d, h, w);
}





std::string int_bit_str_cpu(int num) {
  std::stringstream bit_repr;
  for(unsigned int bit = 0; bit < 8*sizeof(int); ++bit) {
    bit_repr << (num & 0x01);
    num = num >> 1;
  }
  return bit_repr.str();
}

std::string tree_bit_str_cpu(const ot_tree_t* tree) {
  std::stringstream bit_repr;
  
  bit_repr << (tree_isset_bit(tree, 0) ? '1' : '0') << ' ';
  
  for(int bit_idx = 1; bit_idx <= 8; ++bit_idx) {
    bit_repr << (tree_isset_bit(tree, bit_idx) ? '1' : '0');
  }
  bit_repr << ' ';

  for(int bit_idx = 1; bit_idx <= 8; ++bit_idx) {
    int child_idx = tree_child_bit_idx(bit_idx); 
    for(int idx = 0; idx < 8; ++idx) {
      bit_repr << (tree_isset_bit(tree, child_idx + idx) ? '1' : '0');
    }
    if(bit_idx < 8) {
      bit_repr << ' ';
    }
  }


  return bit_repr.str();
}


void octree_print_rec_cpu(const ot_tree_t* tree, const ot_data_t* data, int feature_size, int bit_idx, int bds, int bhs, int bws, int depth) {
  int size = width_from_depth(depth);
  int indent = depth == 0 ? 0 : (1 << depth);
  if(depth < 3 && tree_isset_bit(tree, bit_idx)) {
    for(int ind = 0; ind < indent; ++ind) { printf(" "); }
    printf("|- split node [%d,%d],[%d,%d],[%d,%d] %d\n", bds,bds+size, bhs,bhs+size, bws,bws+size, bit_idx);

    int child_idx = tree_child_bit_idx(bit_idx);
    for(int ad = 0; ad < 2; ++ad) {
      for(int ah = 0; ah < 2; ++ah) {
        for(int aw = 0; aw < 2; ++aw) {
          octree_print_rec_cpu(tree, data, feature_size, child_idx, bds+ad*size/2, bhs+ah*size/2, bws+aw*size/2, depth+1);
          child_idx++;
        }
      }
    }
  }
  else {
    for(int ind = 0; ind < indent; ++ind) { printf(" "); }
    int data_idx = tree_data_idx(tree, bit_idx, feature_size);
    printf("|- data node [%d,%d],[%d,%d],[%d,%d] %d -> %d, %p: [", bds,bds+size, bhs,bhs+size, bws,bws+size, bit_idx, data_idx, data + data_idx);
    const ot_data_t* vec = data + data_idx;
    for(int f = 0; f < feature_size; ++f) {
      printf("%f", vec[f]);
      if(f < feature_size - 1) {
        printf(", ");
      }
    }
    printf("]\n");
  }
}

void octree_print_cpu(const octree* grid_h) {
  int n_blocks = octree_num_blocks(grid_h);
  for(int grid_idx = 0; grid_idx < n_blocks; ++grid_idx) {
    const ot_tree_t* tree = octree_get_tree(grid_h, grid_idx);
    printf("block % 3d\n", grid_idx);
    // std::cout << tree_bit_str(tree) << std::endl;
    octree_print_rec_cpu(tree, octree_get_data(grid_h, grid_idx), grid_h->feature_size, 0, 0,0,0, 0);
  }
}


extern "C"
octree* octree_new_cpu() {
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
void octree_free_cpu(octree* grid_h) {
  delete[] grid_h->trees;
  delete[] grid_h->prefix_leafs;
  delete[] grid_h->data;
  delete grid_h;
}


extern "C"
void octree_clr_trees_cpu(octree* grid_h) {
  memset(grid_h->trees, 0, octree_num_blocks(grid_h) * N_TREE_INTS * sizeof(ot_tree_t));
}

extern "C"
void octree_fill_data_cpu(octree* grid_h, ot_data_t fill_value) {
  int n = grid_h->feature_size * grid_h->n_leafs;
  #pragma omp parallel for
  for(int idx = 0; idx < n; ++idx) {
    grid_h->data[idx] = fill_value;
  }
}

extern "C"
void octree_upd_n_leafs_cpu(octree* grid_h) {
  int n_blocks = octree_num_blocks(grid_h);
  grid_h->n_leafs = 0;
  for(int grid_idx = 0; grid_idx < n_blocks; ++grid_idx) {
    const ot_tree_t* tree = octree_get_tree(grid_h, grid_idx);
    grid_h->n_leafs += tree_n_leafs(tree);
  }
}

extern "C"
void octree_upd_prefix_leafs_cpu(octree* grid_h) {
  int n_blocks = octree_num_blocks(grid_h);
  if(n_blocks > 0) {
    grid_h->prefix_leafs[0] = 0;
    for(int grid_idx = 1; grid_idx < n_blocks; ++grid_idx) {
      const ot_tree_t* tree = octree_get_tree(grid_h, grid_idx-1);
      grid_h->prefix_leafs[grid_idx] = grid_h->prefix_leafs[grid_idx-1] + tree_n_leafs(tree);
    }
  }
}


extern "C"
void octree_resize_cpu(int n, int grid_depth, int grid_height, int grid_width, int feature_size, int n_leafs, octree* dst) {
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
      delete[] dst->trees;
    }
    dst->trees = new ot_tree_t[grid_capacity * N_TREE_INTS];

    if(dst->prefix_leafs != 0) {
      delete[] dst->prefix_leafs;
    }
    dst->prefix_leafs = new ot_size_t[grid_capacity];
  }

  int data_capacity = n_leafs * feature_size;
  if(dst->data_capacity < data_capacity) {
    dst->data_capacity = data_capacity;

    if(dst->data != 0) {
      delete[] dst->data;
    }
    dst->data = new ot_data_t[data_capacity];
  }
}

extern "C"
void octree_resize_as_cpu(const octree* src, octree* dst) {
  octree_resize_cpu(src->n, src->grid_depth, src->grid_height, src->grid_width, src->feature_size, src->n_leafs, dst);
}

extern "C"
void octree_copy_cpu(const octree* src, octree* dst) {
  octree_resize_as_cpu(src, dst);
  octree_cpy_trees_cpu_cpu(src, dst);
  octree_cpy_prefix_leafs_cpu_cpu(src, dst);
  octree_cpy_data_cpu_cpu(src, dst);
}


extern "C"
void octree_cpy_trees_cpu_cpu(const octree* src_h, octree* dst_h) {
  memcpy(dst_h->trees, src_h->trees, octree_num_blocks(src_h) * N_TREE_INTS * sizeof(ot_tree_t));
}

extern "C"
void octree_cpy_prefix_leafs_cpu_cpu(const octree* src_h, octree* dst_h) {
  memcpy(dst_h->prefix_leafs, src_h->prefix_leafs, octree_num_blocks(src_h) * sizeof(ot_size_t));
}

extern "C"
void octree_cpy_data_cpu_cpu(const octree* src_h, octree* dst_h) {
  memcpy(dst_h->data, src_h->data, src_h->n_leafs * src_h->feature_size * sizeof(ot_data_t));
}


extern "C"
void octree_cpy_sup_to_sub_cpu(const octree* sup, octree* sub) {
  #pragma omp parallel for
  for(int sub_leaf_idx = 0; sub_leaf_idx < sub->n_leafs; ++sub_leaf_idx) {
    int grid_idx = leaf_idx_to_grid_idx(sub, sub_leaf_idx);
    int data_idx = sub_leaf_idx - sub->prefix_leafs[grid_idx];
    int sub_bit_idx = data_idx_to_bit_idx(octree_get_tree(sub, grid_idx), data_idx);

    const ot_tree_t* sup_tree = octree_get_tree(sup, grid_idx);
    int sup_bit_idx = tree_bit_idx_leaf(sup_tree, sub_bit_idx);
    int sup_data_idx = tree_data_idx(sup_tree, sup_bit_idx, sup->feature_size);

    octree_cpy_leaf(octree_get_data(sup, grid_idx) + sup_data_idx, sup->feature_size, sub->data + sub_leaf_idx * sub->feature_size);
  }
}



extern "C"
void octree_cpy_sub_to_sup_sum_cpu(const octree* sub, octree* sup) {
  octree_fill_data_cpu(sup, 0);

  #pragma omp parallel for
  for(int sub_leaf_idx = 0; sub_leaf_idx < sub->n_leafs; ++sub_leaf_idx) {
    int grid_idx = leaf_idx_to_grid_idx(sub, sub_leaf_idx);
    int data_idx = sub_leaf_idx - sub->prefix_leafs[grid_idx];
    int sub_bit_idx = data_idx_to_bit_idx(octree_get_tree(sub, grid_idx), data_idx);

    const ot_tree_t* sup_tree = octree_get_tree(sup, grid_idx);
    int sup_bit_idx = tree_bit_idx_leaf(sup_tree, sub_bit_idx);
    int sup_data_idx = tree_data_idx(sup_tree, sup_bit_idx, sup->feature_size);
    ot_data_t* sup_data = octree_get_data(sup, grid_idx);

    for(int f = 0; f < sup->feature_size; ++f) {
      #pragma omp atomic
      sup_data[sup_data_idx + f] += sub->data[sub_leaf_idx * sub->feature_size + f];
    }
  }
}


extern "C"
bool octree_equal_trees_cpu(const octree* in1, const octree* in2) {
  if(in1->n_leafs != in2->n_leafs) {
    return false;
  }

  int n_blocks1 = octree_num_blocks(in1);
  int n_blocks2 = octree_num_blocks(in2);
  if(n_blocks1 != n_blocks2) {
    return false;
  }

  for(int tree_idx = 0; tree_idx < n_blocks1 * N_TREE_INTS; ++tree_idx) {
    if(in1->trees[tree_idx] != in2->trees[tree_idx]) {
      return false;
    }
  }

  return true;
}

extern "C"
bool octree_equal_data_cpu(const octree* in1, const octree* in2) {
  if(in1->feature_size * in1->n_leafs != in2->feature_size * in2->n_leafs) {
    return false;
  }

  for(int data_idx = 0; data_idx < in1->feature_size * in1->n_leafs; ++data_idx) {
    if(in1->data[data_idx] != in2->data[data_idx]) {
      return false;
    }
  }

  return true;
}

extern "C"
bool octree_equal_prefix_leafs_cpu(const octree* in1, const octree* in2) {
  int n_blocks1 = octree_num_blocks(in1);
  int n_blocks2 = octree_num_blocks(in2);
  if(n_blocks1 != n_blocks2) {
    return false;
  }

  for(int grid_idx = 0; grid_idx < n_blocks1; ++grid_idx) {
    if(in1->prefix_leafs[grid_idx] != in2->prefix_leafs[grid_idx]) {
      return false;
    }
  }

  return true;
}

extern "C"
bool octree_equal_cpu(const octree* in1, const octree* in2) {
  if(!octree_equal_shape(in1, in2)) { return false; }
  if(in1->n_leafs != in2->n_leafs) { return false; }

  if(!octree_equal_trees_cpu(in1, in2)) {
    return false;
  }

  if(!octree_equal_prefix_leafs_cpu(in1, in2)) {
    return false;
  }

  if(!octree_equal_data_cpu(in1, in2)) {
    return false;
  }

  return true;
}
