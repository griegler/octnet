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

#ifndef OCTREE_CPU_H
#define OCTREE_CPU_H

#include "octnet/core/core.h"

extern "C" {

/// Function to expose @see tree_isset_bit from core.h
bool tree_isset_bit_cpu(const ot_tree_t* num, int pos);
/// Function to expose @see tree_set_bit from core.h
void tree_set_bit_cpu(ot_tree_t* num, int pos);
/// Function to expose @see tree_unset_bit from core.h
void tree_unset_bit_cpu(ot_tree_t* num, int pos);
/// Function to expose @see octree_get_tree from core.h
ot_tree_t* octree_get_tree_cpu(const octree* grid, ot_size_t grid_idx); 
/// Function to expose @see tree_n_leafs from core.h
int tree_n_leafs_cpu(const ot_tree_t* tree);
/// Function to expose @see tree_data_idx from core.h
int tree_data_idx_cpu(const ot_tree_t* tree, const int bit_idx, ot_size_t feature_size);
/// Function to expose @see octree_mem_capacity from core.h
int octree_mem_capacity_cpu(const octree* grid);
/// Function to expose @see octree_mem_using from core.h
int octree_mem_using_cpu(const octree* grid);
/// Function to expose @see leaf_idx_to_grid_idx from core.h
int leaf_idx_to_grid_idx_cpu(const octree* grid, const int leaf_idx);
/// Function to expose @see data_idx_to_bit_idx from core.h
int data_idx_to_bit_idx_cpu(const ot_tree_t* tree, int data_idx);
/// Function to expose @see depth_from_bit_idx from core.h
int depth_from_bit_idx_cpu(const int bit_idx);
/// Function to expose @see octree_split_grid_idx from core.h
void octree_split_grid_idx_cpu(const octree* in, const int grid_idx, int* n, int* d, int* h, int* w);
/// Function to expose @see bdhw_from_idx_l1 from core.h
void bdhw_from_idx_l1_cpu(const int bit_idx, int* d, int* h, int* w);
/// Function to expose @see bdhw_from_idx_l2 from core.h
void bdhw_from_idx_l2_cpu(const int bit_idx, int* d, int* h, int* w);
/// Function to expose @see bdhw_from_idx_l3 from core.h
void bdhw_from_idx_l3_cpu(const int bit_idx, int* d, int* h, int* w);


/// Allocates memory for a new octree struct and initializes all values to 0.
/// @note use octree_resize_cpu to allocate memory for arrays.
/// @return pointer to new octree struct.
octree* octree_new_cpu();

/// Frees all the memory associated with the given grid-octree struct.
/// @param grid_h
void octree_free_cpu(octree* grid_h);

/// Resizes the arrays of the given grid-octree structure dst to fit the given
/// dimensions. This method only allocates new memory if the new shape requires
/// more memory than already associated, otherwise it changes only the scalar
/// values.
/// @param n
/// @param grid_depth
/// @param grid_height
/// @param grid_width
/// @param feature_size
/// @param n_leafs
/// @param dst
void octree_resize_cpu(int n, int grid_depth, int grid_height, int grid_width, int feature_size, int n_leafs, octree* dst);

/// Resizes the arrays of the given grid-octree structure dst to fit the shape
/// of the given src. This method only allocates new memory if the new shape 
/// requires more memory than already associated, otherwise it changes only the 
/// scalar values.
/// @param src
/// @param dst
void octree_resize_as_cpu(const octree* src, octree* dst);


/// Prints the given grid-octree structure in a pretty format to the stdout.
/// @param grid_h
void octree_print_cpu(const octree* grid_h);

/// Copy the shape and the data from src to dst. The function calls 
/// @see octree_resize_as_cpu. Both structures are supposed to be on the host.
/// @param src
/// @param dst
void octree_copy_cpu(const octree* src, octree* dst);

/// Copy the trees from the host structure src_h to the host structure dst_h.
/// @note This function assumes that dst_h has the right size already.
/// @param src_h
/// @param dst_h
void octree_cpy_trees_cpu_cpu(const octree* src_h, octree* dst_h);

/// Copy the prefix_leafs from the host structure src_h to the host structure dst_h.
/// @note This function assumes that dst_h has the right size already.
/// @param src_h
/// @param dst_h
void octree_cpy_prefix_leafs_cpu_cpu(const octree* src_h, octree* dst_h);

/// Copy the data from the host structure src_h to the host structure dst_h.
/// @note This function assumes that dst_h has the right size already.
/// @param src_h
/// @param dst_h
void octree_cpy_data_cpu_cpu(const octree* src_h, octree* dst_h);

/// Clears the bit strings of all shallow octrees in the given grid-octree 
/// structure grid_h. Therefore, the array is all 0s.
/// @param grid_h
void octree_clr_trees_cpu(octree* grid_h);

/// This function updates the scalar n_leafs of the given grid-octree structure
/// grid_h based on the shallow octree bit strings in trees.
/// @param grid_h
void octree_upd_n_leafs_cpu(octree* grid_h);

/// This function updates the array prefix_leafs of the given grid-octree 
/// structure grid_h based on the shallow octree bit strings in trees.
/// @param grid_h
void octree_upd_prefix_leafs_cpu(octree* grid_h);

/// Sets all values in the data array of grid_h to the given fill_value.
/// @param grid_h
/// @param fill_value
void octree_fill_data_cpu(octree* grid_h, ot_data_t fill_value);

/// Copy data from one octree sup to another sub,
/// where sub is a subtree of sup, ie, the sup and sub have the same structure,
/// but the leafs of sup can be split nodes in sub.
/// @param sup
/// @param sub
void octree_cpy_sup_to_sub_cpu(const octree* sup, octree* sub);

/// Copy data from one grid-octree sub to another sup, where sub is a subtree
/// of sup, ie, the sup and sub have the same structure, but the leafs of sup 
/// can be split nodes in sub. Applies sum pooling where necessary.
/// @param sub
/// @param sup
void octree_cpy_sub_to_sup_sum_cpu(const octree* sub, octree* sup);



/// Compares the tree bit strings in the tree arrays of in1 and in2.
/// @param in1
/// @param in2
/// @return true, if the bit strings of all shallow octrees in in1 and in2 are
///         equal. 
bool octree_equal_trees_cpu(const octree* in1, const octree* in2);

/// Compares the data arrays of in1 and in2.
/// @param in1
/// @param in2
/// @return true, if data arrays in in1 and in2 are equal. 
bool octree_equal_data_cpu(const octree* in1, const octree* in2);

/// Compares the prefix_leafs arrays of in1 and in2.
/// @param in1
/// @param in2
/// @return true, if prefix_leafs arrays in in1 and in2 are equal. 
bool octree_equal_prefix_leafs_cpu(const octree* in1, const octree* in2);

/// Compares the two given grid-octree structures in1 and in2 if they are equal.
/// This involves the scalar values, as well all arrays of the structure.
/// @param in1 
/// @param in2
/// @return true, if in1 and in2 are identical.
bool octree_equal_cpu(const octree* in1, const octree* in2);

} // extern "C"


/// Converts the integer num to a binary representation as std::string.
/// @param num
/// @return the bit string of num as std::string.
std::string int_bit_str_cpu(int num);

/// Converts the ot_tree_t array tree to a binary representation as std::string.
/// @param tree
/// @return the bit string of the tree array as std::string.
std::string tree_bit_str_cpu(const ot_tree_t* tree);

#endif
