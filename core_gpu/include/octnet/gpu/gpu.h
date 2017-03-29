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

#ifndef OCTREE_GPU_H
#define OCTREE_GPU_H

#include "octnet/core/core.h"
#include "common.h"

/// Function to expose @see device_malloc for wrapper code with 
/// template ot_data_t.
inline ot_data_t* device_malloc_ot_data_t(int N) {
  return device_malloc<ot_data_t>(N);
}
/// Function to expose @see defice_free for wrapper code with 
/// template ot_data_t.
inline void device_free_ot_data_t(ot_data_t* dptr) {
  device_free(dptr);
}
/// Function to expose @see device_to_host for wrapper code with 
/// template ot_data_t.
inline void device_to_host_ot_data_t(const ot_data_t* dptr, ot_data_t* hptr, int N) {
  device_to_host(dptr, hptr, N);
}
/// Function to expose @see host_to_device for wrapper code with 
/// template ot_data_t.
inline void host_to_device_ot_data_t(const ot_data_t* hptr, ot_data_t* dptr, int N) {
  host_to_device(hptr, dptr, N);
}
/// Function to expose @see device_to_host_malloc for wrapper code with 
/// template ot_data_t.
inline ot_data_t* device_to_host_malloc_ot_data_t(const ot_data_t* dptr, int N) {
  return device_to_host_malloc(dptr, N);
}

/// Function to expose @see octree_leaf_idx_to_grid_idx for wrapper code. 
template <typename T>
void octree_leaf_idx_to_grid_idx_gpu(const octree* in, const int stride, const int inds_length, T* inds);



extern "C" {

/// Allocates memory for a new octree struct and initializes all values to 0.
/// @note use octree_resize_gpu to allocate memory for arrays.
/// @return pointer to new octree struct.
octree* octree_new_gpu();

/// Frees all the memory associated with the given grid-octree struct.
/// @param grid_h
void octree_free_gpu(octree* grid_d);

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
void octree_resize_gpu(int n, int grid_depth, int grid_height, int grid_width, int feature_size, int n_leafs, octree* dst);

/// Resizes the arrays of the given grid-octree structure dst to fit the shape
/// of the given src. This method only allocates new memory if the new shape 
/// requires more memory than already associated, otherwise it changes only the 
/// scalar values.
/// @param src
/// @param dst
void octree_resize_as_gpu(const octree* src, octree* dst);

/// Copy the shape and the data from src to dst. The function calls 
/// @see octree_resize_as_gpu. Both structures are supposed to be on the device.
/// @param src
/// @param dst
void octree_copy_gpu(const octree* src, octree* dst);

/// Copy the shape and the data from grid_h to grid_d. The function calls 
/// @see octree_resize_as_gpu. grid_h is supposed to be on the host and grid_d
/// on the device, respectively. 
/// @param grid_h
/// @param grid_d
void octree_to_gpu(const octree* grid_h, octree* grid_d);

/// Copy the shape and the data from grid_d to grid_h. The function calls 
/// @see octree_resize_as_cpu. grid_d is supposed to be on the device and grid_h
/// on the host, respectively.
/// @param grid_h
/// @param grid_d
void octree_to_cpu(const octree* grid_d, octree* grid_h);


/// Clears the bit strings of all shallow octrees in the given grid-octree 
/// structure grid_h. Therefore, the array is all 0s.
/// @param grid_d
void octree_clr_trees_gpu(octree* grid_d);

/// This function updates the scalar n_leafs of the given grid-octree structure
/// grid_d based on the shallow octree bit strings in trees.
/// @param grid_d
void octree_upd_n_leafs_gpu(octree* grid_d);

/// This function updates the array prefix_leafs of the given grid-octree 
/// structure grid_d based on the shallow octree bit strings in trees.
/// @param grid_d
void octree_upd_prefix_leafs_gpu(octree* grid_d);

/// Sets all values in the data array of grid_d to the given fill_value.
/// @param grid_d
/// @param fill_value
void octree_fill_data_gpu(octree* grid_d, ot_data_t fill_value);


/// Copy the trees from the host structure src_h to the device structure dst_d.
/// @note This function assumes that dst_d has the right size already.
/// @param src_h
/// @param dst_d
void octree_cpy_trees_cpu_gpu(const octree* src_h, octree* dst_d);

/// Copy the prefix_leafs from the host structure src_h to the device structure dst_d.
/// @note This function assumes that dst_d has the right size already.
/// @param src_h
/// @param dst_d
void octree_cpy_prefix_leafs_cpu_gpu(const octree* src_h, octree* dst_d);

/// Copy the data from the host structure src_h to the device structure dst_d.
/// @note This function assumes that dst_d has the right size already.
/// @param src_h
/// @param dst_d
void octree_cpy_data_cpu_gpu(const octree* src_h, octree* dst_d);


/// Copy the trees from the device structure src_d to the host structure dst_h.
/// @note This function assumes that dst_h has the right size already.
/// @param src_d
/// @param dst_h
void octree_cpy_trees_gpu_cpu(const octree* src_d, octree* dst_h);

/// Copy the prefix_leafs from the device structure src_d to the host structure dst_h.
/// @note This function assumes that dst_d has the right size already.
/// @param src_d
/// @param dst_h
void octree_cpy_prefix_leafs_gpu_cpu(const octree* src_d, octree* dst_h);

/// Copy the data from the device structure src_d to the host structure dst_h.
/// @note This function assumes that dst_h has the right size already.
/// @param src_d
/// @param dst_h
void octree_cpy_data_gpu_cpu(const octree* src_d, octree* dst_h);

/// Copy the trees from the device structure src_d to the device structure dst_d.
/// @note This function assumes that dst_d has the right size already.
/// @param src_d
/// @param dst_d
void octree_cpy_trees_gpu_gpu(const octree* src_d, octree* dst_d);

/// Copy the prefix_leafs from the device structure src_d to the device structure dst_d.
/// @note This function assumes that dst_d has the right size already.
/// @param src_d
/// @param dst_d
void octree_cpy_prefix_leafs_gpu_gpu(const octree* src_d, octree* dst_d);

/// Copy the data from the device structure src_d to the device structure dst_d.
/// @note This function assumes that dst_d has the right size already.
/// @param src_d
/// @param dst_d
void octree_cpy_data_gpu_gpu(const octree* src_d, octree* dst_d);


/// Copy data from one octree sup to another sub,
/// where sub is a subtree of sup, ie, the sup and sub have the same structure,
/// but the leafs of sup can be split nodes in sub
/// @param sup
/// @param sub
void octree_cpy_sup_to_sub_gpu(const octree* sup, octree* sub);

/// Copy data from one grid-octree sub to another sup, where sub is a subtree
/// of sup, ie, the sup and sub have the same structure, but the leafs of sup 
/// can be split nodes in sub. Applies sum pooling where necessary.
/// @param sub
/// @param sup
void octree_cpy_sub_to_sup_sum_gpu(const octree* sub, octree* sup);


/// Compares the tree bit strings in the tree arrays of in1 and in2.
/// @param in1
/// @param in2
/// @return true, if the bit strings of all shallow octrees in in1 and in2 are
///         equal. 
bool octree_equal_trees_gpu(const octree* in1, const octree* in2);

/// Compares the data arrays of in1 and in2.
/// @param in1
/// @param in2
/// @return true, if data arrays in in1 and in2 are equal. 
bool octree_equal_data_gpu(const octree* in1, const octree* in2);

/// Compares the prefix_leafs arrays of in1 and in2.
/// @param in1
/// @param in2
/// @return true, if prefix_leafs arrays in in1 and in2 are equal. 
bool octree_equal_prefix_leafs_gpu(const octree* in1, const octree* in2);

/// Compares the two given grid-octree structures in1 and in2 if they are equal.
/// This involves the scalar values, as well all arrays of the structure.
/// @param in1 
/// @param in2
/// @return true, if in1 and in2 are identical.
bool octree_equal_gpu(const octree* in1, const octree* in2);


} // extern "C"



#endif
