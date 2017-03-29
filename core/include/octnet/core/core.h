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

#ifndef OCTREE_H
#define OCTREE_H

#ifndef DEBUG 
#define DEBUG 0
#endif

#include "types.h"

#include <string>
#include <cstdio>
#include <cmath>

#include <smmintrin.h>


#ifdef __CUDA_ARCH__
#define OCTREE_FUNCTION __host__ __device__
#define FMIN(a,b) fminf(a,b)
#define FMAX(a,b) fmaxf(a,b)
#define IMIN(a,b) min(a,b)
#define IMAX(a,b) max(a,b)
#else
#define OCTREE_FUNCTION 
#define FMIN(a,b) fminf(a,b)
#define FMAX(a,b) fmaxf(a,b)
#define IMIN(a,b) (((a)<(b))?(a):(b))
#define IMAX(a,b) (((a)>(b))?(a):(b))
#endif

/// number of values for a 3x3x3 kernel
#define K333 27

/// using average reduce operation
#define REDUCE_AVG 0
/// using maximum reduce operation
#define REDUCE_MAX 1
/// using sum reduce operation
#define REDUCE_SUM 2

const int N_TREE_INTS = 4;
const int N_OC_TREE_T_BITS = 8*sizeof(ot_tree_t);

/// Core struct that encodes the hybrid grid-octree data structure
/// 
/// The grid-octree structure consists of multiple shallow octrees that are 
/// arranged in a uniform grid. This yields a good trade-off between memory
/// consumption and performance. 
/// Each shallow octree has a maximum depth of 3 and therefore, can comprise
/// 512 (8x8x8) cells at maximum. The number of shallow
/// octrees is given by n x grid_depth x grid_height x grid_width.
/// Hence, this structure corresponds to a dense tensor with shape
/// n x 8 grid_depth x 8 grid_height x 8 grid_width x feature_size.
typedef struct {
  ot_size_t n;             ///< number of grid-octrees (batch size).
  ot_size_t grid_depth;    ///< number of shallow octrees in the depth dimension.
  ot_size_t grid_height;   ///< number of shallow octrees in the height dimension.
  ot_size_t grid_width;    ///< number of shallow octrees in the width dimension.

  ot_size_t feature_size;  ///< length of the data vector associated with a single cell.

  ot_size_t n_leafs;       ///< number of leaf nodes in the complete struct.

  ot_tree_t* trees;        ///< array of length octree_num_blocks(grid) x N_TREE_INTS that encode the structure of the shallow octrees as bit strings.
  ot_size_t* prefix_leafs; ///< prefix sum of the number of leafs in each shallow octree.
  ot_data_t* data;         ///< contiguous data array, all feature vectors associated with the grid-octree data structure.

  ot_size_t grid_capacity; ///< Indicates how much memory is allocated for the trees and prefix_leafs array
  ot_size_t data_capacity; ///< Indicates how much memory is allocated for the data array
} octree;


/// Computes the number of shallow octrees in the grid.
///
/// @param grid 
/// @return the number of shallow octrees in the grid.
OCTREE_FUNCTION
inline ot_size_t octree_num_blocks(const octree* grid) {
  return grid->n * grid->grid_depth * grid->grid_height * grid->grid_width;
}

/// Computes the number of voxels that are comprised by grid.
/// 8*8*8 * octree_num_blocks(grid)
///
/// @param grid 
/// @return the number of voxels that are comprised by grid.
OCTREE_FUNCTION
inline ot_size_t octree_num_voxels(const octree* grid) {
  return octree_num_blocks(grid) * 8*8*8;
}

/// Computes the number data elements are comprised by the grid.
/// grid->feature_size * octree_num_voxels(grid)
///
/// @param grid 
/// @return the number data elements are comprised by the grid.
OCTREE_FUNCTION
inline ot_size_t octree_num_elems(const octree* grid) {
  return octree_num_voxels(grid) * grid->feature_size;
}

/// Computes the flat index of the shallow octree given the subscript indices
/// grid->feature_size * octree_num_voxels(grid).
///
/// @param gn
/// @param gd
/// @param gh
/// @param gw
/// @return flat index of shallow octree.
OCTREE_FUNCTION
inline int octree_grid_idx(const octree* grid, const int gn, const int gd, const int gh, const int gw) {
  return ((gn * grid->grid_depth + gd) * grid->grid_height + gh) * grid->grid_width + gw;
}

/// Computes the offset in the grids trees array.
/// Therefore, it returns the array to the bit string that defines the shallow
/// octree structure.
///
/// @param grid
/// @param grid_idx flat index of the shallow octree.
/// @return array that encodes the tree structure as bit string.
OCTREE_FUNCTION
inline ot_tree_t* octree_get_tree(const octree* grid, const ot_size_t grid_idx) {
  return grid->trees + grid_idx * N_TREE_INTS;
}

/// Returns the offset in the data array that belongs to a shallow octree.
///
/// @param grid
/// @param grid_idx flat index of the shallow octree.
/// @return data array associated with the shallow octree at grid_idx.
OCTREE_FUNCTION
inline ot_data_t* octree_get_data(const octree* grid, const ot_size_t grid_idx) {
  return grid->data + grid->feature_size * grid->prefix_leafs[grid_idx];
}

/// Copy all scalar values of the data structure from src to dst.
/// This includes n, grid_depth, grid_height, grid_width, feature_size and n_leafs.
///
/// @param src
/// @param dst
/// @return void
OCTREE_FUNCTION
inline void octree_cpy_scalars(const octree* src, octree* dst) {
  dst->n = src->n;
  dst->grid_depth = src->grid_depth;
  dst->grid_height = src->grid_height;
  dst->grid_width = src->grid_width;
  dst->feature_size = src->feature_size;
  dst->n_leafs = src->n_leafs;
}

/// Tests if two octree objects have the same shape (nxdxhxw).
///
/// @param in1
/// @param in2
/// @return true, if the shape of in1 and in2 are the same, otherwise false.
OCTREE_FUNCTION
inline bool octree_equal_shape(const octree* in1, const octree* in2) {
  return in1->n == in2->n && 
         in1->grid_depth == in2->grid_depth && 
         in1->grid_height == in2->grid_height && 
         in1->grid_width == in2->grid_width &&
         in1->feature_size == in2->feature_size;
}

/// Copies the data of one leaf cell to the data array of another leaf cell.
///
/// @param data_in
/// @param feature_size length of the data array
/// @param data_out
/// @return void
OCTREE_FUNCTION
inline void octree_cpy_leaf(const ot_data_t* data_in, const ot_size_t feature_size, ot_data_t* data_out) {
  for(int f = 0; f < feature_size; ++f) {
    data_out[f] = data_in[f];
  }
}


/// Determines the memory capacity (in bytes) of grid, and therefore also the 
/// number of bytes that are currently occupied by the grid on the heap.
/// Note that octree_mem_using(grid) <= octree_mem_capacity(grid).
///
/// @param grid
/// @return the number of bytes allocated for grid on the heap.
OCTREE_FUNCTION
inline int octree_mem_capacity(const octree* grid) {
  return grid->grid_capacity * (N_TREE_INTS * sizeof(ot_tree_t) + sizeof(ot_data_t*)) + 
         grid->data_capacity * sizeof(ot_data_t) + 
         7 * sizeof(ot_size_t);
}

/// Determines the number of bytes that are needed by grid on the heap.
/// Note that octree_mem_using(grid) <= octree_mem_capacity(grid).
///
/// @param grid
/// @return the number of bytes needed for grid on the heap.
OCTREE_FUNCTION
inline int octree_mem_using(const octree* grid) {
  const int n_blocks = octree_num_blocks(grid);
  return n_blocks * (N_TREE_INTS * sizeof(ot_tree_t) + sizeof(ot_data_t*)) + 
         grid->n_leafs * grid->feature_size * sizeof(ot_data_t) + 
         7 * sizeof(ot_size_t);
}


/// Sets the bit of num on pos to 1.
///
/// @param num
/// @param pos
/// @return updated num.
OCTREE_FUNCTION
inline int set_bit(int num, int pos) {
  return num | (1 << pos); 
}

/// Sets the bit of num on pos to 0.
///
/// @param num
/// @param pos
/// @return updated num.
OCTREE_FUNCTION
inline int unset_bit(int num, int pos) {
  return num & (~(1 << pos)); 
}

/// Toogles bit of num on pos.
///
/// @param num
/// @param pos
/// @return updated num.
OCTREE_FUNCTION
inline int toogle_bit(int num, int pos) {
  return num ^ (1 << pos); 
}

/// Checks if the bit on pos of num is set, or not.
///
/// @param num
/// @param pos
/// @return true, if bit is set, otherwise false
OCTREE_FUNCTION
inline bool isset_bit(const int num, const int pos) {
  return (num & (1 << pos)) != 0; 
}

/// Sets the bit of num on pos to 1.
///
/// @param num
/// @param pos
/// @return updated num.
OCTREE_FUNCTION
inline void tree_set_bit(ot_tree_t* num, const int pos) {
  num[pos / N_OC_TREE_T_BITS] |= (1 << (pos % N_OC_TREE_T_BITS)); 
}

/// Sets the bit of num on pos to 0.
///
/// @param num
/// @param pos
/// @return updated num.
OCTREE_FUNCTION
inline void tree_unset_bit(ot_tree_t* num, const int pos) {
  num[pos / N_OC_TREE_T_BITS] &= ~(1 << (pos % N_OC_TREE_T_BITS)); 
}

/// Toogles the bit of num on pos.
///
/// @param num
/// @param pos
/// @return updated num.
OCTREE_FUNCTION
inline void tree_toogle_bit(ot_tree_t* num, const int pos) {
  num[pos / N_OC_TREE_T_BITS] ^= (1 << (pos % N_OC_TREE_T_BITS)); 
}

/// Checks if the bit on pos of num is set, or not.
///
/// @param num
/// @param pos
/// @return true, if bit is set, otherwise false
OCTREE_FUNCTION
inline bool tree_isset_bit(const ot_tree_t* num, const int pos) {
  return ( num[pos / N_OC_TREE_T_BITS] & (1 << (pos % N_OC_TREE_T_BITS))) != 0; 
}

/// Counts the number of bits == 1 in the tree array in the range [from, to).
///
/// @param tree
/// @param from
/// @param to
/// @return number of ones in tree in the range [from, to).
OCTREE_FUNCTION
inline int tree_cnt1(const ot_tree_t* tree, const int from, const int to) {
  int cnt = 0;
  int from_, range_;
  unsigned int mask_;
  from_ = IMAX(from, 0); range_ = -from_ + IMIN(to, N_OC_TREE_T_BITS);
  mask_ = range_ <= 0 ? 0 : ((0xFFFFFFFF >> (N_OC_TREE_T_BITS - range_)) << from_);
#ifdef __CUDA_ARCH__
  cnt += __popc(tree[0] & mask_);
#else
  cnt += _mm_popcnt_u32(tree[0] & mask_);
#endif

  from_ = IMAX(from - N_OC_TREE_T_BITS, 0); range_ = -from_ + IMIN(to - N_OC_TREE_T_BITS, N_OC_TREE_T_BITS);
  mask_ = range_ <= 0 ? 0 : ((0xFFFFFFFF >> (N_OC_TREE_T_BITS - range_)) << from_);
#ifdef __CUDA_ARCH__
  cnt += __popc(tree[1] & mask_);
#else
  cnt += _mm_popcnt_u32(tree[1] & mask_);
#endif

  from_ = IMAX(from - 2*N_OC_TREE_T_BITS, 0); range_ = -from_ + IMIN(to - 2*N_OC_TREE_T_BITS, N_OC_TREE_T_BITS);
  mask_ = range_ <= 0 ? 0 : ((0xFFFFFFFF >> (N_OC_TREE_T_BITS - range_)) << from_);
#ifdef __CUDA_ARCH__
  cnt += __popc(tree[2] & mask_);
#else
  cnt += _mm_popcnt_u32(tree[2] & mask_);
#endif
  
  return cnt;
}

/// Counts the number of bits == 0 in the tree array in the range [from, to).
///
/// @param tree
/// @param from
/// @param to
/// @return number of zeros in tree in the range [from, to).
OCTREE_FUNCTION
inline int tree_cnt0(const ot_tree_t* tree, const int from, const int to) {
  int cnt = 0;
  int from_, range_;
  unsigned int mask_;
  from_ = IMAX(from, 0); range_ = -from_ + IMIN(to, N_OC_TREE_T_BITS);
  mask_ = range_ <= 0 ? 0 : ((0xFFFFFFFF >> (N_OC_TREE_T_BITS - range_)) << from_);
#ifdef __CUDA_ARCH__
  cnt += __popc(~tree[0] & mask_);
#else
  cnt += _mm_popcnt_u32(~tree[0] & mask_);
#endif

  from_ = IMAX(from - N_OC_TREE_T_BITS, 0); range_ = -from_ + IMIN(to - N_OC_TREE_T_BITS, N_OC_TREE_T_BITS);
  mask_ = range_ <= 0 ? 0 : ((0xFFFFFFFF >> (N_OC_TREE_T_BITS - range_)) << from_);
#ifdef __CUDA_ARCH__
  cnt += __popc(~tree[1] & mask_);
#else
  cnt += _mm_popcnt_u32(~tree[1] & mask_);
#endif

  from_ = IMAX(from - 2*N_OC_TREE_T_BITS, 0); range_ = -from_ + IMIN(to - 2*N_OC_TREE_T_BITS, N_OC_TREE_T_BITS);
  mask_ = range_ <= 0 ? 0 : ((0xFFFFFFFF >> (N_OC_TREE_T_BITS - range_)) << from_);
#ifdef __CUDA_ARCH__
  cnt += __popc(~tree[2] & mask_);
#else
  cnt += _mm_popcnt_u32(~tree[2] & mask_);
#endif
  
  return cnt;
}

/// Computes the bit index of the first child for the given bit_idx.
/// Used to traverse a shallow octree.
///
/// @param bit_idx
/// @return child bit_idx of bit_idx
OCTREE_FUNCTION
inline int tree_child_bit_idx(const int bit_idx) {
  return 8 * bit_idx + 1;
}

/// Computes the bit index of the parent for the given bit_idx.
/// Used to traverse a shallow octree.
/// @warning does not check the range of bit_idx, and will return an invalid
/// result if for example no parent exists (e.g. for bit_idx=0).
///
/// @param bit_idx
/// @return parent bit_idx of bit_idx
OCTREE_FUNCTION
inline int tree_parent_bit_idx(const int bit_idx) {
  return (bit_idx - 1) / 8;
}



/// Computes the first valid bit_idx for a given bit_idx that is a valid leaf node.
/// Therefore, it recursively checks the parent bit_idx until it is set, or bit_idx
/// is the root (bit_idx=0).
///
/// @param tree shallo octree structure, bit string.
/// @param bit_idx
/// @return valid bit_idx that corresponds to a leaf node
OCTREE_FUNCTION
inline int tree_bit_idx_leaf(const ot_tree_t* tree, const int bit_idx) {
  if(tree_isset_bit(tree, tree_parent_bit_idx(bit_idx))) {
    return bit_idx;
  }
  else if(tree_isset_bit(tree, tree_parent_bit_idx(tree_parent_bit_idx(bit_idx)))) {
    return tree_parent_bit_idx(bit_idx);
  }
  else if(tree_isset_bit(tree, 0)) {
    return tree_parent_bit_idx(tree_parent_bit_idx(bit_idx));
  }
  else {
    return 0;
  }
}

/// Computes the bit_idx of voxel at (bd,bh,bw) within a shallow octree.
/// Note: This will be not a useable bit_idx, one hast to call tree_bit_idx_leaf,
/// or use tree_bit_idx instead.
/// @param bd
/// @param bh
/// @param bw
/// @return bit_idx 
OCTREE_FUNCTION
inline int tree_bit_idx_(const int bd, const int bh, const int bw) {
  // return (1 + 8 + 64) + 
  //        isset_bit(bw, 0) * 1 + isset_bit(bw, 1) * 8  + isset_bit(bw, 2) * 64  +
  //        isset_bit(bh, 0) * 2 + isset_bit(bh, 1) * 16 + isset_bit(bh, 2) * 128 +
  //        isset_bit(bd, 0) * 4 + isset_bit(bd, 1) * 32 + isset_bit(bd, 2) * 256;
  return (1 + 8 + 64) + 
         (bw%2 == 1) * 1 + (bw/2%2 == 1) * 8  + (bw/4%2 == 1) * 64  +
         (bh%2 == 1) * 2 + (bh/2%2 == 1) * 16 + (bh/4%2 == 1) * 128 +
         (bd%2 == 1) * 4 + (bd/2%2 == 1) * 32 + (bd/4%2 == 1) * 256;
}



/// Computes the bit_idx in a shallow octree as encoded in tree using the subscript
/// indices bd, bh, and bw.
///
/// @param tree shallow octree structure, bit string.
/// @param bd
/// @param bh
/// @param bw
/// @return bit_idx that corresponds to the subscript indices.
OCTREE_FUNCTION
inline int tree_bit_idx(const ot_tree_t* tree, const int bd, const int bh, const int bw) {
  // const int bit_idx = (1 + 8 + 64) + 
  //        isset_bit(bw, 0) * 1 + isset_bit(bw, 1) * 8  + isset_bit(bw, 2) * 64  +
  //        isset_bit(bh, 0) * 2 + isset_bit(bh, 1) * 16 + isset_bit(bh, 2) * 128 +
  //        isset_bit(bd, 0) * 4 + isset_bit(bd, 1) * 32 + isset_bit(bd, 2) * 256;
  const int bit_idx = tree_bit_idx_(bd,bh,bw);
  return tree_bit_idx_leaf(tree, bit_idx);
}

/// Computes the offset in the data array of a shallow octree for a specific 
/// octree cell corresponding to bit_idx. 
///
/// As one can not use linear subscript indices in shallow octree structure, the
/// offset in the shallow octree data array has to be computed based on the 
/// bit string that encodes the structure.
/// This function takes already the feature_size of the data array for the 
/// offset into account.
///
/// @param tree
/// @param bit_idx
/// @param feature_size
/// @return data_idx, the offset in the shallow octree data array.
/// @see octree_get_data
OCTREE_FUNCTION
inline int tree_data_idx(const ot_tree_t* tree, const int bit_idx, ot_size_t feature_size) {
  if(bit_idx == 0) {
    return 0;
  }
  int data_idx = tree_cnt0(tree, 0, IMIN(bit_idx, 73)); 
  if(tree_parent_bit_idx(bit_idx) > 1) {
    data_idx -= 8 * tree_cnt0(tree, 1, tree_parent_bit_idx(bit_idx));
  }
  if(bit_idx > 72) {
    data_idx += bit_idx - 73;
  }

  return data_idx * feature_size;
}


/// Computes the number of leaf cells in a shallow octree by parsing the 
/// bit string that corresponds to the structure.
///
/// @param tree
/// @return number of leaf cells in the shallow octree
OCTREE_FUNCTION
inline int tree_n_leafs(const ot_tree_t* tree) {
  int n = tree_cnt1(tree, 0, 73);
  return n * 8 - n + 1;
}
/// Computes the number of split nodes in a shallow octree by parsing the 
/// bit string that corresponds to the structure.
///
/// @param tree
/// @return number of split nodes in the shallow octree
OCTREE_FUNCTION
inline int tree_n_splits(const ot_tree_t* tree) {
  return tree_cnt1(tree, 0, 73);
}



/// This function computes for a given leaf_idx the corresponding grid_idx.
/// Therefore, it computes to which shallow octree a cell belongs using binary
/// search.
///
/// @param grid
/// @param leaf_idx flat index of the leaf cell in the complete grid-octree structure
/// @return grid_idx
OCTREE_FUNCTION
inline int leaf_idx_to_grid_idx(const octree* grid, const int leaf_idx) {
  const int n_blocks = octree_num_blocks(grid);
  
  int l = 0;
  int r = n_blocks;
  while(l <= r) {
    const int m = (l + r) / 2;
    const int am = grid->prefix_leafs[m];
    if(am <= leaf_idx && (m == n_blocks-1 || leaf_idx < grid->prefix_leafs[m+1])) {
      return m;
    }
    else {
      if(am < leaf_idx) {
        l = m + 1;
      }
      else {
        r = m - 1;
      }
    }
  }
  return -1;
}


/// Given the data_idx of a shallow octree, this function computes the corresponding
/// bit_idx for the shallow octree encoded in tree.
///
/// @param tree
/// @param data_idx
/// @return bit_idx
OCTREE_FUNCTION
inline int data_idx_to_bit_idx(const ot_tree_t* tree, int data_idx) {
  if(!tree_isset_bit(tree, 0)) {
    return 0;
  }

  const int n_leafs_l1 = 8 - tree_cnt1(tree, 1, 9);
  const int n_leafs_l2 = (8 - n_leafs_l1) * 8 - tree_cnt1(tree, 9, 73);

  int bit_idx;
  if(data_idx < n_leafs_l1) {
    //bit_idx in l1
    bit_idx = 1;
  }
  else if(data_idx < n_leafs_l1 + n_leafs_l2) {
    //bit_idx in l2
    data_idx -= n_leafs_l1;
    bit_idx = 9;
  }
  else {
    //bit_idx in l3
    data_idx -= (n_leafs_l1 + n_leafs_l2);
    bit_idx = 73;
  }

  while(data_idx >= 0) {
    if(tree_isset_bit(tree, tree_parent_bit_idx(bit_idx))) {
      for(int idx = 0; idx < 8; ++idx) { 
        if(bit_idx > 72 || !tree_isset_bit(tree, bit_idx)) {
          data_idx--;
        }
        if(data_idx < 0) {
          return bit_idx;
        }
        bit_idx++;
      }
    }
    else {
      bit_idx += 8;
    }
  }

  return bit_idx;
}

/// Computes the depth of a leaf cell in the shallow octree corresponding to
/// the bit_idx.
///
/// @param bit_idx
/// @return depth of the leaf cell.
OCTREE_FUNCTION
inline int depth_from_bit_idx(const int bit_idx) {
  return bit_idx == 0 ? 0 : (bit_idx < 9 ? 1 : (bit_idx < 73 ? 2 : 3));
}

/// The width of a shallow octree cell is determined by its depth.
///
/// @param depth
/// @param width of the shallow octree cell (number of voxels in one dimension).
/// @return cell width
OCTREE_FUNCTION
inline int width_from_depth(const int depth) {
  return (1 << (3 - depth));
}

/// Computes the width of a shallow octree cell corresponding to the bit_idx and
/// therefore to the depth of the cell in the octree.
///
/// @param depth
/// @param width of the shallow octree cell (number of voxels in one dimension).
/// @return cell width
OCTREE_FUNCTION
inline int width_from_bit_idx(const int bit_idx) {
  return bit_idx == 0 ? 8 : (bit_idx < 9 ? 4 : (bit_idx < 73 ? 2 : 1));
}

/// Splits the flat grid_idx of the grid-octree structure into the corresponding
/// subscript indices n,d,h,w.
///
/// @param in
/// @param grid_idx
/// @param n
/// @param d
/// @param h
/// @param w
/// return void
OCTREE_FUNCTION
inline void octree_split_grid_idx(const octree* in, const int grid_idx, int* n, int* d, int* h, int* w) {
  (*w) = (grid_idx % in->grid_width);
  (*h) = ((grid_idx - (*w)) / in->grid_width) % in->grid_height;
  (*d) = (((grid_idx - (*w)) / in->grid_width - (*h)) / in->grid_height) % in->grid_depth;
  (*n) = (grid_idx / (in->grid_depth * in->grid_height * in->grid_width));
}

/// Computes the subscript indices the correspond to a bit index, assuming the
/// bit_idx is on depth 1.
/// @attention adds the value to d,h,w
///
/// @param bit_idx
/// @param d
/// @param h
/// @param w
/// @return void
OCTREE_FUNCTION
inline void bdhw_from_idx_l1(const int bit_idx, int* d, int* h, int* w) {
  (*d) += 4 * ((bit_idx - 1)/4);
  (*h) += 4 * ((bit_idx - 1)%4/2);
  (*w) += 4 * ((bit_idx - 1)%2);
}

/// Computes the subscript indices the correspond to a bit index, assuming the
/// bit_idx is on depth 2.
/// @attention adds the value to d,h,w
///
/// @param bit_idx
/// @param d
/// @param h
/// @param w
/// @return void
OCTREE_FUNCTION
inline void bdhw_from_idx_l2(const int bit_idx, int* d, int* h, int* w) {
  (*d) += 4 * ((bit_idx - 9) / 8/4);
  (*h) += 4 * ((bit_idx - 9) / 8%4/2);
  (*w) += 4 * ((bit_idx - 9) / 8%2);

  (*d) += 2 * (((bit_idx - 9) % 8)/4);
  (*h) += 2 * (((bit_idx - 9) % 8)%4/2);
  (*w) += 2 * (((bit_idx - 9) % 8)%2);
}

/// Computes the subscript indices the correspond to a bit index, assuming the
/// bit_idx is on depth 3.
/// @attention adds the value to d,h,w
///
/// @param bit_idx
/// @param d
/// @param h
/// @param w
/// @return void
OCTREE_FUNCTION
inline void bdhw_from_idx_l3(const int bit_idx, int* d, int* h, int* w) {
  (*d) += (4 * (((tree_parent_bit_idx(bit_idx) - 9) / 8)/4));
  (*h) += (4 * (((tree_parent_bit_idx(bit_idx) - 9) / 8)%4/2));
  (*w) += (4 * (((tree_parent_bit_idx(bit_idx) - 9) / 8)%2));

  (*d) += (2 * (((tree_parent_bit_idx(bit_idx) - 9) % 8)/4));
  (*h) += (2 * (((tree_parent_bit_idx(bit_idx) - 9) % 8)%4/2));
  (*w) += (2 * (((tree_parent_bit_idx(bit_idx) - 9) % 8)%2));

  (*d) += ((bit_idx - 73) % 8)/4;
  (*h) += ((bit_idx - 73) % 8)%4/2;
  (*w) += ((bit_idx - 73) % 8)%2;
}

/// Computes the subscript indices for a dense volume for the grid-octree 
/// structure grid given the grid_idx and bit_idx.
///
/// @param grid
/// @param grid_idx
/// @param bit_idx
/// @param n
/// @param d
/// @param h
/// @param w
/// @return depth of the corresponding shallow octree cell (bit_idx).
OCTREE_FUNCTION
inline int octree_ind_to_dense_ind(const octree* grid, const int grid_idx, const int bit_idx, int* n, int* d, int* h, int* w) {
  octree_split_grid_idx(grid, grid_idx, n,d,h,w); 
  d[0] *= 8;
  h[0] *= 8;
  w[0] *= 8;
  
  int depth = depth_from_bit_idx(bit_idx);
  if(depth == 1) {
    bdhw_from_idx_l1(bit_idx, d,h,w);
  }
  else if(depth == 2) {
    bdhw_from_idx_l2(bit_idx, d,h,w);
  }
  else if(depth == 3) {
    bdhw_from_idx_l3(bit_idx, d,h,w);
  }

  return depth;
}

#endif
