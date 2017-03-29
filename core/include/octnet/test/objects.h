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

#ifndef OCTREE_TEST_OBJECTS
#define OCTREE_TEST_OBJECTS

#include "octnet/cpu/cpu.h"

float randf() {
  return float(rand()) / float(RAND_MAX);
}

/// Creates a random test octree instance with the given shape.
///
/// @param gn batch size, number of shallow octree grids.
/// @param gd number of shallow octrees in depth dimension.
/// @param gh number of shallow octrees in depth dimension.
/// @param gw number of shallow octrees in depth dimension.
/// @param fs feature size.
/// @param sp0 split probability of the shallow octree cell on depth 0.
/// @param sp1 split probability of the shallow octree cell on depth 1.
/// @param sp2 split probability of the shallow octree cell on depth 2.
/// @param min_val minimum value for the random octree data.
/// @param max_val maximum value for the random octree data.
/// @return an octree*.
octree* create_test_octree_rand(int gn, int gd, int gh, int gw, int fs, float sp0, float sp1, float sp2, float min_val=-1.f, float max_val=1.f) {
  octree* grid = octree_new_cpu();
  octree_resize_cpu(gn, gd, gh, gw, fs, 0, grid);
  
  octree_clr_trees_cpu(grid);
  for(int grid_idx = 0; grid_idx < octree_num_blocks(grid); ++grid_idx) {
    ot_tree_t* tree = octree_get_tree(grid, grid_idx);
    if(randf() <= sp0) {
      tree_set_bit(tree, 0);
      for(int bit_idx_l1 = 1; bit_idx_l1 <= 8; ++bit_idx_l1) {
        if(randf() <= sp1) {
          tree_set_bit(tree, bit_idx_l1);
          int bit_idx_l2 = tree_child_bit_idx(bit_idx_l1);
          for(int idx_l2 = 0; idx_l2 < 8; ++idx_l2) {
            if(randf() <= sp2) {
              tree_set_bit(tree, bit_idx_l2);
            }
            bit_idx_l2++; 
          }
        }
      }
    }
  }

  octree_upd_n_leafs_cpu(grid);
  octree_resize_as_cpu(grid, grid);
  octree_upd_prefix_leafs_cpu(grid);

  for(int idx = 0; idx < grid->n_leafs * grid->feature_size; ++idx) {
    grid->data[idx] = randf() * (max_val - min_val) + min_val;
  }
  
  return grid;
}

#endif 
