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
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <limits>

#include <smmintrin.h>

#include "octnet/gpu/gpu.h"
#include "octnet/gpu/common.h"
#include "octnet/cpu/cpu.h"
#include "octnet/test/objects.h"


inline int leaf_idx_to_grid_idx2(const octree* grid, const int leaf_idx) {
  // printf("----------\n");
  const int n_blocks = octree_num_blocks(grid);
  
  int l = 0;
  int r = n_blocks;
  while(l <= r) {
    const int m = (l + r) / 2;
    // const int am = n_leafs_upto(grid, m);
    const int am = grid->prefix_leafs[m];
    // printf("%d: %d,%d %d, %d\n", leaf_idx, l,r, m, am);
    // printf("  %d <= %d && (%d == %d || %d < %d)\n", am, leaf_idx, m, n_blocks-1, am, n_leafs_upto(grid, m+1));
    // if(am <= leaf_idx && (m == n_blocks-1 || leaf_idx < n_leafs_upto(grid, m+1))) {
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


void correctness_cpu(const octree* grid) {
  for(int leaf_idx = 0; leaf_idx < grid->n_leafs; ++leaf_idx) {
    int grid_idx = leaf_idx_to_grid_idx(grid, leaf_idx);
    int grid_idx2 = leaf_idx_to_grid_idx2(grid, leaf_idx);
    if(grid_idx != grid_idx2) {
      std::cout << "[ERROR] grid_idx for leaf_idx " << leaf_idx << " does not match " << grid_idx2 << " vs. " << grid_idx << std::endl;
    }
  }
}

void speed_cpu(const octree* grid) {
  int reps = 1000;
  
  int di = 0;
  auto t1 = std::chrono::high_resolution_clock::now();
  for(int rep = 0; rep < reps; ++rep) {
    for(int leaf_idx = 0; leaf_idx < grid->n_leafs; ++leaf_idx) {
      int grid_idx = leaf_idx_to_grid_idx(grid, leaf_idx);
      di += grid_idx;
    }
  }
  auto t2 = std::chrono::high_resolution_clock::now();
  auto time_span = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1);
  std::cout << "cpu old took " << time_span.count() << "[s]" << std::endl;
  std::cout << di << std::endl; 

  di = 0;
  t1 = std::chrono::high_resolution_clock::now();
  for(int rep = 0; rep < reps; ++rep) {
    for(int leaf_idx = 0; leaf_idx < grid->n_leafs; ++leaf_idx) {
      int grid_idx = leaf_idx_to_grid_idx2(grid, leaf_idx);
      di += grid_idx;
    }
  }
  t2 = std::chrono::high_resolution_clock::now();
  time_span = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1);
  std::cout << "cpu new took " << time_span.count() << "[s]" << std::endl;
  std::cout << di << std::endl;
}


int main(int argc, char** argv) {
  std::cout << "[IMPROVE] tree_data_idx" << std::endl;
  srand(42);

  octree* grid;

  grid = create_test_octree_rand(1,2,1,1, 2, 1.0,0.0,0.0);
  correctness_cpu(grid);
  octree_free_cpu(grid);
  
  //test set 1
  std::cout << "test 1" << std::endl;
  grid = create_test_octree_1();
  correctness_cpu(grid);
  speed_cpu(grid);
  octree_free_cpu(grid);

  //test set 2
  std::cout << "test 2" << std::endl;
  grid = create_test_octree_2();
  correctness_cpu(grid);
  speed_cpu(grid);
  octree_free_cpu(grid);

  //test set 3
  std::cout << "test 3" << std::endl;
  grid = create_test_octree_3();
  correctness_cpu(grid);
  speed_cpu(grid);
  octree_free_cpu(grid);

  //test set 4
  std::cout << "test 4" << std::endl;
  grid = create_test_octree_4();
  correctness_cpu(grid);
  speed_cpu(grid);
  octree_free_cpu(grid);

  //test set 5
  std::cout << "test 5" << std::endl;
  grid = create_test_octree_rand(2,4,5,6, 3, 0.5,0.2,0.1);
  correctness_cpu(grid);
  speed_cpu(grid);
  octree_free_cpu(grid);

  //test set 6
  std::cout << "test 6" << std::endl;
  grid = create_test_octree_rand(2,4,5,6, 3, 0.5,0.5,0.5);
  correctness_cpu(grid);
  speed_cpu(grid);
  octree_free_cpu(grid);

  //test set 7
  std::cout << "test 7" << std::endl;
  grid = create_test_octree_rand(2,4,5,6, 3, 0.75,0.75,0.75);
  correctness_cpu(grid);
  speed_cpu(grid);
  octree_free_cpu(grid);

  
  std::cout << "[DONE]" << std::endl;
  return 0;
}
