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
#include <chrono>
#include <limits>

#include <smmintrin.h>

#include "octnet/gpu/gpu.h"
#include "octnet/gpu/common.h"
#include "octnet/cpu/cpu.h"
#include "octnet/test/objects.h"


OCTREE_FUNCTION
inline int data_idx_to_bit_idx2(const ot_tree_t* tree, int data_idx) {
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
#pragma unroll
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




void correctness_cpu(ot_tree_t* tree) {
  std::cout << "---------------------- test correctness cpu --------------------" << std::endl;
  std::cout << tree_bit_str_cpu(tree) << std::endl;
  int n_leafs = tree_n_leafs(tree);
  for(int data_idx = 0; data_idx < n_leafs; ++data_idx) {
    int di_gt = data_idx_to_bit_idx(tree, data_idx);
    int di2 = data_idx_to_bit_idx2(tree, data_idx); 

    if(di2 != di_gt) {
      std::cout << "[ERROR_CPU] data_idx=" << data_idx << ": " << di2 << " should be " << di_gt << std::endl;
    }
  }
}

void speed_cpu(ot_tree_t* tree) {
  int reps = 10000;
  
  int n_leafs = tree_n_leafs(tree);

  int di = 0;
  auto t1 = std::chrono::high_resolution_clock::now();
  for(int rep = 0; rep < reps; ++rep) {
    for(int data_idx = 0; data_idx < n_leafs; ++data_idx) {
      int tmp = data_idx_to_bit_idx(tree, data_idx);
      di += tmp;
    }
  }
  auto t2 = std::chrono::high_resolution_clock::now();
  auto time_span = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1);
  std::cout << "cpu old took " << time_span.count() << "[s]" << std::endl;
  std::cout << di << std::endl; 

  di = 0;
  t1 = std::chrono::high_resolution_clock::now();
  for(int rep = 0; rep < reps; ++rep) {
    for(int data_idx = 0; data_idx < n_leafs; ++data_idx) {
      int tmp = data_idx_to_bit_idx2(tree, data_idx);
      di += tmp;
    }
  }
  t2 = std::chrono::high_resolution_clock::now();
  time_span = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1);
  std::cout << "cpu new took " << time_span.count() << "[s]" << std::endl;
  std::cout << di << std::endl;
}





__global__ void kernel_correctness(const ot_tree_t* tree, int n_leafs) {
  CUDA_KERNEL_LOOP(data_idx, n_leafs) { 
    int di_gt = data_idx_to_bit_idx(tree, data_idx);
    int di2 = data_idx_to_bit_idx2(tree, data_idx);
    if(di2 != di_gt) {
      printf("[ERROR_GPU] data_idx=%d: %d should be %d\n", data_idx, di2, di_gt);
    }
  }
}

void correctness_gpu(ot_tree_t* tree_h) {
  std::cout << "---------------------- test correctness gpu --------------------" << std::endl;
  std::cout << tree_bit_str_cpu(tree_h) << std::endl;
  int n_leafs = tree_n_leafs(tree_h);
  
  ot_tree_t* tree_d = host_to_device_malloc(tree_h, N_TREE_INTS);

  kernel_correctness<<<GET_BLOCKS(n_leafs), CUDA_NUM_THREADS>>>(
      tree_d, n_leafs 
  );
  CUDA_POST_KERNEL_CHECK;

  device_free(tree_d);
}

__global__ void kernel_speed1(const ot_tree_t* tree, int n_leafs) {
  CUDA_KERNEL_LOOP(data_idx, n_leafs) { 
    int di = data_idx_to_bit_idx(tree, data_idx);
    if(di > 1000000 ) {
      printf("[ERROR_GPU] you summoned an evil demon\n");
    }
  }
}
__global__ void kernel_speed2(const ot_tree_t* tree, int n_leafs) {
  CUDA_KERNEL_LOOP(data_idx, n_leafs) { 
    int di = data_idx_to_bit_idx2(tree, data_idx);
    if(di > 1000000 ) {
      printf("[ERROR_GPU] you summoned an evil demon\n");
    }
  }
}

void speed_gpu(ot_tree_t* tree_h) {
  int reps = 10000;
  int n_leafs = tree_n_leafs(tree_h);
  
  ot_tree_t* tree_d = host_to_device_malloc(tree_h, N_TREE_INTS);

  kernel_speed1<<<GET_BLOCKS(n_leafs), CUDA_NUM_THREADS>>>(tree_d, n_leafs);
  auto t1 = std::chrono::high_resolution_clock::now();
  for(int rep = 0; rep < reps; ++rep) {
    kernel_speed1<<<GET_BLOCKS(n_leafs), CUDA_NUM_THREADS>>>(tree_d, n_leafs);
    CUDA_POST_KERNEL_CHECK;
  }
  auto t2 = std::chrono::high_resolution_clock::now();
  auto time_span = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1);
  std::cout << "gpu old took " << time_span.count() << "[s]" << std::endl;

  kernel_speed2<<<GET_BLOCKS(n_leafs), CUDA_NUM_THREADS>>>(tree_d, n_leafs);
  t1 = std::chrono::high_resolution_clock::now();
  for(int rep = 0; rep < reps; ++rep) {
    kernel_speed2<<<GET_BLOCKS(n_leafs), CUDA_NUM_THREADS>>>(tree_d, n_leafs);
    CUDA_POST_KERNEL_CHECK;
  }
  t2 = std::chrono::high_resolution_clock::now();
  time_span = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1);
  std::cout << "gpu new took " << time_span.count() << "[s]" << std::endl;
  
  device_free(tree_d);
}



int main(int argc, char** argv) {
  std::cout << "[IMPROVE] data_idx_to_bit_idxd" << std::endl;
  srand(42);

  octree* grid;

  //test set 1
  std::cout << "test 1" << std::endl;
  grid = create_test_octree_1();
  correctness_cpu(octree_get_tree(grid, 0));
  speed_cpu(octree_get_tree(grid, 0));
  correctness_gpu(octree_get_tree(grid, 0));
  speed_gpu(octree_get_tree(grid, 0));
  octree_free_cpu(grid);

  //test set 2
  std::cout << "test 2" << std::endl;
  grid = create_test_octree_2();
  correctness_cpu(octree_get_tree(grid, 0));
  speed_cpu(octree_get_tree(grid, 0));
  correctness_gpu(octree_get_tree(grid, 0));
  speed_gpu(octree_get_tree(grid, 0));
  octree_free_cpu(grid);

  //test set 3
  std::cout << "test 3" << std::endl;
  grid = create_test_octree_3();
  correctness_cpu(octree_get_tree(grid, 0));
  speed_cpu(octree_get_tree(grid, 0));
  correctness_gpu(octree_get_tree(grid, 0));
  speed_gpu(octree_get_tree(grid, 0));
  octree_free_cpu(grid);

  //test set 4
  std::cout << "test 4" << std::endl;
  grid = create_test_octree_4();
  correctness_cpu(octree_get_tree(grid, 0));
  speed_cpu(octree_get_tree(grid, 0));
  correctness_gpu(octree_get_tree(grid, 0));
  speed_gpu(octree_get_tree(grid, 0));
  octree_free_cpu(grid);

  //test set 5
  std::cout << "test 5" << std::endl;
  grid = create_test_octree_rand(1, 4,5,6, 3, 0.5,0.2,0.1);
  correctness_cpu(octree_get_tree(grid, 0));
  speed_cpu(octree_get_tree(grid, 0));
  correctness_gpu(octree_get_tree(grid, 0));
  speed_gpu(octree_get_tree(grid, 0));
  octree_free_cpu(grid);

  //test set 6
  std::cout << "test 6" << std::endl;
  grid = create_test_octree_rand(1, 4,5,6, 3, 0.5,0.5,0.5);
  correctness_cpu(octree_get_tree(grid, 0));
  speed_cpu(octree_get_tree(grid, 0));
  correctness_gpu(octree_get_tree(grid, 0));
  speed_gpu(octree_get_tree(grid, 0));
  octree_free_cpu(grid);

  //test set 7
  std::cout << "test 7" << std::endl;
  grid = create_test_octree_rand(1, 4,5,6, 3, 0.75,0.75,0.75);
  for(int grid_idx = 0; grid_idx < 4; ++grid_idx) {
    correctness_cpu(octree_get_tree(grid, grid_idx));
    speed_cpu(octree_get_tree(grid, grid_idx));
    correctness_gpu(octree_get_tree(grid, grid_idx));
    speed_gpu(octree_get_tree(grid, grid_idx));
  }
  octree_free_cpu(grid);

  std::cout << "[DONE]" << std::endl;
  
  return 0;
}
