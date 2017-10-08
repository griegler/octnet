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
#include <vector>
#include <string.h>

#include "octnet/test/objects.h"

#include "octnet/cpu/cpu.h"
#include "octnet/cpu/combine.h"
#include "octnet/cpu/dense.h"
#include "octnet/cpu/io.h"
#include "octnet/cpu/unpool.h"
#include "octnet/cpu/split.h"

void test_split_grid_idx_(int n, int grid_depth, int grid_height, int grid_width) {
  octree grid;
  grid.n = n;
  grid.grid_depth = grid_depth;
  grid.grid_height = grid_height;
  grid.grid_width = grid_width;

  int grid_idx = 0;
  for(int gn = 0; gn < n; ++gn) {
    for(int gd = 0; gd < grid_depth; ++gd) {
      for(int gh = 0; gh < grid_height; ++gh) {
        for(int gw = 0; gw < grid_width; ++gw) {
          int tn, td, th, tw;
          octree_split_grid_idx(&grid, grid_idx, &tn,&td,&th,&tw);
          if(!(gn == tn && gd == td && gh == th && gw == tw)) {
            printf("[ERROR in test_split_grid_idx, grid_idx=%d, gt %d,%d,%d,%d, cm %d,%d,%d,%d\n", grid_idx, gn,gd,gh,gw, tn,td,th,tw);
            exit(-1);
          }
          grid_idx++;
        }
      }
    }
  }
}

void test_split_grid_idx() {
  std::cout << "[INFO] test_split_grid_idx" << std::endl;

  test_split_grid_idx_(1,1,1,1);
  test_split_grid_idx_(1,8,8,8);
  test_split_grid_idx_(1,8,16,32);
  test_split_grid_idx_(1,11,23,31);
  test_split_grid_idx_(4,1,1,1);
  test_split_grid_idx_(4,8,8,8);
  test_split_grid_idx_(4,8,16,32);
  test_split_grid_idx_(4,11,23,31);
  
  std::cout << "[INFO] done test_split_grid_idx" << std::endl;
}


void test_cdhw_to_octree() {
  std::cout << "[INFO] test_cdhw_to_octree" << std::endl;
  
  int gn = 2;
  int gd = 1;
  int gh = 1;
  int gw = 1;
  int fs = 1;
  octree* grid = create_test_octree_rand(gn,gd,gh,gw, fs, 0,0,0);
  ot_data_t* dense = new ot_data_t[gn*gd*8*gh*8*gw*8*fs];
  octree_to_cdhw_cpu(grid, gd*8,gh*8,gw*8, dense);

  octree* grid2 = octree_new_cpu();
  cdhw_to_octree_avg_cpu(grid, gd*8, gh*8, gw*8, dense, fs, grid2);

  octree_free_cpu(grid);
  delete[] dense;
  octree_free_cpu(grid2);
  
  std::cout << "[INFO] done test_split_grid_idx" << std::endl;
}

void test_combine_extract_n() {
  std::cout << "[INFO] test_combine_extract_n" << std::endl;
  int n_grids = 16;
  int gn = 1; int gd = 8; int gh = 16; int gw = 32; int fs = 8;
  octree** grids = new octree*[n_grids];
  for(int idx = 0; idx < n_grids; ++idx) {
    grids[idx] = create_test_octree_rand(gn,gd,gh,gw, fs, 0.5,0.5,0.5);
    // octree_print_cpu(grids[idx]);
    if(!octree_equal_cpu(grids[idx], grids[idx])) {
      printf("[ERROR] in generate\n");
    }
  }

  octree* cmb = octree_new_cpu();
  octree_combine_n_cpu(grids, n_grids, cmb);

  for(int idx = 0; idx < n_grids; ++idx) {
    octree* ext = octree_new_cpu();
    octree_extract_n_cpu(cmb, idx, idx+1, ext);

    if(!octree_equal_cpu(grids[idx], ext)) {
      printf("[ERROR] extracted grid does not match generated grid\n");
    }
    
    octree_free_cpu(ext);
  }
  
  octree_free_cpu(cmb);

  for(int idx = 0; idx < n_grids; ++idx) {
    octree_free_cpu(grids[idx]);
  }
  delete[] grids;
  std::cout << "[DONE]" << std::endl;
}

void test_IO(int n_threads) {
  std::cout << "[INFO] test_IO" << std::endl;
  int n = 8;
  int gd = 8; int gh = 16; int gw = 32; int fs = 8;
  octree** grids = new octree*[n];
  std::vector<std::string> paths;
  char** paths_c = new char*[n];
  for(int idx = 0; idx < n; ++idx) {
    std::stringstream ss;
    ss << "test_" << idx << ".oc";
    paths.push_back(ss.str());
    paths_c[idx] = new char[paths[idx].size() + 1];
    strcpy(paths_c[idx], paths[idx].c_str());

    grids[idx] = create_test_octree_rand(1,gd,gh,gw, fs, 0.5,0.5,0.5);
    printf("  write oc to %s\n", paths[idx].c_str());
    octree_write_cpu(paths[idx].c_str(), grids[idx]);
  }

  for(int idx = 0; idx < n; ++idx) {
    printf("  read oc to %s\n", paths[idx].c_str());
    octree* grid = octree_new_cpu();
    octree_read_cpu(paths[idx].c_str(), grid);
    if(!octree_equal_cpu(grids[idx], grid)) {
      printf("[ERROR] octrees do not match\n");
    }
    octree_free_cpu(grid);
  }

  octree* grid = octree_new_cpu();
  octree_read_batch_cpu(n, paths_c, n_threads, grid);
  for(int idx = 0; idx < n; ++idx) {
    octree* grid_ext = octree_new_cpu();
    octree_extract_n_cpu(grid, idx, idx+1, grid_ext);
    if(!octree_equal_cpu(grids[idx], grid_ext)) {
      printf("[ERROR] grid_ext is not the same as grids[%d]\n", idx);
    }
    octree_free_cpu(grid_ext);
  }
  octree_free_cpu(grid);
  
  for(int idx = 0; idx < n; ++idx) {
    octree_free_cpu(grids[idx]);
    delete[] paths_c[idx];
  }
  delete[] paths_c;
  delete[] grids;
  std::cout << "[DONE]" << std::endl;
}

void test_split_rec_surf() {
  
  octree* rec = create_test_octree_rand(1, 1,1,1, 1, 0,0,0);
  octree* in = octree_new_cpu();
  octree_gridunpool2x2x2_cpu(rec, in);

  octree* out = octree_new_cpu();
  octree_split_reconstruction_surface_cpu(in, rec, 0, 1e9, out);

  octree_free_cpu(rec);
  octree_free_cpu(in);
  octree_free_cpu(out);
}

int main() {
  srand(time(NULL));
  
  test_split_grid_idx();
  test_cdhw_to_octree();
  test_combine_extract_n();
  test_IO(1); test_IO(4);
  test_split_rec_surf();

  return 0;
}
