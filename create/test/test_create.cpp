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

#include "octnet/cpu/cpu.h"
#include "octnet/cpu/io.h"
#include "octnet/create/create.h"

#include <cstring>

int main(int argc, char** argv) {    
  char* off_path = argv[1];
  printf("off_path: %s\n", off_path);

  // int depth =  8*4;
  // int height = 8*4;
  // int width =  8*4;
  // float R[] = {1,0,0, 0,1,0, 0,0,1};
  // bool fit = !strcmp(argv[2], "1");
  // int fit_multiply = 2;
  // bool pack = false;
  // int pad = 1;
  // int n_threads = 1;
  // octree* grid = octree_create_from_off_cpu(off_path, depth, height, width, R, fit, fit_multiply, pack, pad, n_threads);
  // printf("created_octree: %d,%d,%d, leafs %d\n", grid->grid_depth, grid->grid_height, grid->grid_width, grid->n_leafs);
  
  // octree_write_ply_boxes_cpu("test.ply", grid);
  // octree_free_cpu(grid);
  

  // OctreeFromPC create(42);
  // octree* grid = create(false, 1, false, 12);
  // octree_free_cpu(grid);

  return 0;
}
