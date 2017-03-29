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

#include "octnet/gpu/gpu.h"
#include "octnet/gpu/buffer.h"
#include "octnet/gpu/conv.h"
#include "octnet/cpu/cpu.h"
#include "octnet/cpu/io.h"

#include <cublas_v2.h>

#include <iostream>

inline int z_curve_x(const int z) {
  // return (((z & 21) & 1)) + (((z & 21) & 4) >> 1) + (((z & 21) & 16) >> 2);
  int x = z & 0x55555555;
  x = (x ^ (x >> 1)) & 0x33333333;
  x = (x ^ (x >> 2)) & 0x0f0f0f0f;
  return x;
}

inline int z_curve_y(const int z) {
  // return (((z & 42) & 2) >> 1) + (((z & 42) & 8) >> 2) + (((z & 42) & 32) >> 3);
  return z_curve_x(z >> 1);
}

int main(int argc, char *argv[]) {
  // test z-curve iteration
  for(int z = 0; z < 64; ++z) {
    printf("%d -> %d,%d\n", z, z_curve_x(z), z_curve_y(z));
  }


  const char* grid_path = "../th/oc/test_octrees/table_128.bin";
  // const char* grid_path = "../th/oc/test_octrees/table_256.bin";

  octree* grid_h = octree_new_cpu();
  octree_read_cpu(grid_path, grid_h); 

  octree* grid_d_in = octree_new_gpu();
  octree_to_gpu(grid_h, grid_d_in);

  ot_data_t_buffer* col_buffer = new_ot_data_t_buffer_gpu();

  octree* grid_d_out = octree_new_gpu();

  int cin = 1;
  int cout = 1;
  ot_data_t* weights = device_malloc<ot_data_t>(cin*cout*3*3*3);
  ot_data_t* bias = device_malloc<ot_data_t>(cout);

  cublasHandle_t handle;
  cublasCreate(&handle);

  for(int rep = 0; rep < 10; ++rep) {
    octree_conv_mm_gpu(handle, grid_d_in, weights, bias, col_buffer, cout, grid_d_out);
  }

  cublasDestroy(handle);

  octree_free_cpu(grid_h);
  octree_free_gpu(grid_d_in);
  octree_free_gpu(grid_d_out);

  free_ot_data_t_buffer_gpu(col_buffer);

  return 0;
}
