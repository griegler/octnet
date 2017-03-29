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

#include "octnet/cpu/math.h"
#include "octnet/cpu/cpu.h"

#include <cstdio>
#include <cstdlib>

#if defined(_OPENMP)
#include <omp.h>
#endif


extern "C"
void octree_add_cpu(const octree* in1, ot_data_t fac1, const octree* in2, ot_data_t fac2, bool check, octree* out) {
  if(check && (in1->feature_size != in2->feature_size || !octree_equal_trees_cpu(in1, in2))) {
    printf("[ERROR] add - tree structure of inputs do not match\n");
    exit(-1);
  }

  //check if inplace
  if(out != in1 && out != in2) {
    octree_resize_as_cpu(in1, out);
    octree_cpy_scalars(in1, out);
    octree_cpy_trees_cpu_cpu(in1, out);
    octree_cpy_prefix_leafs_cpu_cpu(in1, out);
  }

  ot_size_t n = in1->n_leafs * in1->feature_size;
  #pragma omp parallel for
  for(int idx = 0; idx < n; ++idx) {
    out->data[idx] = fac1 * in1->data[idx] + fac2 * in2->data[idx];
  }
}


extern "C"
void octree_scalar_mul_cpu(octree* grid, const ot_data_t scalar) {
  int n = grid->n_leafs * grid->feature_size;
  #pragma omp parallel for
  for(int idx = 0; idx < n; ++idx) {
    grid->data[idx] *= scalar; 
  }
}

extern "C"
void octree_scalar_add_cpu(octree* grid, const ot_data_t scalar) {
  int n = grid->n_leafs * grid->feature_size;
  #pragma omp parallel for
  for(int idx = 0; idx < n; ++idx) {
    grid->data[idx] += scalar; 
  }
}


extern "C"
ot_data_t octree_min_cpu(const octree* grid_in) {
  int n = grid_in->n_leafs * grid_in->feature_size;
  ot_data_t min = grid_in->data[0];
  for(int idx = 1; idx < n; ++idx) {
    ot_data_t val = grid_in->data[idx];
    if(val < min) {
      min = val;
    }
  }
  return min;
}

extern "C"
ot_data_t octree_max_cpu(const octree* grid_in) {
  int n = grid_in->n_leafs * grid_in->feature_size;
  ot_data_t max = grid_in->data[0];
  for(int idx = 1; idx < n; ++idx) {
    ot_data_t val = grid_in->data[idx];
    if(val > max) {
      max = val;
    }
  }
  return max;
}
