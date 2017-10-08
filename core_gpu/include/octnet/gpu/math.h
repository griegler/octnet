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

#ifndef OCTREE_MATH_GPU_H
#define OCTREE_MATH_GPU_H

#include "octnet/core/core.h"

extern "C" {

/// Computes fac1 * in1 + fac2 * in2.
/// in1 and in2 are expected to have the same tree structure.
/// @param in1
/// @param fac1
/// @param in2
/// @param fac2
/// @param check if true, test if tree structure of in1 and in2 are compatible.
/// @param out
void octree_add_gpu(const octree* in1, ot_data_t fac1, const octree* in2, ot_data_t fac2, bool check, octree* out);

/// Adds a scalar value to all octree cells.
/// @param grid
/// @param scalar
void octree_scalar_add_gpu(octree* grid, const ot_data_t scalar);

/// Multiplies a scalar value to all octree cells.
/// @param grid
/// @param scalar
void octree_scalar_mul_gpu(octree* grid, const ot_data_t scalar);

/// Computes the sign of the octree cells in-place
/// @param grid
void octree_sign_gpu(octree* grid);

/// Computes the abs of the octree cells in-place
/// @param grid
void octree_abs_gpu(octree* grid);

/// Computes the log of the octree cells in-place
/// @param grid
void octree_log_gpu(octree* grid);

/// Computes the minimum cell value in the grid-octree.
/// @param grid_in
/// @param minimum cell value.
ot_data_t octree_min_gpu(const octree* grid_in);

/// Computes the maximum cell value in the grid-octree.
/// @param grid_in
/// @param maximum cell value.
ot_data_t octree_max_gpu(const octree* grid_in);

}

#endif 
