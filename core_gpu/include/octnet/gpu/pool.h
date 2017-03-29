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

#ifndef OCTREE_POOL_GPU_H
#define OCTREE_POOL_GPU_H

#include "octnet/core/core.h"
#include "octnet/core/pool.h"

extern "C" {

/// This method implements a average pooling operation for grid-octree 
/// structures. This is realized by pooling together 8 neighbouring shallow
/// octrees (hence, the name gridpool). 
/// This is equivalent to using o2d, volumentric pooling and d2o, but more 
/// memory efficient.
/// @param in input grid-octree structure. grid_depth % 2 == 0, 
///           grid_height % 2 == 0, and grid_height % 2 == 0 must satisfied.
/// @param out output of this operation.
void octree_gridpool2x2x2_avg_gpu(const octree* in, octree* out);

/// Backward pass of @see octree_gridpool2x2x2_avg_gpu.
/// @param in input grid-octree structure. grid_depth % 2 == 0, 
///           grid_height % 2 == 0, and grid_height % 2 == 0 must satisfied.
/// @param grad_out gradient with respect to forward pass output.
/// @param grad_in gradient with respect to the input.
void octree_gridpool2x2x2_avg_bwd_gpu(const octree* in, const octree* grad_out, octree* grad_in);

/// This method implements a max pooling operation for grid-octree 
/// structures. This is realized by pooling together 8 neighbouring shallow
/// octrees (hence, the name gridpool). 
/// This is equivalent to using o2d, volumentric pooling and d2o, but more 
/// memory efficient.
/// @param in input grid-octree structure. grid_depth % 2 == 0, 
///           grid_height % 2 == 0, and grid_height % 2 == 0 must satisfied.
/// @param out output of this operation.
void octree_gridpool2x2x2_max_gpu(const octree* in, octree* out);

/// Backward pass of @see octree_gridpool2x2x2_max_gpu.
/// @param in input grid-octree structure. grid_depth % 2 == 0, 
///           grid_height % 2 == 0, and grid_height % 2 == 0 must satisfied.
/// @param grad_out gradient with respect to forward pass output.
/// @param grad_in gradient with respect to the input.
void octree_gridpool2x2x2_max_bwd_gpu(const octree* in, const octree* grad_out, octree* grad_in);


/// This function implements an average pooling on shallow octree cells, it pools
/// 8 neighbouring octree cells on the same level (indicated by the bool flags).
/// Therefore, this operation increases the sparsity of the data.
/// @param in input grid-octree structure.
/// @param level_0 if true, cells on level 1 are pooled together
/// @param level_1 if true, cells on level 2 are pooled together
/// @param level_2 if true, cells on level 3 are pooled together
/// @param out output 
void octree_pool2x2x2_avg_gpu(const octree* in, bool level_0, bool level_1, bool level_2, octree* out);

/// Backward pass for @see octreee_pool2x2x2_avg_gpu. 
/// @param in input grid-octree structure.
/// @param grad_out gradient with respect to forward pass output.
/// @param grad_in gradient with respect to the input.
void octree_pool2x2x2_avg_bwd_gpu(const octree* in, const octree* grad_out, octree* grad_in);

/// This function implements a max pooling on shallow octree cells, it pools
/// 8 neighbouring octree cells on the same level (indicated by the bool flags).
/// Therefore, this operation increases the sparsity of the data.
/// @param in input grid-octree structure.
/// @param level_0 if true, cells on level 1 are pooled together
/// @param level_1 if true, cells on level 2 are pooled together
/// @param level_2 if true, cells on level 3 are pooled together
/// @param out output 
void octree_pool2x2x2_max_gpu(const octree* in, bool level_0, bool level_1, bool level_2, octree* out);

/// Backward pass for @see octreee_pool2x2x2_max_gpu. 
/// @param in input grid-octree structure.
/// @param grad_out gradient with respect to forward pass output.
/// @param grad_in gradient with respect to the input.
void octree_pool2x2x2_max_bwd_gpu(const octree* in, const octree* grad_out, octree* grad_in);



} 

#endif
