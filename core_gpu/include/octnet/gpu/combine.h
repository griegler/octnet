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

#ifndef OCTREE_COMBINE_GPU_H
#define OCTREE_COMBINE_GPU_H

#include "octnet/core/core.h"

extern "C" {

/// Concatenates two grid-octrees with the same structure to one grid-octree
/// along the feature dimension.
/// @param in1 
/// @param in2 
/// @param check if True the tree structures of in1 and in2 are tested
///              for equivalence.
/// @param out
void octree_concat_gpu(const octree* grid_in1, const octree* grid_in2, bool check, octree* grid_out);

/// Computes the gradient to the concatenation operation octree_concat_gpu.
/// @param in1 input 1 to the concatenation operation.
/// @param in2 input 2 to the concatenation operation. 
/// @param grad_out gradient of the output
/// @param do_grad_in2 if true, computes the gradient for grad_in1 and grad_in2,
///                    otherwise, only the gradient for grad_in1 is computed.
/// @param grad_in1
/// @param grad_in2
void octree_concat_bwd_gpu(const octree* in1, const octree* in2, const octree* grad_out, bool do_grad_in2, octree* grad_in1, octree* grad_in2);


/// Concatenates two grid-octrees to one grid-octree along the feature dimension.
/// The tree structure might differ, and the tree structure of the first 
/// input is used for the output.
/// @param in1 
/// @param in2 
/// @param check if True the tree structures of in1 and in2 are tested
///              for equivalence.
/// @param out
void octree_concat_ds_gpu(const octree* in1, const octree* in2, octree* out);

/// Computes the gradient to the concatenation operation octree_concat_ds_gpu.
/// @param in1 input 1 to the concatenation operation.
/// @param in2 input 2 to the concatenation operation. 
/// @param grad_out gradient of the output
/// @param do_grad_in2 if true, computes the gradient for grad_in1 and grad_in2,
///                    otherwise, only the gradient for grad_in1 is computed.
/// @param grad_in1
/// @param grad_in2
void octree_concat_ds_bwd_gpu(const octree* in1, const octree* in2, const octree* grad_out, bool do_grad_in2, octree* grad_in1, octree* grad_in2);


/// Concatenates the grid-octree in1 with the dense tensor in2 along the feature
/// dimension. in2 is assumed to comprise the same volume as in1.
/// @param in1
/// @param in2
/// @param feature_size2 number of feature channels in in2
/// @param out
void octree_concat_dense_gpu(const octree* in1, const ot_data_t* in2, ot_size_t feature_size2, octree* out);

/// Computes the gradient to the concatenation operation octree_concat_dense_gpu.
/// @param in1 input 1 to the concatenation operation.
/// @param in2 input 2 to the concatenation operation. 
/// @param feature_size2 number of feature channels in in2
/// @param grad_out gradient of the output
/// @param do_grad_in2 if true, computes the gradient for grad_in1 and grad_in2,
///                    otherwise, only the gradient for grad_in1 is computed.
/// @param grad_in1
/// @param grad_in2
void octree_concat_dense_bwd_gpu(const octree* in1, const ot_data_t* in2, ot_size_t feature_size2, const octree* grad_out, bool do_grad_in2, octree* grad_in1, ot_data_t* grad_in2);

} // extern "C"

#endif 
