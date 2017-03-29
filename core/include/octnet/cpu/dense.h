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

#ifndef OCTREE_DENSE_CPU_H
#define OCTREE_DENSE_CPU_H

#include "octnet/core/core.h"
#include "octnet/core/dense.h"

extern "C" {

//------------------------------------------------------------------------------
// Octree to Dense
//------------------------------------------------------------------------------

/// Converts grid-octree representation to dense with DHWC format.
/// @param in
/// @param dense_depth
/// @param dense_height
/// @param dense_width
/// @param out
void octree_to_dhwc_cpu(const octree* in, const int dense_depth, const int dense_height, const int dense_width, ot_data_t* out);

/// Converts grid-octree representation to dense with DHWC format where all
/// voxels are inversely scaled to the size of the octree cell. 
/// @param in
/// @param dense_depth
/// @param dense_height
/// @param dense_width
/// @param out
void octree_to_dhwc_avg_cpu(const octree* in, const int dense_depth, const int dense_height, const int dense_width, ot_data_t* out);

/// Computes the gradient of octree_ot_dhwc_cpu.
/// @param in
/// @param dense_depth
/// @param dense_height
/// @param dense_width
/// @param grad_out 
/// @param grad_in
void octree_to_dhwc_bwd_cpu(const octree* in, const int dense_depth, const int dense_height, const int dense_width, const ot_data_t* grad_out, octree* grad_in);


/// Converts grid-octree representation to dense with CDHW format.
/// @param in
/// @param dense_depth
/// @param dense_height
/// @param dense_width
/// @param out
void octree_to_cdhw_cpu(const octree* in, const int dense_depth, const int dense_height, const int dense_width, ot_data_t* out);

/// Converts grid-octree representation to dense with CDHW format where all
/// voxels are inversely scaled to the size of the octree cell. 
/// @param in
/// @param dense_depth
/// @param dense_height
/// @param dense_width
/// @param out
void octree_to_cdhw_avg_cpu(const octree* in, const int dense_depth, const int dense_height, const int dense_width, ot_data_t* out);

/// Computes the gradient of octree_ot_cdhw_cpu.
/// @param in
/// @param dense_depth
/// @param dense_height
/// @param dense_width
/// @param grad_out 
/// @param grad_in
void octree_to_cdhw_bwd_cpu(const octree* in, const int dense_depth, const int dense_height, const int dense_width, const ot_data_t* grad_out, octree* grad_in);



//------------------------------------------------------------------------------
// Dense to Octree
//------------------------------------------------------------------------------

/// Converts a tensor in DHWC format to a grid-octree, where the tree structure
/// is given by in_struct. The octree cell takes the value of the sum of all 
/// voxels that fall into it.
/// @param in_struct
/// @param dense_depth
/// @param dense_height
/// @param dense_width
/// @param in
/// @param dense_feature_size
/// @param out
void dhwc_to_octree_sum_cpu(const octree* in_struct, const int dense_depth, const int dense_height, const int dense_width, const ot_data_t* in, int dense_feature_size, octree* out);

/// Converts a tensor in DHWC format to a grid-octree, where the tree structure
/// is given by in_struct. The octree cell takes the value of the average of all 
/// voxels that fall into it.
/// @param in_struct
/// @param dense_depth
/// @param dense_height
/// @param dense_width
/// @param in
/// @param dense_feature_size
/// @param out
void dhwc_to_octree_avg_cpu(const octree* in_struct, const int dense_depth, const int dense_height, const int dense_width, const ot_data_t* in, int dense_feature_size, octree* out);

/// Converts a tensor in DHWC format to a grid-octree, where the tree structure
/// is given by in_struct. The octree cell takes the value of the maximum of all 
/// voxels that fall into it.
/// @param in_struct
/// @param dense_depth
/// @param dense_height
/// @param dense_width
/// @param in
/// @param dense_feature_size
/// @param out
void dhwc_to_octree_max_cpu(const octree* in_struct, const int dense_depth, const int dense_height, const int dense_width, const ot_data_t* in, int dense_feature_size, octree* out);

/// Converts a tensor in CDHW format to a grid-octree, where the tree structure
/// is given by in_struct. The octree cell takes the value of the sum of all 
/// voxels that fall into it.
/// @param in_struct
/// @param dense_depth
/// @param dense_height
/// @param dense_width
/// @param in
/// @param dense_feature_size
/// @param out
void cdhw_to_octree_sum_cpu(const octree* in_struct, const int dense_depth, const int dense_height, const int dense_width, const ot_data_t* in, int dense_feature_size, octree* out);

/// Converts a tensor in CDHW format to a grid-octree, where the tree structure
/// is given by in_struct. The octree cell takes the value of the average of all 
/// voxels that fall into it.
/// @param in_struct
/// @param dense_depth
/// @param dense_height
/// @param dense_width
/// @param in
/// @param dense_feature_size
/// @param out
void cdhw_to_octree_avg_cpu(const octree* in_struct, const int dense_depth, const int dense_height, const int dense_width, const ot_data_t* in, int dense_feature_size, octree* out);

/// Converts a tensor in CDHW format to a grid-octree, where the tree structure
/// is given by in_struct. The octree cell takes the value of the maximum of all 
/// voxels that fall into it.
/// @param in_struct
/// @param dense_depth
/// @param dense_height
/// @param dense_width
/// @param in
/// @param dense_feature_size
/// @param out
void cdhw_to_octree_max_cpu(const octree* in_struct, const int dense_depth, const int dense_height, const int dense_width, const ot_data_t* in, int dense_feature_size, octree* out);


/// Computes the gradient wrt. to dhwc_to_octree_sum_cpu.
/// @param grad_out
/// @param dense_depth
/// @param dense_height
/// @param dense_width
/// @param grad_in
void dhwc_to_octree_sum_bwd_cpu(const octree* grad_out, const int dense_depth, const int dense_height, const int dense_width, ot_data_t* grad_in);

/// Computes the gradient wrt. to dhwc_to_octree_avg_cpu.
/// @param grad_out
/// @param dense_depth
/// @param dense_height
/// @param dense_width
/// @param grad_in
void dhwc_to_octree_avg_bwd_cpu(const octree* grad_out, const int dense_depth, const int dense_height, const int dense_width, ot_data_t* grad_in);


/// Computes the gradient wrt. to cdhw_to_octree_sum_cpu.
/// @param grad_out
/// @param dense_depth
/// @param dense_height
/// @param dense_width
/// @param grad_in
void cdhw_to_octree_sum_bwd_cpu(const octree* grad_out, const int dense_depth, const int dense_height, const int dense_width, ot_data_t* grad_in);

/// Computes the gradient wrt. to cdhw_to_octree_avg_cpu.
/// @param grad_out
/// @param dense_depth
/// @param dense_height
/// @param dense_width
/// @param grad_in
void cdhw_to_octree_avg_bwd_cpu(const octree* grad_out, const int dense_depth, const int dense_height, const int dense_width, ot_data_t* grad_in);

} // extern "C"


template <int dense_format, bool avg_vol>
void octree_to_dense_cpu(const octree* in, const int dense_depth, const int dense_height, const int dense_width, ot_data_t* out_data);

#endif
