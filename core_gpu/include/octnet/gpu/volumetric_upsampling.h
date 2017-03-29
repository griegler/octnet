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

#ifndef VOLUMETC_UPSAMPLING_GPU_H
#define VOLUMETC_UPSAMPLING_GPU_H

#include "octnet/core/core.h"

extern "C" {

/// 3D nearest neighbour interpolation along depth, height, width.
/// @param in
/// @param n batch size
/// @param in_depth
/// @param in_height
/// @param in_width
/// @param feature_size
/// @param upsampling factor should be > 1.
/// @param out
void volumetric_nn_upsampling_cdhw_gpu(const ot_data_t* in, int n, int in_depth, int in_height, int in_width, int feature_size, int upsampling_factor, ot_data_t* out);

/// Computes the gradient of the 3D nearest neighbour interpolation.
/// @param grad_out
/// @param n batch size
/// @param in_depth
/// @param in_height
/// @param in_width
/// @param feature_size
/// @param upsampling factor should be > 1.
/// @param grad_in
void volumetric_nn_upsampling_cdhw_bwd_gpu(const ot_data_t* grad_out, int n, int in_depth, int in_height, int in_width, int feature_size, int upsampling_factor, ot_data_t* grad_in);

}

#endif 
