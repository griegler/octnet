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

#ifndef OCTREE_CONV_CPU_H
#define OCTREE_CONV_CPU_H

#include "octnet/core/core.h"

extern "C" {

/// Forward pass of a 3x3s3 convolution on the given grid-octree structure 
/// grid_in. The convolution in bigger octree cells are sum pooled.
/// @param grid_in input to convolution operation.
/// @param weights channels_out x channels_in x 3 x 3 x 3 conv weights matrix, 
///                where channels_in is given by grid_in->feature_size.
/// @param bias channels_out x 1 conv bias vector.
/// @param channels_out number of output channels for this convolution.
/// @param grid output of the convolution operation.
void octree_conv3x3x3_sum_cpu(const octree* grid_in, const ot_data_t* weights, const ot_data_t* bias, int channels_out, octree* grid);

/// Backward pass of a 3x3x3 convolution on a grid-octree structure with 
/// sum pooling wrt. the operation input.
/// @param weights channels_out x channels_in x 3 x 3 x 3 conv weights matrix, 
///                where channels_out is given by grid_out->feature_size.
/// @param grad_out gradient wrt. the output of the forward pass.
/// @param channels_in number of input channels.
/// @param grad_in gradient wrt. to the input of this operation. 
void octree_conv3x3x3_sum_bwd_cpu(const ot_data_t* weights, const octree* grad_out, int channels_in, octree* grad_in);

/// Backward pass of a 3x3x3 convolution on a grid-octree structure with 
/// sum pooling wrt. weights and bias parameters. 
/// @param grid_in input to convolution operation.
/// @param grad_out gradient wrt. the output of the forward pass.
/// @param scale factor multiplied to the parameter gradient before accumulation.
/// @param grad_weights gradients wrt. weights parameters.
/// @param grad_bias gradients wrt. bias parameters.
void octree_conv3x3x3_sum_wbwd_cpu(const octree* grid_in, const octree* grad_out, ot_data_t scale, ot_data_t* grad_weights, ot_data_t* grad_bias);


/// Forward pass of a 3x3s3 convolution on the given grid-octree structure 
/// grid_in. The convolution in bigger octree cells are average pooled.
/// @param grid_in input to convolution operation.
/// @param weights channels_out x channels_in x 3 x 3 x 3 conv weights matrix, 
///                where channels_in is given by grid_in->feature_size.
/// @param bias channels_out x 1 conv bias vector.
/// @param channels_out number of output channels for this convolution.
/// @param grid output of the convolution operation.
void octree_conv3x3x3_avg_cpu(const octree* grid_in, const ot_data_t* weights, const ot_data_t* bias, int channels_out, octree* grid);

/// Backward pass of a 3x3x3 convolution on a grid-octree structure with 
/// average poolin wrt. the operation input.
/// @param weights channels_out x channels_in x 3 x 3 x 3 conv weights matrix, 
///                where channels_out is given by grid_out->feature_size.
/// @param grad_out gradient wrt. the output of the forward pass.
/// @param channels_in number of input channels.
/// @param grad_in gradient wrt. to the input of this operation. 
void octree_conv3x3x3_avg_bwd_cpu(const ot_data_t* weights, const octree* grad_out, int channels_in, octree* grad_in);

/// Backward pass of a 3x3x3 convolution on a grid-octree structure with 
/// average pooling wrt. weights and bias parameters. 
/// @param grid_in input to convolution operation.
/// @param grad_out gradient wrt. the output of the forward pass.
/// @param scale factor multiplied to the parameter gradient before accumulation.
/// @param grad_weights gradients wrt. weights parameters.
/// @param grad_bias gradients wrt. bias parameters.
void octree_conv3x3x3_avg_wbwd_cpu(const octree* grid_in, const octree* grad_out, ot_data_t scale, ot_data_t* grad_weights, ot_data_t* grad_bias);

}

#endif 
