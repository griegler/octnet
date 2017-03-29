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

#ifndef OCTREE_LOSS_GPU_H
#define OCTREE_LOSS_GPU_H

#include "octnet/core/core.h"

extern "C" {

/// Computes the mean squared error of the input and target. 
/// Input and target are supposed to have the same tree structure.
/// @param input
/// @param target
/// @param size_average if true, the loss is averaged by the size of the volume
/// @param check if true, the tree structures of input and target are checked for 
///              equality 
/// @return mean squared error 
ot_data_t octree_mse_loss_gpu(const octree* input, const octree* target, bool size_average, bool check);

/// Computes the gradient of the  mean squared error wrt. input and target. 
/// Input and target are supposed to have the same tree structure.
/// @param input
/// @param target
/// @param size_average if true, the loss is averaged by the size of the volume
/// @param check if true, the tree structures of input and target are checked for 
///              equality 
/// @param grad the gradient wrt. input
void octree_mse_loss_bwd_gpu(const octree* input, const octree* target, bool size_average, bool check, octree* grad);


/// Computes the mean squared error of the input and target. 
/// Input and target can have different tree structures.
/// @param input
/// @param target
/// @param size_average if true, the loss is averaged by the size of the volume
/// @return mean squared error 
ot_data_t octree_mse_ds_loss_gpu(const octree* input, const octree* target, bool size_average);

/// Computes the gradient of the  mean squared error wrt. input and target. 
/// Input and target can have different tree structures.
/// @param input
/// @param target
/// @param size_average if true, the loss is averaged by the size of the volume
/// @param grad the gradient wrt. input
void octree_mse_loss_ds_bwd_gpu(const octree* input, const octree* target, bool size_average, octree* grad);


/// Computes the negative log-likelihood of the input and target. 
/// Input and target are supposed to have the same tree structure.
/// @param input
/// @param target
/// @param weights weighting coefficients for the different classes
/// @param class_base the index of the first class (usually 0, or 1)
/// @param size_average if true, the loss is averaged by the size of the volume
/// @param check if true, the tree structures of input and target are checked for 
///              equality 
/// @param output negative log-ligkelihood
/// @param total_weight sum of all weight coefficients over the volume
void octree_nll_loss_gpu(const octree* input, const octree* target, const ot_data_t* weights, int class_base, bool size_average, bool check, ot_data_t* output, ot_data_t* total_weight);

/// Computes the gradient of the negative log-likelihood wrt. input and target. 
/// Input and target are supposed to have the same tree structure.
/// @param input
/// @param target
/// @param weights weighting coefficients for the different classes
/// @param total_weight sum of all weight coefficients over the volume
/// @param class_base the index of the first class (usually 0, or 1)
/// @param size_average if true, the loss is averaged by the size of the volume
/// @param check if true, the tree structures of input and target are checked for 
///              equality 
/// @param grad the gradient wrt. input
void octree_nll_loss_bwd_gpu(const octree* input, const octree* target, const ot_data_t* weights, const ot_data_t total_weight, int class_base, bool size_average, bool check, octree* grad);



/// Computes the binary cross-entropy of the input and target. 
/// Input and target are supposed to have the same tree structure.
/// @param input
/// @param target
/// @param size_average if true, the loss is averaged by the size of the volume
/// @param check if true, the tree structures of input and target are checked for 
///              equality 
/// @param output binary cross-entropy
/// @param total_weight sum of all weight coefficients over the volume
void octree_bce_loss_gpu(const octree* input, const octree* target, bool size_average, bool check, ot_data_t* output, ot_data_t* total_weight);

/// Computes the gradient of the binary cross-entropy wrt. input and target. 
/// Input and target are supposed to have the same tree structure.
/// @param input
/// @param target
/// @param size_average if true, the loss is averaged by the size of the volume
/// @param check if true, the tree structures of input and target are checked for 
///              equality 
/// @param grad the gradient wrt. input
void octree_bce_loss_bwd_gpu(const octree* input, const octree* target, bool size_average, bool check, octree* grad);


/// Computes the binary cross-entropy of the input and target. 
/// Target is supposed to be a tensor comprising the same volume.
/// @param input
/// @param target
/// @param size_average if true, the loss is averaged by the size of the volume
/// @param output binary cross-entropy
/// @param total_weight sum of all weight coefficients over the volume
void octree_bce_dense_loss_gpu(const octree* input, const ot_data_t* target, bool size_average, ot_data_t* output, ot_data_t* total_weight);

/// Computes the gradient of the binary cross-entropy wrt. input and target. 
/// Target is supposed to be a tensor comprising the same volume.
/// @param input
/// @param target
/// @param size_average if true, the loss is averaged by the size of the volume
/// @param grad the gradient wrt. input
void octree_bce_dense_loss_bwd_gpu(const octree* input, const ot_data_t* target, bool size_average, octree* grad);


/// Computes the binary cross-entropy of the input and target. 
/// Input and target can have different tree structures.
/// @param input
/// @param target
/// @param weights cell-wise weights (expected in same structure as target)
/// @param size_average if true, the loss is averaged by the size of the volume
/// @param output binary cross-entropy
/// @param total_weight sum of all weight coefficients over the volume
void octree_bce_ds_loss_gpu(const octree* input, const octree* target, const octree* weights, bool size_average, ot_data_t* output, ot_data_t* total_weight);

/// Computes the gradient of the binary cross-entropy wrt. input and target. 
/// Input and target can have different tree structures.
/// @param input
/// @param target
/// @param weights cell-wise weights (expected in same structure as target)
/// @param size_average if true, the loss is averaged by the size of the volume
/// @param total_weight sum of all weight coefficients over the volume
/// @param grad the gradient wrt. input
void octree_bce_ds_loss_bwd_gpu(const octree* input, const octree* target, const octree* weights, bool size_average, ot_data_t total_weight, octree* grad);


/// Computes the binary cross-entropy of the input and target. 
/// @param input
/// @param target
/// @param weights voxel-wise weights
/// @param N number of voxels
/// @param output binary cross-entropy
/// @param total_weight sum of all weight coefficients over the volume
void dense_bce_loss_gpu(const ot_data_t* input, const ot_data_t* target, const ot_data_t* weights, ot_size_t N, ot_data_t* output, ot_data_t* total_weight);

/// Computes the gradient of the binary cross-entropy wrt. input and target. 
/// @param input
/// @param target
/// @param weights voxel-wise weights
/// @param N number of voxels
/// @param total_weight sum of all weight coefficients over the volume
/// @param grad the gradient wrt. input
void dense_bce_loss_bwd_gpu(const ot_data_t* input, const ot_data_t* target, const ot_data_t* weights, ot_size_t N, ot_data_t total_weight, ot_data_t* grad); 
}

#endif 
