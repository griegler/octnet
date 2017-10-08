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

#ifndef OCTREE_SPLIT_GPU_H
#define OCTREE_SPLIT_GPU_H

#include "octnet/core/core.h"

extern "C" {

void octree_split_by_prob_gpu(const octree* in, const octree* prob, const ot_data_t thr, bool check, octree* out);
void octree_split_full_gpu(const octree* in, octree* out);
void octree_split_reconstruction_surface_gpu(const octree* in, const octree* rec, ot_data_t rec_thr_from, ot_data_trec_thr_to, octree* out);

void octree_split_bwd_gpu(const octree* in, const octree* grad_out, octree* grad_in);


void octree_split_dense_reconstruction_surface_gpu(const ot_data_t* features, const ot_data_t* reconstruction, int dense_depth, int dense_height, int dense_width, int feature_size, ot_data_t rec_thr_from, ot_data_t rec_thr_to, int structure_type, octree* out);
void octree_split_dense_reconstruction_surface_bwd_gpu(const octree* grad_out, ot_data_t* grad_in);




void octree_split_dense_reconstruction_surface_fres_gpu(const ot_data_t* features, const ot_data_t* reconstruction, int n, int dense_depth, int dense_height, int dense_width, int feature_size, ot_data_t rec_thr_from, ot_data_t rec_thr_to, int band, octree* out);
void octree_split_dense_reconstruction_surface_fres_bwd_gpu(const octree* grad_out, ot_data_t* grad_in);



void octree_split_tsdf_gpu(const ot_data_t* features, const ot_data_t* reconstruction, const octree* guide, int n, int dense_depth, int dense_height, int dense_width, int feature_size, int band, octree* out);

}

#endif 
