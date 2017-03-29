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

#ifndef OCTREE_UNPOOL_CPU_H
#define OCTREE_UNPOOL_CPU_H

#include "octnet/core/core.h"

extern "C" {

/// Performs a nearest-neighbour unpooling operation. 
/// The number of shallow octrees is increased 8-times, but after this operation
/// there are no more leaf cells on the finest resolution. 
/// @param in
/// @param out
void octree_gridunpool2x2x2_cpu(const octree* in, octree* out);

/// Computes the gradient of the nearest neighbour unpooling operation.
/// @param in
/// @param grad_out
/// @param grad_in
void octree_gridunpool2x2x2_bwd_cpu(const octree* in, const octree* grad_out, octree* grad_in);


/// Performs a nearest-neighbour unpooling operation. 
/// However, the structure of the unpooled grid-octree is the same as the one
/// from in_struct. Useful for U-shaped networks.
/// @param in 
/// @param in_struct guidance
/// @param out
void octree_gridunpoolguided2x2x2_cpu(const octree* in, const octree* in_struct, octree* out);

/// Computes the gradient of the guided nearest neighbour unpooling operation.
/// @param in
/// @param in_struct guidance
/// @param grad_out
/// @param grad_in
void octree_gridunpoolguided2x2x2_bwd_cpu(const octree* in, const octree* in_struct, const octree* grad_out, octree* grad_in);

}

#endif 
