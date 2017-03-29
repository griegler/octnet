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

#ifndef OCTREE_IO_CPU_H
#define OCTREE_IO_CPU_H

#include "octnet/core/core.h"

extern "C" {

/// Reads a grid-octree from the given path.
/// @deprecated 
/// @param path 
/// @param grid
void octree_read_deprecated_cpu(const char* path, octree* grid);

/// Reads a tensor with given dimensions from path.
/// @deprecated
/// @param path
/// @param n_dim
/// @param dims
/// @param data
void dense_read_prealloc_deprecated_cpu(const char* path, int n_dim, const int* dims, ot_data_t* data);

/// Reads the header of a tensor from the given path
/// @deprecated
/// @param path
/// @param n_dim
int* dense_read_header_deprecated_cpu(const char* path, int* n_dim);


/// Reads the header information of a grid-octree from a given path.
/// @param path
/// @param grid
void octree_read_header_cpu(const char* path, octree* grid);

/// Reads a grid-octree from a given path.
/// @param path
/// @param grid
void octree_read_cpu(const char* path, octree* grid);

/// Write a grid-octree to a given path.
/// @param path
/// @param grid
void octree_write_cpu(const char* path, const octree* grid);

/// Reads a batch of grid-octree structures.
/// @param n_paths
/// @param paths
/// @param n_threads number of OpenMP threads used for I/O
/// @param grid
void octree_read_batch_cpu(int n_paths, char** paths, int n_threads, octree* grid);


/// Writes the given grid-octree as tensor in DHWC format to path.
/// @param path
/// @param grid
void octree_dhwc_write_cpu(const char* path, const octree* grid);

/// Writes the given grid-octree as tensor in CDHW format to path.
/// @param path
/// @param grid
void octree_cdhw_write_cpu(const char* path, const octree* grid);

/// Writes a given tensor to path.
/// @param path
/// @param n_dim number of tensor dimensions.
/// @param dims tensor dimensions.
/// @param data tensor data
void dense_write_cpu(const char* path, int n_dim, const int* dims, const ot_data_t* data);

/// Reads a tensor from the given path. Allocates memory as needed.
/// @param path
/// @return tensor data
ot_data_t* dense_read_cpu(const char* path);

/// Reads a tensor to a preallocated array.
/// @param path
/// @param n_dim number of tensor dimensions.
/// @param dims tensor dimensions.
/// @param data tensor data
void dense_read_prealloc_cpu(const char* path, int n_dim, const int* dims, ot_data_t* data);

/// Reads a batch of tensors to a preallocated array.
/// @param n_paths
/// @param paths
/// @param n_threads number of OpenMP threads used for I/O
/// @param n_dim number of tensor dimensions.
/// @param dims tensor dimensions.
/// @param data tensor data
void dense_read_prealloc_batch_cpu(int n_paths, char** paths, int n_threads, int n_dim, const int* dims, ot_data_t* data); 

} //extern "C"

#endif
