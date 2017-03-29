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

#ifndef OCTREE_BUFFER_GPU_H
#define OCTREE_BUFFER_GPU_H

#include "octnet/gpu/common.h"

/// Singleton buffer that is dynamically resized as needed of ot_data_t.
/// Mainly used in oc2col, col2oc.
class ot_data_t_buffer_gpu {
public:
  /// Get singleton instance of buffer.
  /// @return singleton instance.
  static ot_data_t_buffer_gpu& i() {
    if(DEBUG) { printf("[DEBUG] ot_data_t_buffer_gpu.i\n"); }
    static ot_data_t_buffer_gpu instance;
    // printf("[INFO][BUFFER] getting instance\n");
    return instance;
  }

  /// Destructor
  virtual ~ot_data_t_buffer_gpu() {
    if(DEBUG) { printf("[DEBUG] ot_data_t_buffer_gpu.~\n"); }
    if(data_) {
      // printf("[INFO][BUFFER] calling destructor\n");
      device_free(data_);
    }
  }

  /// Get data pointer.
  /// @return data pointer.
  ot_data_t* data() {
    return data_;
  }

  /// Returns the capacity of the buffer.
  /// @return capacity
  long capacity() {
    return capacity_;
  }

  /// If N is smaller than the current capacity of the buffer the data pointer
  /// is unchanged, otherwise a new memory is allocated and the old one is 
  /// freed. Note that no memory is copied.
  void resize(long N) {
    if(DEBUG) { printf("[DEBUG] ot_data_t_buffer_gpu.resize to N=%ld, capacity=%ld\n", N, capacity_); }
    if(N > capacity_) {
      // printf("[INFO][BUFFER] resize buffer from %d to %d\n", capacity_, N);
      device_free(data_);
      data_ = device_malloc<ot_data_t>(N);
      capacity_ = N;
    }
  }

private:
  ot_data_t_buffer_gpu() : data_(0), capacity_(0) {}

  ot_data_t_buffer_gpu(ot_data_t_buffer_gpu const&);
  void operator=(ot_data_t_buffer_gpu const&);

private:
  /// data buffer.
  ot_data_t* data_;
  /// current capacity of the buffer.
  long capacity_;
};

#endif 
