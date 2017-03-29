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

#ifndef OCTREE_POOL_H
#define OCTREE_POOL_H

#include "core.h"



/// Pools (sum) the values of 8 sibling octree leaf cells.
///
/// @param data_in data of the 8 sibling leaf cells (length = 8 * feature_size).
/// @param feature_size length of a single data vector for one leaf cell.
/// @param data_out 
OCTREE_FUNCTION
inline void octree_pool2x2x2_sum(const ot_data_t* data_in, ot_size_t feature_size, ot_data_t* data_out) {
  for(int f = 0; f < feature_size; ++f) {
    ot_data_t sum = 0;
    for(int idx = 0; idx < 8; ++idx) {
      sum += data_in[idx * feature_size + f];
    }
    data_out[f] = sum;
  }
}

/// Pools (avg) the values of 8 sibling octree leaf cells.
///
/// @param data_in data of the 8 sibling leaf cells (length = 8 * feature_size).
/// @param feature_size length of a single data vector for one leaf cell.
/// @param data_out 
OCTREE_FUNCTION
inline void octree_pool2x2x2_avg(const ot_data_t* data_in, ot_size_t feature_size, ot_data_t* data_out) {
  for(int f = 0; f < feature_size; ++f) {
    ot_data_t avg = 0;
    for(int idx = 0; idx < 8; ++idx) {
      avg += data_in[idx * feature_size + f];
    }
    avg /= 8.f;
    data_out[f] = avg;
  }
}

/// Pools (max) the values of 8 sibling octree leaf cells.
///
/// @param data_in data of the 8 sibling leaf cells (length = 8 * feature_size).
/// @param feature_size length of a single data vector for one leaf cell.
/// @param data_out 
OCTREE_FUNCTION
inline void octree_pool2x2x2_max(const ot_data_t* data_in, ot_size_t feature_size, ot_data_t* data_out) {
  for(int f = 0; f < feature_size; ++f) {
    ot_data_t max = data_in[f];
    for(int idx = 1; idx < 8; ++idx) {
      max = FMAX(max, data_in[idx * feature_size + f]);
    }
    data_out[f] = max;
  }
}

/// Pools the values of 8 sibling octree leaf cells.
///
/// @tparam pool_fcn 
/// @param data_in data of the 8 sibling leaf cells (length = 8 * feature_size).
/// @param feature_size length of a single data vector for one leaf cell.
/// @param data_out 
template <int pool_fcn>
OCTREE_FUNCTION
inline void octree_pool2x2x2(const ot_data_t* data_in, ot_size_t feature_size, ot_data_t* data_out) {
  if(pool_fcn == REDUCE_AVG) {
    octree_pool2x2x2_avg(data_in, feature_size, data_out);
  }
  else if(pool_fcn == REDUCE_MAX) {
    octree_pool2x2x2_max(data_in, feature_size, data_out);
  }
  else if(pool_fcn == REDUCE_SUM) {
    octree_pool2x2x2_sum(data_in, feature_size, data_out);
  }
}



/// Backward function of the pool (avg) operation.
///
/// @param data_in input data of the pool fwd operation.
/// @param data_grad_out gradient data of the successive operation.
/// @param feature_size length of a single data vector for one leaf cell.
/// @param data_grad_in computed gradient through this operation
OCTREE_FUNCTION
inline void octree_pool2x2x2_avg_bwd(const ot_data_t* data_in, const ot_data_t* data_grad_out, ot_size_t feature_size, ot_data_t* data_grad_in) {
  for(int f = 0; f < feature_size; ++f) {
    for(int idx = 0; idx < 8; ++idx) {
      data_grad_in[idx * feature_size + f] = data_grad_out[f] / 8.f;
    }
  }
}

/// Backward function of the pool (max) operation.
///
/// @param data_in input data of the pool fwd operation.
/// @param data_grad_out gradient data of the successive operation.
/// @param feature_size length of a single data vector for one leaf cell.
/// @param data_grad_in computed gradient through this operation
OCTREE_FUNCTION
inline void octree_pool2x2x2_max_bwd(const ot_data_t* data_in, const ot_data_t* data_grad_out, ot_size_t feature_size, ot_data_t* data_grad_in) {
  for(int f = 0; f < feature_size; ++f) {
    ot_data_t max_val = data_in[f];
    int max_idx = 0;
    data_grad_in[f] = 0;
    for(int idx = 1; idx < 8; ++idx) {
      data_grad_in[idx * feature_size + f] = 0;

      ot_data_t val = data_in[idx * feature_size + f];
      max_idx = val > max_val ? idx : max_idx;
      max_val = FMAX(max_val, val);
    }
    data_grad_in[max_idx * feature_size + f] = data_grad_out[f];
  }
}

/// Backward function of the pool operation.
///
/// @tparam pool_fcn 
/// @param data_in input data of the pool fwd operation.
/// @param data_grad_out gradient data of the successive operation.
/// @param feature_size length of a single data vector for one leaf cell.
/// @param data_grad_in computed gradient through this operation
template <int pool_fcn>
OCTREE_FUNCTION
inline void octree_pool2x2x2_bwd(const ot_data_t* data_in, const ot_data_t* data_grad_out, ot_size_t feature_size, ot_data_t* data_grad_in) {
  if(pool_fcn == REDUCE_AVG) {
    octree_pool2x2x2_avg_bwd(data_in, data_grad_out, feature_size, data_grad_in);
  }
  else if(pool_fcn == REDUCE_MAX) {
    octree_pool2x2x2_max_bwd(data_in, data_grad_out, feature_size, data_grad_in);
  }
}


#endif
