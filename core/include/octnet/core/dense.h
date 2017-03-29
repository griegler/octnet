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

#ifndef OCTREE_DENSE_H
#define OCTREE_DENSE_H

#include "core.h"

#define DENSE_FORMAT_DHWC 0
#define DENSE_FORMAT_CDHW 1


/// Pools (sum) all the values of the tensor data dense into the corresponding 
/// shallow octree cell (out).
///
/// @tparam dense_format DHWC or CDHW.
/// @param dense the data of the tensor.
/// @param n batch index.
/// @param dense_depth the depth of the tensor.
/// @param dense_height the height of the tensor.
/// @param dense_width the width of the tensor.
/// @param d1 start index for the pooling in depth.
/// @param h1 start index for the pooling in height.
/// @param w1 start index for the pooling in width.
/// @param d1 end index for the pooling in depth.
/// @param h1 end index for the pooling in height.
/// @param w1 end index for the pooling in width.
/// @param out data array of the output octree cell.
template <int dense_format>
OCTREE_FUNCTION
inline void dense_to_octree_sum_fcn(const ot_data_t* dense, int n, int dense_depth, int dense_height, int dense_width, int feature_size, int d1, int d2, int h1, int h2, int w1, int w2, ot_data_t* out) {
  for(int f = 0; f < feature_size; ++f) { out[f] = 0; }

  for(int d = d1; d < d2; ++d) {
    for(int h = h1; h < h2; ++h) {
      for(int w = w1; w < w2; ++w) {
        for(int f = 0; f < feature_size; ++f) {
          float val;
          if(dense_format == DENSE_FORMAT_DHWC) {
            val = dense[(((n * dense_depth + d) * dense_height + h) * dense_width + w) * feature_size + f];
          }
          else if (dense_format == DENSE_FORMAT_CDHW) {
            val = dense[(((n * feature_size + f) * dense_depth + d) * dense_height + h) * dense_width + w];
          }
          out[f] += val;
        }
      }
    }
  }
}

/// Pools (avg) all the values of the tensor data dense into the corresponding 
/// shallow octree cell (out).
///
/// @tparam dense_format DHWC or CDHW.
/// @param dense the data of the tensor.
/// @param n batch index.
/// @param dense_depth the depth of the tensor.
/// @param dense_height the height of the tensor.
/// @param dense_width the width of the tensor.
/// @param d1 start index for the pooling in depth.
/// @param h1 start index for the pooling in height.
/// @param w1 start index for the pooling in width.
/// @param d1 end index for the pooling in depth.
/// @param h1 end index for the pooling in height.
/// @param w1 end index for the pooling in width.
/// @param out data array of the output octree cell.
template <int dense_format>
OCTREE_FUNCTION
inline void dense_to_octree_avg_fcn(const ot_data_t* dense, int n, int dense_depth, int dense_height, int dense_width, int feature_size, int d1, int d2, int h1, int h2, int w1, int w2, ot_data_t* out) {
  dense_to_octree_sum_fcn<dense_format>(dense, n, dense_depth, dense_height, dense_width, feature_size, d1,d2, h1,h2, w1,w2, out);
  float norm = (d2-d1) * (h2-h1) * (w2-w1);
  for(int f = 0; f < feature_size; ++f) { out[f] /= norm; }
}

/// Pools (max) all the values of the tensor data dense into the corresponding 
/// shallow octree cell (out).
///
/// @tparam dense_format DHWC or CDHW.
/// @param dense the data of the tensor.
/// @param n batch index.
/// @param dense_depth the depth of the tensor.
/// @param dense_height the height of the tensor.
/// @param dense_width the width of the tensor.
/// @param d1 start index for the pooling in depth.
/// @param h1 start index for the pooling in height.
/// @param w1 start index for the pooling in width.
/// @param d1 end index for the pooling in depth.
/// @param h1 end index for the pooling in height.
/// @param w1 end index for the pooling in width.
/// @param out data array of the output octree cell.
template <int dense_format>
OCTREE_FUNCTION
inline void dense_to_octree_max_fcn(const ot_data_t* dense, int n, int dense_depth, int dense_height, int dense_width, int feature_size, int d1, int d2, int h1, int h2, int w1, int w2, ot_data_t* out) {
  for(int f = 0; f < feature_size; ++f) { out[f] = 0; }

  for(int d = d1; d < d2; ++d) {
    for(int h = h1; h < h2; ++h) {
      for(int w = w1; w < w2; ++w) {
        for(int f = 0; f < feature_size; ++f) {
          float val;
          if(dense_format == DENSE_FORMAT_DHWC) {
            val = dense[(((n * dense_depth + d) * dense_height + h) * dense_width + w) * feature_size + f];
          }
          else if (dense_format == DENSE_FORMAT_CDHW) {
            val = dense[(((n * feature_size + f) * dense_depth + d) * dense_height + h) * dense_width + w];
          }
          out[f] = FMAX(out[f], val);
        }
      }
    }
  }
}


/// Pools all the values of the tensor data dense into the corresponding 
/// shallow octree cell (out).
///
/// @tparam reduce_fcn
/// @tparam dense_format DHWC or CDHW.
/// @param dense the data of the tensor.
/// @param n batch index.
/// @param dense_depth the depth of the tensor.
/// @param dense_height the height of the tensor.
/// @param dense_width the width of the tensor.
/// @param d1 start index for the pooling in depth.
/// @param h1 start index for the pooling in height.
/// @param w1 start index for the pooling in width.
/// @param d1 end index for the pooling in depth.
/// @param h1 end index for the pooling in height.
/// @param w1 end index for the pooling in width.
/// @param out data array of the output octree cell.
template <int reduce_fcn, int dense_format>
OCTREE_FUNCTION
inline void dense_to_octree_fcn(const ot_data_t* dense, int n, int dense_depth, int dense_height, int dense_width, int feature_size, int d1, int d2, int h1, int h2, int w1, int w2, ot_data_t* out) {
  if(reduce_fcn == REDUCE_AVG) {
    dense_to_octree_avg_fcn<dense_format>(dense, n, dense_depth, dense_height, dense_width, feature_size, d1,d2, h1,h2, w1,w2, out);  
  }
  else if(reduce_fcn == REDUCE_MAX) {
    dense_to_octree_max_fcn<dense_format>(dense, n, dense_depth, dense_height, dense_width, feature_size, d1,d2, h1,h2, w1,w2, out);  
  }
  else if(reduce_fcn == REDUCE_SUM) {
    dense_to_octree_sum_fcn<dense_format>(dense, n, dense_depth, dense_height, dense_width, feature_size, d1,d2, h1,h2, w1,w2, out);  
  }
}

#endif
