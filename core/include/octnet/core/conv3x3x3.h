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

#ifndef OCTREE_CONV3X3X3_H
#define OCTREE_CONV3X3X3_H

#include "core.h"

#define INV_FILTER_TRUE true
#define INV_FILTER_FALSE false

#define ADD_BIAS_TRUE true
#define ADD_BIAS_FALSE false


/// Get 0-padded data from an octree for given dense subscript indices.
/// @deprecated
///
/// @param grid_in
/// @param n
/// @param d
/// @param h
/// @param w
/// @param f
/// @return 0 if the subscript indices are out of bounds, otherwise the data.
OCTREE_FUNCTION
inline ot_data_t conv3x3x3_get_padded_data(const octree* grid_in, int n, int d, int h, int w, int f) {
  const bool in_vol = d >= 0 && h >= 0 && w >= 0 && 
      d < 8 * grid_in->grid_depth && h < 8 * grid_in->grid_height && w < 8 * grid_in->grid_width;

  if(in_vol) {
    const int grid_idx = octree_grid_idx(grid_in, n, d / 8, h / 8, w / 8);
    // printf("%d,%d,%d => %d, %d\n", d, h, w, grid_idx, bit_idx);
    const ot_tree_t* tree = octree_get_tree(grid_in, grid_idx);
    const int bit_idx = tree_bit_idx(tree, d % 8, h % 8, w % 8);
    /* ot_data_t* data_in = grid_in->data_ptrs[grid_idx]; */
    ot_data_t* data_in = octree_get_data(grid_in, grid_idx);
    const int data_idx = tree_data_idx(tree, bit_idx, grid_in->feature_size);
    return (data_in + data_idx)[f];
  }
  else {
    return 0;
  }

}


/// Applies a 3x3x3 convolution on a given position in grid grid_in.
/// @deprecated
///
/// @tparam inv_filter if filter weights should be transposed
/// @param n
/// @param ds
/// @param hs
/// @param ws
/// @param grid_in
/// @param weights convolution weights.
/// @param channels_out number of output channels after convolution.
/// @param factor scaling factor of weights.
/// @param out convolution result
template <bool inv_filter>
OCTREE_FUNCTION
inline void conv3x3x3_point(int n, int ds, int hs, int ws, const octree* grid_in, const ot_data_t* weights, int channels_out, float factor, ot_data_t* out) {
  const int channels_in = grid_in->feature_size;
  
  for(int kd = 0; kd < 3; ++kd) {
    for(int kh = 0; kh < 3; ++kh) {
      for(int kw = 0; kw < 3; ++kw) {
        int d, h, w;
        d = ds - 1 + kd;
        h = hs - 1 + kh;
        w = ws - 1 + kw;        

        int k_idx;
        if(inv_filter) {
          k_idx = ((2 - kd) * 3 + (2 - kh)) * 3 + (2 - kw);
        }
        else {
          k_idx = (kd * 3 + kh) * 3 + kw;
        }

        for(int ci = 0; ci < channels_in; ++ci) {
          ot_data_t in = factor * conv3x3x3_get_padded_data(grid_in, n, d, h, w, ci);
          for(int co = 0; co < channels_out; ++co) {
            int weights_idx;
            if(inv_filter) {
              weights_idx = (ci * channels_out + co) * 3 * 3 * 3 + k_idx;
            }
            else {
              weights_idx = (co * channels_in + ci) * 3 * 3 * 3 + k_idx;
            }
            out[co] += weights[weights_idx] * in;
          }
        }
      }
    }
  }

}

/// Performs a 3x3x3 convolution along an octree cell surface.
/// @deprecated
///
/// @tparam inv_filter if filter weights should be transposed.
/// @tparam add_bias if bias term should be added after convolution.
/// @param bd1 start index of surface in shallow octree.
/// @param bd2 end index of surface in shallow octree.
/// @param bh1 start index of surface in shallow octree.
/// @param bh2 end index of surface in shallow octree.
/// @param bw1 start index of surface in shallow octree.
/// @param bw2 end index of surface in shallow octree.
/// @param n batch index.
/// @param ds subscript index of tensor corresponding to shallow octree to locate surface.
/// @param hs subscript index of tensor corresponding to shallow octree to locate surface.
/// @param ws subscript index of tensor corresponding to shallow octree to locate surface.
/// @param grid_in 
/// @param weights convolution weights.
/// @param bias bias value.
/// @param channels_out number of output channels after conv.
/// @param factor scaling factor for convolution weights.
/// @param out convolution result
template <bool inv_filter, bool add_bias>
OCTREE_FUNCTION
inline void conv3x3x3_border(const int bd1, const int bd2, const int bh1, 
    const int bh2, const int bw1, const int bw2, const int n, const int ds, const int hs, const int ws,
    const octree* grid_in, const ot_data_t* weights, const ot_data_t* bias, int channels_out, float factor, ot_data_t* out) {
  // weights \in R^{channels_out \times channels_in \times kd=3 \times kh=3 \times kw=3}
  // out     \in R^{channels_out}
  
  int d, h, w;

  // Attention at start and end indices, so we convolve every border point only once
  d = ds + bd1;
  for(h = hs + bh1; h < hs + bh2; ++h) {
    for(w = ws + bw1; w < ws + bw2; ++w) {
      conv3x3x3_point<inv_filter>(n, d, h, w, grid_in, weights, channels_out, factor, out);
    }
  }

  d = ds + bd2 - 1;
  for(h = hs + bh1; h < hs + bh2; ++h) {
    for(w = ws + bw1; w < ws + bw2; ++w) {
      conv3x3x3_point<inv_filter>(n, d, h, w, grid_in, weights, channels_out, factor, out);
    }
  }

  h = hs + bh1;
  for(d = ds + bd1 + 1; d < ds + bd2 - 1; ++d) {
    for(w = ws + bw1; w < ws + bw2; ++w) {
      conv3x3x3_point<inv_filter>(n, d, h, w, grid_in, weights, channels_out, factor, out);
    }
  }

  h = hs + bh2 - 1;
  for(d = ds + bd1 + 1; d < ds + bd2 - 1; ++d) {
    for(w = ws + bw1; w < ws + bw2; ++w) {
      conv3x3x3_point<inv_filter>(n, d, h, w, grid_in, weights, channels_out, factor, out);
    }
  }

  w = ws + bw1;
  for(d = ds + bd1 + 1; d < ds + bd2 - 1; ++d) {
    for(h = hs + bh1 + 1; h < hs + bh2 - 1; ++h) {
      conv3x3x3_point<inv_filter>(n, d, h, w, grid_in, weights, channels_out, factor, out);
    }
  }

  w = ws + bw2 - 1;
  for(d = ds + bd1 + 1; d < ds + bd2 - 1; ++d) {
    for(h = hs + bh1 + 1; h < hs + bh2 - 1; ++h) {
      conv3x3x3_point<inv_filter>(n, d, h, w, grid_in, weights, channels_out, factor, out);
    }
  }


  if(add_bias) {
    ot_data_t bfactor = factor *
        (2 * (bh2 - bh1) * (bw2 - bw1) +
         2 * (bd2 - bd1 - 2) * (bw2 - bw1) + 
         2 * (bd2 - bd1 - 2) * (bh2 - bh1 - 2));
    for(int co = 0; co < channels_out; ++co) {
      out[co] += bfactor * bias[co];
    }
  }
}

/// Performs the convolution on the constant part of the octree cell.
/// @deprecated
///
/// @tparam inv_filter if filter weights should be transposed.
/// @tparam add_bias if bias term should be added after convolution.
/// @param in_data constant input cell data.
/// @param weights convolution weights.
/// @param bias bias term.
/// @param channels_in number of input channels for convolution.
/// @param channels_out number of output channels for convolution.
/// @param factor scaling factor for conv weights.
/// @param out convolution result
template <bool inv_filter, bool add_bias>
OCTREE_FUNCTION
inline void conv3x3x3_const(const ot_data_t* in_data, const ot_data_t* weights, const ot_data_t* bias, int channels_in, int channels_out, float factor, ot_data_t* out) {

  for(int kd = 0; kd < 3; ++kd) {
    for(int kh = 0; kh < 3; ++kh) {
      for(int kw = 0; kw < 3; ++kw) {
        
        int k_idx = (kd * 3 + kh) * 3 + kw;
        for(int ci = 0; ci < channels_in; ++ci) {
          ot_data_t in = factor * in_data[ci];
          for(int co = 0; co < channels_out; ++co) {
            int weights_idx;
            if(inv_filter) {
              weights_idx = (ci * channels_out + co) * 3 * 3 * 3 + k_idx;
            }
            else {
              weights_idx = (co * channels_in + ci) * 3 * 3 * 3 + k_idx;
            }
            out[co] += weights[weights_idx] * in;
          }
        }

      }
    }
  }

  if(add_bias) {
    for(int co = 0; co < channels_out; ++co) {
      out[co] += factor * bias[co];
      // printf("  b %f * %f => %f\n", factor, bias[co], out[co]);
    }
  }

}






/// Backward function for the convolution of a single point in the grid-octree
/// data structure. 
/// Computes the gradient wrt. to the weights.
/// @deprecated
///
/// @param n
/// @param ds subscript index of corresponding tensor.
/// @param hs subscript index of corresponding tensor.
/// @param ws subscript index of corresponding tensor.
/// @param grid_in input to the convolution operation.
/// @param grad_out gradient of the successive operation.
/// @param channels_out number of output channels of the convolution operation.
/// @param scale scale factor for the gradient update.
/// @param grad_weights resulting gradient of convolution weights.
OCTREE_FUNCTION
inline void conv3x3x3_point_wbwd(int n, int ds, int hs, int ws, const octree* grid_in, const ot_data_t* grad_out, int channels_out, ot_data_t scale, ot_data_t* grad_weights) {
  const int channels_in = grid_in->feature_size;
  for(int kd = 0; kd < 3; ++kd) {
    for(int kh = 0; kh < 3; ++kh) {
      for(int kw = 0; kw < 3; ++kw) {
        int d, h, w;
        d = ds - 1 + kd;
        h = hs - 1 + kh;
        w = ws - 1 + kw;        

        int k_idx = (kd * 3 + kh) * 3 + kw;

        for(int ci = 0; ci < channels_in; ++ci) {
          ot_data_t in = conv3x3x3_get_padded_data(grid_in, n, d, h, w, ci);
          for(int co = 0; co < channels_out; ++co) {
            int weights_idx;
            weights_idx = (co * channels_in + ci) * 3 * 3 * 3 + k_idx;

            ot_data_t val = scale * grad_out[co] * in;

            #if defined(__CUDA_ARCH__)
            atomicAdd(grad_weights + weights_idx, val);
            #elif defined(_OPENMP)
            #pragma omp atomic
            grad_weights[weights_idx] += val;
            #else
            grad_weights[weights_idx] += val;
            #endif
          }
        }

      }
    }
  }

}


/// Backward function for the convolution along a octree cell surface.
/// Computes the gradient wrt. to the weights.
/// @deprecated
///
/// @param bd1 start index of surface in shallow octree.
/// @param bd2 end index of surface in shallow octree.
/// @param bh1 start index of surface in shallow octree.
/// @param bh2 end index of surface in shallow octree.
/// @param bw1 start index of surface in shallow octree.
/// @param bw2 end index of surface in shallow octree.
/// @param n batch index.
/// @param ds subscript index of tensor corresponding to shallow octree to locate surface.
/// @param hs subscript index of tensor corresponding to shallow octree to locate surface.
/// @param ws subscript index of tensor corresponding to shallow octree to locate surface.
/// @param grid_in octree input to the convolution.
/// @param grad_out gradient of the successive operation.
/// @param channels_out number of output channels of the convolution.
/// @param scale scale factor of the gradient update.
/// @param grad_weights resulting gradient weights.
/// @param grad_bias resulting gradient bias.
OCTREE_FUNCTION
inline void conv3x3x3_border_wbwd(const int bd1, const int bd2, const int bh1, 
    const int bh2, const int bw1, const int bw2, const int n, const int ds, const int hs, const int ws,
    const octree* grid_in, const ot_data_t* grad_out, int channels_out, ot_data_t scale, ot_data_t* grad_weights, ot_data_t* grad_bias) {
  
  int d, h, w;

  // Attention at start and end indices, so we convolve every border point only once
  d = ds + bd1;
  for(h = hs + bh1; h < hs + bh2; ++h) {
    for(w = ws + bw1; w < ws + bw2; ++w) {
      conv3x3x3_point_wbwd(n, d, h, w, grid_in, grad_out, channels_out, scale, grad_weights);
    }
  }

  d = ds + bd2 - 1;
  for(h = hs + bh1; h < hs + bh2; ++h) {
    for(w = ws + bw1; w < ws + bw2; ++w) {
      conv3x3x3_point_wbwd(n, d, h, w, grid_in, grad_out, channels_out, scale, grad_weights);
    }
  }

  h = hs + bh1;
  for(d = ds + bd1 + 1; d < ds + bd2 - 1; ++d) {
    for(w = ws + bw1; w < ws + bw2; ++w) {
      conv3x3x3_point_wbwd(n, d, h, w, grid_in, grad_out, channels_out, scale, grad_weights); 
    }
  }

  h = hs + bh2 - 1;
  for(d = ds + bd1 + 1; d < ds + bd2 - 1; ++d) {
    for(w = ws + bw1; w < ws + bw2; ++w) {
      conv3x3x3_point_wbwd(n, d, h, w, grid_in, grad_out, channels_out, scale, grad_weights); 
    }
  }

  w = ws + bw1;
  for(d = ds + bd1 + 1; d < ds + bd2 - 1; ++d) {
    for(h = hs + bh1 + 1; h < hs + bh2 - 1; ++h) {
      conv3x3x3_point_wbwd(n, d, h, w, grid_in, grad_out, channels_out, scale, grad_weights); 
    }
  }

  w = ws + bw2 - 1;
  for(d = ds + bd1 + 1; d < ds + bd2 - 1; ++d) {
    for(h = hs + bh1 + 1; h < hs + bh2 - 1; ++h) {
      conv3x3x3_point_wbwd(n, d, h, w, grid_in, grad_out, channels_out, scale, grad_weights); 
    }
  }


  ot_data_t factor = scale * 
      (2 * (bh2 - bh1) * (bw2 - bw1) +
       2 * (bd2 - bd1 - 2) * (bw2 - bw1) + 
       2 * (bd2 - bd1 - 2) * (bh2 - bh1 - 2));
  for(int co = 0; co < channels_out; ++co) {
    ot_data_t val = factor * grad_out[co];

    #if defined(__CUDA_ARCH__)
    atomicAdd(grad_bias + co, val);
    #elif defined(_OPENMP)
    #pragma omp atomic
    grad_bias[co] += val;
    #else
    grad_bias[co] += val;
    #endif
  }
}


/// Backward function for the constant convolution.
/// @deprecated
///
/// @param in_data constant input cell data.
/// @param grad_out the gradient of the successive operation.
/// @param channels_in the number of input channels
/// @param channels_out number of output channels for convolution.
/// @param scale scale factor of the gradient update.
/// @param grad_weights resulting gradient weights.
/// @param grad_bias resulting gradient bias.
OCTREE_FUNCTION
inline void conv3x3x3_const_wbwd(const ot_data_t* in_data, const ot_data_t* grad_out, int channels_in, int channels_out, float factor, ot_data_t* grad_weights, ot_data_t* grad_bias) {    
  for(int kd = 0; kd < 3; ++kd) {
    for(int kh = 0; kh < 3; ++kh) {
      for(int kw = 0; kw < 3; ++kw) {
        
        int k_idx = (kd * 3 + kh) * 3 + kw;
        for(int ci = 0; ci < channels_in; ++ci) {
          ot_data_t in = factor * in_data[ci];
          for(int co = 0; co < channels_out; ++co) {
            int weights_idx;
            weights_idx = (co * channels_in + ci) * 3 * 3 * 3 + k_idx;

            ot_data_t val = grad_out[co] * in;

            #if defined(__CUDA_ARCH__)
            atomicAdd(grad_weights + weights_idx, val);
            #elif defined(_OPENMP)
            #pragma omp atomic
            grad_weights[weights_idx] += val;
            #else
            grad_weights[weights_idx] += val;
            #endif
          }
        }

      }
    }
  }

  for(int co = 0; co < channels_out; ++co) {
    ot_data_t val = factor * grad_out[co];

    #if defined(__CUDA_ARCH__)
    atomicAdd(grad_bias + co, val);
    #elif defined(_OPENMP)
    #pragma omp atomic
    grad_bias[co] += val;
    #else
    grad_bias[co] += val;
    #endif
  }
}


#endif 
