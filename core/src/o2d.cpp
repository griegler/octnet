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

#include "octnet/cpu/dense.h"


template <int dense_format, bool avg_vol>
void octree_to_dense_cpu(const octree* grid_h, const int dense_depth, const int dense_height, const int dense_width, ot_data_t* out_data) {
  int n_blocks = octree_num_blocks(grid_h);
  int n = grid_h->n;
  int grid_height = grid_h->grid_height;
  int grid_width = grid_h->grid_width;
  int grid_depth = grid_h->grid_depth;

  if(dense_depth < grid_depth * 8 || dense_height < grid_height * 8 || dense_width < grid_width * 8) {
    printf("[ERROR] dense dim (%d,%d,%d) is smaller then dim of grid (%d,%d,%d)\n", 
        dense_depth, dense_height, dense_width, grid_depth*8, grid_height*8, grid_width*8);
    exit(-1);
  }
  int vx_depth_off = (dense_depth - grid_depth * 8) / 2;
  int vx_height_off = (dense_height - grid_height * 8) / 2;
  int vx_width_off = (dense_width - grid_width * 8) / 2;

  int feature_size = grid_h->feature_size;
  
  int n_dense_elems = n * feature_size * dense_depth * dense_height * dense_width;
  #pragma omp parallel for
  for(int idx = 0; idx < n_dense_elems; ++idx) {
    out_data[idx] = 0;
  }

  #pragma omp parallel for
  for(int grid_idx = 0; grid_idx < n_blocks; ++grid_idx) {
    int gn, gd, gh, gw;
    octree_split_grid_idx(grid_h, grid_idx, &gn, &gd, &gh, &gw); 
    
    ot_tree_t* tree = octree_get_tree(grid_h, grid_idx);

    for(int bd = 0; bd < 8; ++bd) {
      for(int bh = 0; bh < 8; ++bh) {
        for(int bw = 0; bw < 8; ++bw) {
          int vx_d = (gd * 8) + bd + vx_depth_off;
          int vx_h = (gh * 8) + bh + vx_height_off;
          int vx_w = (gw * 8) + bw + vx_width_off;
            
          int bit_idx = tree_bit_idx(tree, bd, bh, bw);

          float vol = 1;
          if(avg_vol) {
            vol = bit_idx == 0 ? 512 : (bit_idx < 9 ? 64 : (bit_idx < 73 ? 8 : 1));
          }

          int data_idx = tree_data_idx(tree, bit_idx, feature_size);
          // const ot_data_t* data = grid_h->data_ptrs[grid_idx] + data_idx;
          const ot_data_t* data = octree_get_data(grid_h, grid_idx) + data_idx;
          
          for(int f = 0; f < feature_size; ++f) {
            ot_data_t val = data[f];
            int out_idx; 
            if(dense_format == DENSE_FORMAT_DHWC) {
              out_idx = (((gn * dense_depth + vx_d) * dense_height + vx_h) * dense_width + vx_w) * feature_size + f;
            }
            else if(dense_format == DENSE_FORMAT_CDHW) {
              out_idx = (((gn * feature_size + f) * dense_depth + vx_d) * dense_height + vx_h) * dense_width + vx_w;
            }
            if(avg_vol) {
              out_data[out_idx] = val / vol;
            }
            else {
              out_data[out_idx] = val;
            }
          }
        }
      }
    }
  }

}

extern "C"
void octree_to_dhwc_cpu(const octree* grid_h, const int dense_depth, const int dense_height, const int dense_width, ot_data_t* out_data) {
  octree_to_dense_cpu<DENSE_FORMAT_DHWC, false>(grid_h, dense_depth, dense_height, dense_width, out_data); 
}

extern "C"
void octree_to_cdhw_cpu(const octree* grid_h, const int dense_depth, const int dense_height, const int dense_width, ot_data_t* out_data) {
  octree_to_dense_cpu<DENSE_FORMAT_CDHW, false>(grid_h, dense_depth, dense_height, dense_width, out_data); 
}


extern "C"
void octree_to_dhwc_avg_cpu(const octree* grid_h, const int dense_depth, const int dense_height, const int dense_width, ot_data_t* out_data) {
  octree_to_dense_cpu<DENSE_FORMAT_DHWC, true>(grid_h, dense_depth, dense_height, dense_width, out_data); 
}

extern "C"
void octree_to_cdhw_avg_cpu(const octree* grid_h, const int dense_depth, const int dense_height, const int dense_width, ot_data_t* out_data) {
  octree_to_dense_cpu<DENSE_FORMAT_CDHW, true>(grid_h, dense_depth, dense_height, dense_width, out_data); 
}


extern "C"
void octree_to_dhwc_bwd_cpu(const octree* grid_h, const int dense_depth, const int dense_height, const int dense_width, const ot_data_t* grad_data, octree* grad_grid_h) {
  dhwc_to_octree_sum_cpu(grid_h, dense_depth, dense_height, dense_width, grad_data, grid_h->feature_size, grad_grid_h);
}

extern "C"
void octree_to_cdhw_bwd_cpu(const octree* grid_h, const int dense_depth, const int dense_height, const int dense_width, const ot_data_t* grad_data, octree* grad_grid_h) {
  cdhw_to_octree_sum_cpu(grid_h, dense_depth, dense_height, dense_width, grad_data, grid_h->feature_size, grad_grid_h);
}

