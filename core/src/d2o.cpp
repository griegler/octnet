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
#include "octnet/cpu/cpu.h"

#include <iostream>
#include <sstream>
#include <cstring>
#include <cstdio>
#include <cstdlib>

#if defined(_OPENMP)
#include <omp.h>
#endif

template <int dense_format, int reduce_fcn>
void dense_to_octree_cpu(const octree* grid_h_in, const int dense_depth, const int dense_height, const int dense_width, const ot_data_t* dense, int out_feature_size, octree* grid_h) {

  octree_resize_cpu(grid_h_in->n, grid_h_in->grid_depth, grid_h_in->grid_height, grid_h_in->grid_width, out_feature_size, grid_h_in->n_leafs, grid_h);
  grid_h->feature_size = out_feature_size;
  octree_cpy_trees_cpu_cpu(grid_h_in, grid_h);
  octree_cpy_prefix_leafs_cpu_cpu(grid_h_in, grid_h);

  int n_blocks = octree_num_blocks(grid_h_in);
  ot_size_t feature_size = grid_h->feature_size;

  int vx_depth_off = (dense_depth - grid_h_in->grid_depth * 8) / 2;
  int vx_height_off = (dense_height - grid_h_in->grid_height * 8) / 2;
  int vx_width_off = (dense_width - grid_h_in->grid_width * 8) / 2;

  #pragma omp parallel for
  for(int grid_idx = 0; grid_idx < n_blocks; ++grid_idx) {
    int gn, dl0, hl0, wl0;
    octree_split_grid_idx(grid_h_in, grid_idx, &gn, &dl0, &hl0, &wl0); 

    ot_tree_t* tree = octree_get_tree(grid_h, grid_idx);
    // ot_data_t* data = grid_h->data_ptrs[grid_idx];
    ot_data_t* data = octree_get_data(grid_h, grid_idx);

    dl0 *= 8;
    hl0 *= 8;
    wl0 *= 8;
    dl0 += vx_depth_off;
    hl0 += vx_height_off;
    wl0 += vx_width_off;

    // see if first level is set
    int bit_idx_l0 = 0;
    if(!tree_isset_bit(tree, bit_idx_l0)) { 
      //if not set -> average content
      int data_idx = 0;
      ot_data_t* out = data + data_idx;
      dense_to_octree_fcn<reduce_fcn, dense_format>(dense, gn, dense_depth, dense_height, dense_width, feature_size, dl0,dl0+8, hl0,hl0+8, wl0,wl0+8, out);  
    }
    else { 
      // std::cout << "bit_idx_l0 is SET" << std::endl;

      // see child nodes if are set - second level
      for(int dl1 = 0; dl1 < 2; ++dl1) {
        for(int hl1 = 0; hl1 < 2; ++hl1) {
          for(int wl1 = 0; wl1 < 2; ++wl1) {
            int bit_idx_l1 = 1 + (dl1 * 2 + hl1) * 2 + wl1;
            int dl1_1 = dl0 + dl1*4;
            int hl1_1 = hl0 + hl1*4;
            int wl1_1 = wl0 + wl1*4;

            // see if second level is set
            if(!tree_isset_bit(tree, bit_idx_l1)) {
              int data_idx = tree_data_idx(tree, bit_idx_l1, feature_size);
              ot_data_t* out = data + data_idx;
              dense_to_octree_fcn<reduce_fcn, dense_format>(dense, gn, dense_depth, dense_height, dense_width, feature_size, dl1_1,dl1_1+4, hl1_1,hl1_1+4, wl1_1,wl1_1+4, out);  
            }
            else {

              for(int dl2 = 0; dl2 < 2; ++dl2) {
                for(int hl2 = 0; hl2 < 2; ++hl2) {
                  for(int wl2 = 0; wl2 < 2; ++wl2) {
                    int bit_idx_l2 = tree_child_bit_idx(bit_idx_l1) + (dl2 * 2 + hl2) * 2 + wl2;
                    int dl2_1 = dl1_1 + dl2*2;
                    int hl2_1 = hl1_1 + hl2*2;
                    int wl2_1 = wl1_1 + wl2*2;
                    if(!tree_isset_bit(tree, bit_idx_l2)) {
                      int data_idx = tree_data_idx(tree, bit_idx_l2, feature_size);
                      ot_data_t* out = data + data_idx;
                      dense_to_octree_fcn<reduce_fcn, dense_format>(dense, gn, dense_depth, dense_height, dense_width, feature_size, dl2_1,dl2_1+2, hl2_1,hl2_1+2, wl2_1,wl2_1+2, out);  
                    }
                    else {

                      for(int dl3 = 0; dl3 < 2; ++dl3) {
                        for(int hl3 = 0; hl3 < 2; ++hl3) {
                          for(int wl3 = 0; wl3 < 2; ++wl3) {
                            int bit_idx_l3 = tree_child_bit_idx(bit_idx_l2) + (dl3 * 2 + hl3) * 2 + wl3;
                            int d = dl2_1 + dl3;
                            int h = hl2_1 + hl3;
                            int w = wl2_1 + wl3;
                            
                            int data_idx = tree_data_idx(tree, bit_idx_l3, feature_size);
                            ot_data_t* out = data + data_idx;
                            if(dense_format == DENSE_FORMAT_DHWC) {
                              for(int f = 0; f < feature_size; ++f) {
                                out[f] = dense[(((gn * dense_depth + d) * dense_height + h) * dense_width + w) * feature_size + f];
                              }
                            }
                            else if(dense_format == DENSE_FORMAT_CDHW) {
                              for(int f = 0; f < feature_size; ++f) {
                                out[f] = dense[(((gn * feature_size + f) * dense_depth + d) * dense_height + h) * dense_width + w];
                              }
                            }

                          } // for wl3
                        } // for hl3
                      } // for dl3

                    } // else of if bit_idx_l2

                  } // for wl2
                } // for hl2
              } // for dl2

            } // else of if bit_idx_l1

          } // for wl1
        } // for hl1
      } // for dl1

    } //else of if bit_idx_l0
  } //for grid_idx

}



extern "C"
void dhwc_to_octree_avg_cpu(const octree* grid_h_in, const int dense_depth, const int dense_height, const int dense_width, const ot_data_t* data, int out_feature_size, octree* grid_h_out) {
  dense_to_octree_cpu<DENSE_FORMAT_DHWC, REDUCE_AVG>(grid_h_in, dense_depth, dense_height, dense_width, data, out_feature_size, grid_h_out);
}

extern "C"
void cdhw_to_octree_avg_cpu(const octree* grid_h_in, const int dense_depth, const int dense_height, const int dense_width, const ot_data_t* data, int out_feature_size, octree* grid_h_out) {
  dense_to_octree_cpu<DENSE_FORMAT_CDHW, REDUCE_AVG>(grid_h_in, dense_depth, dense_height, dense_width, data, out_feature_size, grid_h_out);
}


extern "C"
void dhwc_to_octree_max_cpu(const octree* grid_h_in, const int dense_depth, const int dense_height, const int dense_width, const ot_data_t* data, int out_feature_size, octree* grid_h_out) {
  dense_to_octree_cpu<DENSE_FORMAT_DHWC, REDUCE_MAX>(grid_h_in, dense_depth, dense_height, dense_width, data, out_feature_size, grid_h_out);
}

extern "C"
void cdhw_to_octree_max_cpu(const octree* grid_h_in, const int dense_depth, const int dense_height, const int dense_width, const ot_data_t* data, int out_feature_size, octree* grid_h_out) {
  dense_to_octree_cpu<DENSE_FORMAT_CDHW, REDUCE_MAX>(grid_h_in, dense_depth, dense_height, dense_width, data, out_feature_size, grid_h_out);
}


extern "C"
void dhwc_to_octree_sum_cpu(const octree* grid_h_in, const int dense_depth, const int dense_height, const int dense_width, const ot_data_t* data, int out_feature_size, octree* grid_h_out) {
  dense_to_octree_cpu<DENSE_FORMAT_DHWC, REDUCE_SUM>(grid_h_in, dense_depth, dense_height, dense_width, data, out_feature_size, grid_h_out);
}

extern "C"
void cdhw_to_octree_sum_cpu(const octree* grid_h_in, const int dense_depth, const int dense_height, const int dense_width, const ot_data_t* data, int out_feature_size, octree* grid_h_out) {
  dense_to_octree_cpu<DENSE_FORMAT_CDHW, REDUCE_SUM>(grid_h_in, dense_depth, dense_height, dense_width, data, out_feature_size, grid_h_out);
}

extern "C"
void dhwc_to_octree_sum_bwd_cpu(const octree* grad_out_grid_h, const int dense_depth, const int dense_height, const int dense_width, ot_data_t* grad_in_data) {
  // octree_to_dense_cpu<DENSE_FORMAT_DHWC, false>(grad_out_grid_h, grad_in_data); 
  octree_to_dhwc_cpu(grad_out_grid_h, dense_depth, dense_height, dense_width, grad_in_data);
}

extern "C"
void cdhw_to_octree_sum_bwd_cpu(const octree* grad_out_grid_h, const int dense_depth, const int dense_height, const int dense_width, ot_data_t* grad_in_data) {
  // octree_to_dense_cpu<DENSE_FORMAT_CDHW, false>(grad_out_grid_h, grad_in_data); 
  octree_to_cdhw_cpu(grad_out_grid_h, dense_depth, dense_height, dense_width, grad_in_data);
}


extern "C"
void dhwc_to_octree_avg_bwd_cpu(const octree* grad_out_grid_h, const int dense_depth, const int dense_height, const int dense_width, ot_data_t* grad_in_data) {
  // octree_to_dense_cpu<DENSE_FORMAT_DHWC, true>(grad_out_grid_h, grad_in_data); 
  octree_to_dhwc_avg_cpu(grad_out_grid_h, dense_depth, dense_height, dense_width, grad_in_data);
}

extern "C"
void cdhw_to_octree_avg_bwd_cpu(const octree* grad_out_grid_h, const int dense_depth, const int dense_height, const int dense_width, ot_data_t* grad_in_data) {
  // octree_to_dense_cpu<DENSE_FORMAT_CDHW, true>(grad_out_grid_h, grad_in_data); 
  octree_to_cdhw_avg_cpu(grad_out_grid_h, dense_depth, dense_height, dense_width, grad_in_data);
}

