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

#include "octnet/core/conv3x3x3.h"
#include "octnet/cpu/cpu.h"

#include <cstdlib>
#include <cstdio>
#include <cstring>

#if defined(_OPENMP)
#include <omp.h>
#endif


template <int rdc_fcn>
void octree_conv3x3x3_cpu(const octree* grid_in, const ot_data_t* weights, const ot_data_t* bias, int channels_out, octree* grid) {
  octree_resize_cpu(grid_in->n, grid_in->grid_depth, grid_in->grid_height, grid_in->grid_width, channels_out, grid_in->n_leafs, grid);
  octree_cpy_scalars(grid_in, grid);
  grid->feature_size = channels_out;
  octree_cpy_trees_cpu_cpu(grid_in, grid);
  octree_cpy_prefix_leafs_cpu_cpu(grid_in, grid);

  int n_blocks = octree_num_blocks(grid_in);
  const int channels_in = grid_in->feature_size;

  octree_fill_data_cpu(grid, 0);

  #pragma omp parallel for
  for(int grid_idx = 0; grid_idx < n_blocks; ++grid_idx) {
    ot_tree_t* tree = octree_get_tree(grid, grid_idx);
    // ot_data_t* in_data = grid_in->data_ptrs[grid_idx];
    ot_data_t* in_data = octree_get_data(grid_in, grid_idx);
    // ot_data_t* out_data = grid->data_ptrs[grid_idx];
    ot_data_t* out_data = octree_get_data(grid, grid_idx);

    int gn, gd, gh, gw;
    octree_split_grid_idx(grid_in, grid_idx, &gn, &gd, &gh, &gw); 

    int ds = gd * 8;
    int hs = gh * 8;
    int ws = gw * 8;
    
    //check if L0 split is set
    if(!tree_isset_bit(tree, 0)) {
      // if NOT set
      float factor;
      if(rdc_fcn == REDUCE_SUM) {
        factor = 1;
      }
      else if(rdc_fcn == REDUCE_AVG) {
        factor = 1.f / (8*8*8);
      }
      conv3x3x3_border<INV_FILTER_FALSE, ADD_BIAS_TRUE>(0, 8, 0, 8, 0, 8, gn, ds, hs, ws, grid_in, weights, bias, channels_out, factor, out_data);
      conv3x3x3_const<INV_FILTER_FALSE, ADD_BIAS_TRUE>(in_data, weights, bias, channels_in, channels_out, factor*6*6*6, out_data);
    }
    else {

      int bit_idx_l1 = 1;
      for(int bdl1 = 0; bdl1 < 2; ++bdl1) {
        for(int bhl1 = 0; bhl1 < 2; ++bhl1) {
          for(int bwl1 = 0; bwl1 < 2; ++bwl1) {

            if(!tree_isset_bit(tree, bit_idx_l1)) {
              int out_data_idx = tree_data_idx(tree, bit_idx_l1, channels_out);
              int in_data_idx = out_data_idx / channels_out * channels_in;
              float factor;
              if(rdc_fcn == REDUCE_SUM) {
                factor = 1;
              }
              else if(rdc_fcn == REDUCE_AVG) {
                factor = 1.f / (4*4*4);
              }
              conv3x3x3_border<INV_FILTER_FALSE, ADD_BIAS_TRUE>(bdl1*4, bdl1*4+4, bhl1*4, bhl1*4+4, bwl1*4, bwl1*4+4, 
                  gn, ds, hs, ws, grid_in, weights, bias, channels_out, factor, out_data + out_data_idx);
              conv3x3x3_const<INV_FILTER_FALSE, ADD_BIAS_TRUE>(in_data + in_data_idx, weights, bias, channels_in, channels_out, factor*2*2*2, out_data + out_data_idx);
            }
            else {

              int bit_idx_l2 = tree_child_bit_idx(bit_idx_l1);
              for(int bdl2 = 0; bdl2 < 2; ++bdl2) {
                for(int bhl2 = 0; bhl2 < 2; ++bhl2) {
                  for(int bwl2 = 0; bwl2 < 2; ++bwl2) {

                    if(!tree_isset_bit(tree, bit_idx_l2)) {
                      int out_data_idx = tree_data_idx(tree, bit_idx_l2, channels_out);
                      // int in_data_idx = out_data_idx / channels_out * channels_in;
                      float factor;
                      if(rdc_fcn == REDUCE_SUM) {
                        factor = 1;
                      }
                      else if(rdc_fcn == REDUCE_AVG) {
                        factor = 1.f / (2*2*2);
                      }
                      conv3x3x3_border<INV_FILTER_FALSE, ADD_BIAS_TRUE>(bdl1*4+bdl2*2, bdl1*4+bdl2*2+2, bhl1*4+bhl2*2, bhl1*4+bhl2*2+2, bwl1*4+bwl2*2, bwl1*4+bwl2*2+2, 
                          gn, ds, hs, ws, grid_in, weights, bias, channels_out, factor, out_data + out_data_idx);
                    }
                    else {

                      int bit_idx_l3 = tree_child_bit_idx(bit_idx_l2);
                      for(int bdl3 = 0; bdl3 < 2; ++bdl3) {
                        for(int bhl3 = 0; bhl3 < 2; ++bhl3) {
                          for(int bwl3 = 0; bwl3 < 2; ++bwl3) {
                            int out_data_idx = tree_data_idx(tree, bit_idx_l3, channels_out);
                            // printf("%d, %d,%d,%d\n", bit_idx_l3, ds+bdl1*4+bdl2*2+bdl3, hs+bhl1*4+bhl2*2+bhl3, ws+bwl1*4+bwl2*2+bwl3);
                            conv3x3x3_point<INV_FILTER_FALSE>(gn, ds+bdl1*4+bdl2*2+bdl3, hs+bhl1*4+bhl2*2+bhl3, ws+bwl1*4+bwl2*2+bwl3, 
                                grid_in, weights, channels_out, 1, out_data + out_data_idx);  
                            for(int co = 0; co < channels_out; ++co) {
                              out_data[out_data_idx + co] += bias[co];
                            }
                            bit_idx_l3++;
                          }
                        }
                      }

                    }
                    bit_idx_l2++;

                  }
                }
              } 

            } // else if isset L1
            bit_idx_l1++;

          } // for bwl1
        } // for bhl1
      } // for bdl1

    } // if isset L0
  } // for grid_idx

}



template <int rdc_fcn>
void octree_conv3x3x3_bwd_cpu(const ot_data_t* weights, const octree* grad_out, int channels_in, octree* grad_in) {
  octree_resize_cpu(grad_out->n, grad_out->grid_depth, grad_out->grid_height, grad_out->grid_width, channels_in, grad_out->n_leafs, grad_in);
  octree_cpy_scalars(grad_out, grad_in);
  grad_in->feature_size = channels_in;
  octree_cpy_trees_cpu_cpu(grad_out, grad_in);
  octree_cpy_prefix_leafs_cpu_cpu(grad_out, grad_in);

  int n_blocks = octree_num_blocks(grad_out);
  const int channels_out = grad_out->feature_size;

  octree_fill_data_cpu(grad_in, 0);

  #pragma omp parallel for
  for(int grid_idx = 0; grid_idx < n_blocks; ++grid_idx) {
    ot_tree_t* tree = octree_get_tree(grad_in, grid_idx);
    // ot_data_t* grad_in_data = grad_in->data_ptrs[grid_idx];
    ot_data_t* grad_in_data = octree_get_data(grad_in, grid_idx);
    // ot_data_t* grad_out_data = grad_out->data_ptrs[grid_idx];
    ot_data_t* grad_out_data = octree_get_data(grad_out, grid_idx);

    int gn, gd, gh, gw;
    octree_split_grid_idx(grad_in, grid_idx, &gn, &gd, &gh, &gw); 

    int ds = gd * 8;
    int hs = gh * 8;
    int ws = gw * 8;
    
    //check if L0 split is set
    if(!tree_isset_bit(tree, 0)) {
      // if NOT set
      float factor;
      if(rdc_fcn == REDUCE_SUM) {
        factor = 1;
      }
      else if(rdc_fcn == REDUCE_AVG) {
        factor = 1.f / (8*8*8);
      }
      conv3x3x3_border<INV_FILTER_TRUE, ADD_BIAS_FALSE>(0, 8, 0, 8, 0, 8, gn, ds, hs, ws, grad_out, weights, 0, channels_in, factor, grad_in_data);
      conv3x3x3_const<INV_FILTER_TRUE, ADD_BIAS_FALSE>(grad_out_data, weights, 0, channels_out, channels_in, factor*6*6*6, grad_in_data);
    }
    else {

      int bit_idx_l1 = 1;
      for(int bdl1 = 0; bdl1 < 2; ++bdl1) {
        for(int bhl1 = 0; bhl1 < 2; ++bhl1) {
          for(int bwl1 = 0; bwl1 < 2; ++bwl1) {

            if(!tree_isset_bit(tree, bit_idx_l1)) {
              int out_data_idx = tree_data_idx(tree, bit_idx_l1, channels_out);
              int in_data_idx = out_data_idx / channels_out * channels_in;
              float factor;
              if(rdc_fcn == REDUCE_SUM) {
                factor = 1;
              }
              else if(rdc_fcn == REDUCE_AVG) {
                factor = 1.f / (4*4*4);
              }
              conv3x3x3_border<INV_FILTER_TRUE, ADD_BIAS_FALSE>(bdl1*4, bdl1*4+4, bhl1*4, bhl1*4+4, bwl1*4, bwl1*4+4, 
                  gn, ds, hs, ws, grad_out, weights, 0, channels_in, factor, grad_in_data + in_data_idx);
              conv3x3x3_const<INV_FILTER_TRUE, ADD_BIAS_FALSE>(grad_out_data + out_data_idx, weights, 0, channels_out, channels_in, factor*2*2*2, grad_in_data + in_data_idx);
            }
            else {

              int bit_idx_l2 = tree_child_bit_idx(bit_idx_l1);
              for(int bdl2 = 0; bdl2 < 2; ++bdl2) {
                for(int bhl2 = 0; bhl2 < 2; ++bhl2) {
                  for(int bwl2 = 0; bwl2 < 2; ++bwl2) {

                    if(!tree_isset_bit(tree, bit_idx_l2)) {
                      int out_data_idx = tree_data_idx(tree, bit_idx_l2, channels_out);
                      int in_data_idx = out_data_idx / channels_out * channels_in;
                      float factor;
                      if(rdc_fcn == REDUCE_SUM) {
                        factor = 1;
                      }
                      else if(rdc_fcn == REDUCE_AVG) {
                        factor = 1.f / (2*2*2);
                      }
                      conv3x3x3_border<INV_FILTER_TRUE, ADD_BIAS_FALSE>(bdl1*4+bdl2*2, bdl1*4+bdl2*2+2, bhl1*4+bhl2*2, bhl1*4+bhl2*2+2, bwl1*4+bwl2*2, bwl1*4+bwl2*2+2, 
                          gn, ds, hs, ws, grad_out, weights, 0, channels_in, factor, grad_in_data + in_data_idx);
                    }
                    else {

                      int bit_idx_l3 = tree_child_bit_idx(bit_idx_l2);
                      for(int bdl3 = 0; bdl3 < 2; ++bdl3) {
                        for(int bhl3 = 0; bhl3 < 2; ++bhl3) {
                          for(int bwl3 = 0; bwl3 < 2; ++bwl3) {
                            int in_data_idx = tree_data_idx(tree, bit_idx_l3, channels_in);
                            conv3x3x3_point<INV_FILTER_TRUE>(gn, ds+bdl1*4+bdl2*2+bdl3, hs+bhl1*4+bhl2*2+bhl3, ws+bwl1*4+bwl2*2+bwl3, 
                                grad_out, weights, channels_in, 1, grad_in_data + in_data_idx);  
                            bit_idx_l3++;
                          }
                        }
                      }

                    }
                    bit_idx_l2++;

                  }
                }
              } 

            } // else if isset L1
            bit_idx_l1++;

          } // for bwl1
        } // for bhl1
      } // for bdl1

    } // if isset L0
  } // for grid_idx
}




template <int rdc_fcn>
void octree_conv3x3x3_wbwd_cpu(const octree* grid_in, const octree* grad_out, ot_data_t scale, ot_data_t* grad_weights, ot_data_t* grad_bias) {
  int n_blocks = octree_num_blocks(grad_out);
  const int channels_out = grad_out->feature_size;
  const int channels_in = grid_in->feature_size;

  #pragma omp parallel for
  for(int grid_idx = 0; grid_idx < n_blocks; ++grid_idx) {
    ot_tree_t* tree = octree_get_tree(grid_in, grid_idx);
    // ot_data_t* grid_in_data = grid_in->data_ptrs[grid_idx];
    ot_data_t* grid_in_data = octree_get_data(grid_in, grid_idx);
    // ot_data_t* grad_out_data = grad_out->data_ptrs[grid_idx];
    ot_data_t* grad_out_data = octree_get_data(grad_out, grid_idx);

    int gn, gd, gh, gw;
    octree_split_grid_idx(grad_out, grid_idx, &gn, &gd, &gh, &gw); 

    int ds = gd * 8;
    int hs = gh * 8;
    int ws = gw * 8;

    
    //check if L0 split is set
    if(!tree_isset_bit(tree, 0)) {
      // if NOT set
      // conv3x3x3_border_wbwd(0, 8, 0, 8, 0, 8, ds, hs, ws, grad_out, weights, 0, channels_in, grad_in_data);
      // conv3x3x3_const<INV_FILTER_TRUE, ADD_BIAS_FALSE>(grad_out_data, weights, 0, channels_out, channels_in, 6*6*6, grad_in_data);
      float factor;
      if(rdc_fcn == REDUCE_SUM) {
        factor = scale;
      }
      else if(rdc_fcn == REDUCE_AVG) {
        factor = scale / (8*8*8);
      }
      conv3x3x3_border_wbwd(0, 8, 0, 8, 0, 8, gn, ds, hs, ws, grid_in, grad_out_data, channels_out, factor, grad_weights, grad_bias);
      conv3x3x3_const_wbwd(grid_in_data, grad_out_data, channels_in, channels_out, factor*6*6*6, grad_weights, grad_bias);
    }
    else {

      int bit_idx_l1 = 1;
      for(int bdl1 = 0; bdl1 < 2; ++bdl1) {
        for(int bhl1 = 0; bhl1 < 2; ++bhl1) {
          for(int bwl1 = 0; bwl1 < 2; ++bwl1) {

            if(!tree_isset_bit(tree, bit_idx_l1)) {
              float factor;
              if(rdc_fcn == REDUCE_SUM) {
                factor = scale;
              }
              else if(rdc_fcn == REDUCE_AVG) {
                factor = scale / (4*4*4);
              }
              int out_data_idx = tree_data_idx(tree, bit_idx_l1, channels_out);
              int in_data_idx = out_data_idx / channels_out * channels_in;
              // conv3x3x3_border_wbwd(bdl1*4, bdl1*4+4, bhl1*4, bhl1*4+4, bwl1*4, bwl1*4+4, 
              //     ds, hs, ws, grad_out, weights, 0, channels_in, grad_in_data + in_data_idx);
              // conv3x3x3_const_wbwd(grad_out_data + out_data_idx, weights, 0, channels_out, channels_in, 2*2*2, grad_in_data + in_data_idx);
              conv3x3x3_border_wbwd(bdl1*4, bdl1*4+4, bhl1*4, bhl1*4+4, bwl1*4, bwl1*4+4,
                  gn, ds, hs, ws, grid_in, grad_out_data + out_data_idx, channels_out, factor, grad_weights, grad_bias);
              conv3x3x3_const_wbwd(grid_in_data + in_data_idx, grad_out_data + out_data_idx, channels_in, channels_out, factor*2*2*2, grad_weights, grad_bias);
            }
            else {

              int bit_idx_l2 = tree_child_bit_idx(bit_idx_l1);
              for(int bdl2 = 0; bdl2 < 2; ++bdl2) {
                for(int bhl2 = 0; bhl2 < 2; ++bhl2) {
                  for(int bwl2 = 0; bwl2 < 2; ++bwl2) {

                    if(!tree_isset_bit(tree, bit_idx_l2)) {
                      int out_data_idx = tree_data_idx(tree, bit_idx_l2, channels_out);
                      // int in_data_idx = out_data_idx / channels_out * channels_in;
                      // conv3x3x3_border_wbwd(bdl1*4+bdl2*2, bdl1*4+bdl2*2+2, bhl1*4+bhl2*2, bhl1*4+bhl2*2+2, bwl1*4+bwl2*2, bwl1*4+bwl2*2+2, 
                      //     ds, hs, ws, grad_out, weights, 0, channels_in, grad_in_data + in_data_idx);
                      float factor;
                      if(rdc_fcn == REDUCE_SUM) {
                        factor = scale;
                      }
                      else if(rdc_fcn == REDUCE_AVG) {
                        factor = scale / (2*2*2);
                      }
                      conv3x3x3_border_wbwd(bdl1*4+bdl2*2, bdl1*4+bdl2*2+2, bhl1*4+bhl2*2, bhl1*4+bhl2*2+2, bwl1*4+bwl2*2, bwl1*4+bwl2*2+2,
                          gn, ds, hs, ws, grid_in, grad_out_data + out_data_idx, channels_out, factor, grad_weights, grad_bias);
                    }
                    else {

                      int bit_idx_l3 = tree_child_bit_idx(bit_idx_l2);
                      for(int bdl3 = 0; bdl3 < 2; ++bdl3) {
                        for(int bhl3 = 0; bhl3 < 2; ++bhl3) {
                          for(int bwl3 = 0; bwl3 < 2; ++bwl3) {
                            int out_data_idx = tree_data_idx(tree, bit_idx_l3, channels_out);
                            conv3x3x3_point_wbwd(gn, ds+bdl1*4+bdl2*2+bdl3, hs+bhl1*4+bhl2*2+bhl3, ws+bwl1*4+bwl2*2+bwl3, 
                                grid_in, grad_out_data + out_data_idx, channels_out, scale, grad_weights);  
                            for(int co = 0; co < channels_out; ++co) {
                              ot_data_t val = scale * grad_out_data[out_data_idx + co];
                              #if defined(_OPENMP)
                              #pragma omp atomic
                              grad_bias[co] += val;
                              #else
                              grad_bias[co] += val;
                              #endif
                            }
                            bit_idx_l3++;
                          }
                        }
                      }

                    }
                    bit_idx_l2++;

                  }
                }
              } 

            } // else if isset L1
            bit_idx_l1++;

          } // for bwl1
        } // for bhl1
      } // for bdl1

    } // if isset L0
  } // for grid_idx
}





extern "C"
void octree_conv3x3x3_sum_cpu(const octree* grid_in_h, const ot_data_t* weights, const ot_data_t* bias, int channels_out, octree* grid) {
  octree_conv3x3x3_cpu<REDUCE_SUM>(grid_in_h, weights, bias, channels_out, grid);
}
extern "C"
void octree_conv3x3x3_sum_bwd_cpu(const ot_data_t* weights, const octree* grad_out, int channels_in, octree* grad_in) {
  octree_conv3x3x3_bwd_cpu<REDUCE_SUM>(weights, grad_out, channels_in, grad_in);
}
extern "C"
void octree_conv3x3x3_sum_wbwd_cpu(const octree* grid_in, const octree* grad_out, ot_data_t scale, ot_data_t* grad_weights, ot_data_t* grad_bias) {
  octree_conv3x3x3_wbwd_cpu<REDUCE_SUM>(grid_in, grad_out, scale, grad_weights, grad_bias); 
}

extern "C"
void octree_conv3x3x3_avg_cpu(const octree* grid_in_h, const ot_data_t* weights, const ot_data_t* bias, int channels_out, octree* grid) {
  octree_conv3x3x3_cpu<REDUCE_AVG>(grid_in_h, weights, bias, channels_out, grid);
}
extern "C"
void octree_conv3x3x3_avg_bwd_cpu(const ot_data_t* weights, const octree* grad_out, int channels_in, octree* grad_in) {
  octree_conv3x3x3_bwd_cpu<REDUCE_AVG>(weights, grad_out, channels_in, grad_in);
}
extern "C"
void octree_conv3x3x3_avg_wbwd_cpu(const octree* grid_in, const octree* grad_out, ot_data_t scale, ot_data_t* grad_weights, ot_data_t* grad_bias) {
  octree_conv3x3x3_wbwd_cpu<REDUCE_AVG>(grid_in, grad_out, scale, grad_weights, grad_bias); 
}
