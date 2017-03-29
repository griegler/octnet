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

#include "octnet/cpu/pool.h"
#include "octnet/cpu/cpu.h"

#include <cstdlib>

#if defined(_OPENMP)
#include <omp.h>
#endif

template <int pool_fcn>
void octree_gridpool2x2x2_data_cpu(const octree* in, octree* out) {
  int n_blocks = octree_num_blocks(out);
  int feature_size = in->feature_size;

  #pragma omp parallel for
  for(int out_grid_idx = 0; out_grid_idx < n_blocks; ++out_grid_idx) {
    ot_tree_t* otree = octree_get_tree(out, out_grid_idx);
    // ot_data_t* odata = out->data_ptrs[out_grid_idx];
    ot_data_t* odata = octree_get_data(out, out_grid_idx);

    int gn, ogd, ogh, ogw;
    octree_split_grid_idx(out, out_grid_idx, &gn, &ogd, &ogh, &ogw); 
    
    int obit_idx_l1 = 1;
    for(int dgd = 0; dgd < 2; ++dgd) {
      for(int hgh = 0; hgh < 2; ++hgh) {
        for(int wgw = 0; wgw < 2; ++wgw) {

          int igd = 2*ogd + dgd;
          int igh = 2*ogh + hgh;
          int igw = 2*ogw + wgw;
          int in_grid_idx = octree_grid_idx(in, gn, igd, igh, igw);
          ot_tree_t* itree = octree_get_tree(in, in_grid_idx);
          // ot_data_t* idata = in->data_ptrs[in_grid_idx];
          ot_data_t* idata = octree_get_data(in, in_grid_idx);
          
          if(tree_isset_bit(itree, 0)) {
            int obit_idx_l2 = tree_child_bit_idx(obit_idx_l1);
            for(int ibit_idx_l1 = 1; ibit_idx_l1 < 9; ++ibit_idx_l1) {

              if(tree_isset_bit(itree, ibit_idx_l1)) {

                int obit_idx_l3 = tree_child_bit_idx(obit_idx_l2);
                for(int idx = 0; idx < 8; ++idx) {

                  int ibit_idx_l2 = tree_child_bit_idx(ibit_idx_l1) + idx;
                  int out_data_idx = tree_data_idx(otree, obit_idx_l3, feature_size);
                  if(tree_isset_bit(itree, ibit_idx_l2)) {
                    int in_data_idx = tree_data_idx(itree, tree_child_bit_idx(ibit_idx_l2), feature_size);
                    octree_pool2x2x2<pool_fcn>(idata + in_data_idx, feature_size, odata + out_data_idx);
                  }
                  else {
                    int in_data_idx = tree_data_idx(itree, ibit_idx_l2, feature_size);
                    // for(int f = 0; f < feature_size; ++f) {
                    //   odata[out_data_idx + f] = idata[in_data_idx + f];
                    // }
                    octree_cpy_leaf(idata + in_data_idx, feature_size, odata + out_data_idx);
                  }
                  obit_idx_l3++;

                }

              }
              else {
                int out_data_idx = tree_data_idx(otree, obit_idx_l2, feature_size);
                int in_data_idx = tree_data_idx(itree, ibit_idx_l1, feature_size);
                // for(int f = 0; f < feature_size; ++f) {
                //   odata[out_data_idx + f] = idata[in_data_idx + f];
                // }
                octree_cpy_leaf(idata + in_data_idx, feature_size, odata + out_data_idx);
              }
              obit_idx_l2++;

            }
          }
          else {
            int out_data_idx = tree_data_idx(otree, obit_idx_l1, feature_size);
            // for(int f = 0; f < feature_size; ++f) {
            //   odata[out_data_idx + f] = idata[f];
            // }
            octree_cpy_leaf(idata, feature_size, odata + out_data_idx);
          }
          obit_idx_l1++;

        }
      }
    }
  }
}



template <int pool_fcn>
void octree_gridpool2x2x2_cpu(const octree* in, octree* out) {
  if(in->grid_depth % 2 != 0 || in->grid_height % 2 != 0 || in->grid_width % 2 != 0) {
    printf("[ERROR] octree_gridpool2x2x2_cpu grid dimension should be a multiply of 2\n");
    exit(-1);
  }
  if(in->grid_depth / 2 == 0 || in->grid_height / 2 == 0 || in->grid_width / 2 == 0) {
    printf("[ERROR] octree_gridpool2x2x2_cpu grid dimension have to be at least 2x2x2\n");
    exit(-1);
  }

  //copy scalars
  out->n = in->n;
  out->grid_depth = in->grid_depth / 2;
  out->grid_height = in->grid_height / 2;
  out->grid_width = in->grid_width / 2;
  out->feature_size = in->feature_size;

  int n_blocks = octree_num_blocks(out);

  //compute out structure
  octree_resize_as_cpu(out, out);
  octree_clr_trees_cpu(out);

  #pragma omp parallel for
  for(int out_grid_idx = 0; out_grid_idx < n_blocks; ++out_grid_idx) {
    ot_tree_t* otree = octree_get_tree(out, out_grid_idx);

    int gn, ogd, ogh, ogw;
    octree_split_grid_idx(out, out_grid_idx, &gn, &ogd, &ogh, &ogw); 

    // first bit is always set, because out block consists of 8 in blocks
    tree_set_bit(otree, 0); 

    int obit_idx_l1 = 1;
    for(int dgd = 0; dgd < 2; ++dgd) {
      for(int hgh = 0; hgh < 2; ++hgh) {
        for(int wgw = 0; wgw < 2; ++wgw) {
          int igd = 2*ogd + dgd;
          int igh = 2*ogh + hgh;
          int igw = 2*ogw + wgw;
          int in_grid_idx = octree_grid_idx(in, gn, igd, igh, igw);
          ot_tree_t* itree = octree_get_tree(in, in_grid_idx);

          //check if first bit in in blocks is set
          if(tree_isset_bit(itree, 0)) {
            tree_set_bit(otree, obit_idx_l1);

            int obit_idx_l2 = tree_child_bit_idx(obit_idx_l1);
            for(int ibit_idx_l1 = 1; ibit_idx_l1 < 9; ++ibit_idx_l1) {
              //check if l1 bits are set in in blocks
              if(tree_isset_bit(itree, ibit_idx_l1)) {
                tree_set_bit(otree, obit_idx_l2);
              }
              obit_idx_l2++;
            }
          }
          obit_idx_l1++;
        }
      }
    }
  }


  //pool/copy data
  octree_upd_n_leafs_cpu(out);
  octree_resize_as_cpu(out, out);
  octree_upd_prefix_leafs_cpu(out);

  octree_gridpool2x2x2_data_cpu<pool_fcn>(in, out);
}


template <int pool_fcn>
void octree_gridpool2x2x2_bwd_cpu(const octree* in, const octree* grad_out, octree* grad_in) {
  octree_cpy_scalars(in, grad_in);
  octree_resize_as_cpu(in, grad_in);
  octree_cpy_trees_cpu_cpu(in, grad_in);
  octree_cpy_prefix_leafs_cpu_cpu(in, grad_in);
  
  int n_blocks = octree_num_blocks(grad_out);
  int feature_size = in->feature_size;

  #pragma omp parallel for
  for(int out_grid_idx = 0; out_grid_idx < n_blocks; ++out_grid_idx) {
    ot_tree_t* otree = octree_get_tree(grad_out, out_grid_idx);
    // ot_data_t* godata = grad_out->data_ptrs[out_grid_idx];
    ot_data_t* godata = octree_get_data(grad_out, out_grid_idx);

    int gn, ogd, ogh, ogw;
    octree_split_grid_idx(grad_out, out_grid_idx, &gn, &ogd, &ogh, &ogw); 
    
    int obit_idx_l1 = 1;
    for(int dgd = 0; dgd < 2; ++dgd) {
      for(int hgh = 0; hgh < 2; ++hgh) {
        for(int wgw = 0; wgw < 2; ++wgw) {

          int igd = 2*ogd + dgd;
          int igh = 2*ogh + hgh;
          int igw = 2*ogw + wgw;
          int in_grid_idx = octree_grid_idx(in, gn, igd, igh, igw);
          ot_tree_t* itree = octree_get_tree(in, in_grid_idx);
          // ot_data_t* gidata = grad_in->data_ptrs[in_grid_idx];
          ot_data_t* gidata = octree_get_data(grad_in, in_grid_idx);
          // ot_data_t* idata = in->data_ptrs[in_grid_idx];
          ot_data_t* idata = octree_get_data(in, in_grid_idx);
          
          if(tree_isset_bit(itree, 0)) {
            int obit_idx_l2 = tree_child_bit_idx(obit_idx_l1);
            for(int ibit_idx_l1 = 1; ibit_idx_l1 < 9; ++ibit_idx_l1) {

              if(tree_isset_bit(itree, ibit_idx_l1)) {

                int obit_idx_l3 = tree_child_bit_idx(obit_idx_l2);
                for(int idx = 0; idx < 8; ++idx) {

                  int ibit_idx_l2 = tree_child_bit_idx(ibit_idx_l1) + idx;
                  int out_data_idx = tree_data_idx(otree, obit_idx_l3, feature_size);
                  if(tree_isset_bit(itree, ibit_idx_l2)) {
                    int in_data_idx = tree_data_idx(itree, tree_child_bit_idx(ibit_idx_l2), feature_size);
                    octree_pool2x2x2_bwd<pool_fcn>(idata + in_data_idx, godata + out_data_idx, feature_size, gidata + in_data_idx);
                  }
                  else {
                    int in_data_idx = tree_data_idx(itree, ibit_idx_l2, feature_size);
                    // for(int f = 0; f < feature_size; ++f) {
                    //   gidata[in_data_idx + f] = godata[out_data_idx + f];
                    // }
                    octree_cpy_leaf(godata + out_data_idx, feature_size, gidata + in_data_idx);
                  }
                  obit_idx_l3++;

                }

              }
              else {
                int out_data_idx = tree_data_idx(otree, obit_idx_l2, feature_size);
                int in_data_idx = tree_data_idx(itree, ibit_idx_l1, feature_size);
                // for(int f = 0; f < feature_size; ++f) {
                //   gidata[in_data_idx + f] = godata[out_data_idx + f];
                // }
                octree_cpy_leaf(godata + out_data_idx, feature_size, gidata + in_data_idx);
              }
              obit_idx_l2++;

            }
          }
          else {
            int out_data_idx = tree_data_idx(otree, obit_idx_l1, feature_size);
            // for(int f = 0; f < feature_size; ++f) {
            //   gidata[f] = godata[out_data_idx + f];
            // }
            octree_cpy_leaf(godata + out_data_idx, feature_size, gidata);
          }
          obit_idx_l1++;

        }
      }
    }
  }

}


void octree_gridpool2x2x2_avg_cpu(const octree* in, octree* out) {
  octree_gridpool2x2x2_cpu<REDUCE_AVG>(in, out);
}
void octree_gridpool2x2x2_max_cpu(const octree* in, octree* out){
  octree_gridpool2x2x2_cpu<REDUCE_MAX>(in, out);
}
void octree_gridpool2x2x2_sum_cpu(const octree* in, octree* out) {
  octree_gridpool2x2x2_cpu<REDUCE_SUM>(in, out);
}


void octree_gridpool2x2x2_avg_bwd_cpu(const octree* in, const octree* grad_out, octree* grad_in) {
  octree_gridpool2x2x2_bwd_cpu<REDUCE_AVG>(in, grad_out, grad_in);
}
void octree_gridpool2x2x2_max_bwd_cpu(const octree* in, const octree* grad_out, octree* grad_in) {
  octree_gridpool2x2x2_bwd_cpu<REDUCE_AVG>(in, grad_out, grad_in);
}
