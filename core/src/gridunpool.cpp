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

#include "octnet/cpu/unpool.h"
#include "octnet/cpu/cpu.h"

#include <cstdlib>

#if defined(_OPENMP)
#include <omp.h>
#endif

void octree_gridunpool2x2x2_do_cpu(const octree* in, octree* out) {
  const int out_n_blocks = octree_num_blocks(out);
  const int feature_size = in->feature_size;

  #pragma omp parallel for
  for(int out_grid_idx = 0; out_grid_idx < out_n_blocks; ++out_grid_idx) {
    int gn, ogd, ogh, ogw;
    octree_split_grid_idx(out, out_grid_idx, &gn, &ogd, &ogh, &ogw); 

    int igd = ogd / 2;
    int igh = ogh / 2;
    int igw = ogw / 2;
    int in_grid_idx = octree_grid_idx(in, gn, igd, igh, igw);

    // printf("  o %d,%d,%d, i %d,%d,%d\n", ogd,ogh,ogw, igd,igh,igw);
    
    ot_tree_t* otree = octree_get_tree(out, out_grid_idx);
    // ot_data_t* odata = out->data_ptrs[out_grid_idx];
    ot_data_t* odata = octree_get_data(out, out_grid_idx);
    ot_tree_t* itree = octree_get_tree(in, in_grid_idx);
    // ot_data_t* idata = in->data_ptrs[in_grid_idx];
    ot_data_t* idata = octree_get_data(in, in_grid_idx);
    
    int in_bit_idx_l0 = 1 + (ogd % 2) * 4 + (ogh % 2) * 2 + (ogw % 2);
    if(!tree_isset_bit(otree, 0)) {
      int odata_idx = 0;
      int in_bit_idx = tree_bit_idx_leaf(itree, in_bit_idx_l0);
      int idata_idx = tree_data_idx(itree, in_bit_idx, feature_size);
      // for(int f = 0; f < feature_size; ++f) { odata[odata_idx + f] = idata[idata_idx + f]; }
      octree_cpy_leaf(idata + idata_idx, feature_size, odata + odata_idx);
    }
    else {
      for(int dl1 = 0; dl1 < 2; ++dl1) {
        for(int hl1 = 0; hl1 < 2; ++hl1) {
          for(int wl1 = 0; wl1 < 2; ++wl1) {
            int out_bit_idx_l1 = 1 + dl1 * 4 + hl1 * 2 + wl1;
            int in_bit_idx_l1 = tree_child_bit_idx(in_bit_idx_l0) + dl1 * 4 + hl1 * 2 + wl1;
            if(!tree_isset_bit(otree, out_bit_idx_l1)) {
              int odata_idx = tree_data_idx(otree, out_bit_idx_l1, feature_size);
              int in_bit_idx = tree_bit_idx_leaf(itree, in_bit_idx_l1);
              int idata_idx = tree_data_idx(itree, in_bit_idx, feature_size);
              // for(int f = 0; f < feature_size; ++f) { odata[odata_idx + f] = idata[idata_idx + f]; }
              octree_cpy_leaf(idata + idata_idx, feature_size, odata + odata_idx);
            }
            else {
              for(int dl2 = 0; dl2 < 2; ++dl2) {
                for(int hl2 = 0; hl2 < 2; ++hl2) {
                  for(int wl2 = 0; wl2 < 2; ++wl2) {
                    int out_bit_idx_l2 = tree_child_bit_idx(out_bit_idx_l1) + dl2 * 4 + hl2 * 2 + wl2;
                    int in_bit_idx_l2 = tree_child_bit_idx(in_bit_idx_l1) + dl2 * 4 + hl2 * 2 + wl2;
                    // printf("%d,%d\n", out_bit_idx_l2, in_bit_idx_l2);
                    if(!tree_isset_bit(otree, out_bit_idx_l2)) {
                      int odata_idx = tree_data_idx(otree, out_bit_idx_l2, feature_size);
                      int in_bit_idx = tree_bit_idx_leaf(itree, in_bit_idx_l2);
                      int idata_idx = tree_data_idx(itree, in_bit_idx, feature_size);
                      // for(int f = 0; f < feature_size; ++f) { odata[odata_idx + f] = idata[idata_idx + f]; }
                      octree_cpy_leaf(idata + idata_idx, feature_size, odata + odata_idx);
                    }
                    else {
                      for(int bit_add = 0; bit_add < 8; ++bit_add) {
                        int out_bit_idx_l3 = tree_child_bit_idx(out_bit_idx_l2) + bit_add;
                        int in_bit_idx_l3 = in_bit_idx_l2 > 72 ? in_bit_idx_l2 : tree_child_bit_idx(in_bit_idx_l2);
                        // printf("  %d (%d), %d\n", out_bit_idx_l3, in_bit_idx_l2, in_bit_idx_l3);
                        int odata_idx = tree_data_idx(otree, out_bit_idx_l3, feature_size);
                        int in_bit_idx = tree_bit_idx_leaf(itree, in_bit_idx_l3);
                        int idata_idx = tree_data_idx(itree, in_bit_idx, feature_size);
                        // for(int f = 0; f < feature_size; ++f) { odata[odata_idx + f] = idata[idata_idx + f]; }
                        octree_cpy_leaf(idata + idata_idx, feature_size, odata + odata_idx);
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  } //for out_grid_idx
}



extern "C"
void octree_gridunpool2x2x2_cpu(const octree* in, octree* out) {
  out->n = in->n;
  out->grid_depth = in->grid_depth * 2;
  out->grid_height = in->grid_height * 2;
  out->grid_width = in->grid_width * 2;
  out->feature_size = in->feature_size;

  octree_resize_as_cpu(out, out);

  // update tree structure
  octree_clr_trees_cpu(out);
  int out_n_blocks = octree_num_blocks(out);
  #pragma omp parallel for
  for(int out_grid_idx = 0; out_grid_idx < out_n_blocks; ++out_grid_idx) {
    int gn, ogd, ogh, ogw;
    octree_split_grid_idx(out, out_grid_idx, &gn, &ogd, &ogh, &ogw); 

    int igd = ogd / 2;
    int igh = ogh / 2;
    int igw = ogw / 2;
    int in_grid_idx = octree_grid_idx(in, gn, igd, igh, igw);

    const ot_tree_t* in_tree = octree_get_tree(in, in_grid_idx);
    ot_tree_t* out_tree = octree_get_tree(out, out_grid_idx);

    int in_bit_idx = 1 + (ogd % 2) * 4 + (ogh % 2) * 2 + (ogw % 2);
    if(tree_isset_bit(in_tree, in_bit_idx)) {
      tree_set_bit(out_tree, 0);

      in_bit_idx = tree_child_bit_idx(in_bit_idx);
      for(int out_bit_idx = 1; out_bit_idx < 9; ++out_bit_idx) {
        if(tree_isset_bit(in_tree, in_bit_idx)) {
          tree_set_bit(out_tree, out_bit_idx);
        }
        in_bit_idx++;
      }
    }
  }

  octree_upd_n_leafs_cpu(out);
  octree_resize_as_cpu(out, out);
  octree_upd_prefix_leafs_cpu(out);

  octree_gridunpool2x2x2_do_cpu(in, out);
}




extern "C"
void octree_gridunpoolguided2x2x2_cpu(const octree* in, const octree* in_struct, octree* out) {
  if(in->grid_depth != in_struct->grid_depth / 2 || in->grid_height != in_struct->grid_height / 2 || in->grid_width != in_struct->grid_width / 2) {
    printf("[ERROR] octree_gridunpoolguided2x2x2_cpu in dim does not fit in_struct dim (%d,%d,%d), (%d,%d,%d)\n", in->grid_depth,in->grid_height,in->grid_width, in_struct->grid_depth,in_struct->grid_height,in_struct->grid_width);
    exit(-1);
  }

  octree_cpy_scalars(in_struct, out);
  octree_resize_as_cpu(in_struct, out);
  octree_cpy_trees_cpu_cpu(in_struct, out);
  octree_cpy_prefix_leafs_cpu_cpu(in_struct, out);

  octree_gridunpool2x2x2_do_cpu(in, out);
}







void octree_gridunpool2x2x2_do_bwd_cpu(const octree* in, const octree* grad_out, octree* grad_in) {
  octree_cpy_scalars(in, grad_in);
  octree_resize_as_cpu(in, grad_in);
  octree_cpy_trees_cpu_cpu(in, grad_in);
  octree_cpy_prefix_leafs_cpu_cpu(in, grad_in);

  const int out_n_blocks = octree_num_blocks(grad_out);
  const int feature_size = in->feature_size;

  octree_fill_data_cpu(grad_in, 0);

  #pragma omp parallel for
  for(int out_grid_idx = 0; out_grid_idx < out_n_blocks; ++out_grid_idx) {
    int gn, ogd, ogh, ogw;
    octree_split_grid_idx(grad_out, out_grid_idx, &gn, &ogd, &ogh, &ogw); 

    int igd = ogd / 2;
    int igh = ogh / 2;
    int igw = ogw / 2;
    int in_grid_idx = octree_grid_idx(in, gn, igd, igh, igw);

    // printf("  o %d,%d,%d, i %d,%d,%d\n", ogd,ogh,ogw, igd,igh,igw);
    
    ot_tree_t* otree = octree_get_tree(grad_out, out_grid_idx);
    // ot_data_t* odata = grad_out->data_ptrs[out_grid_idx];
    ot_data_t* odata = octree_get_data(grad_out, out_grid_idx);
    ot_tree_t* itree = octree_get_tree(grad_in, in_grid_idx);
    // ot_data_t* idata = grad_in->data_ptrs[in_grid_idx];
    ot_data_t* idata = octree_get_data(grad_in, in_grid_idx);
    
    int in_bit_idx_l0 = 1 + (ogd % 2) * 4 + (ogh % 2) * 2 + (ogw % 2);
    if(!tree_isset_bit(otree, 0)) {
      int odata_idx = 0;
      int in_bit_idx = tree_bit_idx_leaf(itree, in_bit_idx_l0);
      int idata_idx = tree_data_idx(itree, in_bit_idx, feature_size);
      for(int f = 0; f < feature_size; ++f) { 
        #pragma omp atomic
        idata[idata_idx + f] += odata[odata_idx + f]; 
      }
    }
    else {
      for(int dl1 = 0; dl1 < 2; ++dl1) {
        for(int hl1 = 0; hl1 < 2; ++hl1) {
          for(int wl1 = 0; wl1 < 2; ++wl1) {
            int out_bit_idx_l1 = 1 + dl1 * 4 + hl1 * 2 + wl1;
            int in_bit_idx_l1 = tree_child_bit_idx(in_bit_idx_l0) + dl1 * 4 + hl1 * 2 + wl1;
            if(!tree_isset_bit(otree, out_bit_idx_l1)) {
              int odata_idx = tree_data_idx(otree, out_bit_idx_l1, feature_size);
              int in_bit_idx = tree_bit_idx_leaf(itree, in_bit_idx_l1);
              int idata_idx = tree_data_idx(itree, in_bit_idx, feature_size);
              for(int f = 0; f < feature_size; ++f) { 
                // odata[odata_idx + f] = idata[idata_idx + f]; 
                #pragma omp atomic
                idata[idata_idx + f] += odata[odata_idx + f]; 
              }
            }
            else {
              for(int dl2 = 0; dl2 < 2; ++dl2) {
                for(int hl2 = 0; hl2 < 2; ++hl2) {
                  for(int wl2 = 0; wl2 < 2; ++wl2) {
                    int out_bit_idx_l2 = tree_child_bit_idx(out_bit_idx_l1) + dl2 * 4 + hl2 * 2 + wl2;
                    int in_bit_idx_l2 = tree_child_bit_idx(in_bit_idx_l1) + dl2 * 4 + hl2 * 2 + wl2;
                    // printf("%d,%d\n", out_bit_idx_l2, in_bit_idx_l2);
                    if(!tree_isset_bit(otree, out_bit_idx_l2)) {
                      int odata_idx = tree_data_idx(otree, out_bit_idx_l2, feature_size);
                      int in_bit_idx = tree_bit_idx_leaf(itree, in_bit_idx_l2);
                      int idata_idx = tree_data_idx(itree, in_bit_idx, feature_size);
                      for(int f = 0; f < feature_size; ++f) { 
                        // odata[odata_idx + f] = idata[idata_idx + f]; 
                        #pragma omp atomic
                        idata[idata_idx + f] += odata[odata_idx + f]; 
                      }
                    }
                    else {
                      for(int bit_add = 0; bit_add < 8; ++bit_add) {
                        int out_bit_idx_l3 = tree_child_bit_idx(out_bit_idx_l2) + bit_add;
                        // int in_bit_idx_l3 = tree_child_bit_idx(in_bit_idx_l2);
                        int in_bit_idx_l3 = in_bit_idx_l2 > 72 ? in_bit_idx_l2 : tree_child_bit_idx(in_bit_idx_l2);
                        // printf("  %d,%d\n", out_bit_idx_l3, in_bit_idx_l3);
                        int odata_idx = tree_data_idx(otree, out_bit_idx_l3, feature_size);
                        int in_bit_idx = tree_bit_idx_leaf(itree, in_bit_idx_l3);
                        int idata_idx = tree_data_idx(itree, in_bit_idx, feature_size);
                        for(int f = 0; f < feature_size; ++f) { 
                          // odata[odata_idx + f] = idata[idata_idx + f]; 
                          #pragma omp atomic
                          idata[idata_idx + f] += odata[odata_idx + f]; 
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }

  } //for out_grid_idx
}


extern "C"
void octree_gridunpool2x2x2_bwd_cpu(const octree* in, const octree* grad_out, octree* grad_in) {
  octree_gridunpool2x2x2_do_bwd_cpu(in, grad_out, grad_in);    
}

extern "C"
void octree_gridunpoolguided2x2x2_bwd_cpu(const octree* in, const octree* in_struct, const octree* grad_out, octree* grad_in) {
  octree_gridunpool2x2x2_do_bwd_cpu(in, grad_out, grad_in);    
}
