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

#include "octnet/create/create.h"
#include "octnet/cpu/cpu.h"
#include "octnet/cpu/io.h"
#include <cstring>

#if defined(_OPENMP)
#include <omp.h>
#endif

octree* OctreeCreateCpu::alloc_grid() {
  octree* grid = octree_new_cpu();
  octree_resize_cpu(1, grid_depth, grid_height, grid_width, feature_size, 0, grid);
  octree_clr_trees_cpu(grid);
  return grid;
}


void OctreeCreateCpu::create_octree_structure(octree* grid, OctreeCreateHelperCpu* helper) {
  int n_blocks = octree_num_blocks(grid);

  #pragma omp parallel for
  for(int grid_idx = 0; grid_idx < n_blocks; ++grid_idx) {
    ot_tree_t* tree = octree_get_tree(grid, grid_idx);

    int gd = grid_idx / (grid_height * grid_width);
    int gh = (grid_idx / grid_width) % grid_height;
    int gw = grid_idx % grid_width;
    
    float cx = gw * 8 + 4;
    float cy = gh * 8 + 4;
    float cz = gd * 8 + 4;
    if(is_occupied(cx,cy,cz, 8,8,8, gd,gh,gw, helper)) {
      tree_set_bit(tree, 0);

      int bit_idx_l1 = 1;
      for(int dl1 = 0; dl1 < 2; ++dl1) {
        for(int hl1 = 0; hl1 < 2; ++hl1) {
          for(int wl1 = 0; wl1 < 2; ++wl1) {
            float cx_l1 = cx + (wl1 * 4) - 2;
            float cy_l1 = cy + (hl1 * 4) - 2;
            float cz_l1 = cz + (dl1 * 4) - 2;

            if(is_occupied(cx_l1,cy_l1,cz_l1, 4,4,4, gd,gh,gw, helper)) {
              tree_set_bit(tree, bit_idx_l1);

              int bit_idx_l2 = tree_child_bit_idx(bit_idx_l1);
              for(int dl2 = 0; dl2 < 2; ++dl2) {
                for(int hl2 = 0; hl2 < 2; ++hl2) {
                  for(int wl2 = 0; wl2 < 2; ++wl2) {
                    float cx_l2 = cx_l1 + (wl2 * 2) - 1;
                    float cy_l2 = cy_l1 + (hl2 * 2) - 1;
                    float cz_l2 = cz_l1 + (dl2 * 2) - 1;

                    if(is_occupied(cx_l2,cy_l2,cz_l2, 2,2,2, gd,gh,gw, helper)) {
                      tree_set_bit(tree, bit_idx_l2);
                    }
                    bit_idx_l2++;
                  }
                }
              }
            }

            bit_idx_l1++;
          }
        }
      }

    }
  }
}


void OctreeCreateCpu::fit_octree(octree* grid, int fit_multiply, OctreeCreateHelperCpu* helper) {
  if(grid->grid_depth % fit_multiply != 0 || grid->grid_height % fit_multiply != 0 || grid->grid_width % fit_multiply != 0) {
    printf("original grid dimensions are not a multiply of fit_multiply\n");
    exit(-1);
  }

  int min[] = {grid->grid_depth, grid->grid_height, grid->grid_width};
  int max[] = {0,0,0};
  
  for(int d = 0; d < grid->grid_depth; ++d) {
    for(int h = 0; h < grid->grid_height; ++h) {
      for(int w = 0; w < grid->grid_width; ++w) {
        const int grid_idx = octree_grid_idx(grid, 1,d,h,w);
        const ot_tree_t* tree = octree_get_tree(grid, grid_idx);
        if(tree_isset_bit(tree, 0)) {
          min[0] = IMIN(min[0], d);
          min[1] = IMIN(min[1], h);
          min[2] = IMIN(min[2], w);

          max[0] = IMAX(max[0], d);
          max[1] = IMAX(max[1], h);
          max[2] = IMAX(max[2], w);
        }
      }
    }
  }

  printf("    min: %d,%d,%d, max: %d,%d,%d\n", min[0],min[1],min[2], max[0],max[1],max[2]);
  int grid_depth  = max[0] - min[0] + 1;
  int grid_height = max[1] - min[1] + 1;
  int grid_width  = max[2] - min[2] + 1;

  //adjust to fit_multiply
  if(grid_depth % fit_multiply != 0) {
    int target_grid_depth = fit_multiply * (grid_depth / fit_multiply + 1);
    int add_min = (target_grid_depth - grid_depth) / 2;
    int add_max = target_grid_depth - grid_depth - add_min;
    min[0] -= add_min;
    max[0] += add_max;
    if(min[0] < 0) {
      int add = -min[0];
      min[0] += add;
      max[0] += add;
    }
    if(max[0] >= target_grid_depth) {
      int add = max[0] - target_grid_depth + 1;
      min[0] -= add;
      max[0] -= add;
    }
  }
  if(grid_height % fit_multiply != 0) {
    int target_grid_height = fit_multiply * (grid_height / fit_multiply + 1);
    int add_min = (target_grid_height - grid_height) / 2;
    int add_max = target_grid_height - grid_height - add_min;
    min[1] -= add_min;
    max[1] += add_max;
    if(min[1] < 0) {
      int add = -min[1];
      min[1] += add;
      max[1] += add;
    }
    if(max[1] >= target_grid_height) {
      int add = max[1] - target_grid_height + 1;
      min[1] -= add;
      max[1] -= add;
    }
  }
  if(grid_width % fit_multiply != 0) {
    int target_grid_width = fit_multiply * (grid_width / fit_multiply + 1);
    int add_min = (target_grid_width - grid_width) / 2;
    int add_max = target_grid_width - grid_width - add_min;
    min[2] -= add_min;
    max[2] += add_max;
    if(min[2] < 0) {
      int add = -min[2];
      min[2] += add;
      max[2] += add;
    }
    if(max[2] >= target_grid_width) {
      int add = max[2] - target_grid_width + 1;
      min[2] -= add;
      max[2] -= add;
    }
  }
  
  printf("    min: %d,%d,%d, max: %d,%d,%d\n", min[0],min[1],min[2], max[0],max[1],max[2]);
  grid_depth  = max[0] - min[0] + 1;
  grid_height = max[1] - min[1] + 1;
  grid_width  = max[2] - min[2] + 1;
  // int n_blocks = grid_depth * grid_height * grid_width;
  // ot_tree_t* trees = new ot_tree_t[n_blocks * N_TREE_INTS];
  // memset(trees, 0, n_blocks * N_TREE_INTS * sizeof(ot_tree_t));

  for(int d = 0; d < grid_depth; ++d) {
    for(int h = 0; h < grid_height; ++h) {
      for(int w = 0; w < grid_width; ++w) {
        int grid_idx = (d * grid_height + h) * grid_width + w;
        ot_tree_t* tree = grid->trees + grid_idx * N_TREE_INTS;
        // ot_tree_t* tree = trees + grid_idx * N_TREE_INTS;

        int old_d = d + min[0];
        int old_h = h + min[1];
        int old_w = w + min[2];
        int old_grid_idx = (old_d * grid->grid_height + old_h) * grid->grid_width + old_w;
        const ot_tree_t* old_tree = grid->trees + old_grid_idx * N_TREE_INTS;
        // printf("  %d (%d) <= %d (%d) \n", grid_idx, grid_depth*grid_height*grid_width, old_grid_idx, octree_num_blocks(grid));

        for(int tidx = 0; tidx < N_TREE_INTS; ++tidx) {
          tree[tidx] = old_tree[tidx];
        }
      }
    }
  }

  helper->update_offsets(min[0], min[1], min[2]);

  grid->grid_depth = grid_depth;
  grid->grid_height = grid_height;
  grid->grid_width = grid_width;
  // delete[] grid->trees;
  // grid->trees = trees;
}


void OctreeCreateCpu::pack_octree(octree* grid, OctreeCreateHelperCpu* helper) {
  int n_blocks = octree_num_blocks(grid);
  
  #pragma omp parallel for
  for(int grid_idx = 0; grid_idx < n_blocks; ++grid_idx) {
    ot_tree_t* tree = octree_get_tree(grid, grid_idx);

    int gd = grid_idx / (grid->grid_height * grid->grid_width);
    int gh = (grid_idx / grid->grid_width) % grid->grid_height;
    int gw = grid_idx % grid->grid_width;

    helper->update_grid_coords(gd,gh,gw);
    
    float cx = gw * 8 + 4;
    float cy = gh * 8 + 4;
    float cz = gd * 8 + 4;

    //check l3
    int bit_idx_l2 = 9;
    for(int dl1 = 0; dl1 < 2; ++dl1) {
      for(int hl1 = 0; hl1 < 2; ++hl1) {
        for(int wl1 = 0; wl1 < 2; ++wl1) {

          for(int dl2 = 0; dl2 < 2; ++dl2) {
            for(int hl2 = 0; hl2 < 2; ++hl2) {
              for(int wl2 = 0; wl2 < 2; ++wl2) {
                
                if(tree_isset_bit(tree, bit_idx_l2)) {
                  // printf("    [pack] tree_bit at %d = %d\n", bit_idx_l2, tree_isset_bit(tree, bit_idx_l2));
                  //check if all leafs are occupied
                  bool all_oc = true;
                  for(int dl3 = 0; dl3 < 2 && all_oc; ++dl3) {
                    for(int hl3 = 0; hl3 < 2 && all_oc; ++hl3) {
                      for(int wl3 = 0; wl3 < 2 && all_oc; ++wl3) {
                        float cx_l3 = cx + (4*wl1 - 2) + (2*wl2 - 1) + (wl3 * 1) - 0.5;
                        float cy_l3 = cy + (4*hl1 - 2) + (2*hl2 - 1) + (hl3 * 1) - 0.5;
                        float cz_l3 = cz + (4*dl1 - 2) + (2*dl2 - 1) + (dl3 * 1) - 0.5;
                        // printf("    [pack] CHECK %f,%f,%f, %d,%d,%d, %d,%d,%d\n", cx,cy,cz, wl2,hl2,dl2, wl3,hl3,dl3); 
                        // printf("    [pack] CHECK %f,%f,%f => %d \n", cx_l3, cy_l3, cz_l3, is_occupied(cx_l3, cy_l3, cz_l3, 1,1,1));
                        
                        all_oc = all_oc && is_occupied(cx_l3, cy_l3, cz_l3, 1,1,1, gd,gh,gw, helper);
                      }
                    }
                  }
                  if(all_oc) {
                    // printf("    [pack] unset tree at %d\n", bit_idx_l2);
                    // printf("      %s\n", tree_bit_str(tree).c_str());
                    tree_unset_bit(tree, bit_idx_l2);
                    // printf("      %s\n", tree_bit_str(tree).c_str());
                  }
                }
                bit_idx_l2++;

              }
            }
          }

        }
      }
    }

    //check l2
    // printf("---------------------------------------------\n");
    int bit_idx_l1 = 1;
    for(int dl1 = 0; dl1 < 2; ++dl1) {
      for(int hl1 = 0; hl1 < 2; ++hl1) {
        for(int wl1 = 0; wl1 < 2; ++wl1) {

          if(tree_isset_bit(tree, bit_idx_l1)) {
            bool all_oc = true;
            int bit_idx_l2 = tree_child_bit_idx(bit_idx_l1);
            for(int dl2 = 0; dl2 < 2 && all_oc; ++dl2) {
              for(int hl2 = 0; hl2 < 2 && all_oc; ++hl2) {
                for(int wl2 = 0; wl2 < 2 && all_oc; ++wl2) {
                  float cx_l2 = cx + (4*wl1 - 2) + (2*wl2 - 1);
                  float cy_l2 = cy + (4*hl1 - 2) + (2*hl2 - 1);
                  float cz_l2 = cz + (4*dl1 - 2) + (2*dl2 - 1);
                  
                  all_oc = all_oc && !tree_isset_bit(tree, bit_idx_l2) && is_occupied(cx_l2, cy_l2, cz_l2, 2,2,2, gd,gh,gw, helper);
                  bit_idx_l2++;
                }
              }
            }
      
            if(all_oc) {
              // printf("    [pack] unset tree at %d\n", bit_idx_l1);
              // printf("      %s\n", tree_bit_str(tree).c_str());
              tree_unset_bit(tree, bit_idx_l1);
              // printf("      %s\n", tree_bit_str(tree).c_str());
            }
          }
          bit_idx_l1++;

        }
      }
    }

    //check l1
    // printf("---------------------------------------------\n");
    if(tree_isset_bit(tree, 0)) {
     bool all_oc = true;
     int bit_idx_l1 = 1;
      for(int dl1 = 0; dl1 < 2 && all_oc; ++dl1) {
        for(int hl1 = 0; hl1 < 2 && all_oc; ++hl1) {
          for(int wl1 = 0; wl1 < 2 && all_oc; ++wl1) {
            float cx_l2 = cx + (4*wl1 - 2);
            float cy_l2 = cy + (4*hl1 - 2);
            float cz_l2 = cz + (4*dl1 - 2);
            
            all_oc = all_oc && !tree_isset_bit(tree, bit_idx_l1) && is_occupied(cx_l2, cy_l2, cz_l2, 2,2,2, gd,gh,gw, helper);
            bit_idx_l1++;
          }
        }
      }

      if(all_oc) {
        // printf("    [pack] unset tree at %d\n", 0);
        // printf("      %s\n", tree_bit_str(tree).c_str());
        tree_unset_bit(tree, 0);
        // printf("      %s\n", tree_bit_str(tree).c_str());
      } 
    }

  } // for grid_idx

}


void OctreeCreateCpu::update_and_resize_octree(octree* grid) {
  octree_upd_n_leafs_cpu(grid);
  octree_resize_as_cpu(grid, grid);
  octree_upd_prefix_leafs_cpu(grid);
}

void OctreeCreateCpu::fill_octree_data(octree* grid, bool packed, OctreeCreateHelperCpu* helper) {
  int n_blocks = octree_num_blocks(grid);

  #pragma omp parallel for
  for(int grid_idx = 0; grid_idx < n_blocks; ++grid_idx) {
    // printf("    [OctreeCreateCpu] fill grid block %d\n", grid_idx);
    ot_tree_t* tree = octree_get_tree(grid, grid_idx);
    // ot_data_t* data = grid->data_ptrs[grid_idx];
    ot_data_t* data = octree_get_data(grid, grid_idx);

    int gd = grid_idx / (grid->grid_height * grid->grid_width);
    int gh = (grid_idx / grid->grid_width) % grid->grid_height;
    int gw = grid_idx % grid->grid_width;
    
    helper->update_grid_coords(gd,gh,gw);
    // if(grid_idx == 1030) printf("      gd,gh,gw, %d,%d,%d\n", gd,gh,gw);
    
    float cx = gw * 8 + 4;
    float cy = gh * 8 + 4;
    float cz = gd * 8 + 4;
    // if(grid_idx == 1030) printf("      cx,cy,cz, %f,%f,%f\n", cx,cy,cz);
    if(tree_isset_bit(tree, 0)) {

      int bit_idx_l1 = 1;
      for(int dl1 = 0; dl1 < 2; ++dl1) {
        for(int hl1 = 0; hl1 < 2; ++hl1) {
          for(int wl1 = 0; wl1 < 2; ++wl1) {
            float cx_l1 = cx + (wl1 * 4) - 2;
            float cy_l1 = cy + (hl1 * 4) - 2;
            float cz_l1 = cz + (dl1 * 4) - 2;
            // if(grid_idx == 1030) printf("        cz_l1 %f = %f + (%d * 4) - 2\n", cz_l1, cz, dl1);

            if(tree_isset_bit(tree, bit_idx_l1)) {

              int bit_idx_l2 = tree_child_bit_idx(bit_idx_l1);
              for(int dl2 = 0; dl2 < 2; ++dl2) {
                for(int hl2 = 0; hl2 < 2; ++hl2) {
                  for(int wl2 = 0; wl2 < 2; ++wl2) {
                    float cx_l2 = cx_l1 + (wl2 * 2) - 1;
                    float cy_l2 = cy_l1 + (hl2 * 2) - 1;
                    float cz_l2 = cz_l1 + (dl2 * 2) - 1;

                    if(tree_isset_bit(tree, bit_idx_l2)) {
  
                      int bit_idx_l3 = tree_child_bit_idx(bit_idx_l2);
                      for(int dl3 = 0; dl3 < 2; ++dl3) {
                        for(int hl3 = 0; hl3 < 2; ++hl3) {
                          for(int wl3 = 0; wl3 < 2; ++wl3) {
                            float cx_l3 = cx_l2 + (wl3 * 1) - 0.5;
                            float cy_l3 = cy_l2 + (hl3 * 1) - 0.5;
                            float cz_l3 = cz_l2 + (dl3 * 1) - 0.5;

                            int data_idx = tree_data_idx(tree, bit_idx_l3, feature_size);
                            bool oc = is_occupied(cx_l3,cy_l3,cz_l3, 1,1,1, gd,gh,gw, helper);

                            // if(grid_idx == 1030) printf("      get_data l3 %f %f %f\n", cx_l3,cy_l3,cz_l3);
                            get_data(oc, cx_l3,cy_l3,cz_l3, 1,1,1, gd,gh,gw, helper, data + data_idx);
                            bit_idx_l3++;
                          }
                        }
                      }

                    }
                    else {
                      int data_idx = tree_data_idx(tree, bit_idx_l2, feature_size);
                      bool oc = packed && is_occupied(cx_l2,cy_l2,cz_l2, 2,2,2, gd,gh,gw, helper);
                      // if(grid_idx == 1030) printf("      get_data l2 %f %f %f\n", cx_l2,cy_l2,cz_l2);
                      get_data(oc, cx_l2,cy_l2,cz_l2, 2,2,2, gd,gh,gw, helper, data + data_idx);
                    }
                    bit_idx_l2++;
                  }
                }
              }
            }
            else {
              int data_idx = tree_data_idx(tree, bit_idx_l1, feature_size);
              bool oc = packed && is_occupied(cx_l1,cy_l1,cz_l1, 4,4,4, gd,gh,gw, helper);
              // if(grid_idx == 1030) printf("      get_data l1 %f %f %f\n", cx_l1,cy_l1,cz_l1);
              get_data(oc, cx_l1,cy_l1,cz_l1, 4,4,4, gd,gh,gw, helper, data + data_idx);
            }

            bit_idx_l1++;
          }
        }
      }

    }
    else {
      bool oc = packed && is_occupied(cx,cy,cz, 8,8,8, gd,gh,gw, helper);
      // if(grid_idx == 1030) printf("      get_data l0 %f %f %f\n", cx,cy,cz);
      get_data(oc, cx,cy,cz, 8,8,8, gd,gh,gw, helper, data);
    }

  }
}





octree* OctreeCreateCpu::create_octree(bool fit, int fit_multiply, bool pack, int n_threads, OctreeCreateHelperCpu* helper) {
  octree* grid = alloc_grid();

  // printf("test feature access\n");
  // float* data = new float[feature_size];
  // get_data(true, 0,0,0, 8,8,8, 0,0,0, helper, data);
  // delete[] data;
  // printf("DONE test feature access\n");

  //create octree structure
  //printf("  [OctreeCreateCpu] create octree structure\n");
#if defined(_OPENMP)
  omp_set_num_threads(n_threads);
#endif
  create_octree_structure(grid, helper);
  
  // printf("test feature access\n");
  // data = new float[feature_size];
  // get_data(true, 0,0,0, 8,8,8, 0,0,0, helper, data);
  // delete[] data;
  // printf("DONE test feature access\n");

  //fit if needed
  if(fit) {
    //printf("  [OctreeCreateCpu] fit octree structure\n");
    fit_octree(grid, fit_multiply, helper);
  }

  //pack if needed
  if(pack) {
    //printf("  [OctreeCreateCpu] pack octree structure\n");
    pack_octree(grid, helper);
  }

  //update leafs, data array, data ptrs basd on structure
  //printf("  [OctreeCreateCpu] update octree data structure\n");
  update_and_resize_octree(grid);

  // printf("test feature access\n");
  // data = new float[feature_size];
  // get_data(true, 0,0,0, 8,8,8, 0,0,0, helper, data);
  // delete[] data;
  // printf("DONE test feature access\n");
  
  // octree_write_cpu("tmp.oc", grid);
  // octree* grid = octree_new_cpu();
  // octree_read_cpu("tmp.oc");


  //read data
  //printf("  [OctreeCreateCpu] fill data\n");
#if defined(_OPENMP)
  omp_set_num_threads(n_threads);
#endif
  fill_octree_data(grid, pack, helper);
  //printf("  [OctreeCreateCpu] done\n");

  return grid;
}


octree* OctreeCreateCpu::operator()(bool fit, int fit_multiply, bool pack, int n_threads) {
  OctreeCreateHelperCpu helper(grid_depth, grid_height, grid_width);
  return create_octree(fit, fit_multiply, pack, n_threads, &helper);
}







