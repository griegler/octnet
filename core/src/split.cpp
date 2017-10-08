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

#include "octnet/cpu/split.h"
#include "octnet/cpu/cpu.h"
#include "octnet/core/z_curve.h"

#include <iostream>
#include <cstdio>
#include <cstdlib>

extern "C"
void octree_split_by_prob_cpu(const octree* in, const octree* prob, const ot_data_t thr, bool check, octree* out) {
  if(prob->feature_size != 1) {
    printf("[ERROR] split_by_prob - prob feature size != 1\n");
    exit(-1);
  }
  
  if(check && !octree_equal_trees_cpu(in, prob)) {
    printf("[ERROR] split_by_prob - tree structure of inputs do not match\n");
    exit(-1);
  }

  //struct
  octree_cpy_scalars(in, out);
  octree_resize_as_cpu(in, out);
  octree_clr_trees_cpu(out);

  int n_blocks = octree_num_blocks(in);

  #pragma omp parallel for
  for(int grid_idx = 0; grid_idx < n_blocks; ++grid_idx) {
    const ot_tree_t* itree = octree_get_tree(in, grid_idx);
    ot_tree_t* otree = octree_get_tree(out, grid_idx);

    // const ot_data_t* prob_data = prob->data_ptrs[grid_idx];
    const ot_data_t* prob_data = octree_get_data(prob, grid_idx); 

    if(!tree_isset_bit(itree, 0)) {
      int data_idx = tree_data_idx(itree, 0, 1);
      if(prob_data[data_idx] >= thr) {
        tree_set_bit(otree, 0);
      }
    }
    else {

      tree_set_bit(otree, 0);
      for(int bit_idx_l1 = 1; bit_idx_l1 < 9; ++bit_idx_l1) {
        if(!tree_isset_bit(itree, bit_idx_l1)) {
          int data_idx = tree_data_idx(itree, bit_idx_l1, 1);
          if(prob_data[data_idx] >= thr) {
            tree_set_bit(otree, bit_idx_l1);
          }
        }
        else {

          tree_set_bit(otree, bit_idx_l1);
          for(int add_bit_idx_l2 = 0; add_bit_idx_l2 < 8; ++add_bit_idx_l2) {
            int bit_idx_l2 = tree_child_bit_idx(bit_idx_l1) + add_bit_idx_l2;
            if(!tree_isset_bit(itree, bit_idx_l2)) {
              int data_idx = tree_data_idx(itree, bit_idx_l2, 1);
              if(prob_data[data_idx] >= thr) {
                tree_set_bit(otree, bit_idx_l2);
              }
            }
            else {
              tree_set_bit(otree, bit_idx_l2);
            }
          }

        }
      }

    }
  }

  octree_upd_n_leafs_cpu(out);
  octree_resize_as_cpu(out, out);
  octree_upd_prefix_leafs_cpu(out);

  octree_cpy_sup_to_sub_cpu(in, out);
}

extern "C"
void octree_split_full_cpu(const octree* in, octree* out) {
  octree_resize_as_cpu(in, out);
  int n_blocks = octree_num_blocks(in);

  #pragma omp parallel for
  for(int grid_idx = 0; grid_idx < n_blocks; ++grid_idx) {
    ot_tree_t* tree = octree_get_tree(out, grid_idx);
    for(int tree_idx = 0; tree_idx < N_TREE_INTS; ++tree_idx) {
      tree[tree_idx] = ~0;
    }
  }

  octree_upd_n_leafs_cpu(out);
  octree_resize_as_cpu(out, out);
  octree_upd_prefix_leafs_cpu(out);

  octree_cpy_sup_to_sub_cpu(in, out);
}


extern "C"
void octree_split_reconstruction_surface_cpu(const octree* in, const octree* rec, ot_data_t rec_thr_from, ot_data_t rec_thr_to, octree* out) {
  if(rec->feature_size != 1) {
    printf("[ERROR] split_reconstruction_surface - feature size of rec has to be 1\n");
    exit(-1);
  }
  if(in->n != rec->n || in->grid_depth/2 != rec->grid_depth || in->grid_height/2 != rec->grid_height || in->grid_width/2 != rec->grid_width) {
    printf("[ERROR] split_reconstruction_surface - shape of in and rec are not compatible\n");
    exit(-1);
  }
  
  octree_resize_as_cpu(in, out);
  octree_cpy_trees_cpu_cpu(in, out);

  // determine tree structure
  for(int leaf_idx = 0; leaf_idx < in->n_leafs; ++leaf_idx) {
    int in_grid_idx = leaf_idx_to_grid_idx(in, leaf_idx);
    const ot_tree_t* in_tree = octree_get_tree(in, in_grid_idx);

    int in_data_idx = leaf_idx - in->prefix_leafs[in_grid_idx];
    int in_bit_idx = data_idx_to_bit_idx(in_tree, in_data_idx);

    //dense ind of input 
    int n, in_d,in_h,in_w;
    int depth = octree_ind_to_dense_ind(in, in_grid_idx, in_bit_idx, &n, &in_d,&in_h,&in_w);
    if(depth == 3) {
      continue;
    }

    //get ind of rec (halve resolution)
    int ds = in_d/2;
    int hs = in_h/2;
    int ws = in_w/2;
    int rec_gd = ds / 8;
    int rec_gh = hs / 8;
    int rec_gw = ws / 8;
    int rec_bd = ds % 8;
    int rec_bh = hs % 8;
    int rec_bw = ws % 8;
    int rec_grid_idx = octree_grid_idx(rec, n, rec_gd,rec_gh,rec_gw);
    const ot_tree_t* rec_tree = octree_get_tree(rec, rec_grid_idx);
    int rec_bit_idx = tree_bit_idx(rec_tree, rec_bd,rec_bh,rec_bw);

    //determine leaf state
    int data_idx = tree_data_idx(rec_tree, rec_bit_idx, rec->feature_size);
    ot_data_t prob = octree_get_data(rec, rec_grid_idx)[data_idx];
    bool leaf_state = (prob >= rec_thr_from && prob <= rec_thr_to);
    bool other_state = leaf_state;

    //check along faces if a different state exists
    int width = width_from_depth(depth_from_bit_idx(rec_bit_idx));

    // along d
    int grid_idx, bit_idx;
    for(int fd = 0; fd < 2; ++fd) {
      int d = ds + (fd*(width+1)-1); 
      int h = hs; 
      int w = ws;
      if(leaf_state == other_state && d >= 0 && h >= 0 && w >= 0 && d < 8 * rec->grid_depth && h < 8 * rec->grid_height && w < 8 * rec->grid_width) {
        grid_idx = octree_grid_idx(rec, n, d / 8, h / 8, w / 8);
        const ot_tree_t* tree = octree_get_tree(rec, grid_idx);
        ot_data_t* data = octree_get_data(rec, grid_idx);
        int z = 0;
        while(leaf_state == other_state && z < width * width) {
          int e1 = z_curve_x(z);
          int e2 = z_curve_y(z);
          h = hs + e1;
          w = ws + e2;

          bit_idx = tree_bit_idx(tree, d % 8, h % 8, w % 8);
          data_idx = tree_data_idx(tree, bit_idx, rec->feature_size);
          prob = data[data_idx];
          other_state = (prob >= rec_thr_from && prob <= rec_thr_to);

          int data_cnt = bit_idx == 0 ? 8 : (bit_idx < 9 ? 4 : (bit_idx < 73 ? 2 : 1));
          data_cnt = IMIN(width * width - z, data_cnt * data_cnt);
          z += data_cnt;
        }
      }
    }

    // along h
    for(int fh = 0; fh < 2; ++fh) {
      int h = hs + (fh*(width+1)-1); 
      int d = ds; 
      int w = ws;
      if(leaf_state == other_state && d >= 0 && h >= 0 && w >= 0 && d < 8 * rec->grid_depth && h < 8 * rec->grid_height && w < 8 * rec->grid_width) {
        grid_idx = octree_grid_idx(rec, n, d / 8, h / 8, w / 8);
        const ot_tree_t* tree = octree_get_tree(rec, grid_idx);
        ot_data_t* data = octree_get_data(rec, grid_idx);
        int z = 0;
        while(leaf_state == other_state && z < width * width) {
          int e1 = z_curve_x(z);
          int e2 = z_curve_y(z);
          d = ds + e1;
          w = ws + e2;

          bit_idx = tree_bit_idx(tree, d % 8, h % 8, w % 8);
          data_idx = tree_data_idx(tree, bit_idx, rec->feature_size);
          prob = data[data_idx];
          other_state = (prob >= rec_thr_from && prob <= rec_thr_to);

          int data_cnt = bit_idx == 0 ? 8 : (bit_idx < 9 ? 4 : (bit_idx < 73 ? 2 : 1));
          data_cnt = IMIN(width * width - z, data_cnt * data_cnt);
          z += data_cnt;
        }
      }
    }

    // along w
    for(int fw = 0; fw < 2; ++fw) {
      int w = ws + (fw*(width+1)-1); 
      int d = ds;
      int h = hs; 
      if(leaf_state == other_state && d >= 0 && h >= 0 && w >= 0 && d < 8 * rec->grid_depth && h < 8 * rec->grid_height && w < 8 * rec->grid_width) {
        grid_idx = octree_grid_idx(rec, n, d / 8, h / 8, w / 8);
        const ot_tree_t* tree = octree_get_tree(rec, grid_idx);
        ot_data_t* data = octree_get_data(rec, grid_idx);
        int z = 0;
        while(leaf_state == other_state && z < width * width) {
          int e1 = z_curve_x(z);
          int e2 = z_curve_y(z);
          d = ds + e2;
          h = hs + e1;

          bit_idx = tree_bit_idx(tree, d % 8, h % 8, w % 8);
          data_idx = tree_data_idx(tree, bit_idx, rec->feature_size);
          prob = data[data_idx];
          other_state = (prob >= rec_thr_from && prob <= rec_thr_to);

          int data_cnt = bit_idx == 0 ? 8 : (bit_idx < 9 ? 4 : (bit_idx < 73 ? 2 : 1));
          data_cnt = IMIN(width * width - z, data_cnt * data_cnt);
          z += data_cnt;
        }
      }
    }

    // if state change occured, then split leaf (for now full split - full split of shallow octree)
    if(leaf_state != other_state) {
      ot_tree_t* out_tree = octree_get_tree(out, in_grid_idx);
      for(int tree_idx = 0; tree_idx < N_TREE_INTS; ++tree_idx) {
        out_tree[tree_idx] = ~0;
      }    
    }
  } // leaf_idx loop
    
  // printf("in: %d, %d,%d,%d, %d, %d\n", in->n, in->grid_depth,in->grid_height,in->grid_width, in->feature_size, in->n_leafs);
  // printf("out: %d, %d,%d,%d, %d, %d\n", out->n, out->grid_depth,out->grid_height,out->grid_width, out->feature_size, out->n_leafs);
  // for(int grid_idx = 0; grid_idx < octree_num_blocks(out); ++grid_idx) {
  //   std::cout << tree_bit_str_cpu(octree_get_tree(out, grid_idx)) << std::endl;
  // }
  octree_upd_n_leafs_cpu(out);
  // printf("out: %d, %d,%d,%d, %d, %d\n", out->n, out->grid_depth,out->grid_height,out->grid_width, out->feature_size, out->n_leafs);
  octree_resize_as_cpu(out, out);
  octree_upd_prefix_leafs_cpu(out);

  octree_cpy_sup_to_sub_cpu(in, out);
}





extern "C"
void octree_split_bwd_cpu(const octree* in, const octree* grad_out, octree* grad_in) {
  octree_cpy_scalars(in, grad_in);
  octree_resize_as_cpu(in, grad_in);
  octree_cpy_trees_cpu_cpu(in, grad_in);
  octree_cpy_prefix_leafs_cpu_cpu(in, grad_in);
  
  octree_cpy_sub_to_sup_sum_cpu(grad_out, grad_in);
}




extern "C"
void octree_split_dense_reconstruction_surface_fres_cpu(const ot_data_t* features, const ot_data_t* reconstruction, int n, int dense_depth, int dense_height, int dense_width, int feature_size, ot_data_t rec_thr_from, ot_data_t rec_thr_to, int band, octree* out) {
  if(dense_depth % 8 != 0 || dense_height % 8 != 0 || dense_width % 8 != 0) {
    printf("[ERROR] octree_split_dense_reconstruction_surface_fres_cpu - dense dims has to be a factor of 8\n");
    exit(-1);
  }

  int band_width = band * 2 + 1;

  octree_resize_cpu(n, dense_depth/8, dense_height/8, dense_width/8, feature_size, 0, out);
  octree_clr_trees_cpu(out);
  int n_blocks = octree_num_blocks(out);

  for(int grid_idx = 0; grid_idx < n_blocks; ++grid_idx) {
    ot_tree_t* tree = octree_get_tree(out, grid_idx);

    int n,gd,gh,gw;
    octree_split_grid_idx(out, grid_idx, &n,&gd,&gh,&gw);

    for(int idx = 0; idx < 8*8*8; ++idx) {
      bool differ = false;
      
      int bw = idx % 8;
      int bh = ((idx - bw) / 8) % 8;
      int bd = idx / (8 * 8);

      int bit_idx_l3 = tree_bit_idx_(bd,bh,bw);
      int bit_idx_l2 = tree_parent_bit_idx(bit_idx_l3);
      int bit_idx_l1 = tree_parent_bit_idx(bit_idx_l2);

      int dc = (gd * 8 + bd);
      int hc = (gh * 8 + bh);
      int wc = (gw * 8 + bw);
      
      for(int off = 0; off < band_width*band_width*band_width && !differ; ++off) {
        int off_w = off % band_width;
        int off_h = ((off - off_w) / band_width) % band_width;
        int off_d = off / (band_width*band_width);

        int doff = dc + (off_d - band);
        int hoff = hc + (off_h - band);
        int woff = wc + (off_w - band);

        bool in_bounds = doff >= 0 && hoff >= 0 && woff >= 0 && doff < dense_depth && hoff < dense_height && woff < dense_width;
        if(!in_bounds) continue;

        int rec_idx = ((n * dense_depth + doff) * dense_height + hoff) * dense_width + woff;
        bool occ_c = reconstruction[rec_idx] >= rec_thr_from && reconstruction[rec_idx] <= rec_thr_to;
        if(!occ_c) continue;

        for(int nb = 0; nb < 3*3*3 && !differ; ++nb) {
          if(nb == 13) continue;
          int nb_w = nb % 3;
          int nb_h = ((nb - nb_w) / 3) % 3;
          int nb_d = nb / (3*3);
          int d = doff + nb_d - 1; 
          int h = hoff + nb_h - 1;
          int w = woff + nb_w - 1;
          rec_idx = ((n * dense_depth + d) * dense_height + h) * dense_width + w;
          bool in_bounds = d >= 0 && h >= 0 && w >= 0 && d < dense_depth && h < dense_height && w < dense_width;
          bool occ = in_bounds && reconstruction[rec_idx] >= rec_thr_from && reconstruction[rec_idx] <= rec_thr_to;
          differ = differ || (in_bounds && (occ_c != occ));
        }
      }

      if(differ) {
        tree_set_bit(tree, 0);
        tree_set_bit(tree, bit_idx_l1);
        tree_set_bit(tree, bit_idx_l2);
      }
    }
  }


  octree_upd_n_leafs_cpu(out);
  octree_upd_prefix_leafs_cpu(out);
  octree_resize_as_cpu(out, out);

  for(int leaf_idx = 0; leaf_idx < out->n_leafs; ++leaf_idx) {
    // const int grid_idx = out.data[leaf_idx * out->feature_size];
    const int grid_idx = leaf_idx_to_grid_idx(out, leaf_idx);
    const ot_tree_t* tree = octree_get_tree(out, grid_idx);

    const int data_idx = leaf_idx - out->prefix_leafs[grid_idx];
    const int bit_idx = data_idx_to_bit_idx(tree, data_idx);

    int n,d,h,w;
    const int cell_depth = octree_ind_to_dense_ind(out, grid_idx, bit_idx, &n, &d,&h,&w);
    const int cell_width = width_from_depth(cell_depth);
    const int cell_width3 = cell_width * cell_width * cell_width;

    ot_data_t* out_data = octree_get_data(out, grid_idx) + data_idx * out->feature_size;
    for(int f = 0; f < feature_size; ++f) {
      ot_data_t val = 0;
      for(int idx = 0; idx < cell_width3; ++idx) {
        int idx_w = idx % cell_width;
        int idx_h = ((idx - idx_w) / cell_width) % cell_width;
        int idx_d = idx / (cell_width*cell_width);
        int dense_w = (w + idx_w);
        int dense_h = (h + idx_h);
        int dense_d = (d + idx_d);
        ot_data_t feat = features[(((n * feature_size + f) * dense_depth + dense_d) * dense_height + dense_h) * dense_width + dense_w];
        val += feat;
      }
      val /= cell_width3;
      out_data[f] = val;
    }
  }
}

extern "C"
void octree_split_dense_reconstruction_surface_fres_bwd_cpu(const octree* grad_out, ot_data_t* grad_in) {
  int dense_depth = 8 * grad_out->grid_depth;
  int dense_height = 8 * grad_out->grid_height;
  int dense_width = 8 * grad_out->grid_width;
  int feature_size = grad_out->feature_size;
  int n_voxels = grad_out->n * dense_depth * dense_height * dense_width;

  for(int vx_idx = 0; vx_idx < n_voxels; ++vx_idx) {
    int n = vx_idx / (dense_depth * dense_height * dense_width);
    int w = vx_idx % dense_width;
    int h = ((vx_idx - w) / dense_width) % dense_height;
    int d = ((((vx_idx - w) / dense_width) - h) / dense_height) % dense_depth;

    for(int f = 0; f < feature_size; ++f) {
      int grad_in_idx = (((n * feature_size + f) * dense_depth + d) * dense_height + h) * dense_width + w;
      grad_in[grad_in_idx] = 0;
    }

    int bd = d % 8;
    int bh = h % 8;
    int bw = w % 8;

    int gd = d / 8;
    int gh = h / 8;
    int gw = w / 8;

    int grid_idx = octree_grid_idx(grad_out, n, gd, gh, gw);
    const ot_tree_t* tree = octree_get_tree(grad_out, grid_idx);
    int bit_idx = tree_bit_idx(tree, bd, bh, bw); 
    int data_idx = tree_data_idx(tree, bit_idx, feature_size);
    const ot_data_t* grad_out_data = octree_get_data(grad_out, grid_idx) + data_idx;

    for(int f = 0; f < feature_size; ++f) {
      int grad_in_idx = (((n * feature_size + f) * dense_depth + d) * dense_height + h) * dense_width + w;
      grad_in[grad_in_idx] += grad_out_data[f];
    }

  }
}
