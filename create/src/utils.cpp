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

#include "octnet/create/utils.h"
#include "octnet/cpu/cpu.h"

extern "C"
void octree_scanline_fill(octree* grid, ot_data_t fill_value) {
  int* votes = new int[grid->n_leafs];
  for(int idx = 0; idx < grid->n_leafs; ++idx) {
    votes[idx] = 0;
  }

  int dense_depth = 8 * grid->grid_depth;
  int dense_height = 8 * grid->grid_height;
  int dense_width = 8 * grid->grid_width;

  //collect votes
  for(int gn = 0; gn < grid->n; ++gn) {
    //along w
    for(int idx = 0; idx < dense_depth * dense_height; ++idx) {
      int d = idx / dense_height;
      int h = idx % dense_height;
      int gd = d / 8;
      int bd = d % 8;
      int gh = h / 8;
      int bh = h % 8;
      int min = dense_width; int max = 0;
      for(int w = 0; w < dense_width; ++w) {
        int gw = w / 8;
        int bw = w % 8;
        int grid_idx = octree_grid_idx(grid, gn, gd, gh, gw);
        const ot_tree_t* tree = octree_get_tree(grid, grid_idx);
        int bit_idx = tree_bit_idx(tree, bd, bh, bw);
        int data_idx = grid->prefix_leafs[grid_idx] + tree_data_idx(tree, bit_idx, 1); 
        ot_data_t* data = grid->data + data_idx * grid->feature_size;
        if(data[0] != 0) {
          min = IMIN(min, w);
          max = IMAX(max, w);
        }
      }
      for(int w = min; w <= max; ++w) {
        int gw = w / 8;
        int bw = w % 8;
        int grid_idx = octree_grid_idx(grid, gn, gd, gh, gw);
        const ot_tree_t* tree = octree_get_tree(grid, grid_idx);
        int bit_idx = tree_bit_idx(tree, bd, bh, bw);
        int data_idx = grid->prefix_leafs[grid_idx] + tree_data_idx(tree, bit_idx, 1); 
        votes[data_idx] += 1;
      }
    }

    //along h
    for(int idx = 0; idx < dense_depth * dense_width; ++idx) {
      int d = idx / dense_width;
      int w = idx % dense_width;
      int gd = d / 8;
      int bd = d % 8;
      int gw = w / 8;
      int bw = w % 8;
      int min = dense_width; int max = 0;
      for(int h = 0; h < dense_height; ++h) {
        int gh = h / 8;
        int bh = h % 8;
        int grid_idx = octree_grid_idx(grid, gn, gd, gh, gw);
        const ot_tree_t* tree = octree_get_tree(grid, grid_idx);
        int bit_idx = tree_bit_idx(tree, bd, bh, bw);
        int data_idx = grid->prefix_leafs[grid_idx] + tree_data_idx(tree, bit_idx, 1); 
        ot_data_t* data = grid->data + data_idx * grid->feature_size;
        if(data[0] != 0) {
          min = IMIN(min, h);
          max = IMAX(max, h);
        }
      }
      for(int h = min; h <= max; ++h) {
        int gh = h / 8;
        int bh = h % 8;
        int grid_idx = octree_grid_idx(grid, gn, gd, gh, gw);
        const ot_tree_t* tree = octree_get_tree(grid, grid_idx);
        int bit_idx = tree_bit_idx(tree, bd, bh, bw);
        int data_idx = grid->prefix_leafs[grid_idx] + tree_data_idx(tree, bit_idx, 1); 
        votes[data_idx] += 1;
      }
    }

    //along d
    for(int idx = 0; idx < dense_height * dense_width; ++idx) {
      int h = idx / dense_height;
      int w = idx % dense_width;
      int gh = h / 8;
      int bh = h % 8;
      int gw = w / 8;
      int bw = w % 8;
      int min = dense_width; int max = 0;
      for(int d = 0; d < dense_depth; ++d) {
        int gd = d / 8;
        int bd = d % 8;
        int grid_idx = octree_grid_idx(grid, gn, gd, gh, gw);
        const ot_tree_t* tree = octree_get_tree(grid, grid_idx);
        int bit_idx = tree_bit_idx(tree, bd, bh, bw);
        int data_idx = grid->prefix_leafs[grid_idx] + tree_data_idx(tree, bit_idx, 1); 
        ot_data_t* data = grid->data + data_idx * grid->feature_size;
        if(data[0] != 0) {
          min = IMIN(min, d);
          max = IMAX(max, d);
        }
      }
      for(int d = min; d <= max; ++d) {
        int gd = d / 8;
        int bd = d % 8;
        int grid_idx = octree_grid_idx(grid, gn, gd, gh, gw);
        const ot_tree_t* tree = octree_get_tree(grid, grid_idx);
        int bit_idx = tree_bit_idx(tree, bd, bh, bw);
        int data_idx = grid->prefix_leafs[grid_idx] + tree_data_idx(tree, bit_idx, 1); 
        votes[data_idx] += 1;
      }
    } 
  }

  //apply majority vote
  for(int leaf_idx = 0; leaf_idx < grid->n_leafs; ++leaf_idx) {
    int grid_idx = leaf_idx_to_grid_idx(grid, leaf_idx);
    const ot_tree_t* tree = octree_get_tree(grid, grid_idx);

    int cum_n_leafs = grid->prefix_leafs[grid_idx];
    int data_idx = leaf_idx - cum_n_leafs;
    int bit_idx = data_idx_to_bit_idx(tree, data_idx);

    int depth = depth_from_bit_idx(bit_idx);
    int vol = depth == 0 ? 512 : (depth == 1 ? 64 : (depth == 2 ? 8 : 1));

    float vote = votes[leaf_idx] / float(vol);
    if(vote >= 2.0) {
      grid->data[leaf_idx * grid->feature_size] = fill_value;
    }
  } 

  delete[] votes;
}


void octree_occupancy_to_surface(octree* in, octree* out) {
  octree_resize_as_cpu(in, out);
  octree_cpy_trees_cpu_cpu(in, out);
  octree_cpy_prefix_leafs_cpu_cpu(in, out);

  int depth = 8 * in->grid_depth;
  int height = 8 * in->grid_height;
  int width = 8 * in->grid_width;

  for(int leaf_idx = 0; leaf_idx < in->n_leafs; ++leaf_idx) {
    int grid_idx = leaf_idx_to_grid_idx(in, leaf_idx);
    const ot_tree_t* tree = octree_get_tree(in, grid_idx);

    int data_idx = leaf_idx - in->prefix_leafs[grid_idx];
    int bit_idx = data_idx_to_bit_idx(tree, data_idx);

    int n, d,h,w;
    int cell_depth = octree_ind_to_dense_ind(in, grid_idx, bit_idx, &n, &d,&h,&w);
    int cell_width = width_from_depth(cell_depth);

    if(in->data[leaf_idx * in->feature_size] == 0) {
       out->data[leaf_idx * in->feature_size] = 0;
       continue;
    }


    bool surf = false;
    for(int od = d-1; od < d+cell_width+1 && !surf; ++od) {  
      for(int oh = h-1; oh < h+cell_width+1 && !surf; ++oh) {  
        for(int ow = w-1; ow < w+cell_width+1 && !surf; ++ow) {  

          if(od < d || od > d + cell_width || oh < h || oh > h + cell_width || ow < w || ow > w + cell_width) { 
            int ogd = od / 8;
            int ogh = oh / 8;
            int ogw = ow / 8;
            int obd = od % 8;
            int obh = oh % 8;
            int obw = ow % 8;

            int ogrid_idx = octree_grid_idx(out, n, ogd,ogh,ogw);
            const ot_tree_t* otree = octree_get_tree(out, ogrid_idx);
            int obit_idx = tree_bit_idx(otree, obd,obh,obw);
            int odata_idx = tree_data_idx(otree, obit_idx, out->feature_size);
            
            if(od < 0 || oh < 0 || ow < 0 || od >= depth || oh >= height || ow >= width || in->data[odata_idx] == 0) {
              surf = true;
            }
          }
        }
      }
    }
    
    out->data[leaf_idx * out->feature_size] = surf ? 1 : 0;
  }
}

void dense_occupancy_to_surface(const ot_data_t* dense, int depth, int height, int width, int n_iter, ot_data_t* surface) {

  if(n_iter != 1) {
    printf("[ERROR] n_iter != 1 not implemented\n");
    exit(-1);
  }

  for(int iter = 0; iter < n_iter; ++iter) {
    for(int in_idx = 0; in_idx < depth*height*width; ++in_idx) {
      if(dense[in_idx] == 0) {
        surface[in_idx] = 0;
        continue;
      }

      int iw = in_idx % (width);
      int id = in_idx / (width * height);
      int ih = ((in_idx - iw) / width) % height;

      bool surf = false;
      for(int od = id-1; od < id+2 && !surf; ++od) {  
        for(int oh = ih-1; oh < ih+2 && !surf; ++oh) {  
          for(int ow = iw-1; ow < iw+2 && !surf; ++ow) {  
            int border_idx = (od * height + oh) * width + ow;
            if((od != id || oh != ih || ow != iw) && 
               (od < 0 || oh < 0 || ow < 0 || od >= depth || oh >= height || ow >= width || dense[border_idx] == 0)) {
              surf = true;
            }
          }
        }
      }
      
      surface[in_idx] = surf ? 1 : 0;
    }
  }
}
