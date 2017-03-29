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

#include "octnet/gpu/dense.h"
#include "octnet/gpu/gpu.h"


template <int dense_format, bool avg_vol>
__global__ void kernel_octree_to_dense(ot_data_t* out_data, int n_voxels, const int dense_depth, const int dense_height, const int dense_width, const octree grid) {
  const int vx_depth_off = (dense_depth - grid.grid_depth * 8) / 2;
  const int vx_height_off = (dense_height - grid.grid_height * 8) / 2;
  const int vx_width_off = (dense_width - grid.grid_width * 8) / 2;

  CUDA_KERNEL_LOOP(vx_idx, n_voxels) {
    const int feature_size = grid.feature_size;

    int n, d, h, w;
    if(dense_format == DENSE_FORMAT_DHWC) {
      n = vx_idx / (dense_depth * dense_height * dense_width);
      w = (vx_idx) % dense_width;
      h = (((vx_idx) - w) / dense_width) % dense_height;
      d = (((((vx_idx) - w) / dense_width) - h) / dense_height) % dense_width;
    }
    else if(dense_format == DENSE_FORMAT_CDHW) {
      n = vx_idx / (dense_depth * dense_height * dense_width);
      w = vx_idx % dense_width;
      h = ((vx_idx - w) / dense_width) % dense_height;
      d = ((((vx_idx - w) / dense_width) - h) / dense_height) % dense_depth;
    }

    int bd = (d - vx_depth_off) % 8;
    int bh = (h - vx_height_off) % 8;
    int bw = (w - vx_width_off) % 8;

    int gd = (d - vx_depth_off) / 8;
    int gh = (h - vx_height_off) / 8;
    int gw = (w - vx_width_off) / 8;
  
    if(bd < 0 || bh < 0 || bw < 0 || bd >= 8 || bh >= 8 || bw >= 8 || gd < 0 || gh < 0 || gw < 0 || gd >= grid.grid_depth || gh >= grid.grid_height || gw >= grid.grid_width) {
      for(int f = 0; f < feature_size; ++f) { 
        int out_idx; 
        if(dense_format == DENSE_FORMAT_DHWC) {
          out_idx = (((n * dense_depth + d) * dense_height + h) * dense_width + w) * feature_size + f;
        }
        else if(dense_format == DENSE_FORMAT_CDHW) {
          out_idx = (((n * feature_size + f) * dense_depth + d) * dense_height + h) * dense_width + w;
        }
        out_data[out_idx] = 0;
      }
    }
    else {
      int grid_idx = octree_grid_idx(&grid, n, gd, gh, gw);

      const ot_tree_t* tree = octree_get_tree(&grid, grid_idx);
      int bit_idx = tree_bit_idx(tree, bd, bh, bw); 
      int data_idx = tree_data_idx(tree, bit_idx, feature_size);
      // const ot_data_t* data = grid.data_ptrs[grid_idx] + data_idx; 
      const ot_data_t* data = octree_get_data(&grid, grid_idx) + data_idx;

      float vol = 1;
      if(avg_vol) {
        vol = bit_idx == 0 ? 512 : (bit_idx < 9 ? 64 : (bit_idx < 73 ? 8 : 1));
      }

      for(int f = 0; f < feature_size; ++f) { 
        int out_idx; 
        if(dense_format == DENSE_FORMAT_DHWC) {
          out_idx = (((n * dense_depth + d) * dense_height + h) * dense_width + w) * feature_size + f;
        }
        else if(dense_format == DENSE_FORMAT_CDHW) {
          out_idx = (((n * feature_size + f) * dense_depth + d) * dense_height + h) * dense_width + w;
        }
        ot_data_t val = data[f];
        if(avg_vol) {
          out_data[out_idx] = val / vol;
        } 
        else{
          out_data[out_idx] = val;
        }
      }
    }

  } // cuda loop
}


void octree_to_dhwc_gpu(const octree* grid_d, const int dense_depth, const int dense_height, const int dense_width, ot_data_t* out_data) {
  if(DEBUG) { printf("[DEBUG] octree_to_dhwc_gpu\n"); }
  if(dense_depth < grid_d->grid_depth * 8 || dense_height < grid_d->grid_height * 8 || dense_width < grid_d->grid_width * 8) {
    printf("[ERROR] dense dim (%d,%d,%d) is smaller then dim of grid (%d,%d,%d)\n", 
        dense_depth, dense_height, dense_width, grid_d->grid_depth*8, grid_d->grid_height*8, grid_d->grid_width*8);
    exit(-1);
  }

  int n_voxels = grid_d->n * dense_depth * dense_height * dense_width;
  kernel_octree_to_dense<DENSE_FORMAT_DHWC, false><<<GET_BLOCKS(n_voxels), CUDA_NUM_THREADS>>>(
      out_data, n_voxels, dense_depth, dense_height, dense_width, *grid_d
  );
  CUDA_POST_KERNEL_CHECK;
}

void octree_to_cdhw_gpu(const octree* grid_d, const int dense_depth, const int dense_height, const int dense_width, ot_data_t* out_data) {
  if(DEBUG) { printf("[DEBUG] octree_to_cdhw_gpu\n"); }
  if(dense_depth < grid_d->grid_depth * 8 || dense_height < grid_d->grid_height * 8 || dense_width < grid_d->grid_width * 8) {
    printf("[ERROR] dense dim (%d,%d,%d) is smaller then dim of grid (%d,%d,%d)\n", 
        dense_depth, dense_height, dense_width, grid_d->grid_depth*8, grid_d->grid_height*8, grid_d->grid_width*8);
    exit(-1);
  }

  int n_voxels = grid_d->n * dense_depth * dense_height * dense_width;
  kernel_octree_to_dense<DENSE_FORMAT_CDHW, false><<<GET_BLOCKS(n_voxels), CUDA_NUM_THREADS>>>(
      out_data, n_voxels, dense_depth, dense_height, dense_width, *grid_d
  );
  CUDA_POST_KERNEL_CHECK;
}



void octree_to_dhwc_avg_gpu(const octree* grid_d, const int dense_depth, const int dense_height, const int dense_width, ot_data_t* out_data) {
  if(DEBUG) { printf("[DEBUG] octree_to_dhwc_avg_gpu\n"); }
  if(dense_depth < grid_d->grid_depth * 8 || dense_height < grid_d->grid_height * 8 || dense_width < grid_d->grid_width * 8) {
    printf("[ERROR] dense dim (%d,%d,%d) is smaller then dim of grid (%d,%d,%d)\n", 
        dense_depth, dense_height, dense_width, grid_d->grid_depth*8, grid_d->grid_height*8, grid_d->grid_width*8);
    exit(-1);
  }

  int n_voxels = grid_d->n * dense_depth * dense_height * dense_width;
  kernel_octree_to_dense<DENSE_FORMAT_DHWC, true><<<GET_BLOCKS(n_voxels), CUDA_NUM_THREADS>>>(
      out_data, n_voxels, dense_depth, dense_height, dense_width, *grid_d
  );
  CUDA_POST_KERNEL_CHECK;
}

void octree_to_cdwh_avg_gpu(const octree* grid_d, const int dense_depth, const int dense_height, const int dense_width, ot_data_t* out_data) {
  if(DEBUG) { printf("[DEBUG] octree_to_cdwh_avg_gpu\n"); }
  if(dense_depth < grid_d->grid_depth * 8 || dense_height < grid_d->grid_height * 8 || dense_width < grid_d->grid_width * 8) {
    printf("[ERROR] dense dim (%d,%d,%d) is smaller then dim of grid (%d,%d,%d)\n", 
        dense_depth, dense_height, dense_width, grid_d->grid_depth*8, grid_d->grid_height*8, grid_d->grid_width*8);
    exit(-1);
  }

  int n_voxels = grid_d->n * dense_depth * dense_height * dense_width;
  kernel_octree_to_dense<DENSE_FORMAT_CDHW, true><<<GET_BLOCKS(n_voxels), CUDA_NUM_THREADS>>>(
      out_data, n_voxels, dense_depth, dense_height, dense_width, *grid_d
  );
  CUDA_POST_KERNEL_CHECK;
}



void octree_to_dhwc_bwd_gpu(const octree* grid_d, const int dense_depth, const int dense_height, const int dense_width, const ot_data_t* grad_data, octree* grad_grid_d) {
  if(DEBUG) { printf("[DEBUG] octree_to_dhwc_bwd_gpu\n"); }
  dhwc_to_octree_sum_gpu(grid_d, dense_depth, dense_height, dense_width, grad_data, grid_d->feature_size, grad_grid_d);
}

void octree_to_cdhw_bwd_gpu(const octree* grid_d, const int dense_depth, const int dense_height, const int dense_width, const ot_data_t* grad_data, octree* grad_grid_d) {
  if(DEBUG) { printf("[DEBUG] octree_to_cdhw_bwd_gpu\n"); }
  cdhw_to_octree_sum_gpu(grid_d, dense_depth, dense_height, dense_width, grad_data, grid_d->feature_size, grad_grid_d);
}
