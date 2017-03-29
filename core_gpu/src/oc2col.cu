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

#include "octnet/gpu/oc2col.h"
#include "octnet/gpu/gpu.h"
#include "octnet/gpu/buffer.h"
#include "octnet/core/z_curve.h"

__device__ 
inline bool oc2col_in_vol(const octree* in, const int d, const int h, const int w) {
  return d >= 0 && h >= 0 && w >= 0 && d < 8 * in->grid_depth && h < 8 * in->grid_height && w < 8 * in->grid_width;
}


__device__
inline void oc2col_leaf(const octree* in, const ot_tree_t* leaf_tree, const int leaf_grid_idx, const int leaf_bit_idx,
    const int n, const int ds, const int hs, const int ws, const int size, 
    ot_data_t* out_shared, ot_data_t* out) {

  const ot_data_t factor = 1.f / (size * size * size);

  int d,h,w, kidx, grid_idx, bit_idx, data_idx, data_cnt, data_cnt_e1, data_cnt_e2;
  ot_tree_t* tree;
  ot_data_t* data_in;
  ot_data_t val_in, val;
  
  for(int f = 0; f < in->feature_size; ++f) {

    //leaf data
    data_idx = tree_data_idx(leaf_tree, leaf_bit_idx, in->feature_size);
    // data_in = in->data_ptrs[leaf_grid_idx] + data_idx;
    data_in = octree_get_data(in, leaf_grid_idx) + data_idx;
    val_in = data_in[f];
    
    //1-center
    val = size*size*size * factor * val_in;
    // (1,1,1)=13
    out_shared[13] = val;

    //6 
    val = (size-1)*size*size * factor * val_in;
    //(0,1,1)=4, (2,1,1)=22, (1,0,1)=10, (1,2,1)=16, (1,1,0)=12, (1,1,2)=14
    out_shared[ 4] = val;
    out_shared[10] = val;
    out_shared[12] = val;
    out_shared[14] = val;
    out_shared[16] = val;
    out_shared[22] = val;

    //8
    val = (size-1)*(size-1)*(size-1) * factor * val_in;
    //(0,0,0)=0, (0,0,2)=2, (0,2,0)=6, (0,2,2)=8, 
    //(2,0,0)=18, (2,0,2)=20, (2,2,0)=24, (2,2,2)=26
    out_shared[ 0] = val;
    out_shared[ 2] = val;
    out_shared[ 6] = val;
    out_shared[ 8] = val;
    out_shared[18] = val;
    out_shared[20] = val;
    out_shared[24] = val;
    out_shared[26] = val;

    //12
    val = (size-1)*(size-1)*(size) * factor * val_in;
    //(0,0,1)=1,  (0,1,0)=3,  (0,1,2)=5,  (0,2,1)=7
    //(1,0,0)=9,  (1,0,2)=11, (1,2,0)=15, (1,2,2)=17
    //(2,0,1)=19, (2,1,0)=21, (2,1,2)=23, (2,2,1)=25
    out_shared[ 1] = val;
    out_shared[ 3] = val;
    out_shared[ 5] = val;
    out_shared[ 7] = val;
    out_shared[ 9] = val;
    out_shared[11] = val;
    out_shared[15] = val;
    out_shared[17] = val;
    out_shared[19] = val;
    out_shared[21] = val;
    out_shared[23] = val;
    out_shared[25] = val;


    //corner data
    for(int cd = 0; cd < 2; ++cd) { 
      for(int ch = 0; ch < 2; ++ch) { 
        for(int cw = 0; cw < 2; ++cw) { 
          d = ds + (cd*(size+1)-1); h = hs + (ch*(size+1)-1); w = ws + (cw*(size+1)-1);
          if(oc2col_in_vol(in, d,h,w)) {
            grid_idx = octree_grid_idx(in, n, d / 8, h / 8, w / 8);
            tree = octree_get_tree(in, grid_idx);
            data_in = octree_get_data(in, grid_idx);
            bit_idx = tree_bit_idx(tree, d % 8, h % 8, w % 8);
            data_idx = tree_data_idx(tree, bit_idx, in->feature_size);
            kidx = (cd*2*3 + ch*2)*3 + cw*2;
            out_shared[kidx] += factor * data_in[data_idx + f];
          }
        }
      }
    }

    //along the edges
    //d
    for(int ch = 0; ch < 2; ++ch) { 
      for(int cw = 0; cw < 2; ++cw) { 
        d = ds; h = hs + (ch*(size+1)-1); w = ws + (cw*(size+1)-1);
        if(oc2col_in_vol(in, d,h,w)) {
          grid_idx = octree_grid_idx(in, n, d / 8, h / 8, w / 8);
          tree = octree_get_tree(in, grid_idx);
          data_in = octree_get_data(in, grid_idx);
          int e = 0;
          while(e < size) {
            d = ds + e;

            bit_idx = tree_bit_idx(tree, d % 8, h % 8, w % 8);
            data_cnt = bit_idx == 0 ? 8 : (bit_idx < 9 ? 4 : (bit_idx < 73 ? 2 : 1));
            data_cnt = IMIN(size - e, data_cnt);
            data_idx = tree_data_idx(tree, bit_idx, in->feature_size);
            val = factor * data_in[data_idx + f];

            kidx = ((0) * 3 + (ch*2)) * 3 + (cw*2);
            out_shared[kidx] += (data_cnt - (e+data_cnt >= size)) * val;
            kidx = ((1) * 3 + (ch*2)) * 3 + (cw*2);
            out_shared[kidx] +=  data_cnt * val;
            kidx = ((2) * 3 + (ch*2)) * 3 + (cw*2);
            out_shared[kidx] += (data_cnt - (e == 0)) * val;

            e += data_cnt;
          }
        }
      }
    }

    //h
    for(int cd = 0; cd < 2; ++cd) { 
      for(int cw = 0; cw < 2; ++cw) { 
        d = ds + (cd*(size+1)-1); h = hs; w = ws + (cw*(size+1)-1);
        if(oc2col_in_vol(in, d,h,w)) {
          grid_idx = octree_grid_idx(in, n, d / 8, h / 8, w / 8);
          tree = octree_get_tree(in, grid_idx);
          data_in = octree_get_data(in, grid_idx);
          int e = 0;
          while(e < size) {
            h = hs + e;

            bit_idx = tree_bit_idx(tree, d % 8, h % 8, w % 8);
            data_cnt = bit_idx == 0 ? 8 : (bit_idx < 9 ? 4 : (bit_idx < 73 ? 2 : 1));
            data_cnt = IMIN(size - e, data_cnt);
            data_idx = tree_data_idx(tree, bit_idx, in->feature_size);
            val = factor * data_in[data_idx + f];

            kidx = ((cd*2) * 3 + (0)) * 3 + (cw*2);
            out_shared[kidx] += (data_cnt - (e+data_cnt >= size)) * val;
            kidx = ((cd*2) * 3 + (1)) * 3 + (cw*2);
            out_shared[kidx] +=  data_cnt * val;
            kidx = ((cd*2) * 3 + (2)) * 3 + (cw*2);
            out_shared[kidx] += (data_cnt - (e == 0)) * val;
            
            e += data_cnt;
          }
        }
      }
    }

    //w
    for(int cd = 0; cd < 2; ++cd) { 
      for(int ch = 0; ch < 2; ++ch) { 
        d = ds + (cd*(size+1)-1); h = hs + (ch*(size+1)-1); w = ws;
        if(oc2col_in_vol(in, d,h,w)) {
          grid_idx = octree_grid_idx(in, n, d / 8, h / 8, w / 8);
          tree = octree_get_tree(in, grid_idx);
          data_in = octree_get_data(in, grid_idx);
          int e = 0;
          while(e < size) {
            w = ws + e;

            bit_idx = tree_bit_idx(tree, d % 8, h % 8, w % 8);
            data_cnt = bit_idx == 0 ? 8 : (bit_idx < 9 ? 4 : (bit_idx < 73 ? 2 : 1));
            data_cnt = IMIN(size - e, data_cnt);
            data_idx = tree_data_idx(tree, bit_idx, in->feature_size);
            val = factor * data_in[data_idx + f];

            kidx = ((cd*2) * 3 + (ch*2)) * 3 + (0);
            out_shared[kidx] += (data_cnt - (e+data_cnt >= size)) * val;
            kidx = ((cd*2) * 3 + (ch*2)) * 3 + (1);
            out_shared[kidx] +=  data_cnt * val;
            kidx = ((cd*2) * 3 + (ch*2)) * 3 + (2);
            out_shared[kidx] += (data_cnt - (e == 0)) * val;

            e += data_cnt;
          }
        }
      }
    }


    //along the faces
    //d
    for(int fd = 0; fd < 2; ++fd) {
      d = ds + (fd*(size+1)-1); h = hs; w = ws;
      if(oc2col_in_vol(in, d,h,w)) {
        grid_idx = octree_grid_idx(in, n, d / 8, h / 8, w / 8);
        tree = octree_get_tree(in, grid_idx);
        data_in = octree_get_data(in, grid_idx);
        int z = 0;
        while(z < size * size) {
          const int e1 = z_curve_x(z);
          const int e2 = z_curve_y(z);
          h = hs + e1;
          w = ws + e2;

          bit_idx = tree_bit_idx(tree, d % 8, h % 8, w % 8);
          data_cnt = bit_idx == 0 ? 8 : (bit_idx < 9 ? 4 : (bit_idx < 73 ? 2 : 1));
          data_cnt_e1 = IMIN(size - e1, data_cnt);
          data_cnt_e2 = IMIN(size - e2, data_cnt);
          data_cnt = IMIN(size * size - z, data_cnt * data_cnt);

          data_idx = tree_data_idx(tree, bit_idx, in->feature_size);
          val = factor * data_in[data_idx + f];

          kidx = ((fd*2) * 3 + (0)) * 3 + (0);
          out_shared[kidx] += (data_cnt_e1 - (e1+data_cnt_e1 >= size)) * (data_cnt_e2 - (e2+data_cnt_e2 >= size)) * val;
          kidx = ((fd*2) * 3 + (0)) * 3 + (1);
          out_shared[kidx] += (data_cnt_e1 - (e1+data_cnt_e1 >= size)) * (data_cnt_e2) * val;
          kidx = ((fd*2) * 3 + (0)) * 3 + (2);
          out_shared[kidx] += (data_cnt_e1 - (e1+data_cnt_e1 >= size)) * (data_cnt_e2 - (e2 == 0)) * val;
          kidx = ((fd*2) * 3 + (1)) * 3 + (0);
          out_shared[kidx] += (data_cnt_e1) * (data_cnt_e2 - (e2+data_cnt_e2 >= size)) * val;
          kidx = ((fd*2) * 3 + (1)) * 3 + (1);
          out_shared[kidx] += data_cnt * val;
          kidx = ((fd*2) * 3 + (1)) * 3 + (2);
          out_shared[kidx] += (data_cnt_e1) * (data_cnt_e2 - (e2 == 0)) * val;
          kidx = ((fd*2) * 3 + (2)) * 3 + (0);
          out_shared[kidx] += (data_cnt_e1 - (e1 == 0)) * (data_cnt_e2 - (e2+data_cnt_e2 >= size)) * val;
          kidx = ((fd*2) * 3 + (2)) * 3 + (1);
          out_shared[kidx] += (data_cnt_e1 - (e1 == 0)) * (data_cnt_e2) * val;
          kidx = ((fd*2) * 3 + (2)) * 3 + (2);
          out_shared[kidx] += (data_cnt_e1 - (e1 == 0)) * (data_cnt_e2 - (e2 == 0)) * val;

          z += data_cnt;
        }
      }
    }

    //h
    for(int fh = 0; fh < 2; ++fh) {
      d = ds; h = hs + (fh*(size+1)-1); w = ws;
      if(oc2col_in_vol(in, d,h,w)) {
        grid_idx = octree_grid_idx(in, n, d / 8, h / 8, w / 8);
        tree = octree_get_tree(in, grid_idx);
        data_in = octree_get_data(in, grid_idx);
        int z = 0;
        while(z < size * size) {
          const int e1 = z_curve_x(z);
          const int e2 = z_curve_y(z);
          d = ds + e1;
          w = ws + e2;

          bit_idx = tree_bit_idx(tree, d % 8, h % 8, w % 8);
          data_cnt = bit_idx == 0 ? 8 : (bit_idx < 9 ? 4 : (bit_idx < 73 ? 2 : 1));
          data_cnt_e1 = IMIN(size - e1, data_cnt);
          data_cnt_e2 = IMIN(size - e2, data_cnt);
          data_cnt = IMIN(size * size - z, data_cnt * data_cnt);

          data_idx = tree_data_idx(tree, bit_idx, in->feature_size);
          val = factor * data_in[data_idx + f];

          kidx = ((0) * 3 + (fh*2)) * 3 + (0);
          out_shared[kidx] += (data_cnt_e1 - (e1+data_cnt_e1 >= size)) * (data_cnt_e2 - (e2+data_cnt_e2 >= size)) * val;
          kidx = ((0) * 3 + (fh*2)) * 3 + (1);
          out_shared[kidx] += (data_cnt_e1 - (e1+data_cnt_e1 >= size)) * (data_cnt_e2) * val;
          kidx = ((0) * 3 + (fh*2)) * 3 + (2);
          out_shared[kidx] += (data_cnt_e1 - (e1+data_cnt_e1 >= size)) * (data_cnt_e2 - (e2 == 0)) * val;
          kidx = ((1) * 3 + (fh*2)) * 3 + (0);
          out_shared[kidx] += (data_cnt_e1) * (data_cnt_e2 - (e2+data_cnt_e2 >= size)) * val;
          kidx = ((1) * 3 + (fh*2)) * 3 + (1);
          out_shared[kidx] += data_cnt * val;
          kidx = ((1) * 3 + (fh*2)) * 3 + (2);
          out_shared[kidx] += (data_cnt_e1) * (data_cnt_e2 - (e2 == 0)) * val;
          kidx = ((2) * 3 + (fh*2)) * 3 + (0);
          out_shared[kidx] += (data_cnt_e1 - (e1 == 0)) * (data_cnt_e2 - (e2+data_cnt_e2 >= size)) * val;
          kidx = ((2) * 3 + (fh*2)) * 3 + (1);
          out_shared[kidx] += (data_cnt_e1 - (e1 == 0)) * (data_cnt_e2) * val;
          kidx = ((2) * 3 + (fh*2)) * 3 + (2);
          out_shared[kidx] += (data_cnt_e1 - (e1 == 0)) * (data_cnt_e2 - (e2 == 0)) * val;

          z += data_cnt;
        }
      }
    }

    //w
    for(int fw = 0; fw < 2; ++fw) {
      d = ds; h = hs; w = ws + (fw*(size+1)-1); 
      if(oc2col_in_vol(in, d,h,w)) {
        grid_idx = octree_grid_idx(in, n, d / 8, h / 8, w / 8);
        tree = octree_get_tree(in, grid_idx);
        data_in = octree_get_data(in, grid_idx);
        int z = 0;
        while(z < size * size) {
          const int e1 = z_curve_x(z);
          const int e2 = z_curve_y(z);
          d = ds + e1;
          h = hs + e2;

          bit_idx = tree_bit_idx(tree, d % 8, h % 8, w % 8);
          data_cnt = bit_idx == 0 ? 8 : (bit_idx < 9 ? 4 : (bit_idx < 73 ? 2 : 1));
          data_cnt_e1 = IMIN(size - e1, data_cnt);
          data_cnt_e2 = IMIN(size - e2, data_cnt);
          data_cnt = IMIN(size * size - z, data_cnt * data_cnt);

          data_idx = tree_data_idx(tree, bit_idx, in->feature_size);
          val = factor * data_in[data_idx + f];

          kidx = ((0) * 3 + (0)) * 3 + (fw*2);
          out_shared[kidx] += (data_cnt_e1 - (e1+data_cnt_e1 >= size)) * (data_cnt_e2 - (e2+data_cnt_e2 >= size)) * val;
          kidx = ((0) * 3 + (1)) * 3 + (fw*2);
          out_shared[kidx] += (data_cnt_e1 - (e1+data_cnt_e1 >= size)) * (data_cnt_e2) * val;
          kidx = ((0) * 3 + (2)) * 3 + (fw*2);
          out_shared[kidx] += (data_cnt_e1 - (e1+data_cnt_e1 >= size)) * (data_cnt_e2 - (e2 == 0)) * val;
          kidx = ((1) * 3 + (0)) * 3 + (fw*2);
          out_shared[kidx] += (data_cnt_e1) * (data_cnt_e2 - (e2+data_cnt_e2 >= size)) * val;
          kidx = ((1) * 3 + (1)) * 3 + (fw*2);
          out_shared[kidx] += data_cnt * val;
          kidx = ((1) * 3 + (2)) * 3 + (fw*2);
          out_shared[kidx] += (data_cnt_e1) * (data_cnt_e2 - (e2 == 0)) * val;
          kidx = ((2) * 3 + (0)) * 3 + (fw*2);
          out_shared[kidx] += (data_cnt_e1 - (e1 == 0)) * (data_cnt_e2 - (e2+data_cnt_e2 >= size)) * val;
          kidx = ((2) * 3 + (1)) * 3 + (fw*2);
          out_shared[kidx] += (data_cnt_e1 - (e1 == 0)) * (data_cnt_e2) * val;
          kidx = ((2) * 3 + (2)) * 3 + (fw*2);
          out_shared[kidx] += (data_cnt_e1 - (e1 == 0)) * (data_cnt_e2 - (e2 == 0)) * val;

          z += data_cnt;
        }
      }
    }


    //copy shared to global
    for(kidx = 0; kidx < K333; ++kidx) {
      out[f * K333 + kidx] = out_shared[kidx];
    }
  }
}




__global__ void kernel_oc2col_leafs(ot_data_t* col_buffer, const octree in, int leafs_offset, int n_leafs) {
  extern __shared__ ot_data_t out_shared[];

  const int out_inc = K333 * in.feature_size;
  
  CUDA_KERNEL_LOOP(leaf_idx, n_leafs) {
    leaf_idx = leaf_idx + leafs_offset;

    // const int grid_idx = col_buffer[leaf_idx * out_inc];
    const int grid_idx = leaf_idx_to_grid_idx(&in, leaf_idx);
    const ot_tree_t* tree = octree_get_tree(&in, grid_idx);

    // const int cum_n_leafs = n_leafs_upto(&in, grid_idx);
    const int cum_n_leafs = in.prefix_leafs[grid_idx];
    const int data_idx = leaf_idx - cum_n_leafs;
    const int bit_idx = data_idx_to_bit_idx(tree, data_idx);

    int n,d,h,w;
    const int depth = octree_ind_to_dense_ind(&in, grid_idx, bit_idx, &n, &d,&h,&w);
    const int size = width_from_depth(depth);

    const int col_buffer_idx = (cum_n_leafs - leafs_offset) * out_inc + tree_data_idx(tree, bit_idx, out_inc);
    oc2col_leaf(&in, tree, grid_idx, bit_idx, n,d,h,w,size, out_shared + threadIdx.x * K333, col_buffer + col_buffer_idx);
  }
}


void oc2col_gpu(const octree* in, ot_data_t* col_buffer, ot_size_t col_buffer_capacity, int leafs_offset, int n_leafs) {
  if(DEBUG) { printf("[DEBUG] oc2col_gpu n_blocks=%d, n_leafs %d\n", octree_num_blocks(in), in->n_leafs); }

  const int n_blocks = octree_num_blocks(in);
  
  // octree_leaf_idx_to_grid_idx_gpu(in, K333 * in->feature_size, col_buffer_capacity, col_buffer);

  kernel_oc2col_leafs<<<GET_BLOCKS_T(n_leafs, 256), 256, 256 * K333 * sizeof(ot_data_t)>>>(col_buffer, *in, leafs_offset, n_leafs);
  CUDA_POST_KERNEL_CHECK;
}
