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

#include "octnet/cpu/io.h"
#include "octnet/cpu/cpu.h"
#include "octnet/cpu/dense.h"

#include <iostream>
#include <sstream>
#include <vector>
#include <cstring>
#include <cstdio>
#include <cstdlib>

#if defined(_OPENMP)
#include <omp.h>
#endif

#define OC2_MAGIC_NUMBER 31193
#define DENSE2_MAGIC_NUMBER 61027


void sfread(void* dst, int size, int count, FILE* fp) {
  int n_read = fread(dst, size, count, fp);
  if(n_read != count) {
    printf("ERROR: number of read bytes differs from count in fread\n");
    exit(-1);
  }
}



extern "C"
void octree_read_deprecated_cpu(const char* path, octree* grid_h) {
  FILE* fp = fopen(path, "rb");  
  
  grid_h->n = 1;
  sfread(&(grid_h->grid_depth), sizeof(ot_size_t), 1, fp);
  sfread(&(grid_h->grid_height), sizeof(ot_size_t), 1, fp);
  sfread(&(grid_h->grid_width), sizeof(ot_size_t), 1, fp);
  sfread(&(grid_h->feature_size), sizeof(ot_size_t), 1, fp);
  sfread(&(grid_h->n_leafs), sizeof(ot_size_t), 1, fp);
  
  octree_resize_as_cpu(grid_h, grid_h);

  int n_blocks = octree_num_blocks(grid_h);
  sfread(grid_h->trees, sizeof(ot_tree_t), N_TREE_INTS * n_blocks, fp);
  sfread(grid_h->data, sizeof(ot_data_t), grid_h->n_leafs * grid_h->feature_size, fp);

  ot_data_t** data_ptrs = new ot_data_t*[n_blocks];
  sfread(data_ptrs, sizeof(ot_data_t*), n_blocks, fp);
  for(int grid_idx = 0; grid_idx < n_blocks; ++grid_idx) {
    grid_h->prefix_leafs[grid_idx] = (data_ptrs[grid_idx] - data_ptrs[0]) / grid_h->feature_size;
  }
  delete[] data_ptrs;

  fclose(fp);
}

extern "C"
void dense_read_prealloc_deprecated_cpu(const char* path, int n_dim, const int* dims, ot_data_t* data) {
  FILE* fp = fopen(path, "rb");

  if(dims[0] != 1) {
    printf("[ERROR] dense1 format supports only dims[0]=1 (n=1) dense_read_prealloc_deprecated_cpu\n");
    exit(-1);
  }
  n_dim--;
  dims = dims + 1;
   
  int size = 1;
  for(int dim_idx = 0; dim_idx < n_dim; ++dim_idx) {
    int dim_tmp;
    sfread(&(dim_tmp), sizeof(int), 1, fp);
    if(dim_tmp != dims[dim_idx]) {
      printf("invalid size in read_prealloc at dim_idx %d, parameter %d, read %d\n", dim_idx, dims[dim_idx], dim_tmp);
      exit(-1);
    }
    size *= dim_tmp;
  }

  sfread(data, sizeof(ot_data_t), size, fp);
  fclose(fp);
}

extern "C"
int* dense_read_header_deprecated_cpu(const char* path, int* n_dim) {
  n_dim[0] = 5;
  int* dims = new int[5];
  FILE* fp = fopen(path, "rb");
  dims[0] = 1;
  sfread(dims + 1, sizeof(ot_size_t), 4, fp);
  fclose(fp);
  return dims;
}

void octree_read_header_(FILE* fp, octree* grid_h) {
  int magic_number = -1;
  sfread(&(magic_number), sizeof(ot_size_t), 1, fp);
  if(magic_number != OC2_MAGIC_NUMBER) {
    printf("[ERROR] invalid magic number %d\n", magic_number);
    exit(-1);
  }
  sfread(&(grid_h->n), sizeof(ot_size_t), 1, fp);
  sfread(&(grid_h->grid_depth), sizeof(ot_size_t), 1, fp);
  sfread(&(grid_h->grid_height), sizeof(ot_size_t), 1, fp);
  sfread(&(grid_h->grid_width), sizeof(ot_size_t), 1, fp);
  sfread(&(grid_h->feature_size), sizeof(ot_size_t), 1, fp);
  sfread(&(grid_h->n_leafs), sizeof(ot_size_t), 1, fp);
}

extern "C"
void octree_read_header_cpu(const char* path, octree* grid_h) {
  FILE* fp = fopen(path, "rb");
  octree_read_header_(fp, grid_h);
  fclose(fp);
}

extern "C"
void octree_read_cpu(const char* path, octree* grid_h) {
  FILE* fp = fopen(path, "rb");
  
  octree_read_header_(fp, grid_h);
  octree_resize_as_cpu(grid_h, grid_h);

  int n_blocks = octree_num_blocks(grid_h);
  sfread(grid_h->trees, sizeof(ot_tree_t), N_TREE_INTS * n_blocks, fp);
  sfread(grid_h->data, sizeof(ot_data_t), grid_h->n_leafs * grid_h->feature_size, fp);
  sfread(grid_h->prefix_leafs, sizeof(ot_size_t), n_blocks, fp);
  fclose(fp);
}

extern "C"
void octree_write_cpu(const char* path, const octree* grid_h) {
  FILE* fp = fopen(path, "wb");

  const ot_size_t magic_number = OC2_MAGIC_NUMBER;
  fwrite(&(magic_number), sizeof(ot_size_t), 1, fp);
  fwrite(&(grid_h->n), sizeof(ot_size_t), 1, fp);
  fwrite(&(grid_h->grid_depth), sizeof(ot_size_t), 1, fp);
  fwrite(&(grid_h->grid_height), sizeof(ot_size_t), 1, fp);
  fwrite(&(grid_h->grid_width), sizeof(ot_size_t), 1, fp);
  fwrite(&(grid_h->feature_size), sizeof(ot_size_t), 1, fp);
  fwrite(&(grid_h->n_leafs), sizeof(ot_size_t), 1, fp);
  
  int n_blocks = octree_num_blocks(grid_h);
  fwrite(grid_h->trees, sizeof(ot_tree_t), N_TREE_INTS * n_blocks, fp);  
  fwrite(grid_h->data, sizeof(ot_data_t), grid_h->n_leafs * grid_h->feature_size, fp);  
  fwrite(grid_h->prefix_leafs, sizeof(ot_data_t), n_blocks, fp);  

  fclose(fp);
}


extern "C"
void octree_dhwc_write_cpu(const char* path, const octree* grid_h) {
  int n = grid_h->n;
  int depth = 8 * grid_h->grid_depth;
  int height = 8 * grid_h->grid_height;
  int width = 8 * grid_h->grid_width;
  int feature_size = grid_h->feature_size;
  
  ot_data_t* voxels = new ot_data_t[n * depth * height * width * feature_size];
  octree_to_dhwc_cpu(grid_h, depth, height, width, voxels);

  int n_dim = 5;
  int dims[] = {n, depth, height, width, feature_size};
  // dense_write_cpu(path, voxels, depth, height, width, feature_size);
  dense_write_cpu(path, n_dim, dims, voxels);
  delete[] voxels;
}

extern "C"
void octree_cdhw_write_cpu(const char* path, const octree* grid_h) {
  int n = grid_h->n;
  int depth = 8 * grid_h->grid_depth;
  int height = 8 * grid_h->grid_height;
  int width = 8 * grid_h->grid_width;
  int feature_size = grid_h->feature_size;
  
  ot_data_t* voxels = new ot_data_t[n * depth * height * width * feature_size];
  octree_to_cdhw_cpu(grid_h, depth, height, width, voxels);
  int n_dim = 5;
  int dims[] = {n, feature_size, depth, height, width};
  // dense4_write_cpu(path, voxels, feature_size, depth, height, width);
  dense_write_cpu(path, n_dim, dims, voxels);
  delete[] voxels;
}




extern "C"
void dense_write_cpu(const char* path, int n_dim, const int* dims, const ot_data_t* data) {
  FILE* fp = fopen(path, "wb");

  const ot_size_t magic_number = DENSE2_MAGIC_NUMBER;
  fwrite(&(magic_number), sizeof(ot_size_t), 1, fp);

  fwrite(&(n_dim), sizeof(ot_size_t), 1, fp);
  fwrite(dims, sizeof(int), n_dim, fp);

  int size = 1;
  for(int dim_idx = 0; dim_idx < n_dim; ++dim_idx) {
    size *= dims[dim_idx];
  }
  fwrite(data, sizeof(ot_data_t), size, fp);

  fclose(fp);
}

extern "C"
ot_data_t* dense_read_cpu(const char* path) {
  FILE* fp = fopen(path, "rb");
  
  int magic_number = -1;
  sfread(&(magic_number), sizeof(ot_size_t), 1, fp);
  if(magic_number != DENSE2_MAGIC_NUMBER) {
    printf("[ERROR] invalid magic number in dense_read_cpu\n");
    exit(-1);
  }

  int n_dim = -1;
  sfread(&(n_dim), sizeof(ot_size_t), 1, fp);

  int size = 1;
  for(int dim_idx = 0; dim_idx < n_dim; ++dim_idx) {
    int dim_tmp;
    sfread(&(dim_tmp), sizeof(int), 1, fp);
    size *= dim_tmp;
  }

  ot_data_t* data = new ot_data_t[size];
  sfread(data, sizeof(ot_data_t), size, fp);
  fclose(fp);

  return data;
}

extern "C"
void dense_read_prealloc_cpu(const char* path, int n_dim, const int* dims, ot_data_t* data) {
  FILE* fp = fopen(path, "rb");

  int magic_number = -1;
  sfread(&(magic_number), sizeof(ot_size_t), 1, fp);
  if(magic_number != DENSE2_MAGIC_NUMBER) {
    printf("[ERROR] invalid magic number in dense_read_cpu\n");
    exit(-1);
  }
  
  int tmp_n_dim = -1;
  sfread(&(tmp_n_dim), sizeof(ot_size_t), 1, fp);
  if(n_dim != tmp_n_dim) {
    printf("[ERROR] invalid n_dim read in dense_read_prealloc_cpu\n");
    exit(-1);
  }

  int size = 1;
  for(int dim_idx = 0; dim_idx < n_dim; ++dim_idx) {
    int dim_tmp;
    sfread(&(dim_tmp), sizeof(int), 1, fp);
    if(dim_tmp != dims[dim_idx]) {
      printf("invalid size in read_prealloc at dim_idx %d, parameter %d, read %d\n", dim_idx, dims[dim_idx], dim_tmp);
      exit(-1);
    }
    size *= dim_tmp;
  }

  sfread(data, sizeof(ot_data_t), size, fp);
  fclose(fp);
}



extern "C"
void octree_read_batch_cpu(int n_paths, char** paths, int n_threads, octree* grid_h) {
  if(n_paths <= 0) {
    printf("[ERROR] n_paths <= 0 in octree_read_batch_cpu\n");
    exit(-1);
  }

  // printf("received %d paths\n", n_paths);
  // for(int path_idx = 0; path_idx < n_paths; ++path_idx) {
  //   printf("  path: '%s'\n", paths[path_idx]);
  // }

  //determine necessary memory
  ot_size_t n;
  ot_size_t n_leafs[n_paths];
  ot_size_t n_blocks[n_paths];

  FILE* fp = fopen(paths[0], "rb");
  int magic_number = -1;
  sfread(&(magic_number), sizeof(ot_size_t), 1, fp);
  if(magic_number != OC2_MAGIC_NUMBER) {
    printf("[ERROR] invalid magic number %d\n", magic_number);
    exit(-1);
  }
  sfread(&n, sizeof(ot_size_t), 1, fp);
  sfread(&(grid_h->grid_depth), sizeof(ot_size_t), 1, fp);
  sfread(&(grid_h->grid_height), sizeof(ot_size_t), 1, fp);
  sfread(&(grid_h->grid_width), sizeof(ot_size_t), 1, fp);
  sfread(&(grid_h->feature_size), sizeof(ot_size_t), 1, fp);
  sfread(n_leafs, sizeof(ot_size_t), 1, fp);
  n_blocks[0] = n * grid_h->grid_depth * grid_h->grid_height * grid_h->grid_width;
  fclose(fp);

  // printf("path 0: %d, %d,%d,%d, %d, %d\n", n, grid_h->grid_depth,grid_h->grid_height,grid_h->grid_width, grid_h->feature_size, n_leafs[0]);

  #if defined(_OPENMP)
  omp_set_num_threads(n_threads);
  #endif
  #pragma omp parallel for reduction(+:n)
  for(int path_idx = 1; path_idx < n_paths; ++path_idx) {
    int tmp_magic_number, tmp_n, tmp_grid_depth, tmp_grid_width, tmp_grid_height, tmp_feature_size;

    FILE* fp = fopen(paths[path_idx], "rb");
    sfread(&(tmp_magic_number), sizeof(ot_size_t), 1, fp);
    if(tmp_magic_number != OC2_MAGIC_NUMBER) {
      printf("[ERROR] invalid magic number %d\n", tmp_magic_number);
      exit(-1);
    }
    sfread(&tmp_n, sizeof(ot_size_t), 1, fp);
    sfread(&(tmp_grid_depth), sizeof(ot_size_t), 1, fp);
    sfread(&(tmp_grid_height), sizeof(ot_size_t), 1, fp);
    sfread(&(tmp_grid_width), sizeof(ot_size_t), 1, fp);
    sfread(&(tmp_feature_size), sizeof(ot_size_t), 1, fp);
    sfread(n_leafs + path_idx, sizeof(ot_size_t), 1, fp);
    fclose(fp);
    
    n += tmp_n;
    n_blocks[path_idx] = tmp_n * tmp_grid_depth * tmp_grid_height * tmp_grid_width;
  
    // printf("path %d: %d, %d,%d,%d, %d, %d\n", path_idx, n, grid_h->grid_depth,grid_h->grid_height,grid_h->grid_width, grid_h->feature_size, n_leafs[path_idx]);
    
    if(tmp_grid_depth != grid_h->grid_depth) {
      printf("[ERROR] grid_depth of path %d does not match in octree_read_batch_cpu\n", path_idx);
      exit(-1);
    }
    if(tmp_grid_height != grid_h->grid_height) {
      printf("[ERROR] grid_height of path %d does not match in octree_read_batch_cpu\n", path_idx);
      exit(-1);
    }
    if(tmp_grid_width != grid_h->grid_width) {
      printf("[ERROR] grid_width of path %d does not match in octree_read_batch_cpu\n", path_idx);
      exit(-1);
    }
    if(tmp_feature_size != grid_h->feature_size) {
      printf("[ERROR] feature_size of path %d does not match in octree_read_batch_cpu (%d, %d)\n", path_idx, grid_h->feature_size, tmp_feature_size);
      exit(-1);
    }
  }

  // prefix sums
  for(int path_idx = 1; path_idx < n_paths; ++path_idx) {
    n_leafs[path_idx] += n_leafs[path_idx-1];
    n_blocks[path_idx] += n_blocks[path_idx-1];
  }
  
  //resize octree
  grid_h->n = n;
  grid_h->n_leafs = n_leafs[n_paths-1];
  octree_resize_as_cpu(grid_h, grid_h);
  
  // get tree/data content
  #if defined(_OPENMP)
  omp_set_num_threads(n_threads);
  #endif
  #pragma omp parallel for
  for(int path_idx = 0; path_idx < n_paths; ++path_idx) {
    FILE* fp = fopen(paths[path_idx], "rb");
    int tmp_magic_number = -1;
    sfread(&(tmp_magic_number), sizeof(ot_size_t), 1, fp);
    fseek(fp, sizeof(ot_size_t) * 7, SEEK_SET);

    ot_size_t n_leafs_offset  = path_idx == 0 ? 0 : n_leafs[path_idx - 1];
    ot_size_t n_leafs_num     = path_idx == 0 ? n_leafs[0] : n_leafs[path_idx] - n_leafs[path_idx-1];
    ot_size_t n_blocks_offset = path_idx == 0 ? 0 : n_blocks[path_idx - 1];
    ot_size_t n_blocks_num    = path_idx == 0 ? n_blocks[0] : n_blocks[path_idx] - n_blocks[path_idx-1];

    sfread(grid_h->trees + n_blocks_offset * N_TREE_INTS, sizeof(ot_tree_t), N_TREE_INTS * n_blocks_num, fp);
    sfread(grid_h->data + n_leafs_offset * grid_h->feature_size, sizeof(ot_data_t), n_leafs_num * grid_h->feature_size, fp);    
    sfread(grid_h->prefix_leafs + n_blocks_offset, sizeof(ot_size_t), n_blocks_num, fp);
    fclose(fp);
    
    for(int grid_idx = n_blocks_offset; grid_idx < n_blocks_offset + n_blocks_num; ++grid_idx) {
      grid_h->prefix_leafs[grid_idx] += n_leafs_offset;
    }
  }
}


extern "C"
void dense_read_prealloc_batch_cpu(int n_paths, char** paths, int n_threads, int n_dim, const int* dims, ot_data_t* data) {
  int offset = 1;
  for(int dim_idx = 1; dim_idx < n_dim; ++dim_idx) {
    offset *= dims[dim_idx];
  }

  int dims_single[n_dim];
  dims_single[0] = 1;
  for(int dim_idx = 1; dim_idx < n_dim; ++dim_idx) {
    dims_single[dim_idx] = dims[dim_idx];
  }
  
  #if defined(_OPENMP)
  omp_set_num_threads(n_threads);
  #endif
  #pragma omp parallel for
  for(int path_idx = 0; path_idx < n_paths; ++path_idx) {
    dense_read_prealloc_cpu(paths[path_idx], n_dim, dims_single, data + path_idx * offset);
  }
}

