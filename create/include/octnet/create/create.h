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

#ifndef OCTREE_CREATE_CPU_H
#define OCTREE_CREATE_CPU_H

#include "octnet/core/core.h"


class OctreeCreateHelperCpu {
public:
  OctreeCreateHelperCpu(ot_size_t grid_depth_, ot_size_t grid_height_, ot_size_t grid_width_) :
    grid_depth(grid_depth_), grid_height(grid_height_), grid_width(grid_width_),
    off_gd(0), off_gh(0), off_gw(0) 
  {}
  virtual ~OctreeCreateHelperCpu() {}

  virtual void update_offsets(int off_gd_, int off_gh_, int off_gw_) {
    off_gd = off_gd_;
    off_gh = off_gh_;
    off_gw = off_gw_;
    // printf("updated helper, off: %d,%d,%d\n", off_gd, off_gh, off_gw);
  }

  virtual void update_grid_coords(int& gd, int& gh, int& gw) {
    gd += off_gd;
    gh += off_gh;
    gw += off_gw;
  }

  virtual int get_grid_idx(int gd, int gh, int gw) {
    int old_grid_idx = (gd * grid_height + gh) * grid_width + gw;
    return old_grid_idx;
  }

public:
  ot_size_t grid_depth;
  ot_size_t grid_height;
  ot_size_t grid_width;

  int off_gd;
  int off_gh;
  int off_gw;
};



class OctreeCreateCpu {
public:
  OctreeCreateCpu(ot_size_t grid_depth_, ot_size_t grid_height_, ot_size_t grid_width_, ot_size_t feature_size_) : 
    grid_depth(grid_depth_), grid_height(grid_height_), grid_width(grid_width_), feature_size(feature_size_) 
  {}

  virtual ~OctreeCreateCpu() {}

  virtual octree* operator()(bool fit=false, int fit_multiply=1, bool pack=false, int n_threads=1);

protected:
  virtual octree* create_octree(bool fit, int fit_multiply, bool pack, int n_threads, OctreeCreateHelperCpu* helper);

  virtual octree* alloc_grid(); 
  virtual void create_octree_structure(octree* grid, OctreeCreateHelperCpu* helper);
  virtual void fit_octree(octree* grid, int fit_multiply, OctreeCreateHelperCpu* helper);
  virtual void pack_octree(octree* grid, OctreeCreateHelperCpu* helper);
  virtual void update_and_resize_octree(octree* grid);
  virtual void fill_octree_data(octree* grid, bool packed, OctreeCreateHelperCpu* helper);

  virtual bool is_occupied(float cx, float cy, float cz, float vd, float vh, float vw, int gd, int gh, int gw, OctreeCreateHelperCpu* helper) = 0;
  virtual void get_data(bool oc, float cx, float cy, float cz, float vd, float vh, float vw, int gd, int gh, int gw, OctreeCreateHelperCpu* helper, ot_data_t* dst) = 0;

  ot_size_t grid_depth;
  ot_size_t grid_height;
  ot_size_t grid_width;
  ot_size_t feature_size;
};



extern "C" {

octree* octree_create_from_dense_cpu(const ot_data_t* data, int feature_size, int depth, int height, int width, bool fit, int fit_multiply, bool pack, int n_threads);
octree* octree_create_from_dense2_cpu(const ot_data_t* occupancy, const ot_data_t* features, int feature_size, int depth, int height, int width, bool fit, int fit_multiply, bool pack, int n_threads);

octree* octree_create_from_mesh_cpu(int n_verts_, float* verts_, int n_faces_, int* faces, bool rescale_verts, ot_size_t depth, ot_size_t height, ot_size_t width, bool fit, int fit_multiply, bool pack, int pad, int n_threads);
octree* octree_create_from_off_cpu(const char* path, ot_size_t depth, ot_size_t height, ot_size_t width, const float R[9], bool fit, int fit_multiply, bool pack, int pad, int n_threads);
octree* octree_create_from_obj_cpu(const char* path, ot_size_t depth, ot_size_t height, ot_size_t width, const float R[9], bool fit, int fit_multiply, bool pack, int pad, int n_threads);

octree* octree_create_from_pc_simple_cpu(float* xyz, int n_pts, int feature_size, ot_size_t depth, ot_size_t height, ot_size_t width, bool normalize, bool normalize_inplace, bool fit, int fit_multiply, bool pack, int pad, int n_threads);
octree* octree_create_from_pc_cpu(float* xyz, const float* features, int n_pts, int feature_size, ot_size_t depth, ot_size_t height, ot_size_t width, bool normalize, bool normalize_inplace, bool fit, int fit_multiply, bool pack, int pad, int n_threads);
void octree_create_dense_from_pc_cpu(const float* xyz, const float* features, float* vol, int n_pts, int feature_size, ot_size_t depth, ot_size_t height, ot_size_t width, int n_threads);


}

#endif 
