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

#ifndef OCTREE_CREATE_FROM_MESH_CPU_H
#define OCTREE_CREATE_FROM_MESH_CPU_H

#include "octnet/create/create.h"
#include "octnet/cpu/cpu.h"
#include "octnet/geometry/geometry.h"

#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <cstdlib>

#if defined(_OPENMP)
#include <omp.h>
#endif



class OctreeCreateFromMeshHelperCpu : public OctreeCreateHelperCpu {
public:
  OctreeCreateFromMeshHelperCpu(ot_size_t grid_depth_, ot_size_t grid_height_, ot_size_t grid_width_) :
    OctreeCreateHelperCpu(grid_depth_, grid_height_, grid_width_), 
    tinds(grid_depth_ * grid_height_ * grid_width_)
  {}
  virtual ~OctreeCreateFromMeshHelperCpu() {}

public:
  std::vector<std::vector<int> > tinds;
};



class OctreeFromMesh : public OctreeCreateCpu {
public:
  OctreeFromMesh(int n_verts_, float* verts_, int n_faces_, int* faces_, bool rescale_verts, 
      ot_size_t depth_, ot_size_t height_, ot_size_t width_, int pad_) : 
      OctreeCreateCpu((depth_ + 7) / 8, (height_ + 7) / 8, (width_ + 7) / 8, 1), 
      depth(depth_), height(height_), width(width_),
      n_verts(n_verts_), verts(verts_), n_faces(n_faces_), faces(faces_), pad(pad_) {
    
    if(rescale_verts) {
      rescale();
    }
  }

  virtual ~OctreeFromMesh() {
  }
  
  virtual octree* operator()(bool fit, int fit_multiply, bool pack, int n_threads) {
    //determine block triangle intersections
    int n_blocks = grid_depth * grid_height * grid_width;
    printf("  [OctreeCreateCpu] determine block triangle intersections\n");
    OctreeCreateFromMeshHelperCpu helper(grid_depth, grid_height, grid_width);

#if defined(_OPENMP)
    omp_set_num_threads(n_threads);
#endif
    #pragma omp parallel for
    for(int grid_idx = 0; grid_idx < n_blocks; ++grid_idx) {
      int gd = grid_idx / (grid_height * grid_width);
      int gh = (grid_idx / grid_width) % grid_height;
      int gw = grid_idx % grid_width;
      
      float cx = gw * 8 + 4;
      float cy = gh * 8 + 4;
      float cz = gd * 8 + 4;
      block_triangles(cx,cy,cz, 8,8,8, helper.tinds[grid_idx]);
    }
    
    return create_octree(fit, fit_multiply, pack, n_threads, &helper);
  }


protected:
  virtual void block_triangles(float cx, float cy, float cz, float vd, float vh, float vw, std::vector<int>& tinds) {
    for(int fidx = 0; fidx < n_faces; ++fidx) {
      float3 vx_c;
      vx_c.x = cx;
      vx_c.y = cy;
      vx_c.z = cz;
      float3 vx_w;
      vx_w.x = vw;
      vx_w.y = vh;
      vx_w.z = vd;

      float3 v0;
      v0.x = verts[faces[fidx * 3 + 0] * 3 + 0];
      v0.y = verts[faces[fidx * 3 + 0] * 3 + 1];
      v0.z = verts[faces[fidx * 3 + 0] * 3 + 2];
      float3 v1;
      v1.x = verts[faces[fidx * 3 + 1] * 3 + 0];
      v1.y = verts[faces[fidx * 3 + 1] * 3 + 1];
      v1.z = verts[faces[fidx * 3 + 1] * 3 + 2];
      float3 v2;
      v2.x = verts[faces[fidx * 3 + 2] * 3 + 0];
      v2.y = verts[faces[fidx * 3 + 2] * 3 + 1];
      v2.z = verts[faces[fidx * 3 + 2] * 3 + 2];

      bool tria_inter = intersection_triangle_voxel(vx_c, vx_w, v0, v1, v2);
      if(tria_inter) {
        tinds.push_back(fidx);
      }
    }
  }


  virtual bool is_occupied(float cx, float cy, float cz, float vd, float vh, float vw, int gd, int gh, int gw, OctreeCreateHelperCpu* helper_) {
    OctreeCreateFromMeshHelperCpu* helper = dynamic_cast<OctreeCreateFromMeshHelperCpu*>(helper_);
    int grid_idx = helper->get_grid_idx(gd, gh, gw);
    std::vector<int>& tinds = helper->tinds[grid_idx];

    for(size_t idx = 0; idx < tinds.size(); ++idx) {
      int fidx = tinds[idx];
      float3 vx_c;
      vx_c.x = cx;
      vx_c.y = cy;
      vx_c.z = cz;
      float3 vx_w;
      vx_w.x = vw;
      vx_w.y = vh;
      vx_w.z = vd;

      float3 v0;
      v0.x = verts[faces[fidx * 3 + 0] * 3 + 0];
      v0.y = verts[faces[fidx * 3 + 0] * 3 + 1];
      v0.z = verts[faces[fidx * 3 + 0] * 3 + 2];
      float3 v1;
      v1.x = verts[faces[fidx * 3 + 1] * 3 + 0];
      v1.y = verts[faces[fidx * 3 + 1] * 3 + 1];
      v1.z = verts[faces[fidx * 3 + 1] * 3 + 2];
      float3 v2;
      v2.x = verts[faces[fidx * 3 + 2] * 3 + 0];
      v2.y = verts[faces[fidx * 3 + 2] * 3 + 1];
      v2.z = verts[faces[fidx * 3 + 2] * 3 + 2]; 

      // printf("[%f,%f,%f] inter [%f,%f,%f], [%f,%f,%f], [%f,%f,%f]\n",
      //     vx_c.x, vx_c.y, vx_c.z,
      //     v0.x, v0.y, v0.z,
      //     v1.x, v1.y, v1.z,
      //     v2.x, v2.y, v2.z);

      bool tria_inter = intersection_triangle_voxel(vx_c, vx_w, v0, v1, v2);
      if(tria_inter) {
        // printf("  triang intersection at (%f,%f,%f)\n", vx_c.x,vx_c.y,vx_c.z);
        return true;
      }
    }
    return false;
  }

  virtual void get_data(bool oc, float cx, float cy, float cz, float vd, float vh, float vw, int gd, int gh, int gw, OctreeCreateHelperCpu* helper, ot_data_t* dst) {
    if(oc) {
      dst[0] = 1;
    }
    else {
      dst[0] = 0;
    }
  }

  void rescale() {
    float min_x = 1e9;  float min_y = 1e9;  float min_z = 1e9;
    float max_x = -1e9; float max_y = -1e9; float max_z = -1e9;
    for(int fidx = 0; fidx < n_faces; ++fidx) {
      for(int vidx = 0; vidx < 3; ++vidx) {
        min_x = FMIN(min_x, verts[faces[fidx * 3 + vidx] * 3 + 0]);
        min_y = FMIN(min_y, verts[faces[fidx * 3 + vidx] * 3 + 1]);
        min_z = FMIN(min_z, verts[faces[fidx * 3 + vidx] * 3 + 2]);

        max_x = FMAX(max_x, verts[faces[fidx * 3 + vidx] * 3 + 0]);
        max_y = FMAX(max_y, verts[faces[fidx * 3 + vidx] * 3 + 1]);
        max_z = FMAX(max_z, verts[faces[fidx * 3 + vidx] * 3 + 2]);
      }
    }

    // rescale vertices
    printf("bb before rescaling [%f,%f], [%f,%f], [%f,%f]\n",
        min_x, max_x, min_y, max_y, min_z, max_z);

    float src_width = FMAX(max_x - min_x, FMAX(max_y - min_y, max_z - min_z));
    float dst_width = FMIN(depth - 2*pad, FMIN(height - 2*pad, width - 2*pad));
    float o_ctr_x = (max_x + min_x)/2.f; float n_ctr_x = width/2.f; 
    float o_ctr_y = (max_y + min_y)/2.f; float n_ctr_y = height/2.f;
    float o_ctr_z = (max_z + min_z)/2.f; float n_ctr_z = depth/2.f;
    for(int vidx = 0; vidx < n_verts; ++vidx) {
      verts[vidx * 3 + 0] = (verts[vidx * 3 + 0] - o_ctr_x) / src_width * dst_width + n_ctr_x;
      verts[vidx * 3 + 1] = (verts[vidx * 3 + 1] - o_ctr_y) / src_width * dst_width + n_ctr_y;
      verts[vidx * 3 + 2] = (verts[vidx * 3 + 2] - o_ctr_z) / src_width * dst_width + n_ctr_z;
    }

    printf("bb after rescaling [%f,%f], [%f,%f], [%f,%f]\n",
        (min_x - o_ctr_x) / src_width * dst_width + n_ctr_x,
        (max_x - o_ctr_x) / src_width * dst_width + n_ctr_x,
        (min_y - o_ctr_y) / src_width * dst_width + n_ctr_y,
        (max_y - o_ctr_y) / src_width * dst_width + n_ctr_y,
        (min_z - o_ctr_z) / src_width * dst_width + n_ctr_z,
        (max_z - o_ctr_z) / src_width * dst_width + n_ctr_z);
  }


protected:
  const ot_size_t depth;
  const ot_size_t height;
  const ot_size_t width;

  ot_size_t n_verts;
  float* verts;
  ot_size_t n_faces;
  int* faces;

  int pad;
};


#endif 
