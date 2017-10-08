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

#include <vector>
#include <cstring>

#if defined(_OPENMP)
#include <omp.h>
#endif


class OctreeCreateFromPCHelperCpu : public OctreeCreateHelperCpu {
public:
  OctreeCreateFromPCHelperCpu(ot_size_t grid_depth_, ot_size_t grid_height_, ot_size_t grid_width_) :
    OctreeCreateHelperCpu(grid_depth_, grid_height_, grid_width_), 
    xyz_inds(grid_depth_ * grid_height_ * grid_width_)
  {}
  virtual ~OctreeCreateFromPCHelperCpu() {}

public:
  std::vector<std::vector<int> > xyz_inds;
};



class OctreeFromPC : public OctreeCreateCpu {
public:
  OctreeFromPC(float* xyz_, const float* features_, int n_pts_, int feature_size_, ot_size_t depth_, ot_size_t height_, ot_size_t width_, bool normalize, bool normalize_inplace, int pad_) : 
      OctreeCreateCpu((depth_ + 7) / 8, (height_ + 7) / 8, (width_ + 7) / 8, feature_size_), 
      depth(depth_), height(height_), width(width_), features(features_), n_pts(n_pts_), pad(pad_) {
    
    xyz = xyz_;
    owns_xyz = false;
    if(normalize) {
      normalize_points(xyz_, n_pts_, normalize_inplace);
    }
  }

  OctreeFromPC(float* xyz_, int n_pts_, int feature_size_, ot_size_t depth_, ot_size_t height_, ot_size_t width_, bool normalize, bool normalize_inplace, int pad_) : 
      OctreeCreateCpu((depth_ + 7) / 8, (height_ + 7) / 8, (width_ + 7) / 8, feature_size_), 
      depth(depth_), height(height_), width(width_), features(0), n_pts(n_pts_), pad(pad_) {
    
    xyz = xyz_;
    owns_xyz = false;
    if(normalize) {
      normalize_points(xyz_, n_pts_, normalize_inplace);
    }
  }

  virtual ~OctreeFromPC() {
    if(owns_xyz) {
      delete[] xyz;
    }
  }

  
  virtual octree* operator()(bool fit, int fit_multiply, bool pack, int n_threads) {
    //determine block triangle intersections
    int n_blocks = grid_depth * grid_height * grid_width;
    printf("  [OctreeFromPC] determine block pts intersections for grid %d,%d,%d - %d pts\n", grid_depth, grid_height, grid_width, n_pts);
    OctreeCreateFromPCHelperCpu helper(grid_depth, grid_height, grid_width);

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
      block_pts(cx,cy,cz, 8,8,8, helper.xyz_inds[grid_idx]);
    }

    return create_octree(fit, fit_multiply, pack, n_threads, &helper);
  }

  virtual void block_pts(float cx, float cy, float cz, float vd, float vh, float vw, std::vector<int>& xyz_inds) {
    float min_x = cx - vw/2;
    float min_y = cy - vh/2;
    float min_z = cz - vd/2;
    float max_x = cx + vw/2;
    float max_y = cy + vh/2;
    float max_z = cz + vd/2;

    for(int xyz_idx = 0; xyz_idx < n_pts; ++xyz_idx) {
      float x = xyz[xyz_idx * 3 + 0];
      float y = xyz[xyz_idx * 3 + 1];
      float z = xyz[xyz_idx * 3 + 2];

      if(x >= min_x && y >= min_y && z >= min_z && x <= max_x && y <= max_y && z <= max_z) {
        xyz_inds.push_back(xyz_idx);
      }
    }
  }

  virtual bool is_occupied(float cx, float cy, float cz, float vd, float vh, float vw, int gd, int gh, int gw, OctreeCreateHelperCpu* helper_) {
    OctreeCreateFromPCHelperCpu* helper = dynamic_cast<OctreeCreateFromPCHelperCpu*>(helper_);
    std::vector<int>& xyz_inds = helper->xyz_inds[helper->get_grid_idx(gd, gh, gw)];

    float min_x = cx - vw/2;
    float min_y = cy - vh/2;
    float min_z = cz - vd/2;
    float max_x = cx + vw/2;
    float max_y = cy + vh/2;
    float max_z = cz + vd/2;

    for(size_t idx = 0; idx < xyz_inds.size(); ++idx) {
      int xyz_idx = (xyz_inds)[idx];
      float x = xyz[xyz_idx * 3 + 0];
      float y = xyz[xyz_idx * 3 + 1];
      float z = xyz[xyz_idx * 3 + 2];

      bool inter = x >= min_x && y >= min_y && z >= min_z && x <= max_x && y <= max_y && z <= max_z;
      if(inter) {
        return true;
      }
    }
    return false;
  }

  virtual void get_data(bool oc, float cx, float cy, float cz, float vd, float vh, float vw, int gd, int gh, int gw, OctreeCreateHelperCpu* helper_, ot_data_t* dst) {
    OctreeCreateFromPCHelperCpu* helper = dynamic_cast<OctreeCreateFromPCHelperCpu*>(helper_);
    std::vector<int>& xyz_inds = helper->xyz_inds[helper->get_grid_idx(gd, gh, gw)];
    
    if(features == 0) {
      for(int f = 0; f < feature_size; ++f) {
        dst[f] = oc ? 1 : 0;
      }
    }
    else {
      for(int f = 0; f < feature_size; ++f) {
        dst[f] = 0;
      }
      
      //if occupied compute average of features
      if(oc) {
        float min_x = cx - vw/2;
        float min_y = cy - vh/2;
        float min_z = cz - vd/2;
        float max_x = cx + vw/2;
        float max_y = cy + vh/2;
        float max_z = cz + vd/2;

        int n = 0;
        for(size_t idx = 0; idx < xyz_inds.size(); ++idx) {
          int xyz_idx = (xyz_inds)[idx];
          float x = xyz[xyz_idx * 3 + 0];
          float y = xyz[xyz_idx * 3 + 1];
          float z = xyz[xyz_idx * 3 + 2];

          bool inter = x >= min_x && y >= min_y && z >= min_z && x <= max_x && y <= max_y && z <= max_z;
          if(inter) {
            for(int f = 0; f < feature_size; ++f) {
              dst[f] += features[xyz_idx * feature_size + f];
            }
            n++;
          }
        }
        
        for(int f = 0; f < feature_size; ++f) {
          dst[f] /= n;
        }
      }
    }
  }

protected:
  virtual void normalize_points(float* xyz_, int n_pts_, bool normalize_inplace) {
    if(normalize_inplace) {
      printf("  [OctreeFromPC] normalize inplace\n");
      xyz = xyz_;
      owns_xyz = false;
    }
    else {
      printf("  [OctreeFromPC] normalize\n");
      xyz = new float[3 * n_pts_];
      memcpy(xyz, xyz_, sizeof(float) * 3 * n_pts_);
      owns_xyz = true;
    }

    // normalize points
    float x, y, z;
    float min_x = 1e9;  float min_y = 1e9;  float min_z = 1e9;
    float max_x = -1e9; float max_y = -1e9; float max_z = -1e9;
    for(int idx = 0; idx < n_pts_; ++idx) {
      x = xyz[idx * 3 + 0]; 
      y = xyz[idx * 3 + 1]; 
      z = xyz[idx * 3 + 2]; 
      
      min_x = FMIN(min_x, x);
      min_y = FMIN(min_y, y);
      min_z = FMIN(min_z, z);
      max_x = FMAX(max_x, x);
      max_y = FMAX(max_y, y);
      max_z = FMAX(max_z, z);
    }

    float src_width = FMAX(max_x - min_x, FMAX(max_y - min_y, max_z - min_z));
    // float dst_width = FMIN(depth, FMIN(height, width));
    float dst_width = FMIN(depth - 2*pad, FMIN(height - 2*pad, width - 2*pad));
    float o_ctr_x = (max_x + min_x)/2.f; float n_ctr_x = width/2.f; 
    float o_ctr_y = (max_y + min_y)/2.f; float n_ctr_y = height/2.f;
    float o_ctr_z = (max_z + min_z)/2.f; float n_ctr_z = depth/2.f;
    for(int idx = 0; idx < n_pts_; ++idx) {
      xyz[idx * 3 + 0] = (xyz[idx * 3 + 0] - o_ctr_x) / src_width * dst_width + n_ctr_x;
      xyz[idx * 3 + 1] = (xyz[idx * 3 + 1] - o_ctr_y) / src_width * dst_width + n_ctr_y;
      xyz[idx * 3 + 2] = (xyz[idx * 3 + 2] - o_ctr_z) / src_width * dst_width + n_ctr_z;
    }

    // min_x = 1e9;  min_y = 1e9;  min_z = 1e9;
    // max_x = -1e9; max_y = -1e9; max_z = -1e9;
    // for(int idx = 0; idx < n_pts_; ++idx) {
    //   x = xyz[idx * 3 + 0]; 
    //   y = xyz[idx * 3 + 1]; 
    //   z = xyz[idx * 3 + 2]; 
      
    //   min_x = FMIN(min_x, x);
    //   min_y = FMIN(min_y, y);
    //   min_z = FMIN(min_z, z);
    //   max_x = FMAX(max_x, x);
    //   max_y = FMAX(max_y, y);
    //   max_z = FMAX(max_z, z);
    // }
  }

  const ot_size_t depth;
  const ot_size_t height;
  const ot_size_t width;

  bool owns_xyz;
  float* xyz;
  const float* features;
  int n_pts;

  int pad;
};


extern "C"
octree* octree_create_from_pc_simple_cpu(float* xyz, int n_pts, int feature_size, ot_size_t depth, ot_size_t height, ot_size_t width, bool normalize, bool normalize_inplace, bool fit, int fit_multiply, bool pack, int pad, int n_threads) {
  printf("pc simple - n_pts: %d, feature_size: %d\n", n_pts, feature_size);
  OctreeFromPC create(xyz, n_pts, feature_size, depth, height, width, normalize, normalize_inplace, pad);
  printf("create octree\n");
  return create(fit, fit_multiply, pack, n_threads);
}

extern "C"
octree* octree_create_from_pc_cpu(float* xyz, const float* features, int n_pts, int feature_size, ot_size_t depth, ot_size_t height, ot_size_t width, bool normalize, bool normalize_inplace, bool fit, int fit_multiply, bool pack, int pad, int n_threads) {
  printf("pc - n_pts: %d, feature_size: %d\n", n_pts, feature_size);
  OctreeFromPC create(xyz, features, n_pts, feature_size, depth, height, width, normalize, normalize_inplace, pad);
  printf("create octree\n");
  return create(fit, fit_multiply, pack, n_threads);
}


extern "C"
void octree_create_dense_from_pc_cpu(const float* xyz, const float* features, float* vol, int n_pts, int feature_size, ot_size_t depth, ot_size_t height, ot_size_t width, int n_threads) {
#if defined(_OPENMP)
  omp_set_num_threads(n_threads);
#endif

  int* norm = new int[depth * height * width];
  #pragma omp parallel for
  for(int idx = 0; idx < depth * height * width; ++idx) {
    norm[idx] = 0;
    for(int f = 0; f < feature_size; ++f) {
      vol[f * depth * height * width + idx] = 0;
    }
  }
  
  #pragma omp parallel for
  for(int pt_idx = 0; pt_idx < n_pts; ++pt_idx) {
    int w = xyz[pt_idx * 3 + 0];
    int h = xyz[pt_idx * 3 + 1];
    int d = xyz[pt_idx * 3 + 2];
    if(w < 0 || w >= width || h < 0 || h >= height || d < 0 || d >= depth) {
      printf("[WARNING] pt_%d=(%d,%d,%d) is out of volume %d,%d,%d; (%f,%f,%f\n", pt_idx, d,h,w, depth,height,width, xyz[pt_idx * 3 + 0],xyz[pt_idx * 3 + 1],xyz[pt_idx * 3 + 2]);
      continue;
    }

    #pragma omp atomic 
    norm[(d * height + h) * width + w] += 1;
    for(int f = 0; f < feature_size; ++f) {
      int vol_idx = ((f * depth + d) * height + h) * width + w;
      #pragma omp atomic 
      vol[vol_idx] += features[pt_idx * feature_size + f];
    }
  }
  
  #pragma omp parallel for
  for(int idx = 0; idx < depth * height * width; ++idx) {
    int n = norm[idx];
    if(n > 0) {
      for(int f = 0; f < feature_size; ++f) {
        vol[f * depth * height * width + idx] /= n;
      }
    }
  }

  delete[] norm;
}

