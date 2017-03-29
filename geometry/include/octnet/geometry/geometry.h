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

#ifndef GEOMETRY_H
#define GEOMETRY_H

#ifdef __CUDA_ARCH__
#define FMIN(a,b) fminf(a,b)
#define FMAX(a,b) fmaxf(a,b)
#define IMIN(a,b) min(a,b)
#define IMAX(a,b) max(a,b)
#else
#define FMIN(a,b) fminf(a,b)
#define FMAX(a,b) fmaxf(a,b)
#define IMIN(a,b) (((a)<(b))?(a):(b))
#define IMAX(a,b) (((a)>(b))?(a):(b))
#endif

#define EPS 1e-9

#include "extern/aabb_triang_akenine-moeller.h"

#ifndef __CUDA_ARCH__
#include <cmath>


typedef struct {
  float x;
  float y;
  float z;
  float& operator[](int idx) {
    return idx <= 0 ? x : (idx == 1 ? y : z);
  }
} float3;

#endif

inline float3 vec_add(float3 a, float3 b) {
  float3 r;
  r.x = a.x + b.x;
  r.y = a.y + b.y;
  r.z = a.z + b.z;
  return r;
}

inline float3 vec_sub(float3 a, float3 b) {
  float3 r;
  r.x = a.x - b.x;
  r.y = a.y - b.y;
  r.z = a.z - b.z;
  return r;
}

inline float vec_dot(float3 a, float3 b) {
  return a.x*b.x + a.y*b.y + a.z*b.z;
}

inline float3 vec_cross(float3 u, float3 v) {
  float3 r;
  r.x = u.y * v.z - u.z * v.y;
  r.y = u.z * v.x - u.x * v.z;
  r.z = u.x * v.y - u.y * v.x;
  return r;
}


template <typename T>
inline void swap(T& x, T& y) {
  T tmp(x);
  x = y;
  y = tmp;
}

inline bool intersection_plane_voxel(float3 vx_c, float3 vx_w, float3 pn, float pd) {
  float3 e; //pos extents
  e.x = fabs(vx_w.x * 0.5);
  e.y = fabs(vx_w.y * 0.5);
  e.z = fabs(vx_w.z * 0.5);

  float r = e.x * fabs(pn.x) + e.y * fabs(pn.y) + e.z * fabs(pn.z);
  float s = vec_dot(vx_c, pn) - pd;
  return fabs(s) <= r;
}



inline bool intersection_triangle_voxel(float3 vx_c, float3 vx_w, float3 a, float3 b, float3 c) {
  float boxcenter[3] = {vx_c.x, vx_c.y, vx_c.z};
  float boxhalfsize[3] = {vx_w.x/2.f, vx_w.y/2.f, vx_w.z/2.f};
  float triverts[3][3] = { {a.x, a.y, a.z}, {b.x, b.y, b.z}, {c.x, c.y, c.z}};
  return triBoxOverlap(boxcenter, boxhalfsize, triverts);
}



inline bool intersection_ray_voxel(float3 p, float3 d, float3 vx, float3 vx_w, float& tmin) {
  tmin = -1e9;
  float tmax = 1e9;

  float t1, t2;

  float vx_min = vx.x - fabs(vx_w.x)/2.f;
  float vx_max = vx.x + fabs(vx_w.x)/2.f;
  if(fabs(d.x) < EPS) {
    if (p.x < vx_min || p.x > vx_max) {
      // printf("false - case 0\n");
      return false;
    }
  }
  else {
    t1 = (vx_min - p.x) / d.x;
    t2 = (vx_max - p.x) / d.x;
    if(t1 > t2) {
      swap(t1, t2);
    }
    tmin = FMAX(t1, tmin);
    tmax = FMIN(t2, tmax);
    if(tmin > tmax) {
      // printf("false - case 1\n");
      return false;
    }
  }

  vx_min = vx.y - fabs(vx_w.y)/2.f;
  vx_max = vx.y + fabs(vx_w.y)/2.f;
  if(fabs(d.y) < EPS) {
    if (p.y < vx_min || p.y > vx_max) {
      // printf("false - case 2\n");
      return false;
    }
  }
  else {
    t1 = (vx_min - p.y) / d.y;
    t2 = (vx_max - p.y) / d.y;
    if(t1 > t2) {
      swap(t1, t2);
    }
    tmin = FMAX(t1, tmin);
    tmax = FMIN(t2, tmax);
    if(tmin > tmax) {
      // printf("false - case 3\n");
      return false;
    }
  }

  vx_min = vx.z - fabs(vx_w.z)/2.f;
  vx_max = vx.z + fabs(vx_w.z)/2.f;
  if(fabs(d.z) < EPS) {
    if (p.z < vx_min || p.z > vx_max) {
      // printf("false - case 3\n");
      return false;
    }
  }
  else {
    t1 = (vx_min - p.z) / d.z;
    t2 = (vx_max - p.z) / d.z;
    if(t1 > t2) {
      swap(t1, t2);
    }
    tmin = FMAX(t1, tmin);
    tmax = FMIN(t2, tmax);
    if(tmin > tmax) {
      // printf("false - case 4\n");
      return false;
    }
  }

  return true;
}


#endif
