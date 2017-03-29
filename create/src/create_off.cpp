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

#include "octnet/create/create_from_mesh.h"


extern "C"
octree* octree_create_from_off_cpu(const char* path, ot_size_t depth, ot_size_t height, ot_size_t width, const float R[9], bool fit, int fit_multiply, bool pack, int pad, int n_threads) {
  std::ifstream file(path);
  std::string line;
  std::stringstream ss;
  int line_nb = 0;

  printf("[INFO] parse off file\n");

  //parse header
  std::getline(file, line); ++line_nb;
  if(line != "off" && line != "OFF") {
    std::cout << "invalid header: " << line << std::endl;
    std::cout << path << std::endl;
    exit(-1);
  }

  //parse n vertices, n faces
  size_t n_verts, n_faces;
  std::getline(file, line); ++line_nb;
  ss << line;
  ss >> n_verts;
  ss >> n_faces;
  int dummy;
  ss >> dummy;

  //reserve memory for vertices and triangs
  std::vector<float> verts;
  std::vector<int> faces;

  //parse vertices
  float x,y,z;
  float x_,y_,z_;
  for(size_t idx = 0; idx < n_verts; ++idx) {
    std::getline(file, line); ++line_nb;
    ss.clear(); ss.str("");
    ss << line;
    ss >> x_;  
    ss >> y_;  
    ss >> z_;  

    x = R[0] * x_ + R[1] * y_ + R[2] * z_;
    y = R[3] * x_ + R[4] * y_ + R[5] * z_;
    z = R[6] * x_ + R[7] * y_ + R[8] * z_;

    verts.push_back(x);
    verts.push_back(y);
    verts.push_back(z);
  }

  //parse faces
  for(size_t idx = 0; idx < n_faces; ++idx) {
    std::getline(file, line); ++line_nb;
    ss.clear(); ss.str("");
    ss << line;
    ss >> dummy;
    if(dummy != 3) {
      std::cout << "not a triangle, has " << dummy << " pts" << std::endl;
      exit(-1);
    }

    ss >> dummy; faces.push_back(dummy);
    ss >> dummy; faces.push_back(dummy);
    ss >> dummy; faces.push_back(dummy);
  }

  if(n_verts != verts.size() / 3) {
    std::cout << "n_verts in header differs from actual n_verts" << std::endl;
    exit(-1);
  }
  if(n_faces != faces.size() / 3) {
    std::cout << "n_faces in header differs from actual n_faces" << std::endl;
    exit(-1);
  }

  file.close();

  bool rescale = true;
  printf("[INFO] create octree from mesh\n");
  octree* grid = octree_create_from_mesh_cpu(n_verts, &(verts[0]), n_faces, &(faces[0]), rescale, depth, height, width, fit, fit_multiply, pack, pad, n_threads);

  return grid;
}





