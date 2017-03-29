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


inline void split(const std::string &s, char delim, std::vector<std::string> &elems) {
  std::stringstream ss;
  ss.str(s);
  std::string item;
  while (std::getline(ss, item, delim)) {
    elems.push_back(item);
  }
}
inline std::vector<std::string> split(const std::string &s, char delim) {
  std::vector<std::string> elems;
  split(s, delim, elems);
  return elems;
}

inline bool starts_with(const std::string& str, const std::string& with) {
  if(str.length() < with.length()) {
    return false;
  }

  for(size_t idx = 0; idx < with.length(); ++idx) {
    if(str[idx] != with[idx]) {
      return false;
    }
  }

  return true;
}


extern "C"
octree* octree_create_from_obj_cpu(const char* path, ot_size_t depth, ot_size_t height, ot_size_t width, const float R[9], bool fit, int fit_multiply, bool pack, int pad, int n_threads) {

  printf("[INFO] parse obj file\n");
  std::vector<float> verts;
  std::vector<int> faces;

  int line_nb = 1;
  std::string line;
  std::stringstream ss;
  std::ifstream file(path);
  while(std::getline(file, line)) {
    std::vector<std::string> splits = split(line, ' ');

    const std::string type = splits[0];

    if(type == "v") {
      if(splits.size() != 4) {
        printf("[ERROR] invalid vertice on line %d\n", line_nb);
        exit(-1);
      }
      float x_,y_,z_;
      ss.clear(); ss.str(""); ss << splits[1]; ss >> x_; 
      ss.clear(); ss.str(""); ss << splits[2]; ss >> y_; 
      ss.clear(); ss.str(""); ss << splits[3]; ss >> z_; 

      float x = R[0] * x_ + R[1] * y_ + R[2] * z_;
      float y = R[3] * x_ + R[4] * y_ + R[5] * z_;
      float z = R[6] * x_ + R[7] * y_ + R[8] * z_;

      verts.push_back(x);
      verts.push_back(y);
      verts.push_back(z);
    }
    else if(type == "f") {
      if(splits.size() != 4) {
        printf("[ERROR] invalid face on line %d\n", line_nb);
        exit(-1);
      }
      int v_idx;
      ss.clear(); ss.str(""); ss << splits[1]; ss >> v_idx; faces.push_back(v_idx - 1);
      ss.clear(); ss.str(""); ss << splits[2]; ss >> v_idx; faces.push_back(v_idx - 1);
      ss.clear(); ss.str(""); ss << splits[3]; ss >> v_idx; faces.push_back(v_idx - 1);
      // if(line_nb >= 10 && line_nb <= 45) {
      //   printf("f %d %d %d \n", faces[faces.size()-3], faces[faces.size()-2], faces[faces.size()-1]);
      // }
    }
    else if(type == "o") {
      // DO NOTHING
    }
    else if(type == "s") {
      // DO NOTHING
    }
    else if(type == "vn") {
      // DO NOTHING
    }
    else if(type == "vt") {
      // DO NOTHING
    }
    else if(type == "g") {
      // DO NOTHING
    }
    else if(type == "#") {
      // DO NOTHING
    }
    else if(starts_with(type, "usemtl")) {
      // DO NOTHING
    }
    else if(starts_with(type, "mtllib")) {
      // DO NOTHING
    }
    else {
      printf("[ERROR] unknown type %s on line %d\n", type.c_str(), line_nb);
      exit(-1);
    }

    line_nb++;
  }

  file.close();

  int n_verts = verts.size() / 3;
  int n_faces = faces.size() / 3;
  printf("[INFO] parsed %d vertices and %d faces (%d lines in total)\n", n_verts, n_faces, (n_verts + n_faces));

  bool rescale = true;
  printf("[INFO] create octree from mesh\n");
  octree* grid = octree_create_from_mesh_cpu(n_verts, &(verts[0]), n_faces, &(faces[0]), rescale, depth, height, width, fit, fit_multiply, pack, pad, n_threads);

  return grid;
}





