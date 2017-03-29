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

#ifndef OCTREE_Z_CURVE_H
#define OCTREE_Z_CURVE_H

#include "core.h"

/// Function used to efficiently iterate the surface of a shallow octree cell.
/// Instead of two loops for the surface iteration, one flatten z index is enough
/// and this function computes the corresponding x index. The z-curve perserves
/// the data locality, which can be used for efficient cashing.
///
/// @param z
/// @return x
/// @see https://en.wikipedia.org/wiki/Z-order_curve
OCTREE_FUNCTION
inline int z_curve_x(const int z) {
  int x = z & 0x55555555;
  x = (x ^ (x >> 1)) & 0x33333333;
  x = (x ^ (x >> 2)) & 0x0f0f0f0f;
  return x;
}

/// Returns the corresponding y value of the z value.
///
/// @param z
/// @return y
/// @see https://en.wikipedia.org/wiki/Z-order_curve
/// @see z_curve_x
OCTREE_FUNCTION
inline int z_curve_y(const int z) {
  return z_curve_x(z >> 1);
}


#endif
