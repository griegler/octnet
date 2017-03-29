#ifndef OCTREE_CREATE_UTILS_H
#define OCTREE_CREATE_UTILS_H

#include "octnet/core/core.h"

extern "C" {

void octree_scanline_fill(octree* grid, ot_data_t fill_value);

void octree_occupancy_to_surface(octree* in, octree* out);

/// Converts a dense occupancy representation to a dense surface representation
/// using a 6-neighbourhood relationship. The n_iter indicates how thick
/// the surface should get (i.e. one iteration adds one voxel in each direction 
/// of the surface, 1 => normal surface).
/// @param dense DxHxW array of 0/1 values
/// @param depth
/// @param height
/// @param width
/// @param surface_width
/// @param surface
void dense_occupancy_to_surface(const ot_data_t* dense, int depth, int height, int width, int n_iter, ot_data_t* surface);

}

#endif
