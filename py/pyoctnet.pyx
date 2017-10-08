# Copyright (c) 2017, The OctNet authors
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the <organization> nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL OCTNET AUTHORS BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

cimport cython
import numpy as np
cimport numpy as np

from libc.stdlib cimport free, malloc
from libcpp cimport bool
from cpython cimport PyObject, Py_INCREF

CREATE_INIT = True # workaround, so cython builds a init function

np.import_array() 

"""
Lightweight wrapper class for native float arrays. Allows numpy style access.
"""
cdef class FloatArrayWrapper:
  """ native float array that is encapsulated. """
  cdef float* data_ptr
  """ size/length of the float array. """
  cdef int size
  """ indicates owner ship of the float array, if true, free array in destructor. """
  cdef int owns

  """
  Set the native data array for this class.
  @param data_ptr native float array
  @param size length of the array
  @param owns if True, the object destructor frees the array
  """
  cdef set_data(self, float* data_ptr, int size, int owns):
    self.data_ptr = data_ptr
    self.size = size
    self.owns = owns

  """
  Method, which is called by np.array to encapsulate the data provided in data_ptr.
  """
  def __array__(self):
    cdef np.npy_intp shape[1]
    shape[0] = <np.npy_intp> self.size
    ndarray = np.PyArray_SimpleNewFromData(1, shape, np.NPY_FLOAT, self.data_ptr)
    return ndarray

  """
  Destructor. Calls free on data array if it is owned (owns == True) by the wrapper.
  """
  def __dealloc__(self):
    cdef float* ptr = self.data_ptr
    if self.owns != 0:
      free(<void*>ptr)




cdef extern from "../core/include/octnet/core/core.h":
  ctypedef int ot_size_t;
  ctypedef float ot_data_t;
  ctypedef int ot_tree_t;
  ctypedef struct octree:
    ot_size_t n;            
    ot_size_t grid_depth;
    ot_size_t grid_height;
    ot_size_t grid_width;
    ot_size_t feature_size;  
    ot_size_t n_leafs;       
    ot_tree_t* trees;        
    ot_size_t* prefix_leafs; 
    ot_data_t* data;         
    ot_size_t grid_capacity;
    ot_size_t data_capacity;

  ot_tree_t* octree_get_tree(const octree* grid, ot_size_t grid_idx);
  int octree_grid_idx(const octree* grid, const int gn, const int gd, const int gh, const int gw);
  int tree_child_bit_idx(const int bit_idx);
  int tree_parent_bit_idx(const int bit_idx);

  void tree_set_bit(ot_tree_t* num, int pos);
  bool tree_isset_bit(const ot_tree_t* num, const int pos);
  int tree_n_leafs(const ot_tree_t* tree);
  int tree_n_splits(const ot_tree_t* tree);
  int tree_data_idx(const ot_tree_t* tree, const int bit_idx, ot_size_t feature_size);
  
  int octree_mem_capacity(const octree* grid);
  int octree_mem_using(const octree* grid);


cdef extern from "../core/include/octnet/cpu/cpu.h":
  void octree_print_cpu(const octree* grid_h);
  octree* octree_new_cpu();
  void octree_free_cpu(octree* grid_h);
  void octree_resize_cpu(int n, int grid_depth, int grid_height, int grid_width, int feature_size, int n_leafs, octree* dst);
  void octree_copy_cpu(const octree* src, octree* dst);
  void octree_upd_n_leafs_cpu(octree* grid_h);
  void octree_upd_prefix_leafs_cpu(octree* grid_h);
  bool octree_equal_cpu(const octree* in1, const octree* in2);

cdef extern from "../core/include/octnet/cpu/dense.h":
  void octree_to_dhwc_cpu(const octree* grid_h, const int dense_depth, const int dense_height, const int dense_width, ot_data_t* data);
  void octree_to_cdhw_cpu(const octree* grid_h, const int dense_depth, const int dense_height, const int dense_width, ot_data_t* data);
  void cdhw_to_octree_avg_cpu(const octree* grid_h_in, const int dense_depth, const int dense_height, const int dense_width, const ot_data_t* data, int out_feature_size, octree* grid_h_out);

cdef extern from "../core/include/octnet/cpu/io.h":
  void octree_read_deprecated_cpu(const char* path, octree* grid_h);
  void dense_read_prealloc_deprecated_cpu(const char* path, int n_dim, const int* dims, ot_data_t* data);
  int* dense_read_header_deprecated_cpu(const char* path, int* n_dim);
  
  void octree_read_header_cpu(const char* path, octree* grid_h);
  void octree_read_cpu(const char* path, octree* grid_h);
  void octree_write_cpu(const char* path, const octree* grid_h);
  void octree_dhwc_write_cpu(const char* path, const octree* grid_h);
  void octree_cdhw_write_cpu(const char* path, const octree* grid_h);

  ot_data_t* dense_read_cpu(const char* path, int n_dim);
  void dense_read_prealloc_cpu(const char* path, int n_dim, const int* dims, ot_data_t* data);
  void dense_write_cpu(const char* path, int n_dim, const int* dims, const ot_data_t* data);


cdef extern from "../core/include/octnet/cpu/misc.h":
  void octree_determine_gt_split_cpu(const octree* struc, const ot_data_t* gt, octree* out);

cdef extern from "../core/include/octnet/cpu/split.h":
  void octree_split_dense_reconstruction_surface_fres_cpu(const ot_data_t* features, const ot_data_t* reconstruction, int n, int dense_depth, int dense_height, int dense_width, int feature_size, ot_data_t rec_thr_from, ot_data_t rec_thr_to, int band, octree* out);

cdef extern from "../core/include/octnet/cpu/pool.h":
  void octree_gridpool2x2x2_max_cpu(const octree* in_oc, octree* out);

cdef extern from "../core/include/octnet/cpu/unpool.h":
  void octree_gridunpool2x2x2_cpu(const octree* in_oc, octree* out);
  void octree_gridunpoolguided2x2x2_cpu(const octree* in_oc, const octree* in_struct, octree* out);

cdef extern from "../core/include/octnet/cpu/conv.h":
  void octree_conv3x3x3_avg_cpu(const octree* grid_in_h, const ot_data_t* weights, const ot_data_t* bias, int channels_out, octree* grid);

cdef extern from "../core/include/octnet/cpu/combine.h":
  void octree_extract_feature_cpu(const octree* grid_in, int feature_from, int feature_to, octree* out);

cdef extern from "../create/include/octnet/create/create.h":
  octree* octree_create_from_dense_cpu(const ot_data_t* data, int feature_size, int depth, int height, int width, bool fit, int fit_multiply, bool pack, int n_threads);
  octree* octree_create_from_dense2_cpu(const ot_data_t* occupancy, const ot_data_t* features, int feature_size, int depth, int height, int width, bool fit, int fit_multiply, bool pack, int n_threads);
  octree* octree_create_from_mesh_cpu(int n_verts_, float* verts_, int n_faces_, int* faces, bool rescale_verts, ot_size_t depth, ot_size_t height, ot_size_t width, bool fit, int fit_multiply, bool pack, int pad, int n_threads);
  octree* octree_create_from_off_cpu(const char* path, ot_size_t depth, ot_size_t height, ot_size_t width, const float R[9], bool fit, int fit_multiply, bool pack, int pad, int n_threads);
  octree* octree_create_from_obj_cpu(const char* path, ot_size_t depth, ot_size_t height, ot_size_t width, const float R[9], bool fit, int fit_multiply, bool pack, int pad, int n_threads);
  octree* octree_create_from_pc_simple_cpu(float* xyz, int n_pts, int feature_size, ot_size_t depth, ot_size_t height, ot_size_t width, bool normalize, bool normalize_inplace, bool fit, int fit_multiply, bool pack, int pad, int n_threads);
  octree* octree_create_from_pc_cpu(float* xyz, const float* features, int n_pts, int feature_size, ot_size_t depth, ot_size_t height, ot_size_t width, bool normalize, bool normalize_inplace, bool fit, int fit_multiply, bool pack, int pad, int n_threads);
  void octree_create_dense_from_pc_cpu(const float* xyz, const float* features, float* vol, int n_pts, int feature_size, ot_size_t depth, ot_size_t height, ot_size_t width, int n_threads);

cdef extern from "../create/include/octnet/create/utils.h":
  void octree_scanline_fill(octree* grid, ot_data_t fill_value);
  void octree_occupancy_to_surface(octree* inp, octree* out);
  void dense_occupancy_to_surface(const ot_data_t* dense, int depth, int height, int width, int n_iter, ot_data_t* surface);





"""
Simple helper function that converts a python list of integers representing the
shape of an array to an native int array.
@param shape list of int.
@return native int array.
"""
cdef int* npy_shape_to_int_array(shape):
  cdef int* dims = <int*> malloc(len(shape) * sizeof(int))
  for idx in range(len(shape)):
    dims[idx] = shape[idx]
  return dims

"""
Reads a dense tensor from a binary file to a preallocated array.
@param path path to binary file.
@param data 5-dimensional, contiguous data array.
"""
def read_dense(char* path, float[:,:,:,:,::1] data):
  cdef int* dims = npy_shape_to_int_array(data.shape)
  dense_read_prealloc_cpu(path, 5, dims, &(data[0,0,0,0,0]))
  free(dims)

"""
Writes a dense tensor to a binary file.
@param path output path.
@param data data tensor.
"""
def write_dense(char* path, float[:,:,:,:,::1] data):
  cdef int* dims = npy_shape_to_int_array(data.shape)
  dense_write_cpu(path, 5, dims, &(data[0,0,0,0,0]))
  free(dims)

"""
Reads a dense tensor from a binary file to a preallocated array.
@deprecated
@param path path to binary file.
@param data 5-dimensional, contiguous data array.
"""
def read_dense_deprecated(char* path, float[:,:,:,:,::1] data):
  cdef int* dims = npy_shape_to_int_array(data.shape)
  dense_read_prealloc_deprecated_cpu(path, 5, dims, &(data[0,0,0,0,0]))
  free(dims)

"""
Reads the header of a dense tensor from a binary file.
@deprecated
@param path path to binary file.
@return shape of the tensor stored at path.
"""
def read_dense_header_deprecated(char* path):
  cdef int n_dim
  cdef int* dims = dense_read_header_deprecated_cpu(path, &n_dim)
  cdef dims_list = []
  for idx in range(n_dim):
    dims_list.append(dims[idx])
  free(dims)
  return dims_list

def occupancy_to_surface(float[:,:,::1] dense):
  surface = np.empty((dense.shape[0],dense.shape[1],dense.shape[2]), dtype=np.float32)
  cdef float[:,:,::1] surface_view = surface
  dense_occupancy_to_surface(&(dense[0,0,0]), dense.shape[0], dense.shape[1], dense.shape[2], 1, &(surface_view[0,0,0]))
  return surface

"""
Reads the header of the serialized grid-octree structure.
@param path path to the serialized grid-octree structure.
@return n, grid_depth, grid_height, grid_width, feature_size, n_leafs
"""
def read_oc_header(char* path):
  cdef octree* grid = octree_new_cpu()
  octree_read_header_cpu(path, grid)
  cdef ot_size_t n = grid.n
  cdef ot_size_t grid_depth = grid.grid_depth
  cdef ot_size_t grid_height = grid.grid_height
  cdef ot_size_t grid_width = grid.grid_width
  cdef ot_size_t feature_size = grid.feature_size
  cdef ot_size_t n_leafs = grid.n_leafs
  return n, grid_depth, grid_height, grid_width, feature_size, n_leafs


"""
This class encapsulates a pointer to a native `struct octree`.
The class provides several `create_*` methods to create hybrid grid-octree 
data structures from various input sources.

  # To create an empty octree call
  Octree.create_empty()

  # To load a serialized octree call
  Octree.create_from_bin('/path/to/octree.oc')
"""
cdef class Octree:
  """ Pointer to native hybrid grid-octree structure. """
  cdef octree* grid

  """ 
  Set the pointer to native octree wrapped by this object.
  @note the octree_free_cpu is called on destruction of this wrapper, therefore
        freeing all the memory associated with the native octree.
  @param grid pointer to native octree structure.
  """
  cdef set_grid(self, octree* grid):
    self.grid = grid 

  """
  Get pointer to native octree structure.
  @return octree*
  """
  cdef octree* get_grid(self):
    return self.grid

  """
  Returns a flat numpy array to the data in the octree. 
  If grid_idx and bit_idx are provided to this function, then only the data
  to the corresponding octree cell is returned.
  @param grid_idx 
  @param bit_idx
  @return numpy array to octree data
  """
  def get_grid_data(self, grid_idx=None, bit_idx=None):
    cdef FloatArrayWrapper wrapper = FloatArrayWrapper()
    wrapper.set_data(self.grid.data, self.grid.n_leafs * self.grid.feature_size, 0)
    cdef np.ndarray array = np.array(wrapper, copy=False)
    array.base = <PyObject*> wrapper
    Py_INCREF(wrapper)

    if grid_idx is not None and bit_idx is not None:
      didx = self.data_idx(grid_idx, bit_idx)
      array = array[didx : didx + self.feature_size()]
    return array

  """
  Returns the offset index in the octree data array for a given octree cell.
  The octree cell is specified by the grid_idx and bit_idx.
  @param grid_idx
  @param bit_idx
  @return data_idx to the specific octree cell. Takes feature_size into account.
  """
  def data_idx(self, int grid_idx, int bit_idx):
    # cdef int n_upto = n_leafs_upto(self.grid, grid_idx)
    cdef int n_upto = self.grid.prefix_leafs[grid_idx]
    cdef ot_tree_t* tree = octree_get_tree(self.grid, grid_idx)
    cdef int data_idx = tree_data_idx(tree, bit_idx, 1)
    return (n_upto + data_idx) * self.grid.feature_size


  """
  Destructor. Calls free on the wrapped octree structure.
  """
  def __dealloc__(self):
    octree_free_cpu(self.grid)

  """
  Prints the octree data structure to stdout.
  """
  def print_tree(self):
    octree_print_cpu(self.grid)

  """ @return the batch size of the octree structure. """
  def n(self):
    return self.grid[0].n
  """ @return the grid depth of the octree structure. """
  def grid_depth(self):
    return self.grid[0].grid_depth
  """ @return the grid height of the octree structure. """
  def grid_height(self):
    return self.grid[0].grid_height
  """ @return the grid width of the octree structure. """
  def grid_width(self):
    return self.grid[0].grid_width
  """ @return the feature size of the octree structure. """
  def feature_size(self):
    return self.grid[0].feature_size
  """ @return the depth of the tensor that corresponds to the octree structure. """
  def vx_depth(self):
    return 8 * self.grid[0].grid_depth
  """ @return the height of the tensor that corresponds to the octree structure. """
  def vx_height(self):
    return 8 * self.grid[0].grid_height
  """ @return the width of the tensor that corresponds to the octree structure. """
  def vx_width(self):
    return 8 * self.grid[0].grid_width
  """ @return the number of octree leaf cells. """
  def n_leafs(self):
    return self.grid[0].n_leafs

  """ @return the number of shallow octrees in the hybrid grid-octree structure. """
  def num_blocks(self):
    return self.n() * self.grid_depth() * self.grid_height() * self.grid_width()

  """
  Converts the subscript indices to the shallow octrees to a flatten index.
  @param gn
  @param gd
  @param gh
  @param gw
  @return grid_idx index to shallow octree.
  """
  def grid_idx(self, int gn, int gd, int gh, int gw):
    return octree_grid_idx(self.grid, gn, gd, gh, gw)

  """ 
  Computes for the given bit_idx the bit index of its first child.
  @param bit_idx
  @return child bit_idx.
  @see tree_child_bit_idx() for more details.
  """
  def tree_child_bit_idx(self, int bit_idx):
    return tree_child_bit_idx(bit_idx)
  
  """ 
  Computes for the given bit_idx the bit index of its parent.
  @param bit_idx
  @return parent bit_idx.
  @see tree_parent_bit_idx() for more details.
  """
  def tree_parent_bit_idx(self, int bit_idx):
    return tree_parent_bit_idx(bit_idx)

  """
  Checks for a shallow octree tree bit string, indexed by grid_idx, if a certain 
  bit (bit_idx) is set, or not.
  @param grid_idx
  @param bit_idx
  @return True, if bit is set, False otherwise.
  """
  def tree_isset_bit(self, int grid_idx, int bit_idx):
    cdef ot_tree_t* tree = octree_get_tree(self.grid, grid_idx)
    return tree_isset_bit(tree, bit_idx)

  """
  Returns the number of leaf cells for a given shallow octree.
  @param grid_idx index to shallow octree.
  @return number of leaf cells.
  """
  def tree_n_leafs(self, int grid_idx):
    cdef ot_tree_t* tree = octree_get_tree(self.grid, grid_idx)
    return tree_n_leafs(tree)

  """
  Returns the number of split nodes for a given shallow octree.
  @param grid_idx index to shallow octree.
  @return number of split nodes.
  """
  def tree_n_splits(self, int grid_idx):
    cdef ot_tree_t* tree = octree_get_tree(self.grid, grid_idx)
    return tree_n_splits(tree)

  """
  Returns the number of bytes that are reserved for this octree instance.
  This is a upper bound of what is really needed by this instance.
  @return number ob bytes allocated for this octree instance.
  """
  def mem_capacity(self):
    return octree_mem_capacity(self.grid)

  """
  Returns the number of bytes that are necessary for this octree instance.
  @return number of bytes needed for this octree instance.
  """
  def mem_using(self):
    return octree_mem_using(self.grid)

  """ 
  Compares this octree with another Octree (shape, structure, data).
  @return True, if Octree other is equal, False otherwise. 
  """
  def equals(self, Octree other):
    return octree_equal_cpu(self.grid, other.grid)

  """
  Class method that creates and wraps an empty native octree.
  @return Octree wrapper.
  """
  @classmethod
  def create_empty(cls):
    cdef octree* ret = octree_new_cpu()
    cdef Octree grid = Octree()
    grid.set_grid(ret)
    return grid

  """
  Class method that reads a native octree from the given path.
  @param path
  @return Octree wrapper.
  """
  @classmethod
  def create_from_bin(cls, char* path):
    cdef octree* ret = octree_new_cpu()
    octree_read_cpu(path, ret)
    cdef Octree grid = Octree()
    grid.set_grid(ret)
    return grid

  """
  Class method that reads a native octree from the given path.
  @deprecated
  @param path
  @return Octree wrapper.
  """
  @classmethod
  def create_from_bin_deprecated(cls, char* path):
    cdef octree* ret = octree_new_cpu()
    octree_read_deprecated_cpu(path, ret)
    cdef Octree grid = Octree()
    grid.set_grid(ret)
    return grid

  """
  Resizes the octree to the given shape. If the new shape is smaller than the 
  current one, the memory is not reallocated. 
  @param n
  @param grid_depth
  @param grid_height
  @param grid_width
  @param feature_size
  @param n_leafs
  @see octree_resize_cpu for more details.
  """
  def resize(self, int n, int grid_depth, int grid_height, int grid_width, int feature_size, int n_leafs):
    octree_resize_cpu(n, grid_depth, grid_height, grid_width, feature_size, n_leafs, self.grid)

  """ 
  Updates the number of leafs for the given octree.
  Usually called after a resize.
  """
  def upd_n_leafs(self):
    octree_upd_n_leafs_cpu(self.grid)

  """ 
  Updates prefix sum array of leafs for the given octree.
  Usually called after a resize.
  """
  def upd_prefix_leafs(self):
    octree_upd_prefix_leafs_cpu(self.grid)

  """
  Creates a native clone of the octree and wraps it in a new Python object.
  @return Octree wrapper.
  """
  def copy(self):
    cdef Octree other = self.create_empty()
    octree_copy_cpu(self.grid, other.grid)
    return other
  
  """
  Converts the octree to a tensor where the features are the last dimension.
  n x depth x height x width x features.
  @return numpy array.
  """
  def to_dhwc(self):
    dense = np.empty((self.n(), self.vx_depth(), self.vx_height(), self.vx_width(), self.feature_size()), dtype=np.float32)
    cdef float[:,:,:,:,::1] dense_view = dense
    octree_to_dhwc_cpu(self.grid, self.vx_depth(), self.vx_height(), self.vx_width(), &(dense_view[0,0,0,0,0]))
    return np.squeeze(dense)
  
  """
  Converts the octree to a tensor where the features are the second dimension.
  n x features x depth x height x width.
  @return numpy array.
  """
  def to_cdhw(self):
    dense = np.empty((self.n(), self.feature_size(), self.vx_depth(), self.vx_height(), self.vx_width()), dtype=np.float32)
    cdef float[:,:,:,:,::1] dense_view = dense
    octree_to_cdhw_cpu(self.grid, self.vx_depth(), self.vx_height(), self.vx_width(), &(dense_view[0,0,0,0,0]))
    return np.squeeze(dense)

  """ 
  Pools the tensor dense into a new octree with the same structure as this instance.
  @param dense tensor with valid shape.
  @return Octree wrapper.
  """
  def cdhw_to_octree(self, float[:,:,:,:,::1] dense):
    cdef octree* ret = octree_new_cpu()
    cdhw_to_octree_avg_cpu(self.grid, dense.shape[2], dense.shape[3], dense.shape[4], &(dense[0,0,0,0,0]), dense.shape[1], ret)
    cdef Octree grid = Octree()
    grid.set_grid(ret)
    return grid

  """
  Serializes the octree to a binary file.
  @param path
  """
  def write_bin(self, char* path):
    octree_write_cpu(path, self.grid)

  """
  First converts the octree to a tensor and then serializes the tensor to a 
  binary file.
  @param path
  """
  def write_to_cdhw(self, char* path):
    octree_cdhw_write_cpu(path, self.grid)

  """
  Experimental function.
  @see octree_determine_gt_split_cpu for more details.
  """
  def determine_gt_split(self, float[:,:,:,::1] gt, Octree out):
    if self.grid[0].n != gt.shape[0]:
      raise Exception('n does not match')
    if self.grid[0].grid_depth * 8 != gt.shape[1]:
      raise Exception('n does not match')
    if self.grid[0].grid_height * 8 != gt.shape[2]:
      raise Exception('n does not match')
    if self.grid[0].grid_width * 8 != gt.shape[3]:
      raise Exception('n does not match')
    octree_determine_gt_split_cpu(self.grid, &(gt[0,0,0,0]), out.grid)

  """
  Experimental function.
  @see octree_split_dense_reconstruction_surface_fres_cpu for more details.
  """
  def split_dense_rec_surf_fres(self, float[:,:,:,:,::1] feat, float[:,:,:,:,::1] rec, float rec_thr_from, float rec_thr_to, int band):
    octree_split_dense_reconstruction_surface_fres_cpu(&(feat[0,0,0,0,0]), &(rec[0,0,0,0,0]), feat.shape[0],feat.shape[2],feat.shape[3],feat.shape[4],feat.shape[1], rec_thr_from, rec_thr_to, band, self.grid)
    return self

  """ 
  Applies 2x2x2 grid pooling (max) on this instance and stores the result in the
  provided out octree. out is resized as needed in this function.
  @param out
  """
  def grid_pool_max(self, Octree out):
    octree_gridpool2x2x2_max_cpu(self.grid, out.get_grid())

  """
  Applies a 3x3x3 convolution on this instance and stores the result in the
  provided out octree. out is resized as needed in this function.
  @param weights convolution weights channels_in x channels_out x 3 x 3 x 3
  @param bias bias weight for each output channel
  @param out
  """
  def conv_avg(self, float[:,:,:,:,::1] weights, float[::1] bias, Octree out):
    cdef int channels_out = weights.shape[0]
    cdef int channels_in = weights.shape[1]
    if weights.shape[2] != 3 or weights.shape[3] != 3 or weights.shape[4] != 3:
      raise Exception('weights not valid for 3x3x3 conv')
    if bias.shape[0] != channels_out:
      raise Exception('bias.shape[0] != channels_out (weights.shape[0])')
    octree_conv3x3x3_avg_cpu(self.grid, &(weights[0,0,0,0,0]), &(bias[0]), channels_out, out.get_grid())

  """ 
  Applies 2x2x2 grid unpooling (nearest n. interpolation) on this instance and 
  stores the result in the provided out octree. out is resized as needed in 
  this function.
  @param out
  """
  def grid_unpool(self, Octree out):
    octree_gridunpool2x2x2_cpu(self.grid, out.get_grid())

  """
  Experimental function.
  @see octree_extract_feature_cpu for more details.
  """
  def extract_feature(self, int feature_from, int feature_to):  
    cdef octree* grid = octree_new_cpu()
    cdef Octree out = Octree()
    out.set_grid(grid)
    octree_extract_feature_cpu(self.grid, feature_from, feature_to, out.grid)
    return out


  """
  Class method to create an octree structure from a tensor.
  @param dense is the 3D data tensor.
  @param ranges intervals that define values for which the octree should be occupied.
  @param fit deprecated
  @param fit_multiply deprecated
  @param pack deprecated
  @param n_threads number of CPU threads that should be used for this function.
  """
  @classmethod
  def create_from_dense(cls, float[:,:,:,::1] dense, bool fit=False, int fit_multiply=1, bool pack=False, int n_threads=1):
    cdef octree* ret = octree_create_from_dense_cpu(&(dense[0,0,0,0]), dense.shape[0], dense.shape[1], dense.shape[2], dense.shape[3], fit, fit_multiply, pack, n_threads)
    cdef Octree grid = Octree()
    grid.set_grid(ret)
    return grid

  """
  Class method to create an octree structure from two tensors, one provides the 
  occupancy (leaf cells of the octree structure) and the other the feature values.
  @param occupancy is the 3D occupancy tensor, entries != 0 are converted to 
                   leaf cells in the resulting octree (DxHxW).
  @param features feature vectores for all voxels (fxDxHxW).
  @param fit deprecated
  @param fit_multiply deprecated
  @param pack deprecated
  @param n_threads number of CPU threads that should be used for this function.
  """
  @classmethod
  def create_from_dense2(cls, float[:,:,::1] occupancy, float[:,:,:,::1] features, bool fit=False, int fit_multiply=1, bool pack=False, int n_threads=1):
    if occupancy.shape[0] != features.shape[1] or occupancy.shape[1] != features.shape[2] or occupancy.shape[2] != features.shape[3]:
      raise Exception('occupancy shape does not match features shape')
    cdef octree* ret = octree_create_from_dense2_cpu(&(occupancy[0,0,0]), &(features[0,0,0,0]), features.shape[0], features.shape[1], features.shape[2], features.shape[3], fit, fit_multiply, pack, n_threads)
    cdef Octree grid = Octree()
    grid.set_grid(ret)
    return grid

  """
  Class method to create an octree structure from a triangle mesh.
  @param verts Nx3 contiguous float array with xyz vertic coordinates.
  @param faces Nx3 contiguous int array with indices to triangle vertices.
  @param rescale if true, vertices are adjusted to fit into bounding volume.
  @param depth number of voxel in depth dimension the octree should comprise.
  @param height number of voxel in height dimension the octree should comprise.
  @param width number of voxel in width dimension the octree should comprise.
  @param fit deprecated
  @param fit_multiply deprecated
  @param pack deprecated
  @param pad number of padding voxels if rescale == True
  @param n_threads number of CPU threads that should be used for this function.
  """
  @classmethod
  def create_from_mesh(cls, float[:,::1] verts, int[:,::1] faces, bool rescale, ot_size_t depth, ot_size_t height, ot_size_t width, bool fit=False, int fit_multiply=1, bool pack=False, int pad=0, int n_threads=1):
    if verts.shape[1] != 3:
      raise Exception('verts must have 3 columns (xyz)')
    if faces.shape[1] != 3:
      raise Exception('verts must have 3 columns (v0,v1,v2)')
    cdef octree* ret = octree_create_from_mesh_cpu(verts.shape[0], &(verts[0,0]), faces.shape[0], &(faces[0,0]), rescale, depth, height, width, fit, fit_multiply, pack, pad, n_threads)
    cdef Octree grid = Octree()
    grid.set_grid(ret)
    return grid

  """
  Class method to create an octree structure from an OFF file (mesh).
  @param path
  @param depth number of voxel in depth dimension the octree should comprise.
  @param height number of voxel in height dimension the octree should comprise.
  @param width number of voxel in width dimension the octree should comprise.
  @param R 3x3 rotation matrix applied to the triangle mesh
  @param fit deprecated
  @param fit_multiply deprecated
  @param pack deprecated
  @param n_threads number of CPU threads that should be used for this function.
  """
  @classmethod
  def create_from_off(cls, char* path, ot_size_t depth, ot_size_t height, ot_size_t width, float[:,::1] R, bool fit=False, int fit_multiply=1, bool pack=False, int pad=0, int n_threads=1):
    if R.shape[0] != 3 or R.shape[1] != 3:
      raise Exception('invalid R shape')
    cdef octree* ret = octree_create_from_off_cpu(path, depth, height, width, &(R[0,0]), fit, fit_multiply, pack, pad, n_threads)
    cdef Octree grid = Octree()
    grid.set_grid(ret)
    return grid

  """
  Class method to create an octree structure from an OBJ file (mesh).
  @param path
  @param depth number of voxel in depth dimension the octree should comprise.
  @param height number of voxel in height dimension the octree should comprise.
  @param width number of voxel in width dimension the octree should comprise.
  @param R 3x3 rotation matrix applied to the triangle mesh
  @param fit deprecated
  @param fit_multiply deprecated
  @param pack deprecated
  @param n_threads number of CPU threads that should be used for this function.
  """ 
  @classmethod
  def create_from_obj(cls, char* path, ot_size_t depth, ot_size_t height, ot_size_t width, float[:,::1] R, bool fit=False, int fit_multiply=1, bool pack=False, int pad=0, int n_threads=1):
    if R.shape[0] != 3 or R.shape[1] != 3:
      raise Exception('invalid R shape')
    cdef octree* ret = octree_create_from_obj_cpu(path, depth, height, width, &(R[0,0]), fit, fit_multiply, pack, pad, n_threads)
    cdef Octree grid = Octree()
    grid.set_grid(ret)
    return grid

  """
  Class method to create an octree structure from point cloud.
  The octree data will be set to 1, if a point is within a leaf cell, and 0 
  otherwise.
  @param xyz Nx3 contiguous float array for the xyz coordinates of the pointcloud.
  @param feature_size the length of the octree feature vectors.
  @param depth number of voxel in depth dimension the octree should comprise.
  @param height number of voxel in height dimension the octree should comprise.
  @param width number of voxel in width dimension the octree should comprise.
  @param normalize if True, the point cloud gets scaled and shifted to fit
                   within the bounding volume.
  @param normalize_inplace if True, the normalization is done inplace (xyz).
  @param fit deprecated
  @param fit_multiply deprecated
  @param pack deprecated
  @param n_threads number of CPU threads that should be used for this function.
  """
  @classmethod
  def create_from_pc_simple(cls, float[:,::1] xyz, ot_size_t feature_size, ot_size_t depth, ot_size_t height, ot_size_t width, bool normalize=False, bool normalize_inplace=False, bool fit=False, int fit_multiply=1, bool pack=False, int pad=0, int n_threads=1):
    if xyz.shape[1] != 3:
      raise Exception('xyz must have 3 columns (xyz)')
    cdef octree* ret = octree_create_from_pc_simple_cpu(&(xyz[0,0]), xyz.shape[0], feature_size, depth, height, width, normalize, normalize_inplace, fit, fit_multiply, pack, pad, n_threads)
    cdef Octree grid = Octree()
    grid.set_grid(ret)
    return grid

  """
  Class method to create an octree structure from point cloud.
  The octree data will be the average of the corresponding feature vectors.
  @param xyz Nx3 contiguous float array for the xyz coordinates of the pointcloud.
  @param features Nxfs contiguous float array that represents a feature vector
                  for each 3D point.
  @param depth number of voxel in depth dimension the octree should comprise.
  @param height number of voxel in height dimension the octree should comprise.
  @param width number of voxel in width dimension the octree should comprise.
  @param normalize if True, the point cloud gets scaled and shifted to fit
                   within the bounding volume.
  @param normalize_inplace if True, the normalization is done inplace (xyz).
  @param fit deprecated
  @param fit_multiply deprecated
  @param pack deprecated
  @param n_threads number of CPU threads that should be used for this function.
  """
  @classmethod
  def create_from_pc(cls, float[:,::1] xyz, float[:,::1] features, ot_size_t depth, ot_size_t height, ot_size_t width, bool normalize, bool normalize_inplace=True, bool fit=False, int fit_multiply=1, bool pack=False, int pad=0, int n_threads=1):
    if xyz.shape[0] != features.shape[0]:
      raise Exception('rows of xyz and features differ')
    if xyz.shape[1] != 3:
      raise Exception('xyz must have 3 columns (xyz)')
    cdef octree* ret = octree_create_from_pc_cpu(&(xyz[0,0]), &(features[0,0]), xyz.shape[0], features.shape[1], depth, height, width, normalize, normalize_inplace, fit, fit_multiply, pack, pad, n_threads)
    cdef Octree grid = Octree()
    grid.set_grid(ret)
    return grid

  @classmethod
  def create_dense_from_pc(cls, float[:,::1] xyz, float[:,::1] features, ot_size_t depth, ot_size_t height, ot_size_t width, int n_threads=1):
    if xyz.shape[0] != features.shape[0]:
      raise Exception('rows of xyz and features differ')
    if xyz.shape[1] != 3:
      raise Exception('xyz must have 3 columns (xyz)')
    dense = np.empty((features.shape[1], depth, height, width), dtype=np.float32)
    cdef float[:,:,:,::1] dense_view = dense
    octree_create_dense_from_pc_cpu(&(xyz[0,0]), &(features[0,0]), &(dense_view[0,0,0,0]), xyz.shape[0], features.shape[1], depth, height, width, n_threads)
    return dense

  """
  Given a voxelized octree uses parity count from the three orthogonal views
  to fill the occupancy grid. For example, create_from_mesh and others only
  voxelize the surface, to also set the interior of objects to occupied use
  this method.
  @param fill_value value used to fill octree
  @return self
  """
  def scanline_fill(self, float fill_value):
    octree_scanline_fill(self.grid, fill_value)
    return self


  def occupancy_to_surface(self, Octree out=None):
    cdef octree* out_ptr
    if out is None:
      out_ptr = octree_new_cpu()
    else:
      out_ptr = out.grid
    octree_occupancy_to_surface(self.grid, out_ptr)
    if out is None:
      out = Octree()
      out.set_grid(out_ptr)
    return out


"""
Warps a native octree struct in a Python Octree class.
@return Octree wrapper.
"""
cdef warp_octree(octree* grid):
  grid_w = Octree()
  grid_w.set_grid(grid)
  return grid_w


"""
Returns an iterator (d,h,w order) over all cells of all shallow octrees in grid.
@param n batch index to the grid of shallow octrees.
@param leafs_only if True, the iterator returns only information on leaf cells,
                  otherwise it returns also for split nodes.
@return (is_leaf, grid_idx, bit_idx, gd,gh,gw, bd,bh,bw, depth)
        is_leaf indicates, if returned value correspond to a leaf cell,
        grid_idx is the flat index to the corresponding shallow octree,
        bit_idx is the index in the shallow octrees tree bit string,
        gd,gh,gw are the subscript indices to the shallow octree,
        bd,bh,bw are the subscript indices within the shallow octree,
        depth is the depth of the octree cell.
"""
def leaf_iterator(grid, n=0, leafs_only=True):
  for gd in range(grid.grid_depth()):
    for gh in range(grid.grid_height()):
      for gw in range(grid.grid_width()):
        grid_idx = grid.grid_idx(n, gd,gh,gw)
        if not grid.tree_isset_bit(grid_idx, 0):
          yield (True, grid_idx, 0, gd,gh,gw, 0,0,0, 0)
        else:
          if not leafs_only:
            yield (False, grid_idx, 0, gd,gh,gw, 0,0,0, 0)
          bit_idx_l1 = 1;
          for bdl1 in [0,4]:
            for bhl1 in [0,4]:
              for bwl1 in [0,4]:
                if not grid.tree_isset_bit(grid_idx, bit_idx_l1):
                  yield (True, grid_idx, bit_idx_l1, gd,gh,gw, bdl1,bhl1,bwl1, 1)
                else:
                  if not leafs_only:
                    yield (False, grid_idx, bit_idx_l1, gd,gh,gw, bdl1,bhl1,bwl1, 1)
                  bit_idx_l2 = grid.tree_child_bit_idx(bit_idx_l1);
                  for bdl2 in [0,2]:
                    for bhl2 in [0,2]:
                      for bwl2 in [0,2]:
                        if not grid.tree_isset_bit(grid_idx, bit_idx_l2):
                          yield (True, grid_idx, bit_idx_l2, gd,gh,gw, bdl1+bdl2,bhl1+bhl2,bwl1+bwl2, 2)
                        else:
                          if not leafs_only:
                            yield (False, grid_idx, bit_idx_l2, gd,gh,gw, bdl1+bdl2,bhl1+bhl2,bwl1+bwl2, 2)
                          bit_idx_l3 = grid.tree_child_bit_idx(bit_idx_l2);
                          for bdl3 in [0,1]:
                            for bhl3 in [0,1]:
                              for bwl3 in [0,1]:
                                yield (True, grid_idx, bit_idx_l3, gd,gh,gw, bdl1+bdl2+bdl3,bhl1+bhl2+bhl3,bwl1+bwl2+bwl3, 3)
                                bit_idx_l3 += 1;
                        bit_idx_l2 += 1;
                bit_idx_l1 += 1;


"""
Returns an iterator (w,d,h order) over all cells of all shallow octrees in grid.
@param n batch index to the grid of shallow octrees.
@param leafs_only if True, the iterator returns only information on leaf cells,
                  otherwise it returns also for split nodes.
@return (is_leaf, grid_idx, bit_idx, gd,gh,gw, bd,bh,bw, depth)
        is_leaf indicates, if returned value correspond to a leaf cell,
        grid_idx is the flat index to the corresponding shallow octree,
        bit_idx is the index in the shallow octrees tree bit string,
        gd,gh,gw are the subscript indices to the shallow octree,
        bd,bh,bw are the subscript indices within the shallow octree,
        depth is the depth of the octree cell.
"""
def leaf_iterator2(grid, n=0, leafs_only=True):
  for gw in range(grid.grid_width()):
    for gd in range(grid.grid_depth()):
      for gh in range(grid.grid_height()-1,-1,-1):
        grid_idx = grid.grid_idx(n, gd,gh,gw)
        if not grid.tree_isset_bit(grid_idx, 0):
          yield (True, grid_idx, 0, gd,gh,gw, 0,0,0, 0)
        else:
          if not leafs_only:
            yield (False, grid_idx, 0, gd,gh,gw, 0,0,0, 0)
          for bwl1_idx in [0,1]:
            for bdl1_idx in [0,1]:
              for bhl1_idx in [1,0]:
                bdl1, bhl1, bwl1 = bdl1_idx * 4, bhl1_idx * 4, bwl1_idx * 4
                bit_idx_l1 = 1 + bdl1_idx * 4 + bhl1_idx * 2 + bwl1_idx;
                if not grid.tree_isset_bit(grid_idx, bit_idx_l1):
                  yield (True, grid_idx, bit_idx_l1, gd,gh,gw, bdl1,bhl1,bwl1, 1)
                else:
                  if not leafs_only:
                    yield (False, grid_idx, bit_idx_l1, gd,gh,gw, bdl1,bhl1,bwl1, 1)
                  for bwl2_idx in [0,1]:
                    for bdl2_idx in [0,1]:
                      for bhl2_idx in [1,0]:
                        bdl2, bhl2, bwl2 = bdl2_idx * 2, bhl2_idx * 2, bwl2_idx * 2
                        bit_idx_l2 = grid.tree_child_bit_idx(bit_idx_l1) + bdl2_idx * 4 + bhl2_idx * 2 + bwl2_idx;
                        if not grid.tree_isset_bit(grid_idx, bit_idx_l2):
                          yield (True, grid_idx, bit_idx_l2, gd,gh,gw, bdl1+bdl2,bhl1+bhl2,bwl1+bwl2, 2)
                        else:
                          if not leafs_only:
                            yield (False, grid_idx, bit_idx_l2, gd,gh,gw, bdl1+bdl2,bhl1+bhl2,bwl1+bwl2, 2)
                          for bwl3_idx in [0,1]:
                            for bdl3_idx in [0,1]:
                              for bhl3_idx in [1,0]:
                                bdl3, bhl3, bwl3 = bdl3_idx, bhl3_idx, bwl3_idx
                                bit_idx_l3 = grid.tree_child_bit_idx(bit_idx_l2) + bdl3_idx * 4 + bhl3_idx * 2 + bwl3_idx;
                                yield (True, grid_idx, bit_idx_l3, gd,gh,gw, bdl1+bdl2+bdl3,bhl1+bhl2+bhl3,bwl1+bwl2+bwl3, 3)
