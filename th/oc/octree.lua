-- Copyright (c) 2017, The OctNet authors
-- All rights reserved.
--
-- Redistribution and use in source and binary forms, with or without
-- modification, are permitted provided that the following conditions are met:
--     * Redistributions of source code must retain the above copyright
--       notice, this list of conditions and the following disclaimer.
--     * Redistributions in binary form must reproduce the above copyright
--       notice, this list of conditions and the following disclaimer in the
--       documentation and/or other materials provided with the distribution.
--     * Neither the name of the <organization> nor the
--       names of its contributors may be used to endorse or promote products
--       derived from this software without specific prior written permission.
--
-- THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
-- ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
-- WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
-- DISCLAIMED. IN NO EVENT SHALL OCTNET AUTHORS BE LIABLE FOR ANY
-- DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
-- (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
-- LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
-- ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
-- (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
-- SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

local ffi = require 'ffi'

function oc.isOctree(t)
  local t = torch.type(t)
  return t == 'oc.FloatOctree' or t == 'oc.CudaOctree' or t == 'oc.Octree'
end

function oc.write_dense_to_bin(path, dense)
  local dense = dense:float()
  local sz = {}
  for idx = 1, dense:nDimension() do table.insert(sz, dense:size(idx)) end
  local dims = ffi.new("int[?]", dense:nDimension(), sz)
  oc.cpu.dense_write_cpu(path, dense:nDimension(), dims, dense:data())
end 

function oc.read_dense_from_bin(path, dense)
  local sz = {}
  for idx = 1, dense:nDimension() do table.insert(sz, dense:size(idx)) end
  local dims = ffi.new("int[?]", dense:nDimension(), sz)
  oc.cpu.dense_read_prealloc_cpu(path, dense:nDimension(), dims, dense:data())
  return dense
end 

function oc.read_dense_from_bin_batch(paths, dense, n_threads)
  local n_threads = n_threads or 1
  local sz = {}
  for idx = 1, dense:nDimension() do table.insert(sz, dense:size(idx)) end
  local dims = ffi.new("int[?]", dense:nDimension(), sz)
  local paths_c = ffi.new("const char*[?]", #paths+1, paths)
  paths_c[#paths] = nil
  oc.cpu.dense_read_prealloc_batch_cpu(#paths, paths_c, n_threads, dense:nDimension(), dims, dense:data())
  return dense
end 



--------------------------------------------------------------------------------
local function free_octree_gpu(obj)
  oc.gpu.octree_free_gpu(obj)
end

local function oc_cuda_gc_wrapper(obj)
  local obj = ffi.gc(obj, free_octree_gpu)
  return obj
end

local function free_octree_cpu(obj)
  oc.cpu.octree_free_cpu(obj)
end

local function oc_float_gc_wrapper(obj)
  local obj = ffi.gc(obj, free_octree_cpu)
  return obj
end

local Octree = torch.class('oc.Octree')
function Octree:__init(oc_type)
  self._type = oc_type
  if self._type == 'oc_float' then
    self.grid = oc_float_gc_wrapper( oc.cpu.octree_new_cpu() ) 
  elseif self._type == 'oc_cuda' then
    self.grid = oc_cuda_gc_wrapper( oc.gpu.octree_new_gpu() ) 
  else
    error('invalid type for octree')
  end
end




local FloatOctree, floc_parent = torch.class('oc.FloatOctree', 'oc.Octree')
function FloatOctree:__init()
  floc_parent.__init(self, 'oc_float')
end 

function FloatOctree:new() 
  return oc.FloatOctree()
end

function FloatOctree:write(file)
  file:writeObject(self._type)
end 

function FloatOctree:read(file)
  self._type = file:readObject()
  if self._type == 'oc_float' then
    self.grid = oc_float_gc_wrapper( oc.cpu.octree_new_cpu() )
  else
    error('invalid type')
  end
end

function FloatOctree:set(other)
  if torch.type(other) == 'oc.FloatOctree' then
    self.grid = other.grid
  elseif not other then
    self.grid = oc_float_gc_wrapper( oc.cpu.octree_new_cpu() )
  else
    error('other is not an oc.FloatOctree, nor a nil')
  end
end 

function FloatOctree:apply(fcn)
  local n = self.grid.n_leafs * self.grid.feature_size
  for idx = 0, n-1 do 
    local val = fcn(idx)
    if val then 
      self.grid.data[idx] = val
    end
  end
  return self
end

function FloatOctree:create_from_dense(array)
  if array:nDimension() ~= 3 then error('invalid tensor in create_from_dense') end
  local grid = oc.FloatOctree()
  grid.grid = oc_float_gc_wrapper( oc.cpu.octree_create_from_dense_cpu(array:data(), array:size(1), array:size(2), array:size(3)) )
  return grid
end

function FloatOctree:equals(other, eps, debug)
  local eps = eps or 1e-4
  local deubg = debug or false
  if self.grid.n            ~= other.grid.n   then 
    print(self.grid.n, other.grid.n)
    if debug then print('[EQ] n unequal') end
    return false 
  end
  if self.grid.grid_depth   ~= other.grid.grid_depth   then 
    if debug then print('[EQ] grid_depth unequal') end
    return false 
  end
  if self.grid.grid_height  ~= other.grid.grid_height  then 
    if debug then print('[EQ] grid_height unequal') end
    return false 
  end
  if self.grid.grid_width   ~= other.grid.grid_width   then 
    if debug then print('[EQ] grid_width unequal') end
    return false 
  end
  if self.grid.feature_size ~= other.grid.feature_size then 
    if debug then print('[EQ] feature_size unequal') end
    return false 
  end
  if self.grid.n_leafs      ~= other.grid.n_leafs      then 
    if debug then print('[EQ] n_leafs unequal') end
    return false 
  end
  for idx = 0, 4 * self:n_blocks() - 1 do
    if self.grid.trees[idx] ~= other.grid.trees[idx] then 
      if debug then print('[EQ] tree at '..math.floor(idx/4.0)..' unequal') end
      return false 
    end
  end
  for grid_idx = 0, self:n_blocks() - 1 do
    if self.grid.prefix_leafs[grid_idx] ~= other.grid.prefix_leafs[grid_idx] then 
      if debug then print('[EQ] prefix_leafs at '..grid_idx..' unequal') end
      return false 
    end
  end
  for idx = 0, (self.grid.n_leafs * self.grid.feature_size) - 1 do
    -- print(idx, self.grid.data[idx], other.grid.data[idx])
    local err = math.abs(self.grid.data[idx] - other.grid.data[idx])
    if err > eps then 
      if debug then print('[EQ] data at '..idx..' unequal - err: '..err..' (values: '..self.grid.data[idx]..', '..other.grid.data[idx]..')') end
      return false 
    end
  end
  return true
end


function FloatOctree:clr_trees()
  oc.cpu.octree_clr_trees_cpu(self.grid)
end

function FloatOctree:tree_isset_bit(grid_idx, pos)
  return oc.cpu.tree_isset_bit_cpu(oc.cpu.octree_get_tree_cpu(self.grid, grid_idx-1), pos-1)
end

function FloatOctree:tree_set_bit(grid_idx, pos)
  oc.cpu.tree_set_bit_cpu(oc.cpu.octree_get_tree_cpu(self.grid, grid_idx-1), pos-1)
end 

function FloatOctree:tree_unset_bit(grid_idx, pos)
  oc.cpu.tree_unset_bit_cpu(oc.cpu.octree_get_tree_cpu(self.grid, grid_idx-1), pos-1)
end 

function FloatOctree:tree_n_leafs(grid_idx) 
  return oc.cpu.tree_n_leafs_cpu(oc.cpu.octree_get_tree_cpu(self.grid, grid_idx-1))
end

function FloatOctree:tree_data_idx(grid_idx, bit_idx)
  return oc.cpu.tree_data_idx_cpu(oc.cpu.octree_get_tree_cpu(self.grid, grid_idx-1), bit_idx-1, self.grid.feature_size)
end


function FloatOctree:leaf_idx_to_grid_bit_idx(leaf_idx)
  local leaf_idx = leaf_idx - 1
  local grid_idx = oc.cpu.leaf_idx_to_grid_idx_cpu(self.grid, leaf_idx)
  local tree = oc.cpu.octree_get_tree_cpu(self.grid, grid_idx)
  local data_idx = leaf_idx - self.grid[0].prefix_leafs[grid_idx]
  local bit_idx = oc.cpu.data_idx_to_bit_idx_cpu(tree, data_idx)
  return grid_idx + 1, bit_idx + 1
end

function FloatOctree:depth_from_bit_idx(bit_idx)
  return oc.cpu.depth_from_bit_idx_cpu(bit_idx-1)
end

function FloatOctree:split_grid_idx(grid_idx)
  local n = ffi.new('int [1]')
  local d = ffi.new('int [1]')
  local h = ffi.new('int [1]')
  local w = ffi.new('int [1]')
  oc.cpu.octree_split_grid_idx_cpu(self.grid, grid_idx-1, n,d,h,w)
  return n[0]+1, d[0]+1 ,h[0]+1 ,w[0]+1
end

function FloatOctree:dense_offset(grid_idx, bit_idx)
  local grid_idx = grid_idx - 1
  local bit_idx = bit_idx - 1
  
  local n = ffi.new('int [1]')
  local d = ffi.new('int [1]')
  local h = ffi.new('int [1]')
  local w = ffi.new('int [1]')
  oc.cpu.octree_split_grid_idx_cpu(self.grid, grid_idx, n,d,h,w)
  d[0] = 8 * d[0]
  h[0] = 8 * h[0]
  w[0] = 8 * w[0]

  local depth = oc.cpu.depth_from_bit_idx_cpu(bit_idx)
  if depth == 1 then
    oc.cpu.bdhw_from_idx_l1_cpu(bit_idx, d,h,w)
  elseif depth == 2 then
    oc.cpu.bdhw_from_idx_l2_cpu(bit_idx, d,h,w)
  elseif depth == 3 then
    oc.cpu.bdhw_from_idx_l3_cpu(bit_idx, d,h,w)
  end
  
  return n[0]+1, d[0]+1 ,h[0]+1 ,w[0]+1
end 


function FloatOctree:tree_bit_string(grid_idx) 
  local grid_idx = grid_idx or error('no grid_idx specified')

  local function bool2int(b) 
    if b then return 1 else return 0 end
  end

  local str = bool2int(self:tree_isset_bit(grid_idx, 1)) .. ' '
  for i = 2, 9 do
    str = str .. bool2int(self:tree_isset_bit(grid_idx, i))
  end
  for c = 2, 9 do
    str = str .. ' '
    local cidx = (c - 1) * 8 + 2
    for i = cidx, cidx + 7 do
      str = str .. bool2int(self:tree_isset_bit(grid_idx, i))
    end
  end
  return str
end

function FloatOctree:print()
  oc.cpu.octree_print_cpu(self.grid)
end

-- used mainly for finite differences
-- can be avoided by overloading functions directly on octree that are needed in fd
function FloatOctree:data()
  local tensor = torch.FloatTensor()
  oc.torch_cpu.THFloatStorage_free(tensor:cdata().storage)
  tensor:cdata().storage = oc.torch_cpu.octree_data_torch_cpu(self.grid)
  tensor:cdata().size = oc.torch_cpu.THAlloc(ffi.sizeof('long') * 1)
  tensor:cdata().size[0] = tensor:cdata().storage.size
  tensor:cdata().stride = oc.torch_cpu.THAlloc(ffi.sizeof('long') * 1)
  tensor:cdata().stride[0] = 1
  tensor:cdata().nDimension = 1
  tensor:cdata().storageOffset = 0
  tensor:cdata().refcount = 1
  return tensor
end 

function FloatOctree:cpy_data(tensor)
  if self.grid.feature_size * self.grid.n_leafs ~= tensor:nElement() then 
    error('number of elements does not match: '..(self.grid.feature_size * self.grid.n_leafs)..' vs '..tensor:nElement()) 
  end 
  if not tensor:isContiguous() then error('tensor is not contiguous') end
  -- for idx = 0, self.grid.n_leafs - 1 do
  --   self.grid.data[idx] = tensor:data()[idx]
  -- end
  ffi.copy(self.grid.data, tensor:data(), self.grid.feature_size * self.grid.n_leafs * ffi.sizeof('float'))
end 


function FloatOctree:cuda(other) 
  local grid = other or oc.CudaOctree()
  oc.gpu.octree_to_gpu(self.grid, grid.grid)
  return grid
end 


local CudaOctree, cuoc_parent = torch.class('oc.CudaOctree', 'oc.Octree')
function CudaOctree:__init()
  cuoc_parent.__init(self, 'oc_cuda')
end 

function CudaOctree:new() 
  return oc.CudaOctree()
end

function CudaOctree:write(file)
  file:writeObject(self._type)
end 

function CudaOctree:read(file)
  self._type = file:readObject()
  if self._type == 'oc_cuda' then
    self.grid = oc_cuda_gc_wrapper( oc.gpu.octree_new_gpu() )
  else
    error('invalid type')
  end
end

function CudaOctree:set(other)
  if torch.type(other) == 'oc.CudaOctree' then
    self.grid = other.grid
  elseif not other then
    self.grid = oc_cuda_gc_wrapper( oc.gpu.octree_new_gpu() )
  else
    print(torch.type(other))
    error('other is not an oc.CudaOctree')
  end
end 

function CudaOctree:float(other) 
  local grid = other or oc.FloatOctree()
  oc.gpu.octree_to_cpu(self.grid, grid.grid)
  return grid
end 



function Octree:to_cdhw()
  local dense_depth = 8*self:grid_depth()
  local dense_height = 8*self:grid_height()
  local dense_width = 8*self:grid_width()
  local out_size = torch.LongStorage({self:n(), self:feature_size(), dense_depth, dense_height, dense_width})
  local out
  if self._type == 'oc_float' then
    out = torch.FloatTensor(out_size)
    oc.cpu.octree_to_cdhw_cpu(self.grid, dense_depth, dense_height, dense_width, out:data())
  elseif self._type == 'oc_cuda' then
    out = torch.CudaTensor(out_size)
    oc.gpu.octree_to_cdhw_gpu(self.grid, dense_depth, dense_height, dense_width, out:data())
  end
  return out
end

function Octree:read_header_from_bin(path)
  oc.cpu.octree_read_header_cpu(path, self.grid)
  if self._type == 'oc_cuda' then
    self.grid = oc_cuda_gc_wrapper( oc.gpu.octree_to_gpu(self.grid) )
  end
  return self
end

function Octree:read_from_bin(path)
  oc.cpu.octree_read_cpu(path, self.grid)
  if self._type == 'oc_cuda' then
    self.grid = oc_cuda_gc_wrapper( oc.gpu.octree_to_gpu(self.grid) )
  end
  return self
end

function Octree:read_from_bin_batch(paths, n_threads)
  local n_threads = n_threads or 1
  local paths_c = ffi.new("const char*[?]", #paths+1, paths)
  paths_c[#paths] = nil
  oc.cpu.octree_read_batch_cpu(#paths, paths_c, n_threads, self.grid)
  if self._type == 'oc_cuda' then
    self.grid = oc_cuda_gc_wrapper( oc.gpu.octree_to_gpu(self.grid) )
  end
  return self
end

function Octree:write_to_bin(path)
  local grid = self
  if self._type == 'oc_cuda' then
    grid = grid:float()
  end
  oc.cpu.octree_write_cpu(path, grid.grid)
end

function Octree:tree_child_bit_idx(bit_idx) 
  return 8 * (bit_idx - 1) + 1 + 1
end

function Octree:tree_parent_bit_idx(bit_idx) 
  return math.floor(((bit_idx - 1) - 1) / 8) + 1
end

function Octree:mem_capacity()
  return oc.cpu.octree_mem_capacity_cpu(self.grid)
end

function Octree:mem_using()
  return oc.cpu.octree_mem_using_cpu(self.grid)
end

function Octree:update_n_leafs()
  if self._type == 'oc_float' then
    oc.cpu.octree_upd_n_leafs_cpu(self.grid)
  elseif self._type == 'oc_cuda' then
    oc.gpu.octree_upd_n_leafs_gpu(self.grid)
  end
end

function Octree:update_prefix_leafs()
  if self._type == 'oc_float' then
    oc.cpu.octree_upd_prefix_leafs_cpu(self.grid)
  elseif self._type == 'oc_cuda' then
    oc.gpu.octree_upd_prefix_leafs_gpu(self.grid)
  end
end

function Octree:fill(fill_value)
  if self._type == 'oc_float' then
    oc.cpu.octree_fill_data_cpu(self.grid, fill_value)
  elseif self._type == 'oc_cuda' then
    oc.gpu.octree_fill_data_gpu(self.grid, fill_value)
  end
  return self
end

function Octree:zero()
  return self:fill(0)
end

function Octree:resize(n, grid_depth, grid_height, grid_width, feature_size, n_leafs)
  if self._type == 'oc_float' then
    oc.cpu.octree_resize_cpu(n, grid_depth, grid_height, grid_width, feature_size, n_leafs, self.grid)
  elseif self._type == 'oc_cuda' then
    oc.gpu.octree_resize_gpu(n, grid_depth, grid_height, grid_width, feature_size, n_leafs, self.grid)
  end
  return self
end

function Octree:resize_as(grid)
  if self._type == 'oc_float' then
    oc.cpu.octree_resize_as_cpu(grid.grid, self.grid)
  elseif self._type == 'oc_cuda' then
    oc.gpu.octree_resize_as_gpu(grid.grid, self.grid)
  end
  return self
end
function Octree:resizeAs(grid)
  self:resize_as(grid)
  if self._type == 'oc_float' then
    oc.cpu.octree_cpy_trees_cpu_cpu(grid.grid, self.grid)
    oc.cpu.octree_cpy_prefix_leafs_cpu_cpu(grid.grid, self.grid)
  elseif self._type == 'oc_cuda' then
    oc.gpu.octree_cpy_trees_gpu_gpu(grid.grid, self.grid)
    oc.gpu.octree_cpy_prefix_leafs_gpu_gpu(grid.grid, self.grid)
  end
  return self
end 

function Octree:copy(other)
  if self._type ~= other._type then
    error('type does not match')
  end

  if self._type == 'oc_float' then
    oc.cpu.octree_copy_cpu(other.grid, self.grid)
  elseif self._type == 'oc_cuda' then
    oc.gpu.octree_copy_gpu(other.grid, self.grid)
  end
end

function Octree:clone() 
  local clone_grid = self:new()
  clone_grid:copy(self)
  return clone_grid
end 


function Octree:size(idx)
  if idx then
    if idx == 1 then
      return self:n()
    elseif idx == 2 then
      return self:feature_size()
    elseif idx == 3 then
      return 8 * self:grid_depth()
    elseif idx == 4 then
      return 8 * self:grid_height()
    elseif idx == 5 then
      return 8 * self:grid_width()
    else
      error('[ERROR] out of bounds in Octree:size() with index '..idx)
    end
  else
    return torch.LongStorage({self:n(), self:feature_size(), 8 * self:grid_depth(), 8 * self:grid_height(), 8 * self:grid_width()})
  end
end 

function Octree:feature_size() return self.grid[0].feature_size end
function Octree:n_leafs() return self.grid[0].n_leafs end

function Octree:n()           return self.grid[0].n  end
function Octree:grid_depth()  return self.grid[0].grid_depth  end
function Octree:grid_height() return self.grid[0].grid_height end
function Octree:grid_width()  return self.grid[0].grid_width  end

function Octree:dense_depth()  return 8 * self.grid[0].grid_depth  end
function Octree:dense_height() return 8 * self.grid[0].grid_height end
function Octree:dense_width()  return 8 * self.grid[0].grid_width  end

function Octree:n_blocks()
  return self:n() * self:grid_depth() * self:grid_height() * self:grid_width()
end

function Octree:n_elems()    
  return self:feature_size() * self:n_leafs() 
end


function Octree:add(fac1, other1, fac2, other2)
  -- only one argument
  if not other1 and not fac2 and not other2 then 
    if oc.isOctree(fac1) then
      other1 = fac1
      fac1 = 1
      fac2 = 1
      other2 = self
    else
      if torch.type(self) == 'oc.FloatOctree' then
        oc.cpu.octree_scalar_add_cpu(self.grid, fac1)
      elseif torch.type(self) == 'oc.CudaOctree' then
        oc.gpu.octree_scalar_add_gpu(self.grid, fac1)
      else
        error('unknown type of self')
      end
      return self
    end
  -- two arguments
  elseif not fac2 and not other2 then 
    if oc.isOctree(fac1) then
      other2 = other1
      other1 = fac1
      fac1 = 1
      fac2 = 1
    else
      fac2 = 1
      other2 = self
    end
  -- three arguments
  elseif not other2 then
    other2 = fac2
    fac2 = other1
    other1 = fac1
    fac1 = 1
  end

  if torch.type(self) == 'oc.FloatOctree' then
    oc.cpu.octree_add_cpu(other1.grid, fac1, other2.grid, fac2, true, self.grid)
  elseif torch.type(self) == 'oc.CudaOctree' then
    oc.gpu.octree_add_gpu(other1.grid, fac1, other2.grid, fac2, true, self.grid)
  else
    error('unknown type of self')
  end

  return self
end

function Octree:mul(val)
  if self._type == 'oc_float' then
    oc.cpu.octree_scalar_mul_cpu(self.grid, val)
  elseif self._type == 'oc_cuda' then
    oc.gpu.octree_scalar_mul_gpu(self.grid, val)
  end
  return self
end

function Octree:sign()
  if self._type == 'oc_float' then
    oc.cpu.octree_sign_cpu(self.grid)
  elseif self._type == 'oc_cuda' then
    oc.gpu.octree_sign_gpu(self.grid)
  end
  return self
end

function Octree:abs()
  if self._type == 'oc_float' then
    oc.cpu.octree_abs_cpu(self.grid)
  elseif self._type == 'oc_cuda' then
    oc.gpu.octree_abs_gpu(self.grid)
  end
  return self
end

function Octree:log()
  if self._type == 'oc_float' then
    oc.cpu.octree_log_cpu(self.grid)
  elseif self._type == 'oc_cuda' then
    oc.gpu.octree_log_gpu(self.grid)
  end
  return self
end

function Octree:min()
  if self._type == 'oc_float' then
    return oc.cpu.octree_min_cpu(self.grid)
  elseif self._type == 'oc_cuda' then
    return oc.gpu.octree_min_gpu(self.grid)
  end
end

function Octree:max()
  if self._type == 'oc_float' then
    return oc.cpu.octree_max_cpu(self.grid)
  elseif self._type == 'oc_cuda' then
    return oc.gpu.octree_max_gpu(self.grid)
  end
end


function Octree:copy_from_sup(sup)
  if self._type == 'oc_float' then
    oc.cpu.octree_cpy_sup_to_sub_cpu(sup.grid, self.grid)
  elseif self._type == 'oc_cuda' then
    oc.gpu.octree_cpy_sup_to_sub_gpu(sup.grid, self.grid)
  end

  return self
end

function Octree:full_split(input)
  if self._type == 'oc_float' then
    oc.cpu.octree_full_split_cpu(input.grid, self.grid)
  elseif self._type == 'oc_cuda' then
    oc.gpu.octree_full_split_gpu(input.grid, self.grid)
  end

  return self
end

function Octree:combine_n(grids_in, grid_out)
  -- self is in1
  local grid_out = grid_out or self:new()

  local grid_in_c = ffi.new('octree*['..#grids_in..']')
  for i = 0, #grids_in-1 do
    grid_in_c[i] = grids_in[i+1].grid
  end 

  if self._type == 'oc_float' then
    oc.cpu.octree_combine_n_cpu(ffi.cast('const octree**', grid_in_c), #grids_in, grid_out.grid)
  elseif self._type == 'oc_cuda' then
    error('not implemented')
  end

  return grid_out
end

function Octree:extract_n(from, to, grid_out)
  -- self is in1
  local grid_out = grid_out or self:new()

  if self._type == 'oc_float' then
    oc.cpu.octree_extract_n_cpu(self.grid, from-1, to-1, grid_out.grid)
  elseif self._type == 'oc_cuda' then
    error('not implemented')
  end

  return grid_out
end


function Octree:mask_by_label(labels, mask_label, check)
  local labels = labels or error('need to pass an labels octree')
  local mask_label = mask_label or error('need to set a mask_label')
  local check = check or false

  if self._type == 'oc_float' then
    oc.cpu.octree_mask_by_label_cpu(labels.grid, mask_label, check, self.grid)
  elseif self._type == 'oc_cuda' then
    oc.gpu.octree_mask_by_label_gpu(labels.grid, mask_label, check, self.grid)
  end

  return self
end

function Octree:determine_gt_split(gt, out)
  local out = out or self:new()

  if self._type == 'oc_float' then
    oc.cpu.octree_determine_gt_split_cpu(self.grid, gt:data(), out.grid)
  elseif self._type == 'oc_cuda' then
    oc.gpu.octree_determine_gt_split_gpu(self.grid, gt:data(), out.grid)
  end

  return out
end


