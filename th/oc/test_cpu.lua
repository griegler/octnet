#!/usr/bin/env th

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

local test_utils = require('test_utils')
local nn = require('nn')

local mytester = torch.Tester()
local octest = torch.TestSuite()


function octest.VolumetricNNUpsampling()
  local function vol_upsample(input, factor)
    local N = input:size(1)
    local fs = input:size(2)
    local in_depth, in_height, in_width = input:size(3), input:size(4), input:size(5)
    local out_height, out_width, out_depth = factor*in_height, factor*in_width, factor*in_depth
    local out = torch.zeros(N, fs, out_depth, out_height, out_width):float()
    local in_data = input:data()
    local out_data = out:data()
    for n = 0, N-1 do
      for f = 0, fs-1 do
        for od = 0, out_depth-1 do
          for oh = 0, out_height-1 do
            for ow = 0, out_width-1 do
              local id, ih, iw = math.floor(od/factor), math.floor(oh/factor), math.floor(ow/factor)
              local out_idx = (((n * fs + f) * out_depth + od) * out_height + oh) * out_width + ow
              local in_idx = (((n * fs + f) * in_depth + id) * in_height + ih) * in_width + iw
              local in_val = in_data[in_idx]
              out_data[out_idx] = in_val
            end 
          end
        end
      end
    end
    return out
  end 

  local function vol_upsample_bwd(input, factor)
    local N = input:size(1)
    local fs = input:size(2)
    local in_depth, in_height, in_width = input:size(3), input:size(4), input:size(5)
    local out_height, out_width, out_depth = math.floor(in_height/factor), math.floor(in_width/factor), math.floor(in_depth/factor)
    local out = torch.zeros(N, fs, out_depth, out_height, out_width):float()
    local in_data = input:data()
    local out_data = out:data()
    for n = 0, N-1 do
      for f = 0, fs-1 do
        for id = 0, in_depth-1 do
          for ih = 0, in_height-1 do
            for iw = 0, in_width-1 do
              local od, oh, ow = math.floor(id/factor), math.floor(ih/factor), math.floor(iw/factor)
              local out_idx = (((n * fs + f) * out_depth + od) * out_height + oh) * out_width + ow
              local in_idx = (((n * fs + f) * in_depth + id) * in_height + ih) * in_width + iw
              out_data[out_idx] = out_data[out_idx] + in_data[in_idx]
            end 
          end
        end
      end
    end
    return out
  end

  local function test(input, factor)
    local mod = oc.VolumetricNNUpsampling(factor):float()
    local mod_out = mod:forward(input)
    local tst_out = vol_upsample(input, factor)

    local err = torch.abs(mod_out - tst_out):max()
    mytester:assert(err < 1e-6, 'VolumetricNNUpsampling forward factor='..factor)

    local grad = tst_out:clone()
    grad:apply(function() torch.uniform(-1,1) end)
    local mod_out = mod:backward(input, grad)
    local tst_out = vol_upsample_bwd(grad, factor)
    
    local err = torch.abs(mod_out - tst_out):max()
    mytester:assert(err < 1e-6, 'VolumetricNNUpsampling backward factor='..factor)
  end 

  for _, factor in ipairs{2, 1, 3, 4} do
    test(torch.randn(1, 1, 1,1,1):float(), factor)
    test(torch.randn(1, 2, 1,1,1):float(), factor)
    test(torch.randn(1, 4, 4,5,6):float(), factor)
    test(torch.randn(3, 4, 4,5,6):float(), factor)
  end 
end


function octest.copy_sup_to_sub()
  local function test(sup)
    local sub = sup:clone()
    for grid_idx = 1, sup:n_blocks() do
      for bit_idx = 1, 73 do 
        local pa_bit_idx = sup:tree_parent_bit_idx(bit_idx)
        if (bit_idx == 1 or sub:tree_isset_bit(grid_idx, pa_bit_idx)) and torch.uniform(0,1) > 0.5 then
          sub:tree_set_bit(grid_idx, bit_idx)
        end
      end 
    end
    -- sub:tree_set_bit(1,1)
    sub:update_n_leafs()
    sub:update_prefix_leafs()
    sub:resize_as(sub)
    sub:apply(function() torch.uniform(-1,1) end)

    sub:copy_from_sup(sup)

    local dsup = oc.OctreeToCDHW():float():forward(sup) 
    local dsub = oc.OctreeToCDHW():float():forward(sub) 
    local err = torch.abs(dsup - dsub):max()
    mytester:assert(err < 1e-6, 'copy_sup_to_sub error='..err)
  end

  for _, n in ipairs{1, 4} do
    test( test_utils.octree_rand(n, 1,1,1, 1, 0,0,0) )
    test( test_utils.octree_rand(n, 2,4,6, 3, 0,0,0) )
    test( test_utils.octree_rand(n, 2,4,6, 3, 1,0,0) )
    test( test_utils.octree_rand(n, 2,4,6, 3, 1,1,0) )
    test( test_utils.octree_rand(n, 2,4,6, 3, 1,1,1) )
    test( test_utils.octree_rand(n, 2,4,6, 3, 0.5,0.5,0.5) )
    test( test_utils.octree_rand(n, 2,4,6, 3, 0.5,0.5,0.5) )
  end
end

function octest.combine_extract_n()
  local function test(grid, reps)
    local grids = { }
    for idx = 1, reps do
      table.insert(grids, grid)
    end
    local grid_cmb = grid:combine_n(grids)

    for idx = 1, #grids do
      local grid_ext = grid_cmb:extract_n(idx, idx+1)
      
      mytester:assert(grid_ext:equals(grids[idx], 1e-4, true), 'combine_extract_n')
    end
  end

  test( test_utils.octree_rand(1, 1,1,1, 1, 0,0,0), 1 )
  test( test_utils.octree_rand(1, 1,1,1, 1, 0,0,0), 4 )
  test( test_utils.octree_rand(1, 2,3,4, 1, 0,0,0), 4 )
  test( test_utils.octree_rand(1, 2,3,4, 3, 0,0,0), 4 )
  test( test_utils.octree_rand(1, 2,3,4, 3, 0.5,0.5,0.5), 4 )
  test( test_utils.octree_rand(1, 4,5,6, 5, 0,0,0), 13 )  
  test( test_utils.octree_rand(1, 8,16,32, 8, 0,0,0), 16 )  

  local grids = { test_utils.octree_rand(1, 2,3,4, 3, 1,1,0),
                  test_utils.octree_rand(2, 2,3,4, 3, 1,1,0), 
                  test_utils.octree_rand(1, 2,3,4, 3, 1,1,0),
                  test_utils.octree_rand(3, 2,3,4, 3, 1,1,0) }
  local grid_cmb = grid:combine_n(grids)

  mytester:assert(grid_cmb:extract_n(1,2):equals(grids[1], 1e-4, true), 'combine_extract_n')
  mytester:assert(grid_cmb:extract_n(2,4):equals(grids[2], 1e-4, true), 'combine_extract_n')
  mytester:assert(grid_cmb:extract_n(4,5):equals(grids[3], 1e-4, true), 'combine_extract_n')
  mytester:assert(grid_cmb:extract_n(5,8):equals(grids[4], 1e-4, true), 'combine_extract_n')
end

function octest.IO()
  local function test_oc(n, grid_depth, grid_height, grid_width, feature_size, n_threads)
    print('')

    local grids = {}
    local oc_paths = {}
    for idx = 1, n do
      local oc_path = string.format('test_grid_%d.oc', idx)
      table.insert(oc_paths, oc_path)

      local grid = test_utils.octree_rand(1,grid_depth,grid_height,grid_width, feature_size, 0.5,0.5,0.5)
      print(string.format('grid_gt %d: %d, %d,%d,%d, %d, %d', idx, grid:n(), grid:grid_depth(),grid:grid_height(),grid:grid_width(), grid:feature_size(), grid:n_leafs()))
      table.insert(grids, grid)
      grid:write_to_bin(oc_path)
    end

    for idx = 1, n do
      local grid = oc.FloatOctree()
      grid:read_from_bin( oc_paths[idx] )
      print(string.format('grid_read %d: %d, %d,%d,%d, %d, %d', idx, grid:n(), grid:grid_depth(),grid:grid_height(),grid:grid_width(), grid:feature_size(), grid:n_leafs()))

      mytester:assert(grid:equals(grids[idx], 1e-4, true), 'read_from_bin')
    end 

    local grid_b = oc.FloatOctree()
    print('read batch')
    grid_b:read_from_bin_batch(oc_paths, n_threads)
    print(string.format('grid_b: %d, %d,%d,%d, %d, %d', grid_b:n(), grid_b:grid_depth(),grid_b:grid_height(),grid_b:grid_width(), grid_b:feature_size(), grid_b:n_leafs()))
    for idx = 1, n do 
      -- print('split '..idx)
      local grid = grid_b:extract_n(idx, idx+1)
      print(string.format('grid_ext %d: %d, %d,%d,%d, %d, %d', idx, grid:n(), grid:grid_depth(),grid:grid_height(),grid:grid_width(), grid:feature_size(), grid:n_leafs()))
      
      mytester:assert(grid:equals(grids[idx], 1e-4, true), 'read_from_bin_batch')
    end 
  end 

  local function test_dense(n, depth, height, width, feature_size, n_threads)
    local tensors = {}
    local tensor_paths = {}

    for idx = 1, n do
      local tensor = torch.randn(1, feature_size, depth,height,width):float()
      table.insert(tensors, tensor)
      local tensor_path = string.format('test_tensor_%d.cdhw', idx)
      table.insert(tensor_paths, tensor_path)
      oc.write_dense_to_bin(tensor_path, tensor)
    end

    for idx = 1, n do 
      local tensor = torch.FloatTensor(1, feature_size, depth,height,width)
      oc.read_dense_from_bin(tensor_paths[idx], tensor)

      mytester:assert(torch.abs(tensor - tensors[idx]):max() < 1e-6, 'read_dense_from_bin')
    end 

    local tensor_b = torch.FloatTensor(n, feature_size, depth,height,width)
    oc.read_dense_from_bin_batch(tensor_paths, tensor_b, n_threads)
    for idx = 1, n do 
      local tensor = tensor_b[idx]
      mytester:assert(torch.abs(tensor - tensors[idx]):max() < 1e-6, 'read_dense_from_bin_batch')
    end
  end 

  for _, n_threads in ipairs{1, 4} do
    test_oc(1, 1,1,1, 1, n_threads)
    test_oc(2, 1,1,1, 1, n_threads)
    test_oc(1, 1,1,1, 2, n_threads)
    test_oc(2, 1,1,1, 2, n_threads)
    test_oc(3, 2,3,4, 2, n_threads)
    test_oc(4, 3,4,5, 4, n_threads)
    test_oc(8, 8,16,32, 8, n_threads)
    
    test_dense(1, 1,1,1, 1, n_threads)
    test_dense(2, 1,1,1, 1, n_threads)
    test_dense(1, 1,1,1, 2, n_threads)
    test_dense(2, 1,1,1, 2, n_threads)
    test_dense(3, 2,3,4, 2, n_threads)
    test_dense(4, 3,4,5, 4, n_threads)
    test_dense(8, 8,16,32, 8, n_threads)
  end
end 

function octest.OctreeCDHW()
  local function test(grid, dense_depth, dense_height, dense_width)
    local input = grid
    
    local o2d = oc.OctreeToCDHW(dense_depth, dense_height, dense_width)
    local d2o = oc.CDHWToOctree(o2d, 'avg')

    local o2d_output = o2d:forward(input)
    local d2o_output = d2o:forward(o2d_output)
    mytester:assert(d2o_output:equals(grid, 1e-4, true), 'error in OctreeCDHW forward')


    local d2o_bwd_output = d2o:backward(o2d_output, d2o_output)
    local o2d_bwd_output = o2d:backward(input, d2o_bwd_output)
    mytester:assert(o2d_bwd_output:equals(grid, 1e-4, true), 'error in OctreeCDHW backward')
  end
  
  local function test_battery(dense_depth, dense_height, dense_width)
    for _, n in ipairs{1,4} do
      test(test_utils.octree_rand(n,1,1,1, 1, 0,0,0), dense_depth, dense_height, dense_width)
      test(test_utils.octree_rand(n,1,1,1, 3, 0,0,0), dense_depth, dense_height, dense_width)
      test(test_utils.octree_rand(n,1,1,1, 1, 1,0,0), dense_depth, dense_height, dense_width)
      test(test_utils.octree_rand(n,1,1,1, 1, 1,1,0), dense_depth, dense_height, dense_width)
      test(test_utils.octree_rand(n,1,1,1, 1, 1,1,1), dense_depth, dense_height, dense_width)
      test(test_utils.octree_rand(n,2,3,4, 3, 0,0,0), dense_depth, dense_height, dense_width)
      test(test_utils.octree_rand(n,2,3,4, 3, 1,0,0), dense_depth, dense_height, dense_width)
      test(test_utils.octree_rand(n,2,3,4, 3, 1,1,0), dense_depth, dense_height, dense_width)
      test(test_utils.octree_rand(n,2,3,4, 3, 1,1,1), dense_depth, dense_height, dense_width)
      test(test_utils.octree_rand(n,2,3,4, 2, 0.5,0.5,0.5), dense_depth, dense_height, dense_width)
      test(test_utils.octree_rand(n,2,3,4, 2, 0.5,0.5,0.5), dense_depth, dense_height, dense_width)
    end
  end
    
  test_battery()
  test_battery(32,32,32)
  test_battery(64,64,64)
end

function octest.Octree2CDHWBwd()
  -- takes a lot of time
  local function test(grid)
    local input = grid
    local o2d = oc.OctreeToCDHW()

    local output = o2d:forward(input)
    local grad_output = output:clone()
    o2d:backward(input, grad_output)

    local fwd_fcn = function() o2d:forward(input) end
    local jac_fwd = test_utils.jacobian_forward(grid.grid.data, grid:n_elems(), output:data(), output:nElement(), fwd_fcn, 1e-2)
    local bwd_fcn = function() o2d:backward(input, grad_output) end
    local jac_bwd = test_utils.jacobian_backward(o2d.gradInput.grid.data, o2d.gradInput:n_elems(), grad_output:data(), grad_output:nElement(), bwd_fcn)
    local max_err = torch.abs(jac_fwd - jac_bwd):max()
    mytester:assert(max_err < 1e-5, 'Octree2DenseBwd error '..max_err)
  end
    
  for _, n in ipairs{1,4} do
    test(test_utils.octree_rand(n,1,1,1, 1, 0,0,0))
    test(test_utils.octree_rand(n,1,1,1, 1, 1,0,0))
    test(test_utils.octree_rand(n,1,1,1, 1, 1,1,0))
    test(test_utils.octree_rand(n,1,1,1, 1, 1,1,1))
    test(test_utils.octree_rand(n,2,3,4, 2, 0.5,0.5,0.5))
  end
end

function octest.OctreeDenseConvolutionBwd()
  -- takes a lot of time
  local function test(grid)
    local out_channels = 1
    local input = grid
    local in_channels = grid:feature_size()
    local conv_dense = oc.OctreeDenseConvolution(in_channels, out_channels, 'avg', false, nn):float()

    local output = conv_dense:forward(input)
    local grad_output = output:clone()
    conv_dense:backward(input, grad_output)

    local fwd_fcn = function() conv_dense:forward(input) end
    local jac_fwd = test_utils.jacobian_forward(grid.grid.data, grid:n_elems(), output.grid.data, output:n_elems(), fwd_fcn, 1e-2)
    local bwd_fcn = function() conv_dense:zeroGradParameters(); conv_dense:backward(input, grad_output) end
    local jac_bwd = test_utils.jacobian_backward(conv_dense.gradInput.grid.data, conv_dense.gradInput:n_elems(), grad_output.grid.data, grad_output:n_elems(), bwd_fcn)
    local max_err = torch.abs(jac_fwd - jac_bwd):max()
    -- print(jac_fwd, jac_bwd)
    mytester:assert(max_err < 1e-2, 'Octree2DenseBwd error '..max_err)
  end    

  for _, n in ipairs{1,4} do
    test(test_utils.octree_rand(n,1,1,1, 1, 0,0,0))
    test(test_utils.octree_rand(n,1,1,1, 1, 1,0,0))
    test(test_utils.octree_rand(n,1,1,1, 1, 1,1,0))
    test(test_utils.octree_rand(n,1,1,1, 1, 1,1,1))
    test(test_utils.octree_rand(n,2,3,4, 2, 0.5,0.5,0.5))
  end
end



function octest.OctreeConvolution3x3x3()
  for _, out_channels in ipairs{2} do
    for _, grid in ipairs{test_utils.octree_rand(1, 3,2,2, 3, 0.5,0.5,0.5)} do
      print('')
    
      local input = grid
      local in_channels = grid:feature_size()
      local rdc_fcn = 'avg'
      
      local conv = oc.OctreeConvolution3x3x3(in_channels, out_channels, rdc_fcn):float()
      local conv_dense = oc.OctreeDenseConvolution(in_channels, out_channels, rdc_fcn, true):float()
      conv.weight:copy(conv_dense.weight)
      conv.bias:copy(conv_dense.bias)

      -- compare forward to OctreeDenseConvolution
      local conv_out = conv:forward(input)
      local conv_dense_out = conv_dense:forward(input)
      
      print('min/max input : '..input:min()..','..input:max())
      print('min/max output: '..conv_dense_out:min()..','..conv_dense_out:max())
      print('min/max output: '..conv_out:min()..','..conv_out:max())

      if rdc_fcn == 'sum' then
        mytester:assert(conv_out:equals(conv_dense_out, 1e-2, true), 'error in OctreeConvolution3x3x3 forward')
      elseif rdc_fcn == 'avg' then
        mytester:assert(conv_out:equals(conv_dense_out, 1e-5, true), 'error in OctreeConvolution3x3x3 forward')
      end


      -- compare backward to OctreeDenseConvolution
      conv:zeroGradParameters()
      conv_dense:zeroGradParameters()
      local grad_out = conv_out:mul(1e-3)
      local conv_bwd_out = conv:backward(input, grad_out)
      local conv_dense_bwd_out = conv_dense:backward(input, grad_out)

      if rdc_fcn == 'sum' then
        mytester:assert(conv_bwd_out:equals(conv_dense_bwd_out, 1e-2, true), 'error in OctreeConvolution3x3x3 backward (sum)')
      elseif rdc_fcn == 'avg' then
        mytester:assert(conv_bwd_out:equals(conv_dense_bwd_out, 1e-5, true), 'error in OctreeConvolution3x3x3 backward (avg)')
      end

      local werr = torch.abs(conv.gradWeight - conv_dense.gradWeight):max()
      local berr = torch.abs(conv.gradBias - conv_dense.gradBias):max()
      if rdc_fcn == 'sum' then
        mytester:assert(werr < 1e-1, 'error in OctreeConvolution3x3x3 weight backward')
        mytester:assert(berr < 1e-1, 'error in OctreeConvolution3x3x3 bias backward')
      elseif rdc_fcn == 'avg' then
        mytester:assert(werr < 1e-4, 'error in OctreeConvolution3x3x3 weight backward')
        mytester:assert(berr < 1e-4, 'error in OctreeConvolution3x3x3 bias backward')
      end
    end
  end
end

function octest.OctreePool2x2x2()
  -- check gradient with finite differences
  -- takes a lot of time
  local function test_pool_fd(pool, input)
    local output = pool:forward(input)
    local grad_output = output:clone()
    for idx = 0, grad_output:n_elems()-1 do
      grad_output.grid.data[idx] = idx + 1
    end
    pool:backward(input, grad_output)

    local fwd_fcn = function() pool:forward(input) end
    local jac_fwd = test_utils.jacobian_forward(grid.grid.data, grid:n_elems(), output.grid.data, output:n_elems(), fwd_fcn, 1e-3)
    local bwd_fcn = function() pool:zeroGradParameters(); pool:backward(input, grad_output) end
    local jac_bwd = test_utils.jacobian_backward(pool.gradInput.grid.data, pool.gradInput:n_elems(), grad_output.grid.data, grad_output:n_elems(), bwd_fcn)
    local max_err = torch.abs(jac_fwd - jac_bwd):max()
    -- print('')
    -- print(jac_fwd)
    -- print(jac_bwd)
    mytester:assert(max_err < 1e-4, 'Octree2DenseBwd error '..max_err)
  end 

  for _, level_0 in ipairs{false, true} do
  for _, level_1 in ipairs{false, true} do
  for _, level_2 in ipairs{false, true} do
    test_pool_fd(oc.OctreePool2x2x2('avg', leve_0, level_1, level_2):float(), test_utils.octree_rand(1, 2,2,3, 3, 0,0,0))
    test_pool_fd(oc.OctreePool2x2x2('avg', leve_0, level_1, level_2):float(), test_utils.octree_rand(1, 2,2,3, 3, 1,0,0))
    test_pool_fd(oc.OctreePool2x2x2('avg', leve_0, level_1, level_2):float(), test_utils.octree_rand(1, 2,2,3, 3, 1,1,0))
    test_pool_fd(oc.OctreePool2x2x2('avg', leve_0, level_1, level_2):float(), test_utils.octree_rand(1, 2,2,3, 3, 1,1,1))
    test_pool_fd(oc.OctreePool2x2x2('avg', leve_0, level_1, level_2):float(), test_utils.octree_rand(1, 2,2,3, 3, 0.5,0,0))
    test_pool_fd(oc.OctreePool2x2x2('avg', leve_0, level_1, level_2):float(), test_utils.octree_rand(1, 2,2,3, 3, 0.5,0.5,0))
    test_pool_fd(oc.OctreePool2x2x2('avg', leve_0, level_1, level_2):float(), test_utils.octree_rand(1, 2,2,3, 3, 0.5,0.5,0.5))
    test_pool_fd(oc.OctreePool2x2x2('avg', leve_0, level_1, level_2):float(), test_utils.octree_rand(2, 2,2,3, 3, 0.5,0.5,0.5))
  end
  end
  end

end

function octest.OctreeGridPool2x2x2()
  local function test(pool_fcn, grid, fd)
    local pool = oc.OctreeGridPool2x2x2(pool_fcn):float()

    local o2d = oc.OctreeToCDHW()
    local dp
    if pool_fcn == 'avg' then
      dp = nn.VolumetricAveragePooling(2,2,2, 2,2,2)
    elseif pool_fcn == 'max' then
      dp = nn.VolumetricMaxPooling(2,2,2, 2,2,2)
    else
      error('unknown pool fcn')
    end
    local d2o = oc.CDHWToOctree(pool, pool_fcn)
    o2d = o2d:float()
    dp = dp:float()
    d2o = d2o:float()

    -- test fwd
    local gp_out = pool:forward(grid)
    pool.received_input = gp_out

    local cdhw = o2d:forward(grid)
    local cdhw_pool = dp:forward(cdhw)
    local dn_out = d2o:forward(cdhw_pool)

    -- print('------ input ------')
    -- grid:print()
    -- print('------ oc out ------')
    -- gp_out:print()
    -- print('------ dense out ------')
    -- dn_out:print()
    mytester:assert(gp_out:equals(dn_out, 1e-6, true), 'error in OctreeGridPool2x2x2 forward')

    -- test bwd
    local grad_out = gp_out:clone()
    local gp_grad_in = pool:backward(grid, grad_out)

    local d2o_grad_in = d2o:backward(cdhw_pool, grad_out)
    local cdhw_pool_grad_in = dp:backward(cdhw, d2o_grad_in)
    local dn_grad_in = o2d:backward(grid, cdhw_pool_grad_in)
  
    mytester:assert(gp_grad_in:equals(dn_grad_in, 1e-4, true), 'error in OctreeGridPool2x2x2 backward')


    -- check gradient with finite differences
    if fd then
      local output = pool:forward(grid)
      local grad_output = output:clone()
      for idx = 0, grad_output:n_elems()-1 do
        grad_output.grid.data[idx] = math.random(-2, 2)
      end
      pool:backward(grid, grad_output)

      local fwd_fcn = function() pool:forward(grid) end
      local jac_fwd = test_utils.jacobian_forward(grid.grid.data, grid:n_elems(), output.grid.data, output:n_elems(), fwd_fcn, 1e-3)
      local bwd_fcn = function() pool:zeroGradParameters(); pool:backward(grid, grad_output) end
      local jac_bwd = test_utils.jacobian_backward(pool.gradInput.grid.data, pool.gradInput:n_elems(), grad_output.grid.data, grad_output:n_elems(), bwd_fcn)
      local max_err = torch.abs(jac_fwd - jac_bwd):max()
      -- print('')
      -- print(jac_fwd)
      -- print(jac_bwd)
      mytester:assert(max_err < 1e-4, 'OctreeGridPool2x2x2 fd error '..max_err)
    end
  end

  
  local pool_fcn = 'avg'
  for _, n in ipairs{1, 4} do
    test(pool_fcn, test_utils.octree_rand(n, 2,2,2, 2, 0,0,0), true)
    test(pool_fcn, test_utils.octree_rand(n, 2,2,2, 2, 1,1,1), true)
    test(pool_fcn, test_utils.octree_rand(n, 2,2,2, 2, 0.5,0.5,0.5), true)
    test(pool_fcn, test_utils.octree_rand(n, 2,4,6, 2, 0.5,0.5,0.5))
    test(pool_fcn, test_utils.octree_rand(n, 4,8,6, 8, 0.5,0.5,0.5))
  end
end



function octest.OctreeGridUnpoolGuided2x2x2()
  local function test(grid_in)
    local function fwd(grid_data, grid_struct)
      local vup = oc.VolumetricNNUpsampling(2):float()
      local o2d = oc.OctreeToCDHW():float()
      local d2o = oc.CDHWToOctree(o2d, 'avg'):float()

      local out_cdhw = o2d:forward(grid_data)
      local out_up = vup:forward(out_cdhw)
      o2d.received_input = grid_struct
      local out = d2o:forward(out_up)
      return out
    end

    local function bwd(grid_data, grid_struct, grad)
      local vup = oc.VolumetricNNUpsampling(2):float()
      local o2d = oc.OctreeToCDHW():float()
      local d2o = oc.CDHWToOctree(o2d, 'avg'):float()

      local out_cdhw = o2d:forward(grid_data)
      local out_up = vup:forward(out_cdhw)
      o2d.received_input = grid_struct
      local out = d2o:forward(out_up)

      local grad_d2o = d2o:backward(out_up, grad) 
      local grad_down = vup:backward(out_cdhw, grad_d2o)
      local grad_out = o2d:backward(grid_data, grad_down)
      return grad_out
    end
    
    local struct = nn.Identity()
    struct:forward(grid)
    local pooling = oc.OctreeGridPool2x2x2('max')
    local grid_pooled = pooling:forward(grid)

    local test_out = fwd(grid_pooled, grid_in)
    local test_grad = bwd(grid_pooled, grid_in, test_out)

    local mod = oc.OctreeGridUnpoolGuided2x2x2(struct)
    local mod_out = mod:forward(grid_pooled)
    local mod_grad = mod:backward(grid_pooled, test_out)

    mytester:assert(test_out:equals(mod_out, 1e-4, true), 'error in OctreeGridUnpoolGuided2x2x2 forward')
    mytester:assert(test_grad:equals(mod_grad, 1e-4, true), 'error in OctreeGridUnpoolGuided2x2x2 backward')
  end


  for _, n in ipairs{1, 4} do
  for _, fs in ipairs{1, 3} do
    test(test_utils.octree_rand(n, 2,2,2, fs, 0.0,0.0,0.0))
    test(test_utils.octree_rand(n, 2,2,2, fs, 0.0,0.0,0.0))
    test(test_utils.octree_rand(n, 2,2,2, fs, 1.0,0.0,0.0))
    test(test_utils.octree_rand(n, 2,2,2, fs, 1.0,1.0,0.0))
    test(test_utils.octree_rand(n, 2,2,2, fs, 1.0,1.0,1.0))

    test(test_utils.octree_rand(n, 2,4,6, fs, 0.0,0.0,0.0))
    test(test_utils.octree_rand(n, 2,4,6, fs, 0.0,0.0,0.0))
    test(test_utils.octree_rand(n, 2,4,6, fs, 1.0,0.0,0.0))
    test(test_utils.octree_rand(n, 2,4,6, fs, 1.0,1.0,0.0))
    test(test_utils.octree_rand(n, 2,4,6, fs, 1.0,1.0,1.0))

    test(test_utils.octree_rand(n, 2,2,2, fs, 0.5,0.5,0.5))
    test(test_utils.octree_rand(n, 2,4,6, fs, 0.5,0.5,0.5))
    test(test_utils.octree_rand(n, 4,4,6, fs, 0.5,0.5,0.5))
  end
  end
end


function octest.OctreeGridUnpool2x2x2()
  local function test(grid_big)
    local grid_big_cpy = grid_big:clone()
    local grid_small = oc.OctreeGridPool2x2x2('max'):float():forward(grid_big)

    for grid_idx = 1, grid_big:n_blocks() do
      for bit_idx = 10, 73 do
        grid_big:tree_unset_bit(grid_idx, bit_idx)
      end 
    end
    grid_big:update_n_leafs()
    grid_big:update_prefix_leafs()
    
    local mod_struct = nn.Identity():float()
    mod_struct:forward(grid_big)
    local mod_guided = oc.OctreeGridUnpoolGuided2x2x2(mod_struct)

    local mod = oc.OctreeGridUnpool2x2x2()

    local test_out = mod_guided:forward(grid_small)
    local mod_out = mod:forward(grid_small)

    local test_grad = mod_guided:backward(grid_small, test_out)
    local mod_grad = mod:backward(grid_small, test_out)
    
    mytester:assert(test_out:equals(mod_out, 1e-4, true), 'error in OctreeGridUnpool2x2x2 forward')
    mytester:assert(test_grad:equals(mod_grad, 1e-4, true), 'error in OctreeGridUnpool2x2x2 backward')
  end

  for _, n in ipairs{1, 4} do
  for _, fs in ipairs{1, 3} do
    test(test_utils.octree_rand(n, 2,2,2, fs, 0.0,0.0,0.0))
    test(test_utils.octree_rand(n, 2,2,2, fs, 0.0,0.0,0.0))
    test(test_utils.octree_rand(n, 2,2,2, fs, 1.0,0.0,0.0))
    test(test_utils.octree_rand(n, 2,2,2, fs, 1.0,1.0,0.0))
    test(test_utils.octree_rand(n, 2,2,2, fs, 1.0,1.0,1.0))

    test(test_utils.octree_rand(n, 2,4,6, fs, 0.0,0.0,0.0))
    test(test_utils.octree_rand(n, 2,4,6, fs, 0.0,0.0,0.0))
    test(test_utils.octree_rand(n, 2,4,6, fs, 1.0,0.0,0.0))
    test(test_utils.octree_rand(n, 2,4,6, fs, 1.0,1.0,0.0))
    test(test_utils.octree_rand(n, 2,4,6, fs, 1.0,1.0,1.0))

    test(test_utils.octree_rand(n, 2,2,2, fs, 0.5,0.5,0.5))
    test(test_utils.octree_rand(n, 2,4,6, fs, 0.5,0.5,0.5))
    test(test_utils.octree_rand(n, 4,4,6, fs, 0.5,0.5,0.5))
  end
  end
end



function octest.OctreeAdd()
  local function test(oc1)
    local todense = function(x) return oc.OctreeToCDHW():forward(x) end
    local oc2 = oc1:clone():apply(function() torch.uniform(-1,1) end)
    local de1 = todense(oc1)
    local de2 = todense(oc2)

    local add1 = torch.uniform(-1,1)
    local ocr = oc1:add(add1)
    local der = de1:add(add1)
    mytester:assert(torch.abs(todense(ocr) - der):max() < 1e-6, 'add1')

    local ocr = oc1:add(oc2) 
    local der = de1:add(de2)
    mytester:assert(torch.abs(todense(ocr) - der):max() < 1e-6, 'add oc2')

    local ocr = oc1:add(oc1, oc2) 
    local der = de1:add(de1, de2)
    mytester:assert(torch.abs(todense(ocr) - der):max() < 1e-6, 'add oc1 and oc2')

    local fac = torch.uniform(-2, 2)
    local ocr = oc1:add(fac, oc2) 
    local der = de1:add(fac, de2)
    mytester:assert(torch.abs(todense(ocr) - der):max() < 1e-6, 'add fac oc2')

    local fac = torch.uniform(-2, 2)
    local ocr = oc1:add(oc1, fac, oc2) 
    local der = de1:add(de1, fac, de2)
    mytester:assert(torch.abs(todense(ocr) - der):max() < 1e-6, 'add oc1 and fac oc2')

    local fac1 = torch.uniform(-2, 2)
    local fac2 = torch.uniform(-2, 2)
    local ocr = oc1:add(fac1, oc1, fac2, oc2) 
    local der = de1:add(de1:mul(fac1), fac2, de2)
    mytester:assert(torch.abs(todense(ocr) - der):max() < 1e-5, 'add fac oc1 and fac oc2')

    for _, ip in ipairs{true, false} do
      local err_str = 'CAddTable'
      if ip then err_str = err_str..'(ip)' end
      local cadd_oc = nn.CAddTable(ip):oc()
      local cadd_de = nn.CAddTable(ip):float()

      local ocr = cadd_oc:forward({oc1, oc2})
      local der = cadd_de:forward({de1, de2})
      mytester:assert(torch.abs(todense(ocr) - der):max() < 1e-5, err_str..' forward')

      local ocr = cadd_oc:backward({oc1, oc2}, oc2)
      local der = cadd_de:backward({de1, de2}, de2)
      mytester:assert(torch.abs(todense(ocr[1]) - der[1]):max() < 1e-6, err_str..' backward')
      mytester:assert(torch.abs(todense(ocr[2]) - der[2]):max() < 1e-6, err_str..' backward')
    end
  end

  for _, n in ipairs{1, 4} do
    test( test_utils.octree_rand(n, 1,1,1, 1, 0,0,0) )
    test( test_utils.octree_rand(n, 2,3,4, 2, 0,0,0) )
    test( test_utils.octree_rand(n, 2,3,4, 2, 1,0,0) )
    test( test_utils.octree_rand(n, 2,3,4, 2, 1,1,0) )
    test( test_utils.octree_rand(n, 2,3,4, 2, 1,1,1) )
    test( test_utils.octree_rand(n, 2,3,4, 2, 0.5,0.5,0.5) )
  end
end


function octest.OctreeConcat()
  local function test(grid_in1)
    local grid_in2 = grid_in1:clone()
    grid_in2:resize(grid_in1:n(), grid_in1:grid_depth(), grid_in1:grid_height(), grid_in1:grid_width(), grid_in1:feature_size()*2, grid_in1:n_leafs())
    grid_in2:update_prefix_leafs()
    for didx = 0, grid_in2:n_elems()-1 do
      grid_in2.grid.data[didx] = didx % 7
    end
    
    local mod = oc.OctreeConcat(true):float()
    local grid_out = mod:forward({grid_in1, grid_in2})

    local grid_i1 = grid_in1.grid
    local grid_i2 = grid_in2.grid
    local grid_o = grid_out.grid
    for lidx = 0, grid_in1:n_leafs()-1 do
      for f = 0, grid_in1:feature_size()-1 do
        local err = math.abs(grid_i1.data[lidx * grid_in1:feature_size() + f] - grid_o.data[lidx * grid_out:feature_size() + f])
        mytester:assertlt(err, 1e-6, 'error in OctreeConcat fwd')
      end
      for f = 0, grid_in2:feature_size()-1 do
        local err = math.abs(grid_i2.data[lidx * grid_in2:feature_size() + f] - grid_o.data[lidx * grid_out:feature_size() + grid_in1:feature_size() + f])
        mytester:assertlt(err, 1e-6, 'error in OctreeConcat fwd')
      end
    end 


    local grid_out = grid_out:clone()
    local grad_in = mod:backward({grid_in1, grid_in2}, grid_out)

    local grid_o = grid_out.grid
    local grid_i1 = grad_in[1].grid
    local grid_i2 = grad_in[2].grid
    for lidx = 0, grid_in1:n_leafs()-1 do
      for f = 0, grid_in1:feature_size()-1 do
        local err = math.abs(grid_i1.data[lidx * grid_in1:feature_size() + f] - grid_o.data[lidx * grid_out:feature_size() + f])
        mytester:assertlt(err, 1e-6, 'error in OctreeConcat bwd')
      end
      for f = 0, grid_in2:feature_size()-1 do
        local err = math.abs(grid_i2.data[lidx * grid_in2:feature_size() + f] - grid_o.data[lidx * grid_out:feature_size() + grid_in1:feature_size() + f])
        mytester:assertlt(err, 1e-6, 'error in OctreeConcat bwd')
      end
    end 
  end

  function test_dense(in1, fs2, do_grad_in2)
    local in2 = torch.zeros(in1:size(1), fs2, in1:size(3), in1:size(4), in1:size(5))
    in2:apply(function() torch.uniform(-1,1) end)
    in2 = in2:float()

    local o2c = oc.OctreeToCDHW():float()
    o2c:forward(in1)
    local c2o = oc.CDHWToOctree(o2c):float()
    local in2_o = c2o:forward(in2)

    local mod1 = oc.OctreeConcat(false, do_grad_in2):float()
    local mod2 = oc.OctreeConcat(false, do_grad_in2):float()

    local out1 = mod1:forward({in1, in2_o})
    local out2 = mod2:forward({in1, in2})
    mytester:assert(out1:equals(out2, 1e-6, true), 'error in OctreeConcat(dense) fwd')

    local test_a = oc.OctreeToCDHW():forward(out2)
    local test_b = torch.cat(oc.OctreeToCDHW():forward(in1), in2, 2)
    local err = torch.abs(test_a - test_b):max()
    mytester:assert(err < 1e-6, 'error in OctreeConcat(dense) fwd test')


    local grad_out = out1:clone()
    grad_out:apply(function() torch.uniform(-1,1) end)
    local out1 = mod1:backward({in1, in2_o}, grad_out)
    local out2 = mod2:backward({in1, in2}, grad_out)
    
    mytester:assert(out1[1]:equals(out2[1], 1e-6, true), 'error in OctreeConcat(dense) bwd1')
    if do_grad_in2 then
      local err = torch.abs(o2c:forward(out1[2]) - out2[2]):max()
      mytester:assertlt(err, 1e-6, 'error in OctreeConcat(dense) bwd2')
    end

    local test_a = oc.OctreeToCDHW():forward(out2[1])
    local test_b = oc.OctreeToCDHW():forward(grad_out)[{{}, {1,in1:feature_size()}}]
    local err = torch.abs(test_a - test_b):max()
    mytester:assert(err < 1e-6, 'error in OctreeConcat(dense) bwd1 test')

    if do_grad_in2 then
      local test_a = out2[2]
      local test_b = oc.OctreeToCDHW():forward(grad_out)[{{}, {in1:feature_size()+1, in1:feature_size()+in2:size(2)}}]
      local err = torch.abs(test_a - test_b):max()
      mytester:assert(err < 1e-6, 'error in OctreeConcat(dense) bwd1 test')
    end

  end


  for _, n in ipairs{1, 4} do
  for fs = 1, 3 do
  for _, do_grad_in2 in ipairs{true, false} do
    test( test_utils.octree_rand(n, 1,1,1, fs, 0,0,0) )
    test( test_utils.octree_rand(n, 2,3,4, fs, 0,0,0) )
    test( test_utils.octree_rand(n, 2,3,4, fs, 1,0,0) )
    test( test_utils.octree_rand(n, 2,3,4, fs, 1,1,0) )
    test( test_utils.octree_rand(n, 2,3,4, fs, 1,1,1) )
    test( test_utils.octree_rand(n, 2,3,4, fs, 0.5,0.5,0.5) ) 
    
    test_dense( test_utils.octree_rand(n, 1,1,1, fs, 0,0,0), 4, do_grad_in2 )
    test_dense( test_utils.octree_rand(n, 2,3,4, fs, 0,0,0), 4, do_grad_in2 )
    test_dense( test_utils.octree_rand(n, 2,3,4, fs, 1,0,0), 4, do_grad_in2 )
    test_dense( test_utils.octree_rand(n, 2,3,4, fs, 1,1,0), 4, do_grad_in2 )
    test_dense( test_utils.octree_rand(n, 2,3,4, fs, 1,1,1), 4, do_grad_in2 )
    test_dense( test_utils.octree_rand(n, 2,3,4, fs, 0.5,0.5,0.5), 4, do_grad_in2 ) 
  end
  end
  end
end


function octest.OctreeReLU()
  local function test(inplace, grid_in)
    local mod_oc = oc.OctreeReLU(inplace):float()
    local mod_de = nn.ReLU():float()

    local out_de = mod_de:forward( oc.OctreeToCDHW():forward(grid_in) )
    local out_oc = mod_oc:forward(grid_in)
    
    local err = torch.abs(oc.OctreeToCDHW():forward(out_oc) - out_de):max()
    mytester:assert(err < 1e-6, 'error in OctreeReLU forward')
    if inplace then
      mytester:assert(out_oc.grid == grid_in.grid, 'error in OctreeReLU forward')
    end

    local grad_out = grid_in:clone():apply(function() return torch.uniform(-1,1) end)
    local grad_de = mod_de:backward(oc.OctreeToCDHW():forward(grid_in), oc.OctreeToCDHW():forward(grad_out))
    local grad_oc = mod_oc:backward(grid_in, grad_out)
    
    local err = torch.abs(oc.OctreeToCDHW():forward(out_oc) - out_de):max()
    mytester:assert(err < 1e-6, 'error in OctreeReLU backward')
    if inplace then
      mytester:assert(out_oc.gradInput == grid_in.gradInput, 'error in OctreeReLU backward')
    end
  end

  for _, n in ipairs{1, 4} do
    for _, inplace in ipairs{false, true} do
      test(inplace, test_utils.octree_rand(n, 1,1,1, 1, 0,0,0, -1,1))
      test(inplace, test_utils.octree_rand(n, 2,3,4, 2, 0,0,0, -1,1))
      test(inplace, test_utils.octree_rand(n, 2,3,4, 2, 1,0,0, -1,1))
      test(inplace, test_utils.octree_rand(n, 2,3,4, 2, 1,1,0, -1,1))
      test(inplace, test_utils.octree_rand(n, 2,3,4, 2, 1,1,1, -1,1))
      test(inplace, test_utils.octree_rand(n, 2,3,4, 2, 0.5,0.5,0.5, -1,1))
    end
  end
end

function octest.OctreeSigmoid()
  local function test(inplace, grid_in)
    local mod_oc = oc.OctreeSigmoid(inplace):float()
    local mod_de = nn.Sigmoid():float()

    local out_de = mod_de:forward( oc.OctreeToCDHW():forward(grid_in) )
    local out_oc = mod_oc:forward(grid_in)
    
    local err = torch.abs(oc.OctreeToCDHW():forward(out_oc) - out_de):max()
    mytester:assert(err < 1e-6, 'error in OctreeReLU forward err='..err)
    if inplace then
      mytester:assert(out_oc.grid == grid_in.grid, 'error in OctreeReLU forward (ip)')
    end

    local grad_out = grid_in:clone():apply(function() return torch.uniform(-1,1) end)
    local grad_de = mod_de:backward(oc.OctreeToCDHW():forward(grid_in), oc.OctreeToCDHW():forward(grad_out))
    local grad_oc = mod_oc:backward(grid_in, grad_out)
    
    local err = torch.abs(oc.OctreeToCDHW():forward(out_oc) - out_de):max()
    mytester:assert(err < 1e-6, 'error in OctreeReLU backward err='..err)
    if inplace then
      mytester:assert(out_oc.gradInput == grid_in.gradInput, 'error in OctreeReLU backward (ip)')
    end
  end

  test(true, test_utils.octree_rand(1, 1,1,1, 1, 0,0,0, -1,1))
  for _, n in ipairs{1, 4} do
    for _, inplace in ipairs{false, true} do
      test(inplace, test_utils.octree_rand(n, 1,1,1, 1, 0,0,0, -1,1))
      test(inplace, test_utils.octree_rand(n, 2,3,4, 2, 0,0,0, -1,1))
      test(inplace, test_utils.octree_rand(n, 2,3,4, 2, 1,0,0, -1,1))
      test(inplace, test_utils.octree_rand(n, 2,3,4, 2, 1,1,0, -1,1))
      test(inplace, test_utils.octree_rand(n, 2,3,4, 2, 1,1,1, -1,1))
      test(inplace, test_utils.octree_rand(n, 2,3,4, 2, 0.5,0.5,0.5, -1,1))
    end
  end
end

function octest.OctreeLogSoftMax()
  local function test(in1)
    local o2d = oc.OctreeToCDHW()
    local d2o = oc.CDHWToOctree(o2d, 'avg')
    
    local dense_mod = nn.LogSoftMax():float()
    local dense_in = o2d:forward(in1):clone()
    local dense_size = dense_in:size()
    local dense_out = dense_mod:forward(dense_in:view(dense_size[1], dense_size[2], dense_size[3]*dense_size[4], dense_size[5]))
    dense_out = dense_out:view(dense_size)

    local oc_mod = oc.OctreeLogSoftMax()
    local oc_out = oc_mod:forward(in1)
    local oc_out_d = o2d:forward(oc_out)

    local max_err = torch.abs(dense_out - oc_out_d):max()
    mytester:assert(max_err < 1e-6, 'error in OctreeLogSoftMax forward')

    
    local dense_grad = dense_mod:backward(dense_in, o2d:forward(oc_out):clone())
    dense_grad = dense_grad:view(dense_size)
    local oc_dense_grad = d2o:forward(dense_grad)
    local oc_grad = oc_mod:backward(in1, oc_out)
    
    mytester:assert(oc_grad:equals(oc_dense_grad, 1e-4, true), 'error in OctreeLogSoftMax backward')
  end

  for _, n in ipairs{1, 4} do
    test(test_utils.octree_rand(n, 2,2,2, 3, 0,0,0))
    test(test_utils.octree_rand(n, 2,2,2, 3, 1,0,0))
    test(test_utils.octree_rand(n, 2,2,2, 3, 1,1,0))
    test(test_utils.octree_rand(n, 2,2,2, 3, 1,1,1))
    test(test_utils.octree_rand(n, 2,2,2, 3, 0.5,0.5,0.5))
    test(test_utils.octree_rand(n, 2,4,6, 3, 0.5,0.5,0.5))
    test(test_utils.octree_rand(n, 4,8,6, 8, 0.5,0.5,0.5))
  end
end


function octest.OctreeSplit()
  local function bwd(data, struct, grad_out)
    local o2d = oc.OctreeToCDHW():float()
    local d2o = oc.CDHWToOctree(o2d, 'avg'):float()

    local out_cdhw = o2d:forward(data)
    o2d.received_input = struct
    local out = d2o:forward(out_cdhw)
    local grad_d2o = d2o:backward(out_cdhw, grad_out)
    local grad_in = o2d:backward(data, grad_d2o)
    return grad_in
  end

  local function test_by_prob(prob, data)
    local threshold = 0
    local mod_prob = nn.Identity():float()
    local mod = oc.OctreeSplitByProb(mod_prob, threshold, true):float()

    mod_prob:forward(prob)
    local mod_out = mod:forward(data)

    -- test structure
    for grid_idx = 1, prob:n_blocks() do
      for bit_idx = 1, 73 do
        local in_bit = prob:tree_isset_bit(grid_idx, bit_idx)
        local in_pa_bit = prob:tree_isset_bit(grid_idx, prob:tree_parent_bit_idx(bit_idx))
        local out_bit = mod_out:tree_isset_bit(grid_idx, bit_idx)
        local out_pa_bit = mod_out:tree_isset_bit(grid_idx, mod_out:tree_parent_bit_idx(bit_idx))
        if in_bit == true then 
          mytester:assert( out_bit == true, 'OctreeSplit: split node in in is not replicated in mod_out' ) 
        end
        if in_bit == false and (bit_idx == 1 or in_pa_bit == true) then
          local in_data_idx = prob:tree_data_idx(grid_idx, bit_idx)
          -- local p = prob.grid.data_ptrs[grid_idx-1][in_data_idx]
          local p = prob.grid.data[prob.grid.prefix_leafs[grid_idx-1] * prob.grid.feature_size + in_data_idx]
          if p >= threshold then
            mytester:assert( out_bit == true, string.format('OctreeSplitByProb: there should be a split because p (%f) > threshold (%f)', p, threshold) ) 
          end 
        end
        if out_bit == true then
          mytester:assert(bit_idx == 1 or out_pa_bit == true, ' OctreeSplitByProb: there can not be a split without a parent split')
        end 
      end 
    end 

    -- test fwd content
    local oc_out = oc.OctreeToCDHW():forward(mod_out)
    local de_out = oc.OctreeToCDHW():forward(data) 
    local err = torch.abs(oc_out - de_out):max()
    mytester:assert(err < 1e-6, 'error in OctreeSplitByProb forward err='..err)

    local grad_out = mod_out:clone():apply(function() return torch.uniform(-1,1) end)
    local oc_grad = bwd(data, mod_out, grad_out:clone())
    local de_grad = mod:backward(data, grad_out)
    mytester:assert(oc_grad:equals(de_grad, 1e-4, true), 'error in OctreeSplitByProb backward')
  end

  local function test_full(data)
    local mod = oc.OctreeSplitFull():float()
    local mod_out = mod:forward(data)

    -- test structure
    mytester:assert(mod_out:n_leafs() == mod_out:n_blocks()*512, 'error in OctreeSplitFull struct')

    -- test fwd content
    local oc_out = oc.OctreeToCDHW():forward(mod_out)
    local de_out = oc.OctreeToCDHW():forward(data) 
    local err = torch.abs(oc_out - de_out):max()
    mytester:assert(err < 1e-6, 'error in OctreeSplitFull forward err='..err)

    local grad_out = mod_out:clone():apply(function() return torch.uniform(-1,1) end)
    local oc_grad = bwd(data, mod_out, grad_out:clone())
    local de_grad = mod:backward(data, grad_out)
    mytester:assert(oc_grad:equals(de_grad, 1e-4, true), 'error in OctreeSplitFull backward')
  end

  local function test_densesurfrecfres(n, fs, vx_res, band)
    local features = torch.rand(n, fs, vx_res,vx_res,vx_res):float()
    local rec = torch.rand(n, 1, vx_res,vx_res,vx_res):float()

    local rec_mod = nn.Identity()
    rec_mod:forward(rec)
    local mod = oc.OctreeDenseSplitSurfFres(rec_mod, 0.95, 1.0, band):float()
    local out = mod:forward(features)
    
    local test_mod = nn.Sequential()
      :add( oc.CDHWToOctree(mod, 'avg') )
    test_mod = test_mod:float()
    mod.received_input = out
    local test_out = test_mod:forward(features)

    -- out = out:float()
    -- test_out = test_out:float()
    -- print(features)
    -- print(rec)
    -- out:print()
    -- test_out:print()
    mytester:assert(out:equals(test_out, 1e-6, true), 'error in OctreeDenseSplitSurfFres forward')


    local grad_out = out:clone()
    grad_out:apply(function(x) return math.random() end)
    -- grad_out = grad_out:float()

    local out = mod:backward(features, grad_out)
    local test_out = mod:backward(features, grad_out)

    -- print(out)
    -- print(test_out)
    local err = torch.abs(out - test_out):max()
    mytester:assert(err < 1e-6, 'error in OctreeDenseSplitSurfFres backward')
  end

  for _, band in ipairs{0,1,3} do
    test_densesurfrecfres(1, 1, 8, band)
    test_densesurfrecfres(1, 1, 16, band)
    test_densesurfrecfres(1, 3, 16, band)
    test_densesurfrecfres(4, 5, 32, band)
  end

  for n in ipairs{1, 4} do
    for _, fs in ipairs{1,3,5} do
      local data = test_utils.octree_rand(n, 1,1,1, fs, 0,0,0, -1,1)
      test_by_prob(test_utils.octree_alter_fs(data, 1), data)
      test_full(data)

      local data = test_utils.octree_rand(n, 1,1,1, fs, 1,0,0, -1,1)
      test_by_prob(test_utils.octree_alter_fs(data, 1), data)
      test_full(data)

      local data = test_utils.octree_rand(n, 1,1,1, fs, 1,1,0, -1,1)
      test_by_prob(test_utils.octree_alter_fs(data, 1), data)
      test_full(data)

      local data = test_utils.octree_rand(n, 1,1,1, fs, 1,1,1, -1,1)
      test_by_prob(test_utils.octree_alter_fs(data, 1), data)
      test_full(data)
      
      local data = test_utils.octree_rand(n, 1,1,1, fs, 0.5,0.5,0.5, -1,1)
      test_by_prob(test_utils.octree_alter_fs(data, 1), data)
      test_full(data)
      
      local data = test_utils.octree_rand(n, 2,3,4, fs, 0.5,0.5,0.5, -1,1)
      test_by_prob(test_utils.octree_alter_fs(data, 1), data)
      test_full(data)
    end
  end
end


function octest.OctreeMaskByLabel()
  local function test(input)
    local labels = input:clone()
    labels.grid.feature_size = 1
    labels:update_prefix_leafs()
    local data = torch.FloatTensor(labels:n_leafs() * labels:feature_size())
    data:apply(function() return torch.random(1, 3) end)
    labels:cpy_data(data)

    local mask_label = 1

    local input_d = oc.OctreeToCDHW():forward(input)
    local labels_d = oc.OctreeToCDHW():forward(labels)

    local out = input:mask_by_label(labels, mask_label, true)
    local out_d = oc.OctreeToCDHW():forward(out)

    for b = 1, input_d:size(1) do
    for c = 1, input_d:size(2) do
    for d = 1, input_d:size(3) do
    for h = 1, input_d:size(4) do
    for w = 1, input_d:size(5) do
      local l = labels_d[{b,1,d,h,w}]
      local i = input_d[{b,c,d,h,w}]
      local o = out_d[{b,c,d,h,w}]
      if l == mask_label then
        mytester:assert(o == 0, 'error in OctreeMaskByLabel')
      else
        mytester:assert(o == i, 'error in OctreeMaskByLabel')
      end
    end
    end
    end
    end
    end
  end

  for _, n in ipairs{1, 4} do
    test(test_utils.octree_rand(n, 2,2,2, 3, 0,0,0))
    test(test_utils.octree_rand(n, 2,2,2, 3, 1,0,0))
    test(test_utils.octree_rand(n, 2,2,2, 3, 1,1,0))
    test(test_utils.octree_rand(n, 2,2,2, 3, 1,1,1))
    test(test_utils.octree_rand(n, 2,2,2, 3, 0.5,0.5,0.5))
    test(test_utils.octree_rand(n, 2,4,6, 3, 0.5,0.5,0.5))
    test(test_utils.octree_rand(n, 4,8,6, 8, 0.5,0.5,0.5))
  end
end


function octest.OctreeMSECriterion()
  local function test(in1)
    local in2 = in1:clone()
    local data = torch.FloatTensor(in2:n_leafs() * in2:feature_size())
    data:apply(function() return torch.uniform(-1, 1) end)
    in2:cpy_data(data)

    local o2d1 = oc.OctreeToCDHW()
    local o2d2 = oc.OctreeToCDHW()
    local d2o = oc.CDHWToOctree(o2d1, 'sum')
    
    local dense_crit = nn.MSECriterion(true):float()
    local dense_out = dense_crit:forward(o2d1:forward(in1), o2d2:forward(in2))

    local oc_crit = oc.OctreeMSECriterion(true, true)
    local oc_out = oc_crit:forward(in1, in2)

    -- print(dense_out, oc_out, torch.abs(dense_out - oc_out))
    mytester:assert(torch.abs(dense_out - oc_out) < 1e-3, 'error in OctreeMSECriterion forward')

    
    local dense_grad = dense_crit:backward(o2d1:forward(in1), o2d2:forward(in2))
    local oc_dense_grad = d2o:forward(dense_grad)
    local oc_grad = oc_crit:backward(in1, in2)
    
    mytester:assert(oc_grad:equals(oc_dense_grad, 1e-5, true), 'error in OctreeMSECriterion backward')
  end

  
  for _, n in ipairs{1, 4} do
    test(test_utils.octree_rand(n, 2,2,2, 2, 0,0,0))
    test(test_utils.octree_rand(n, 2,2,2, 2, 1,0,0))
    test(test_utils.octree_rand(n, 2,2,2, 2, 1,1,0))
    test(test_utils.octree_rand(n, 2,2,2, 2, 1,1,1))
    test(test_utils.octree_rand(n, 2,2,2, 2, 0.5,0.5,0.5))
    test(test_utils.octree_rand(n, 2,4,6, 2, 0.5,0.5,0.5))
  end
end


function octest.OctreeClassNLLCriterion()
  local function test(in1, weights)
    local in2 = in1:clone()
    local data = torch.FloatTensor(in2:n_leafs() * in2:feature_size())
    data:apply(function() return torch.random(1, in1:feature_size()) end)
    in2:cpy_data(data)
    in2.grid.feature_size = 1

    local o2d1 = oc.OctreeToCDHW()
    local o2d2 = oc.OctreeToCDHW()
    local d2o = oc.CDHWToOctree(o2d1, 'sum')
    
    local dense_crit = nn.SpatialClassNLLCriterion(weights):float()
    local dense_in = o2d1:forward(in1)
    local dense_ta = o2d2:forward(in2)
    local dense_in_size = dense_in:size()
    local dense_ta_size = dense_ta:size()
    dense_in = dense_in:view(dense_in_size[1], dense_in_size[2], dense_in_size[3]*dense_in_size[4], dense_in_size[5])
    dense_ta = dense_ta:view(dense_ta_size[1], dense_ta_size[3]*dense_ta_size[4], dense_ta_size[5])
    local dense_out = dense_crit:forward(dense_in, dense_ta)

    local oc_crit = oc.OctreeClassNLLCriterion(weights, true)
    local oc_out = oc_crit:forward(in1, in2)

    mytester:assert(torch.abs(dense_out - oc_out) < 1e-3, 'error in OctreeClassNLLCriterion forward')

    
    local dense_grad = dense_crit:backward(dense_in, dense_ta)
    local oc_dense_grad = d2o:forward(dense_grad:view(dense_in_size))
    local oc_grad = oc_crit:backward(in1, in2)
    
    mytester:assert(oc_grad:equals(oc_dense_grad, 1e-5, true), 'error in OctreeClassNLLCriterion backward')
  end

  
  local fs = 8
  for _, n in ipairs{1, 4} do
    for _, weights in ipairs{torch.ones(fs):float(), torch.rand(fs):float()} do
      test(test_utils.octree_rand(n, 2,2,2, fs, 0,0,0), weights)
      test(test_utils.octree_rand(n, 2,2,2, fs, 1,0,0), weights)
      test(test_utils.octree_rand(n, 2,2,2, fs, 1,1,0), weights)
      test(test_utils.octree_rand(n, 2,2,2, fs, 1,1,1), weights)
      test(test_utils.octree_rand(n, 2,2,2, fs, 0.5,0.5,0.5), weights)
      test(test_utils.octree_rand(n, 2,4,6, fs, 0.5,0.5,0.5), weights)
      test(test_utils.octree_rand(n, 4,8,6, fs, 0.5,0.5,0.5), weights)
    end
  end
end

function octest.OctreeCrossEntropyCriterion()
  local function test(in1, weights)
    local in2 = in1:clone()
    local data = torch.FloatTensor(in2:n_leafs() * in2:feature_size())
    data:apply(function() return torch.random(1, in1:feature_size()) end)
    in2:cpy_data(data)
    in2.grid.feature_size = 1

    local o2d1 = oc.OctreeToCDHW()
    local o2d2 = oc.OctreeToCDHW()
    local d2o = oc.CDHWToOctree(o2d1, 'sum')
    
    local dense_crit = nn.CrossEntropyCriterion(weights):float()
    local dense_in = o2d1:forward(in1)
    local dense_ta = o2d2:forward(in2)
    dense_in = dense_in:transpose(2,3):transpose(3,4):transpose(4,5):contiguous() -- bcdhw - bdchw - bdhcw - bdhwc
    local dense_in_size = dense_in:size()
    dense_in = dense_in:view(dense_in:size(1) * dense_in:size(2) * dense_in:size(3) * dense_in:size(4), dense_in:size(5))
    dense_ta = dense_ta:view(dense_ta:size(1) * dense_ta:size(3) * dense_ta:size(4) * dense_ta:size(5))
    local dense_out = dense_crit:forward(dense_in, dense_ta)

    local oc_crit = oc.OctreeCrossEntropyCriterion(weights, true)
    local oc_out = oc_crit:forward(in1, in2)

    -- print(dense_out, oc_out, torch.abs(dense_out - oc_out))
    mytester:assert(torch.abs(dense_out - oc_out) < 1e-3, 'error in OctreeClassNLLCriterion forward')

    
    local dense_grad = dense_crit:backward(dense_in, dense_ta)
    dense_grad = dense_grad:view(dense_in_size):transpose(4,5):transpose(3,4):transpose(2,3):contiguous() -- bdhwc - bdhcw - bdchw - bcdhw
    local oc_dense_grad = d2o:forward(dense_grad)
    local oc_grad = oc_crit:backward(in1, in2)
    
    mytester:assert(oc_grad:equals(oc_dense_grad, 1e-5, true), 'error in OctreeClassNLLCriterion backward')
  end

  
  local fs = 8
  for _, n in ipairs{1, 4} do
    for _, weights in ipairs{torch.ones(fs):float(), torch.rand(fs):float()} do
      test(test_utils.octree_rand(n, 2,2,2, fs, 0,0,0), weights)
      test(test_utils.octree_rand(n, 2,2,2, fs, 1,0,0), weights)
      test(test_utils.octree_rand(n, 2,2,2, fs, 1,1,0), weights)
      test(test_utils.octree_rand(n, 2,2,2, fs, 1,1,1), weights)
      test(test_utils.octree_rand(n, 2,2,2, fs, 0.5,0.5,0.5), weights)
      test(test_utils.octree_rand(n, 2,4,6, fs, 0.5,0.5,0.5), weights)
    end
  end
end


function octest.OctreeBCECriterion()
  local function test_grid_grid(in1)
    local in2 = in1:clone()
    local data = torch.FloatTensor(in2:n_leafs() * in2:feature_size())
    data:apply(function() return torch.uniform(0, 1) end)
    in2:cpy_data(data)

    local o2d1 = nn.Sequential()
      :add( oc.OctreeToCDHW() )
      :float()
    local o2d2 = oc.OctreeToCDHW()

    local dense_crit = nn.BCECriterion(nil, true):float()
    local dense_out = dense_crit:forward(o2d1:forward(in1), o2d2:forward(in2))

    local oc_crit = oc.OctreeBCECriterion(true, true)
    local oc_out = oc_crit:forward(in1, in2)

    local err = torch.abs(dense_out - oc_out)
    -- print(dense_out, oc_out, err)
    mytester:assert(err < 1e-3, 'error in OctreeBCECriterion grid-grid forward: err='..err)

    
    local dense_grad = dense_crit:backward(o2d1:forward(in1), o2d2:forward(in2))
    local oc_dense_grad = o2d1:backward(in1, dense_grad)
    local oc_grad = oc_crit:backward(in1, in2)
    
    mytester:assert(oc_grad:equals(oc_dense_grad, 1e-4, true), 'error in OctreeBCECriterion grid-grid backward')
  end
  
  local function test_grid_dense(in1)
    local in2 = torch.FloatTensor(in1:n(), in1:feature_size(), in1:dense_depth(), in1:dense_height(), in1:dense_width())
    in2:apply(function() return torch.uniform(0, 1) end)

    local o2d1 = nn.Sequential()
      :add( oc.OctreeToCDHW() )
      :float()
    local o2d2 = oc.OctreeToCDHW()

    local dense_crit = nn.BCECriterion(nil, true):float()
    local dense_out = dense_crit:forward(o2d1:forward(in1), in2)

    local oc_crit = oc.OctreeBCECriterion(true, true)
    local oc_out = oc_crit:forward(in1, in2)

    local err = torch.abs(dense_out - oc_out) 
    -- print(dense_out, oc_out, err)
    mytester:assert(err < 1e-3, 'error in OctreeBCECriterion grid-dense forward: err='..err)

    
    local dense_grad = dense_crit:backward(o2d1:forward(in1), in2)
    local oc_dense_grad = o2d1:backward(in1, dense_grad)
    local oc_grad = oc_crit:backward(in1, in2)
    
    mytester:assert(oc_grad:equals(oc_dense_grad, 1e-5, true), 'error in OctreeBCECriterion grid-dense backward')
  end

  local function test_grid_grid_ds(in1, in2)
    local o2d1 = nn.Sequential()
      :add( oc.OctreeToCDHW() )
      :float()
    local o2d2 = oc.OctreeToCDHW()

    local dense_crit = nn.BCECriterion(nil, true):float()
    local dense_out = dense_crit:forward(o2d1:forward(in1), o2d2:forward(in2))

    local oc_crit = oc.OctreeBCECriterion(true, true, true)
    local oc_out = oc_crit:forward(in1, in2)

    local err = torch.abs(dense_out - oc_out)
    -- print(dense_out, oc_out, err)
    mytester:assert(err < 1e-3, 'error in OctreeBCECriterion grid-grid-ds forward: err='..err)

    
    local dense_grad = dense_crit:backward(o2d1:forward(in1), o2d2:forward(in2))
    local oc_dense_grad = o2d1:backward(in1, dense_grad)
    local oc_grad = oc_crit:backward(in1, in2)
    
    mytester:assert(oc_grad:equals(oc_dense_grad, 1e-5, true), 'error in OctreeBCECriterion grid-grid-ds backward')
  end

  local ocrand = test_utils.octree_rand 
  for _, n in ipairs{1, 4} do
    test_grid_grid(ocrand(n, 1,1,1, 1, 0,0,0))
    test_grid_grid(ocrand(n, 2,2,2, 2, 0,0,0))
    test_grid_grid(ocrand(n, 2,2,2, 2, 1,0,0))
    test_grid_grid(ocrand(n, 2,2,2, 2, 1,1,0))
    test_grid_grid(ocrand(n, 2,2,2, 2, 1,1,1))
    test_grid_grid(ocrand(n, 2,2,2, 2, 0.5,0.5,0.5))
    test_grid_grid(ocrand(n, 2,4,6, 2, 0.5,0.5,0.5))
    test_grid_grid(ocrand(n, 4,8,6, 6, 0.5,0.5,0.5))

    test_grid_dense(ocrand(n, 1,1,1, 1, 0,0,0))
    test_grid_dense(ocrand(n, 2,2,2, 2, 0,0,0)) 
    test_grid_dense(ocrand(n, 2,2,2, 2, 1,0,0))
    test_grid_dense(ocrand(n, 2,2,2, 2, 1,1,0))
    test_grid_dense(ocrand(n, 2,2,2, 2, 1,1,1))
    test_grid_dense(ocrand(n, 2,2,2, 2, 0.5,0.5,0.5))
    test_grid_dense(ocrand(n, 2,4,6, 2, 0.5,0.5,0.5))
    test_grid_dense(ocrand(n, 4,8,6, 6, 0.5,0.5,0.5))

    test_grid_grid_ds(ocrand(n, 1,1,1, 1, 0,0,0), ocrand(n, 1,1,1, 1, 0,0,0))
    test_grid_grid_ds(ocrand(n, 1,1,1, 1, 0,0,0), ocrand(n, 1,1,1, 1, 0.5,0.5,0.5))
    test_grid_grid_ds(ocrand(n, 2,2,2, 2, 0,0,0), ocrand(n, 2,2,2, 2, 0,0,0)) 
    test_grid_grid_ds(ocrand(n, 2,2,2, 2, 0,0,0), ocrand(n, 2,2,2, 2, 0.5,0.5,0.5)) 
    test_grid_grid_ds(ocrand(n, 2,2,2, 2, 1,0,0), ocrand(n, 2,2,2, 2, 1,0,0)) 
    test_grid_grid_ds(ocrand(n, 2,2,2, 2, 1,0,0), ocrand(n, 2,2,2, 2, 0.5,0.5,0.5)) 
    test_grid_grid_ds(ocrand(n, 2,2,2, 2, 1,1,0), ocrand(n, 2,2,2, 2, 1,1,0)) 
    test_grid_grid_ds(ocrand(n, 2,2,2, 2, 1,1,0), ocrand(n, 2,2,2, 2, 0.5,0.5,0.5)) 
    test_grid_grid_ds(ocrand(n, 2,2,2, 2, 1,1,1), ocrand(n, 2,2,2, 2, 1,1,1)) 
    test_grid_grid_ds(ocrand(n, 2,2,2, 2, 1,1,1), ocrand(n, 2,2,2, 2, 0.5,0.5,0.5)) 
    test_grid_grid_ds(ocrand(n, 2,2,2, 2, 0.5,0.5,0.5), ocrand(n, 2,2,2, 2, 0.5,0.5,0.5))
    test_grid_grid_ds(ocrand(n, 2,4,6, 2, 0.5,0.5,0.5), ocrand(n, 2,4,6, 2, 0.5,0.5,0.5))
    test_grid_grid_ds(ocrand(n, 4,8,6, 6, 0.5,0.5,0.5), ocrand(n, 4,8,6, 6, 0.5,0.5,0.5))
    test_grid_grid_ds(ocrand(n, 4,8,6, 6, 0.5,0.5,0.5), ocrand(n, 4,8,6, 6, 1,1,1))
  end
end



local seed = os.time()
print('seed: '..seed)
math.randomseed(seed)
torch.manualSeed(seed)

mytester:add(octest)
-- mytester:run()
mytester:run('OctreeBCECriterion')

