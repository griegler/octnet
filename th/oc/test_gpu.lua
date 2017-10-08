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

-- you can easily test specific units like this:
-- th -lnn -e "nn.test{'LookupTable'}"
-- th -lnn -e "nn.test{'LookupTable', 'Add'}"

require('nn')
require('cunn')
require('cudnn')

local test_utils = require('test_utils')


local mytester = torch.Tester()
local jac
local sjac

local precision = 1e-5
local expprecision = 1e-4
local precision_backward = 1e-2

local cuoctest = torch.TestSuite()



local function criterionJacobianTest(cri, input, target)
  local eps = 1e-6
  local _ = cri:forward(input, target)
  local dfdx = cri:backward(input, target)

  -- for each input perturbation, do central difference
  local centraldiff_dfdx = torch.Tensor():type('torch.CudaTensor'):resizeAs(dfdx)
  local input_s = input:storage()
  local centraldiff_dfdx_s = centraldiff_dfdx:storage()
  for i=1,input:nElement() do
    -- f(xi + h)
    input_s[i] = input_s[i] + eps
    local fx1 = cri:forward(input, target)
    -- f(xi - h)
    input_s[i] = input_s[i] - 2*eps
    local fx2 = cri:forward(input, target)
    -- f'(xi) = (f(xi + h) - f(xi - h)) / 2h
    local cdfx = (fx1 - fx2) / (2*eps)
    -- store f' in appropriate place
    centraldiff_dfdx_s[i] = cdfx
    -- reset input[i]
    input_s[i] = input_s[i] + eps
  end

  -- compare centraldiff_dfdx with :backward()
  print('centraldiff_dfdx')
  print(centraldiff_dfdx)
  print('dfdx')
  print(dfdx)
  local err = (centraldiff_dfdx - dfdx):abs():max()
  mytester:assertlt(err, precision, 'error in difference between central difference and :backward')
end




function cuoctest.test_update_n_leafs()
  local grids = {
    test_utils.octree_rand(1, 1,1,2, 2, 0,0,0),
    test_utils.octree_rand(1, 1,1,2, 2, 0.5,0,0),
    test_utils.octree_rand(1, 2,3,4, 3, 0.5,0.5,0),
    test_utils.octree_rand(1, 2,3,4, 3, 0.5,0.5,0.5),
    test_utils.octree_rand(4, 1,1,2, 2, 0,0,0),
    test_utils.octree_rand(4, 1,1,2, 2, 0.5,0,0),
    test_utils.octree_rand(4, 2,3,4, 3, 0.5,0.5,0),
    test_utils.octree_rand(4, 2,3,4, 3, 0.5,0.5,0.5)
  }
  for _, grid_h in ipairs(grids) do
    local grid_d = grid_h:cuda()

    grid_h.grid.n_leafs = 0;
    grid_d.grid.n_leafs = 0;
    
    grid_h:update_n_leafs()
    grid_d:update_n_leafs()

    -- print(grid_h:n_leafs(), grid_d:n_leafs())
    mytester:asserteq(grid_h:n_leafs(), grid_d:n_leafs(), 'update n leafs failed')
  end
end


function cuoctest.test_update_prefix_leafs()
  local grids = {
    test_utils.octree_rand(1, 1,1,2, 2, 0,0,0),
    test_utils.octree_rand(1, 1,1,2, 2, 0.5,0,0),
    test_utils.octree_rand(1, 2,3,4, 3, 0.5,0.5,0),
    test_utils.octree_rand(1, 2,3,4, 3, 0.5,0.5,0.5),
    test_utils.octree_rand(4, 1,1,2, 2, 0,0,0),
    test_utils.octree_rand(4, 1,1,2, 2, 0.5,0,0),
    test_utils.octree_rand(4, 2,3,4, 3, 0.5,0.5,0),
    test_utils.octree_rand(4, 2,3,4, 3, 0.5,0.5,0.5)
  }
  for sample_idx, grid_h in ipairs(grids) do
    print('test '..sample_idx)
    local grid_d = grid_h:cuda()
    
    grid_h:update_prefix_leafs()
    grid_d:update_prefix_leafs()

    grid_d = grid_d:float()

    -- print('')
    for idx = 0, grid_d:n_blocks()-1 do
      mytester:asserteq(grid_h.grid.prefix_leafs[idx], grid_d.grid.prefix_leafs[idx], 'update prefix leafs failed')
    end 

  end
end

function cuoctest.copy_sup_to_sub()
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

    local sub_h = sub:clone() 
    local sub_d = sub:clone():cuda() 
    local sup_h = sup:clone()
    local sup_d = sup:clone():cuda()

    sub_h:copy_from_sup(sup_h)
    sub_d:copy_from_sup(sup_d)

    sub_d = sub_d:float()
    mytester:assert(sub_d:equals(sub_h, 1e-6, true), 'copy_sup_to_sub')
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


function cuoctest.VolumetricNNUpsampling()
  local function test(input_h, factor)
    local input_d = input_h:clone():cuda()
    local mod_h = oc.VolumetricNNUpsampling(factor):float()
    local mod_d = oc.VolumetricNNUpsampling(factor):cuda()

    local out_h = mod_h:forward(input_h)
    local out_d = mod_d:forward(input_d)

    local grad_h = out_h:clone()
    grad_h:apply(function() torch.uniform(-1, 1) end)
    local grad_d = grad_h:clone():cuda()

    local gout_h = mod_h:backward(input_h, grad_h)
    local gout_d = mod_d:backward(input_d, grad_d)
    
    out_d = out_d:float()
    local err = torch.abs(out_h - out_d):max()
    mytester:assert(err < 1e-6, 'VolumetricNNUpsampling forward factor='..factor)

    gout_d = gout_d:float()
    local err = torch.abs(gout_h - gout_d):max()
    mytester:assert(err < 1e-6, 'VolumetricNNUpsampling backward factor='..factor)
  end

  for _, factor in ipairs{2, 1, 3, 4} do
    test(torch.randn(1, 1, 1,1,1):float(), factor)
    test(torch.randn(1, 2, 1,1,1):float(), factor)
    test(torch.randn(1, 4, 4,5,6):float(), factor)
    test(torch.randn(3, 4, 4,5,6):float(), factor)
  end 
end


function cuoctest.OctreeModuleClone()
  local grid_h = test_utils.octree_rand(1, 1,1,1, 1, 0,0,0)
  local grid_d = grid_h:cuda()

  local mod = oc.OctreeToCDHW()
  local mod_h = mod:float()

  local mod_c = mod_h:clone()

  local grid1 = mod_h:forward(grid_h)
  local grid2 = mod_c:forward(grid_h)
  mytester:assert(torch.abs(grid1 - grid2):max() < 1e-6, 'error in OctreeModuleClone')
end

function cuoctest.OctreeToCDHW()
  local function test(grid_h, dense_depth, dense_height, dense_width)
    local grid_d = grid_h:cuda()
    local mod_h = oc.OctreeToCDHW(dense_depth, dense_height, dense_width):float() 
    local mod_d = oc.OctreeToCDHW(dense_depth, dense_height, dense_width):cuda() 

    local out_h = mod_h:forward(grid_h)
    local out_d = mod_d:forward(grid_d)
    local max_err = torch.abs(out_h - out_d:float()):max()
    -- print(out_h)
    -- print(out_d:float())
    mytester:assert(max_err < 1e-4, 'error in OctreeToCDHW forward cin: '..grid_h:feature_size())

    local grad_out_h = out_h:clone()
    local grad_out_d = out_h:clone():cuda()
    local grad_in_h = mod_h:backward(grid_h, grad_out_h)
    local grad_in_d = mod_d:backward(grid_d, grad_out_d)
    mytester:assert(grad_in_h:equals(grad_in_d:float(), 1e-4), 'error in OctreeToCDHW backward cin: '..grid_h:feature_size())
  end

  local function test_battery(dense_depth, dense_height, dense_width)
    for _, n in ipairs{1, 4} do
      test(test_utils.octree_rand(n, 1,1,1, 1, 0,0,0), dense_depth, dense_height, dense_width)
      test(test_utils.octree_rand(n, 1,1,1, 1, 1,0,0), dense_depth, dense_height, dense_width)
      test(test_utils.octree_rand(n, 1,1,1, 1, 1,1,0), dense_depth, dense_height, dense_width)
      test(test_utils.octree_rand(n, 1,1,1, 1, 1,1,1), dense_depth, dense_height, dense_width)
      test(test_utils.octree_rand(n, 2,3,4, 3, 0,0,0), dense_depth, dense_height, dense_width)
      test(test_utils.octree_rand(n, 2,3,4, 3, 1,0,0), dense_depth, dense_height, dense_width)
      test(test_utils.octree_rand(n, 2,3,4, 3, 1,1,0), dense_depth, dense_height, dense_width)
      test(test_utils.octree_rand(n, 2,3,4, 3, 1,1,1), dense_depth, dense_height, dense_width)
      test(test_utils.octree_rand(n, 2,3,4, 2, 0.5,0.5,0.5), dense_depth, dense_height, dense_width)
      test(test_utils.octree_rand(n, 2,3,4, 2, 0.5,0.5,0.5), dense_depth, dense_height, dense_width)
    end
  end
  
  test_battery()
  test_battery(32,32,32)
  test_battery(64,64,64)
end


function cuoctest.CDHWToOctree()
  local function test(fcn, grid_h, dense_depth, dense_height, dense_width)
    local grid_d = grid_h:cuda()

    local o2d_h = oc.OctreeToCDHW(dense_depth, dense_height, dense_width):float()
    local o2d_d = oc.OctreeToCDHW(dense_depth, dense_height, dense_width):cuda()
    local d2o_h = oc.CDHWToOctree(o2d_h, fcn):float()
    local d2o_d = oc.CDHWToOctree(o2d_d, fcn):cuda()

    local dense_h = o2d_h:forward(grid_h)
    local dense_d = o2d_d:forward(grid_d)

    local out_h = d2o_h:forward(dense_h)
    local out_d = d2o_d:forward(dense_d)

    local grad_out_h = out_h:clone()
    local grad_out_d = out_h:clone():cuda()
    local grad_in_h = d2o_h:backward(dense_h, grad_out_h) 
    local grad_in_d = d2o_d:backward(dense_d, grad_out_d) 
    
    out_d = out_d:float()
    grad_in_d = grad_in_d:float()

    mytester:assert(out_h:equals(out_d, 1e-4, true), 'error in CDHWToOctree forward '..fcn)
    
    local max_err = torch.abs(grad_in_h - grad_in_d):max()
    -- print(grad_in_h)
    -- print(grad_in_d)
    mytester:assert(max_err < 1e-4, 'error in CDHWToOctree backward '..fcn)
  end


  local function test_battery(dense_depth, dense_height, dense_width)
    for _, n in ipairs{1, 4} do
      for _, fcn in ipairs{'avg', 'sum'} do
        test(fcn, test_utils.octree_rand(n, 1,1,1, 1, 0,0,0), dense_depth, dense_height, dense_width)
        test(fcn, test_utils.octree_rand(n, 1,1,1, 1, 1,0,0), dense_depth, dense_height, dense_width)
        test(fcn, test_utils.octree_rand(n, 1,1,1, 1, 1,1,0), dense_depth, dense_height, dense_width)
        test(fcn, test_utils.octree_rand(n, 1,1,1, 1, 1,1,1), dense_depth, dense_height, dense_width)
        test(fcn, test_utils.octree_rand(n, 2,3,4, 3, 0,0,0), dense_depth, dense_height, dense_width)
        test(fcn, test_utils.octree_rand(n, 2,3,4, 3, 1,0,0), dense_depth, dense_height, dense_width)
        test(fcn, test_utils.octree_rand(n, 2,3,4, 3, 1,1,0), dense_depth, dense_height, dense_width)
        test(fcn, test_utils.octree_rand(n, 2,3,4, 3, 1,1,1), dense_depth, dense_height, dense_width)
        test(fcn, test_utils.octree_rand(n, 2,3,4, 2, 0.5,0.5,0.5), dense_depth, dense_height, dense_width)
        test(fcn, test_utils.octree_rand(n, 2,3,4, 2, 0.5,0.5,0.5), dense_depth, dense_height, dense_width)
      end
    end
  end
  
  test_battery()
  test_battery(32,32,32)
  test_battery(64,64,64)
end



function cuoctest.OctreeConvolutionMM()
  local function test_conv(cin,cout, grid_h, eps_fwd, eps_bwd, eps_wbwd)
    local eps_fwd = eps_fwd or 1e-4
    local eps_bwd = eps_bwd or 1e-4
    local eps_wbwd = eps_wbwd or 1e-3

    local grid_d = grid_h:cuda()
    local conv_dense = oc.OctreeDenseConvolution(cin,cout, 'avg', false)

    conv_dense = conv_dense:cuda()
    local conv_mm = oc.OctreeConvolutionMM(cin,cout):cuda()

    conv_mm.weight:copy(conv_dense.weight)
    conv_mm.bias:copy(conv_dense.bias)

    local out_dense = conv_dense:forward(grid_d)
    local out_mm = conv_mm:forward(grid_d)

    out_dense = out_dense:clone():float()
    out_mm = out_mm:clone():float()

    mytester:assert(out_dense:equals(out_mm, eps_fwd, true), 'error in OctreeConvolutionMM forward '..conv_mm.nInputPlane..', '..conv_mm.nOutputPlane)

    -------------------------------
    -- test backward
    local grad_dense_d = out_mm:clone():mul(0.001):cuda()
    local grad_mm_d = out_mm:clone():mul(0.001):cuda()
    conv_dense:zeroGradParameters()
    local out_dense = conv_dense:backward(grid_d, grad_dense_d)
    conv_mm:zeroGradParameters()
    local out_mm = conv_mm:backward(grid_d, grad_mm_d)
    -- print(conv_dense.gradWeight)
    -- print(conv_mm.gradWeight)
    local max_e_w = torch.abs(conv_dense.gradWeight - conv_mm.gradWeight):max()
    if max_e_w >= eps_wbwd then
      print('error in gradWeight', max_e_w, torch.abs(conv_dense.gradWeight):max(), torch.abs(conv_mm.gradWeight):max())
      print(grad_dense_d:min(), grad_dense_d:max())
    end 
    mytester:assertlt(max_e_w, eps_wbwd,'error in OctreeConvolutionMM backward weight '..conv_mm.nInputPlane..', '..conv_mm.nOutputPlane..': '..max_e_w)
    local max_e_b = torch.abs(conv_dense.gradBias - conv_mm.gradBias):max()
    mytester:assertlt(max_e_b, eps_wbwd,'error in OctreeConvolutionMM backward bias '..conv_mm.nInputPlane..', '..conv_mm.nOutputPlane..': '..max_e_b)

    out_dense = out_dense:clone():float()
    out_mm = out_mm:clone():float()
    mytester:assert(out_dense:equals(out_mm, eps_bwd, true), 'error in OctreeConvolutionMM backward '..conv_mm.nInputPlane..', '..conv_mm.nOutputPlane)
  end

  for _, n in ipairs{1, 4} do
    for _, cincout in ipairs{{1,1}, {1,3}, {3,1}, {2,4}, {4,2}, {32,32}} do
      local cin = cincout[1]
      local cout = cincout[2]
      print('test 1')
      test_conv(cin,cout, test_utils.octree_rand(n, 2,3,4, cin, 0.0,0.0,0.0))
      print('test 2')
      test_conv(cin,cout, test_utils.octree_rand(n, 2,3,4, cin, 1.0,0.0,0.0))
      print('test 3')
      test_conv(cin,cout, test_utils.octree_rand(n, 2,3,4, cin, 1.0,1.0,0.0))
      print('test 4')
      test_conv(cin,cout, test_utils.octree_rand(n, 2,3,4, cin, 1.0,1.0,1.0))
      print('test 5')
      test_conv(cin,cout, test_utils.octree_rand(n, 2,3,4, cin, 0.5,0.5,0.0))
      print('test 6')
      test_conv(cin,cout, test_utils.octree_rand(n, 2,3,4, cin, 0.5,0.5,0.5))
      print('test 7')
      test_conv(cin,cout, test_utils.octree_rand(n, 2,3,4, cin, 0.5,0.5,0.5))
    end
  end
end


function cuoctest.OctreePool2x2x2()
  local function test_pool(pool_fcn, level_0, level_1, level_2, grid_h, debug)
    local grid_d = grid_h:cuda()
    local pool_h = oc.OctreePool2x2x2(pool_fcn, level_0,level_1,level_2):float()
    local pool_d = oc.OctreePool2x2x2(pool_fcn, level_0,level_1,level_2):cuda()

    local out_h = pool_h:forward(grid_h)
    local out_d = pool_d:forward(grid_d)
    out_d = out_d:float()

    if debug then
      print(out_h:tree_bit_string(1))
      out_h:print()
      print(out_d:tree_bit_string(1))
      out_d:print()
    end
    mytester:assert(out_h:equals(out_d, 1e-4, true), 'error in OctreePool2x2x2 forward')
  end

  local function test_pool_bwd(pool_fcn, level_0, level_1, level_2, grid_h, debug)
    local grid_d = grid_h:cuda()
    local pool_h = oc.OctreePool2x2x2(pool_fcn, level_0,level_1,level_2):float()
    local pool_d = oc.OctreePool2x2x2(pool_fcn, level_0,level_1,level_2):cuda()

    local out_h = pool_h:forward(grid_h)
    local out_d = pool_d:forward(grid_d)

    local grad_out_h = out_h:clone()
    local grad_out_d = out_d:clone()
    local grad_in_h = pool_h:backward(grid_h, grad_out_h)
    local grad_in_d = pool_d:backward(grid_d, grad_out_d)
    grad_in_d = grad_in_d:float()

    mytester:assert(grad_in_h:equals(grad_in_d, 1e-4, true), 'error in OctreePool2x2x2 backward')
  end
  
  
  for _, n in ipairs{1, 4} do
  for _, level_0 in ipairs{false, true} do
  for _, level_1 in ipairs{false, true} do
  for _, level_2 in ipairs{false, true} do
  for _, pool_fcn in ipairs{'avg', 'max'} do
    print(pool_fcn)
    print('test 1')
    test_pool(pool_fcn, level_0, level_1, level_2, test_utils.octree_rand(n, 1,1,2, 1, 0,0,0))
    print('test 2')
    test_pool(pool_fcn, level_0, level_1, level_2, test_utils.octree_rand(n, 1,1,2, 2, 1,0,0))
    print('test 3')
    test_pool(pool_fcn, level_0, level_1, level_2, test_utils.octree_rand(n, 2,2,3, 3, 0.5,0,0))
    print('test 4')
    test_pool(pool_fcn, level_0, level_1, level_2, test_utils.octree_rand(n, 2,2,3, 3, 0.5,0.5,0))
    print('test 5')
    test_pool(pool_fcn, level_0, level_1, level_2, test_utils.octree_rand(n, 2,2,3, 3, 0.5,0.5,0))
    print('test 6')
    test_pool(pool_fcn, level_0, level_1, level_2, test_utils.octree_rand(n, 2,2,3, 3, 0.5,0.5,0.5), false)

    print('test 7')
    test_pool_bwd(pool_fcn, level_0, level_1, level_2, test_utils.octree_rand(n, 1,1,2, 1, 0,0,0))
    print('test 8')
    test_pool_bwd(pool_fcn, level_0, level_1, level_2, test_utils.octree_rand(n, 1,1,2, 2, 1,0,0))
    print('test 9')
    test_pool_bwd(pool_fcn, level_0, level_1, level_2, test_utils.octree_rand(n, 2,2,3, 3, 0.5,0,0))
    print('test 10')
    test_pool_bwd(pool_fcn, level_0, level_1, level_2, test_utils.octree_rand(n, 2,2,3, 3, 0.5,0.5,0))
    print('test 11')
    test_pool_bwd(pool_fcn, level_0, level_1, level_2, test_utils.octree_rand(n, 2,2,3, 3, 0.5,0.5,0))
    print('test 12')
    test_pool_bwd(pool_fcn, level_0, level_1, level_2, test_utils.octree_rand(n, 2,2,3, 3, 0.5,0.5,0.5))
  end
  end
  end
  end
  end
end


function cuoctest.OctreeGridPool2x2x2()
  local function test(pool_fcn, grid_h)
    local grid_d = grid_h:cuda()
    local pool_h = oc.OctreeGridPool2x2x2(pool_fcn):float()
    local pool_d = oc.OctreeGridPool2x2x2(pool_fcn):cuda()

    -- test fwd
    local out_h = pool_h:forward(grid_h)
    local out_d = pool_d:forward(grid_d)

    -- test bwd
    local grad_out_h = out_h:clone()
    local grad_out_d = out_h:clone():cuda()
    
    local grad_in_h = pool_h:backward(grid_h, grad_out_h)
    local grad_in_d = pool_d:backward(grid_d, grad_out_d)

    out_h = out_h
    out_d = out_d:float()
    mytester:assert(out_h:equals(out_d, 1e-6, true), 'error in OctreeGridPool2x2x2 forward')

    grad_in_h = grad_in_h
    grad_in_d = grad_in_d:float()
    mytester:assert(grad_in_h:equals(grad_in_d, 1e-6, true), 'error in OctreeGridPool2x2x2 backward')
  end

  for _, n in ipairs{1, 4} do
    for _, pool_fcn in ipairs{'avg', 'max'} do
      test(pool_fcn, test_utils.octree_rand(n, 2,2,2, 2, 0,0,0))
      test(pool_fcn, test_utils.octree_rand(n, 2,2,2, 2, 1,0,0))
      test(pool_fcn, test_utils.octree_rand(n, 2,2,2, 2, 1,1,0))
      test(pool_fcn, test_utils.octree_rand(n, 2,2,2, 2, 1,1,1))
      test(pool_fcn, test_utils.octree_rand(n, 2,4,6, 2, 0.5,0.5,0.5))
      test(pool_fcn, test_utils.octree_rand(n, 4,8,6, 8, 0.5,0.5,0.5))
    end
  end
end

function cuoctest.OctreeGridUnpoolGuided2x2x2()
  local function test(grid_in_h)
    local grid_in_d = grid_in_h:cuda()

    local struct_h = nn.Identity():float()
    struct_h:forward(grid_in_h)
    local struct_d = nn.Identity():cuda()
    struct_d:forward(grid_in_d)

    local grid_pooled_h = oc.OctreeGridPool2x2x2('max'):float():forward(grid_in_h)
    local grid_pooled_d = oc.OctreeGridPool2x2x2('max'):cuda():forward(grid_in_d)

    local pool_h = oc.OctreeGridUnpoolGuided2x2x2(struct_h):float()
    local pool_d = oc.OctreeGridUnpoolGuided2x2x2(struct_d):cuda()

    -- test fwd
    local out_h = pool_h:forward(grid_pooled_h)
    local out_d = pool_d:forward(grid_pooled_d)

    -- test bwd
    local grad_out_h = out_h:clone()
    local grad_out_d = out_h:clone():cuda()
    
    local grad_in_h = pool_h:backward(grid_pooled_h, grad_out_h)
    local grad_in_d = pool_d:backward(grid_pooled_d, grad_out_d)

    out_h = out_h
    out_d = out_d:float()
    mytester:assert(out_h:equals(out_d, 1e-6, true), 'error in OctreeGridUnpoolGuided2x2x2ool2x2x2 forward')

    grad_in_h = grad_in_h
    grad_in_d = grad_in_d:float()
    mytester:assert(grad_in_h:equals(grad_in_d, 1e-6, true), 'error in OctreeGridUnpoolGuided2x2x2ool2x2x2 backward')
  end

  local pool_mod = oc.OctreeGridPool2x2x2('max')
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

function cuoctest.OctreeGridUnpool2x2x2()
  local function test(grid_in_h)
    local grid_in_d = grid_in_h:cuda()

    local pool_h = oc.OctreeGridUnpool2x2x2():float()
    local pool_d = oc.OctreeGridUnpool2x2x2():cuda()

    -- test fwd
    local out_h = pool_h:forward(grid_in_h)
    local out_d = pool_d:forward(grid_in_d)

    -- test bwd
    local grad_out_h = out_h:clone()
    local grad_out_d = out_h:clone():cuda()
    
    local grad_in_h = pool_h:backward(grid_in_h, grad_out_h)
    local grad_in_d = pool_d:backward(grid_in_d, grad_out_d)

    out_h = out_h
    out_d = out_d:float()
    mytester:assert(out_h:equals(out_d, 1e-6, true), 'error in OctreeGridUnpool2x2x2ool2x2x2 forward')

    grad_in_h = grad_in_h
    grad_in_d = grad_in_d:float()
    mytester:assert(grad_in_h:equals(grad_in_d, 1e-6, true), 'error in OctreeGridUnpool2x2x2ool2x2x2 backward')
  end

  local pool_mod = oc.OctreeGridPool2x2x2('max')
  for _, n in ipairs{1, 4} do
    for _, fs in ipairs{1, 3} do
      test(test_utils.octree_rand(n, 1,1,1, fs, 0.0,0.0,0.0))
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


function cuoctest.OctreeActivations()
  local function test_relu(inplace, grid_in_h)
    local mod_h = oc.OctreeReLU(inplace):float()
    local mod_d = oc.OctreeReLU(inplace):cuda()
    local grid_in_d = grid_in_h:clone():cuda()

    local grid_out_h = mod_h:forward(grid_in_h)
    local grid_out_d = mod_d:forward(grid_in_d)

    local grad_out = grid_out_h:clone():apply(function() return torch.uniform(-1,1) end)
    local grad_out_h = grad_out:clone()
    local grad_out_d = grad_out:clone():cuda()

    local grad_in_h = mod_h:backward(grid_in_h, grad_out_h)
    local grad_in_d = mod_d:backward(grid_in_d, grad_out_d)

    local grid_out_d = grid_out_d:float() 
    local grad_in_d = grad_in_d:float() 
    
    mytester:assert(grid_out_h:equals(grid_out_d, 1e-4, true), 'error in OctreeReLU forward')
    mytester:assert(grad_in_h:equals(grad_in_d, 1e-4, true), 'error in OctreeReLU backward 1') 
  end

  local function test_sigmoid(inplace, in_h)
    local mod_h = oc.OctreeSigmoid(inplace):float()
    local mod_d = oc.OctreeSigmoid(inplace):cuda()
    local in_d = in_h:clone():cuda()

    local out_h = mod_h:forward(in_h)
    local out_d = mod_d:forward(in_d)
    -- print(in_h.grid.data[0], out_h.grid.data[0], out_d:float().grid.data[0])

    local grad_out = out_h:clone():apply(function() return torch.uniform(-1,1) end)
    local grad_out_h = grad_out:clone()
    local grad_out_d = grad_out:clone():cuda()

    local grad_in_h = mod_h:backward(in_h, grad_out_h)
    local grad_in_d = mod_d:backward(in_d, grad_out_d)

    local out_d = out_d:float() 
    local grad_in_d = grad_in_d:float() 
    
    mytester:assert(out_h:equals(out_d, 1e-6, true), 'error in OctreeSigmoid forward')
    mytester:assert(grad_in_h:equals(grad_in_d, 1e-6, true), 'error in OctreeSigmoid backward 1') 
  end

  local function test_logsoftmax(in_h)
    local mod_h = oc.OctreeLogSoftMax():float()
    local mod_d = oc.OctreeLogSoftMax():cuda()
    local in_d = in_h:clone():cuda()

    local out_h = mod_h:forward(in_h)
    local out_d = mod_d:forward(in_d)
    -- print(in_h.grid.data[0], out_h.grid.data[0], out_d:float().grid.data[0])

    local grad_out = out_h:clone():apply(function() return torch.uniform(-1,1) end)
    local grad_out_h = grad_out:clone()
    local grad_out_d = grad_out:clone():cuda()

    local grad_in_h = mod_h:backward(in_h, grad_out_h)
    local grad_in_d = mod_d:backward(in_d, grad_out_d)

    local out_d = out_d:float() 
    local grad_in_d = grad_in_d:float() 
    
    mytester:assert(out_h:equals(out_d, 1e-4, true), 'error in OctreeLogSoftMax forward')
    mytester:assert(grad_in_h:equals(grad_in_d, 1e-4, true), 'error in OctreeLogSoftMax backward 1') 
  end

  for _, n in ipairs{1, 4} do
    for _, inplace in ipairs{false, true} do
      local grid_in = test_utils.octree_rand(n, 1,1,1, 1, 0,0,0, -1,1)
      test_relu(inplace, grid_in)
      test_sigmoid(inplace, grid_in)
      test_logsoftmax(grid_in)

      local grid_in = test_utils.octree_rand(n, 2,3,4, 2, 0,0,0, -1,1)
      test_relu(inplace, grid_in)
      test_sigmoid(inplace, grid_in)
      test_logsoftmax(grid_in)

      local grid_in = test_utils.octree_rand(n, 2,3,4, 2, 1,0,0, -1,1)
      test_relu(inplace, grid_in)
      test_sigmoid(inplace, grid_in)
      test_logsoftmax(grid_in)

      local grid_in = test_utils.octree_rand(n, 2,3,4, 2, 1,1,0, -1,1)
      test_relu(inplace, grid_in)
      test_sigmoid(inplace, grid_in)
      test_logsoftmax(grid_in)

      local grid_in = test_utils.octree_rand(n, 2,3,4, 2, 1,1,1, -1,1)
      test_relu(inplace, grid_in)
      test_sigmoid(inplace, grid_in)
      test_logsoftmax(grid_in)

      local grid_in = test_utils.octree_rand(n, 2,3,4, 2, 0.5,0.5,0.5, -1,1)
      test_relu(inplace, grid_in)
      test_sigmoid(inplace, grid_in)
      test_logsoftmax(grid_in)
    end
  end
end



function cuoctest.OctreeAdd()
  local function test(oc1)
    local todense = function(x) return oc.OctreeToCDHW():cuda():forward(x) end
    local oc2 = oc1:clone():apply(function() torch.uniform(-1,1) end):cuda()
    local oc1 = oc1:cuda()
    local de1 = todense(oc1)
    local de2 = todense(oc2)

    local add1 = torch.uniform(-1,1)
    local ocr = oc1:add(add1)
    local der = de1:add(add1)
    local err = torch.abs(todense(ocr) - der):max()
    mytester:assert(err < 1e-6, 'add1: '..err)

    local ocr = oc1:add(oc2) 
    local der = de1:add(de2)
    local err = torch.abs(todense(ocr) - der):max()
    mytester:assert(err < 1e-6, 'add oc2: '..err)

    local ocr = oc1:add(oc1, oc2) 
    local der = de1:add(de1, de2)
    local err = torch.abs(todense(ocr) - der):max()
    mytester:assert(err < 1e-6, 'add oc1 and oc2: '..err)

    local fac = torch.uniform(-2, 2)
    local ocr = oc1:add(fac, oc2) 
    local der = de1:add(fac, de2)
    local err = torch.abs(todense(ocr) - der):max()
    mytester:assert(err < 1e-6, 'add fac oc2: '..err)

    local fac = torch.uniform(-2, 2)
    local ocr = oc1:add(oc1, fac, oc2) 
    local der = de1:add(de1, fac, de2)
    local err = torch.abs(todense(ocr) - der):max()
    mytester:assert(err < 1e-6, 'add oc1 and fac oc2: '..err)

    local fac1 = torch.uniform(-2, 2)
    local fac2 = torch.uniform(-2, 2)
    local ocr = oc1:add(fac1, oc1, fac2, oc2) 
    local der = de1:add(de1:mul(fac1), fac2, de2)
    local err = torch.abs(todense(ocr) - der):max()
    mytester:assert(err < 1e-6, 'add oc1 and fac oc2: '..err)

    for _, ip in ipairs{false, true} do
      local cadd_oc = nn.CAddTable(ip):oc():cuda()
      local cadd_de = nn.CAddTable(ip):cuda()

      local ocr = cadd_oc:forward({oc1, oc2})
      local der = cadd_de:forward({de1, de2})
      local err = torch.abs(todense(ocr) - der):max()
      mytester:assert(err < 1e-5, 'CAddTable forward: '..err)

      local ocr = cadd_oc:backward({oc1, oc2}, oc2)
      local der = cadd_de:backward({de1, de2}, de2)
      local err = torch.abs(todense(ocr[1]) - der[1]):max()
      mytester:assert(err < 1e-6, 'CAddTable backward: '..err)
      local err = torch.abs(todense(ocr[2]) - der[2]):max()
      mytester:assert(err < 1e-6, 'CAddTable backward: '..err)
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

function cuoctest.OctreeConcat()
  local function test(in1, dense, do_grad_in2)
    local test_str = 'error in Octreeconcat'
    if dense then test_str = test_str .. '(dense)' end
    if do_grad_in2 then test_str = test_str .. '(do_grad_in2)' end

    local in2 = in1:clone()
    in2:resize(in1:n(), in1:grid_depth(), in1:grid_height(), in1:grid_width(), in1:feature_size()*2, in1:n_leafs())
    in2:update_prefix_leafs()
    in2:apply(function() return torch.uniform(-1,1) end)
    
    if dense then
      in2 = oc.OctreeToCDHW():float():forward(in2)
    end

    local in1_h = in1
    local in2_h = in2
    local in1_d = in1_h:clone():cuda()
    local in2_d = in2_h:clone():cuda()

    local mod_h = oc.OctreeConcat(true, do_grad_in2):float()
    local mod_d = oc.OctreeConcat(true, do_grad_in2):cuda()

    local out_h = mod_h:forward({in1_h, in2_h})
    local out_d = mod_d:forward({in1_d, in2_d})

    mytester:assert(out_h:equals(out_d, 1e-6, true), test_str .. ' forward')


    local grad_out_h = out_h:clone()
    local grad_out_d = out_h:clone():cuda()

    local grad_in_h = mod_h:backward({in1_h, in2_h}, grad_out_h)
    local grad_in_d = mod_d:backward({in1_d, in2_d}, grad_out_d)

    local grid_out_d = out_d:float() 
    local grad_in_d = { grad_in_d[1]:float(), grad_in_d[2]:float() } 
    
    mytester:assert(grad_in_h[1]:equals(grad_in_d[1], 1e-6, true), test_str .. ' backward 1')
    if do_grad_in2 then
      if dense then
        print('test grad_in2 dense')
        local err = torch.abs(grad_in_h[2] - grad_in_d[2]):max()
        mytester:assert(err < 1e-6, test_str .. ' backward 2 ' .. err) 
      else
        print('test grad_in2 oc')
        mytester:assert(grad_in_h[2]:equals(grad_in_d[2], 1e-6, true), test_str .. ' backward 2') 
      end
    end
  end

  local function test_grid_grid(in1, fs2, do_grad_in2) 
    in2 = in1:clone()
    in2:resize(in1:n(), in1:grid_depth(), in1:grid_height(), in1:grid_width(), fs2, in1:n_leafs())
    in2:update_prefix_leafs()
    in2:apply(function() return torch.uniform(-1,1) end)

    in1 = in1:cuda()
    in2 = in2:cuda()
    in1_de = oc.OctreeToCDHW():cuda():forward(in1)
    in2_de = oc.OctreeToCDHW():cuda():forward(in2)

    local mod = oc.OctreeConcat(true, do_grad_in2):cuda()
    local out = mod:forward({in1, in2})

    local mod_de = nn.JoinTable(1,4):cuda()
    local out_de = mod_de:forward({in1_de, in2_de})
    
    local err = torch.abs(oc.OctreeToCDHW():cuda():forward(out) - out_de):max()
    mytester:assert(err < 1e-6, 'OctreeConcat grid-grid forward err=' .. err) 


    local grad_out = out:clone()
    local out = mod:backward({in1, in2}, grad_out)
    
    local grad_out = oc.OctreeToCDHW():cuda():forward(grad_out)
    local out_de = mod_de:backward({in1_de, in2_de}, grad_out)
    
    local err = torch.abs(oc.OctreeToCDHW():cuda():forward(out[1]) - out_de[1]):max()
    mytester:assert(err < 1e-6, 'OctreeConcat grid-grid backward1 err=' .. err) 

    if do_grad_in2 then
      local err = torch.abs(oc.OctreeToCDHW():cuda():forward(out[2]) - out_de[2]):max()
      mytester:assert(err < 1e-6, 'OctreeConcat grid-grid backward2 err=' .. err) 
    end
  end
  
  local function test_grid_dense(in1, fs2, do_grad_in2) 
    in2 = torch.rand(in1:n(), fs2, 8*in1:grid_depth(), 8*in1:grid_height(), 8*in1:grid_width()):cuda()

    in1 = in1:cuda()
    in1_de = oc.OctreeToCDHW():cuda():forward(in1)

    local mod = oc.OctreeConcat(true, do_grad_in2):cuda()
    local out = mod:forward({in1, in2})

    local mod_de = nn.JoinTable(1,4):cuda()
    local out_de = mod_de:forward({in1_de, in2})
    mod.received_input = out
    local de2oc = oc.CDHWToOctree(mod, 'avg'):cuda()
    out_de = de2oc:forward(out_de)
    
    mytester:assert(out:float():equals(out_de:float(), 1e-5, true), 'OctreeConcat grid-dense forward') 


    local grad_out = out:clone()
    local out = mod:backward({in1, in2}, grad_out)
    
    local grad_out = oc.OctreeToCDHW():cuda():forward(grad_out)
    local out_de = mod_de:backward({in1_de, in2}, grad_out)
    local de2oc = oc.CDHWToOctree(mod, 'avg'):cuda()
    out_de[1] = de2oc:forward(out_de[1])
    
    mytester:assert(out[1]:float():equals(out_de[1]:float(), 1e-5, true), 'OctreeConcat grid-dense backward1') 

    if do_grad_in2 then
      local err = torch.abs(out[2] - out_de[2]):max()
      mytester:assert(err < 1e-6, 'OctreeConcat grid-dense backward2 err=' .. err) 
    end
  end

  local function test_grid_grid_ds(in1, in2, do_grad_in2) 
    in1 = in1:cuda()
    in1_de = oc.OctreeToCDHW():cuda():forward(in1)
    in2 = in2:cuda()
    in2_de = oc.OctreeToCDHW():cuda():forward(in2)

    local mod = oc.OctreeConcat(true, do_grad_in2, true):cuda()
    local out = mod:forward({in1, in2})

    local mod_de = nn.JoinTable(1,4):cuda()
    local out_de = mod_de:forward({in1_de, in2_de})
    mod.received_input = out
    local de2oc = oc.CDHWToOctree(mod, 'avg'):cuda()
    out_de = de2oc:forward(out_de)
    
    -- out:float():print()
    -- out_de:float():print()
    mytester:assert(out:float():equals(out_de:float(), 1e-5, true), 'OctreeConcat grid-grid-ds forward') 


    local grad_out = out:clone()
    local out = mod:backward({in1, in2}, grad_out)
    
    local out_de = mod_de:backward({in1_de, in2_de}, oc.OctreeToCDHW():cuda():forward(grad_out))

    mod.received_input = in1
    local de2oc = oc.CDHWToOctree(mod, 'avg'):cuda()
    out_de[1] = de2oc:forward(out_de[1])
    mytester:assert(out[1]:float():equals(out_de[1]:float(), 1e-5, true), 'OctreeConcat grid-grid-ds backward1') 

    if do_grad_in2 then
      mod.received_input = in2
      local de2oc = oc.CDHWToOctree(mod, 'avg'):cuda()
      out_de[2] = de2oc:forward(out_de[2])
      -- grad_out:float():print()
      -- out[2]:float():print()
      -- out_de[2]:float():print()
      mytester:assert(out[2]:float():equals(out_de[2]:float(), 1e-5, true), 'OctreeConcat grid-grid-ds backward2') 
    end
  end

  local ocrnd = test_utils.octree_rand
  
  for _, do_grad_in2 in ipairs{true, false} do
  for _, n in ipairs{1, 4} do
  for fs2 = 1, 3 do
    test_grid_grid( ocrnd(n, 1,1,1, 1, 0,0,0), fs2, do_grad_in2)
    test_grid_grid( ocrnd(n, 1,1,1, 1, 1,0,0), fs2, do_grad_in2)
    test_grid_grid( ocrnd(n, 1,1,1, 1, 1,1,0), fs2, do_grad_in2)
    test_grid_grid( ocrnd(n, 1,1,1, 1, 0.5,0.5,0.5), fs2, do_grad_in2)
    test_grid_grid( ocrnd(n, 2,3,4, 3, 0.5,0.5,0.5), fs2, do_grad_in2)

    test_grid_dense( ocrnd(n, 1,1,1, 1, 0,0,0), fs2, do_grad_in2)
    test_grid_dense( ocrnd(n, 1,1,1, 1, 1,0,0), fs2, do_grad_in2)
    test_grid_dense( ocrnd(n, 1,1,1, 1, 1,1,0), fs2, do_grad_in2)
    test_grid_dense( ocrnd(n, 1,1,1, 1, 0.5,0.5,0.5), fs2, do_grad_in2)
    test_grid_dense( ocrnd(n, 2,3,4, 3, 0.5,0.5,0.5), fs2, do_grad_in2)
    
    test_grid_grid_ds( ocrnd(n, 1,1,1, 1, 0,0,0), ocrnd(n, 1,1,1, fs2, 0,0,0), do_grad_in2)
    test_grid_grid_ds( ocrnd(n, 1,1,1, 1, 1,0,0), ocrnd(n, 1,1,1, fs2, 1,0,0), do_grad_in2)
    test_grid_grid_ds( ocrnd(n, 1,1,1, 1, 1,1,0), ocrnd(n, 1,1,1, fs2, 1,1,0), do_grad_in2)
    test_grid_grid_ds( ocrnd(n, 1,1,1, 1, 1,1,1), ocrnd(n, 1,1,1, fs2, 1,1,1), do_grad_in2)
    test_grid_grid_ds( ocrnd(n, 1,1,1, 1, 0.5,0.5,0.5), ocrnd(n, 1,1,1, fs2, 0.5,0.5,0.5), do_grad_in2)
    test_grid_grid_ds( ocrnd(n, 2,3,4, 2, 0.5,0.5,0.5), ocrnd(n, 2,3,4, fs2, 0.5,0.5,0.5), do_grad_in2)
  end
  end
  end
  
end



function cuoctest.OctreeSplit()
  local function test_by_prob(data_h)
    local prob_h = test_utils.octree_alter_fs(data_h, 1)
    local threshold = 0
    local prob_d = prob_h:cuda()
    local data_d = data_h:cuda()

    local mod_prob_h = nn.Identity():float()
    mod_prob_h:forward(prob_h)
    local mod_prob_d = nn.Identity():cuda()
    mod_prob_d:forward(prob_d)

    local mod_h = oc.OctreeSplitByProb(mod_prob_h, threshold, true):float()
    local mod_d = oc.OctreeSplitByProb(mod_prob_d, threshold, true):cuda()

    local out_h = mod_h:forward(data_h)
    local out_d = mod_d:forward(data_d)

    -- test bwd
    local grad_out_h = out_h:clone():apply(function() return torch.uniform(-1,1) end)
    local grad_out_d = grad_out_h:clone():cuda()
    
    local grad_in_h = mod_h:backward(data_h, grad_out_h)
    local grad_in_d = mod_d:backward(data_d, grad_out_d)

    out_h = out_h
    out_d = out_d:float()
    mytester:assert(out_h:equals(out_d, 1e-6, true), 'error in OctreeSplitByProb forward')

    grad_in_h = grad_in_h
    grad_in_d = grad_in_d:float()
    mytester:assert(grad_in_h:equals(grad_in_d, 1e-6, true), 'error in OctreeSplitByProb backward')
  end

  local function test_full(data_h)
    local data_d = data_h:cuda()

    local mod_h = oc.OctreeSplitFull():float()
    local mod_d = oc.OctreeSplitFull():cuda()

    local out_h = mod_h:forward(data_h)
    local out_d = mod_d:forward(data_d)

    -- test bwd
    local grad_out_h = out_h:clone():apply(function() return torch.uniform(-1,1) end)
    local grad_out_d = grad_out_h:clone():cuda()
    
    local grad_in_h = mod_h:backward(data_h, grad_out_h)
    local grad_in_d = mod_d:backward(data_d, grad_out_d)

    out_h = out_h
    out_d = out_d:float()
    mytester:assert(out_h:equals(out_d, 1e-6, true), 'error in OctreeSplitFull forward')

    grad_in_h = grad_in_h
    grad_in_d = grad_in_d:float()
    mytester:assert(grad_in_h:equals(grad_in_d, 1e-4, true), 'error in OctreeSplitFull backward')
  end

  local function test_recsurf(data_h)
    local rec_h = test_utils.octree_alter_fs(data_h, 1)
    local data_h = oc.OctreeGridUnpool2x2x2():forward(data_h)

    local rec_d = rec_h:clone():cuda()
    local data_d = data_h:clone():cuda()

    local mod_rec_h = nn.Identity():float()
    mod_rec_h:forward(rec_h)
    local mod_rec_d = nn.Identity():cuda()
    mod_rec_d:forward(rec_d)

    local mod_h = oc.OctreeSplitRecSurf(mod_rec_h, 0, 1e9):float()
    local mod_d = oc.OctreeSplitRecSurf(mod_rec_d, 0, 1e9):cuda()

    local out_h = mod_h:forward(data_h)
    local out_d = mod_d:forward(data_d)

    -- test bwd
    local grad_out_h = out_h:clone():apply(function() return torch.uniform(-1,1) end)
    local grad_out_d = grad_out_h:clone():cuda()
    
    local grad_in_h = mod_h:backward(data_h, grad_out_h)
    local grad_in_d = mod_d:backward(data_d, grad_out_d)

    out_h = out_h
    out_d = out_d:float()
    mytester:assert(out_h:equals(out_d, 1e-6, true), 'error in OctreeSplitRecSurf forward')

    grad_in_h = grad_in_h
    grad_in_d = grad_in_d:float()
    mytester:assert(grad_in_h:equals(grad_in_d, 1e-6, true), 'error in OctreeSplitRecSurf backward')
  end

  local function test_densesurfrec(n, fs, vx_res, structure_type)
    local features = torch.rand(n, fs, vx_res,vx_res,vx_res):cuda()
    local rec = torch.rand(n, 1, vx_res,vx_res,vx_res):cuda()

    local rec_mod = nn.Identity()
    rec_mod:forward(rec)
    local mod = oc.OctreeDenseSplitSurf(rec_mod, 0.95, 1, structure_type):cuda()
    local out = mod:forward(features)
    
    local test_mod = nn.Sequential()
      :add( oc.VolumetricNNUpsampling(2,2,2) )
      :add( oc.CDHWToOctree(mod, 'avg') )
    test_mod = test_mod:cuda()
    mod.received_input = out
    local test_out = test_mod:forward(features)

    out = out:float()
    test_out = test_out:float()
    -- print(features)
    -- print(rec)
    -- out:print()
    -- test_out:print()
    mytester:assert(out:equals(test_out, 1e-6, true), 'error in OctreeDenseSplitRecSurf forward')


    local grad_out = out:clone()
    grad_out:apply(function(x) return math.random() end)
    grad_out = grad_out:cuda()

    local out = mod:backward(features, grad_out)
    local test_out = mod:backward(features, grad_out)

    -- print(out)
    -- print(test_out)
    local err = torch.abs(out - test_out):max()
    mytester:assert(err < 1e-6, 'error in OctreeDenseSplitSurf backward')
  end

  local function test_densesurfrecfres(n, fs, vx_res, band)
    local features = torch.rand(n, fs, vx_res,vx_res,vx_res):cuda()
    local rec = torch.rand(n, 1, vx_res,vx_res,vx_res):cuda()

    local rec_mod = nn.Identity()
    rec_mod:forward(rec)
    local mod = oc.OctreeDenseSplitSurfFres(rec_mod, 0.95, 1, band):cuda()
    local out = mod:forward(features)
    
    local test_mod = nn.Sequential()
      :add( oc.CDHWToOctree(mod, 'avg') )
    test_mod = test_mod:cuda()
    mod.received_input = out
    local test_out = test_mod:forward(features)

    out = out:float()
    test_out = test_out:float()
    -- print(features)
    -- print(rec)
    -- out:print()
    -- test_out:print()
    mytester:assert(out:equals(test_out, 1e-6, true), 'error in OctreeDenseSplitSurfFres forward')


    local rec_mod_f = nn.Identity()
    rec_mod_f:forward(rec:float())
    local mod_f = oc.OctreeDenseSplitSurfFres(rec_mod_f, 0.95, 1, band):float()
    local out_f = mod_f:forward(features:float())
    mytester:assert(out_f:equals(out, 1e-6, true), 'error in OctreeDenseSplitSurfFres forward')


    local grad_out = out:clone()
    grad_out:apply(function(x) return math.random() end)
    grad_out = grad_out:cuda()

    local out = mod:backward(features, grad_out)
    local test_out = mod:backward(features, grad_out)

    -- print(out)
    -- print(test_out)
    local err = torch.abs(out - test_out):max()
    mytester:assert(err < 1e-6, 'error in OctreeDenseSplitSurfFres backward')
  end
  
  local function test_tsdf(n, fs, vx_res, guide, band)
    local features, rec
    local guide_mod = nil
    local guide_grid
    if guide then
      guide_grid = test_utils.octree_rand(n, vx_res/4,vx_res/4,vx_res/4, fs, 0.5,0.5,0.5) 
      guide_grid = guide_grid:cuda()
      guide_mod = nn.Identity()
      guide_mod:forward(guide_grid)
      features = torch.randn(n, fs, vx_res,vx_res,vx_res):cuda()
      rec = torch.zeros(n, 1, vx_res,vx_res,vx_res):cuda()
    else
      features = torch.randn(n, fs, vx_res,vx_res,vx_res):cuda()
      rec = torch.randn(n, 1, vx_res,vx_res,vx_res):cuda():add(1.5)
    end

    local rec_mod = nn.Identity()
    rec_mod:forward(rec)
    local mod = oc.OctreeSplitTsdf(rec_mod, guide_mod):cuda()
    local out = mod:forward(features)
    
    local test_mod = nn.Sequential()
      :add( oc.VolumetricNNUpsampling(2,2,2) )
      :add( oc.CDHWToOctree(mod, 'avg') )
    test_mod = test_mod:cuda()
    mod.received_input = out
    local test_out = test_mod:forward(features)

    out = out:float()
    test_out = test_out:float()
    mytester:assert(out:equals(test_out, 1e-6, true), 'error in OctreeSplitTsdf forward')

    if guide then
      guide_grid = guide_grid:float()
      for idx = 0, 4 * guide_grid:n_blocks() - 1 do
        mytester:assert(guide_grid.grid.trees[idx] == out.grid.trees[idx], 'error in OctreeSplitTsdf forward (guide)')
      end
    end 

    local grad_out = out:clone()
    grad_out:apply(function(x) return math.random() end)
    grad_out = grad_out:cuda()

    local out = mod:backward(features, grad_out)
    local test_out = mod:backward(features, grad_out)

    local err = torch.abs(out - test_out):max()
    mytester:assert(err < 1e-6, 'error in OctreeSplitTsdf backward')
  end

  for _, band in ipairs{0, 1, 2} do
    for _, guide in ipairs{false, true} do
      test_tsdf(1, 1,  8, guide)
      test_tsdf(1, 1, 16, guide)
      test_tsdf(1, 3, 16, guide)
      test_tsdf(4, 5, 32, guide)
      test_tsdf(4, 5, 64, guide)
    end
  end

  -- for _, band in ipairs{0,1,3} do
  --   test_densesurfrecfres(1, 1, 8, band)
  --   test_densesurfrecfres(1, 1, 16, band)
  --   test_densesurfrecfres(1, 3, 16, band)
  --   test_densesurfrecfres(4, 5, 32, band)
  --   test_densesurfrecfres(4, 5, 64, band)
  -- end

  -- for _, structure_type in ipairs{'full', 'surface', 'octant'} do
  --   test_densesurfrec(1, 1, 4, structure_type)
  --   test_densesurfrec(1, 1, 16, structure_type)
  --   test_densesurfrec(1, 3, 16, structure_type)
  --   test_densesurfrec(4, 5, 32, structure_type)
  -- end

  -- local tests = {}
  -- for n in ipairs{1, 4} do
  --   for _, fs in ipairs{1,3,5} do
  --     local data = test_utils.octree_rand(n, 1,1,1, fs, 0,0,0, -1,1)
  --     table.insert(tests, data)
  --     local data = test_utils.octree_rand(n, 1,1,1, fs, 1,0,0, -1,1)
  --     table.insert(tests, data)
  --     local data = test_utils.octree_rand(n, 1,1,1, fs, 1,1,0, -1,1)
  --     table.insert(tests, data)
  --     local data = test_utils.octree_rand(n, 1,1,1, fs, 1,1,1, -1,1)
  --     table.insert(tests, data)
  --     local data = test_utils.octree_rand(n, 1,1,1, fs, 0.5,0.5,0.5, -1,1)
  --     table.insert(tests, data)
  --     local data = test_utils.octree_rand(n, 2,3,4, fs, 0.5,0.5,0.5, -1,1)
  --     table.insert(tests, data)
  --   end
  -- end
    
  -- for _, data in ipairs(tests) do
  --   test_by_prob(data)
  --   test_full(data)
  --   test_recsurf(data)
  -- end
end


function cuoctest.OctreeMSECriterion()
  local function test_grid_grid(in1_h)
    local in2_h = in1_h:clone()
    local data = torch.FloatTensor(in2_h:n_leafs() * in2_h:feature_size())
    data:apply(function() return torch.uniform(-1, 1) end)
    in2_h:cpy_data(data)

    local in1_d = in1_h:cuda()
    local in2_d = in2_h:cuda()

    local mod_h = oc.OctreeMSECriterion(true, true):float()
    local mod_d = oc.OctreeMSECriterion(true, true):cuda()
    
    local out_h = mod_h:forward(in1_h, in2_h)
    local out_d = mod_d:forward(in1_d, in2_d)

    -- print(out_h, out_d, torch.abs(out_h - out_d))
    mytester:assert(torch.abs(out_h - out_d) < 1e-6, 'error in OctreeMSECriterion forward')

   
    local out_h = mod_h:backward(in1_h, in2_h)
    local out_d = mod_d:backward(in1_d, in2_d) 
    
    out_d = out_d:float()
    mytester:assert(out_h:equals(out_d, 1e-6, true), 'error in OctreeMSECriterion backward')
  end

  local function test_grid_grid_ds(in1_h, in2_h)
    local in1_d = in1_h:cuda()
    local in2_d = in2_h:cuda()

    local mod_h = nn.MSECriterion():cuda()
    local mod_d = oc.OctreeMSECriterion(true, true, true):cuda()

    local out_h = mod_h:forward(in1_d:to_cdhw(), in2_d:to_cdhw())
    local out_d = mod_d:forward(in1_d, in2_d)

    local err = torch.abs(out_h - out_d)
    if err >= 1e-3 then print(out_h, out_d, err) end
    mytester:assert(err < 1e-3, 'error in OctreeMSECriterion grid_grid_ds forward '..err)
   
    local out_h = mod_h:backward(in1_d:to_cdhw(), in2_d:to_cdhw())
    local out_d = mod_d:backward(in1_d, in2_d) 
    
    local oc2de = nn.Identity()
    oc2de.received_input = out_d
    local de2oc = oc.CDHWToOctree(oc2de, 'sum'):cuda()
    out_h = de2oc:forward(out_h)

    out_d = out_d:float()
    out_h = out_h:float()
    mytester:assert(out_h:equals(out_d, 1e-5, true), 'error in OctreeMSECriterion grid_grid_ds backward')
  end

  local ocrand = test_utils.octree_rand
  for _, n in ipairs{1, 4} do
    test_grid_grid(ocrand(n, 1,1,1, 1, 0,0,0))
    test_grid_grid(ocrand(n, 1,1,1, 1, 1,1,1))
    test_grid_grid(ocrand(n, 2,2,2, 2, 0,0,0))
    test_grid_grid(ocrand(n, 2,2,2, 2, 1,0,0))
    test_grid_grid(ocrand(n, 2,2,2, 2, 1,1,0))
    test_grid_grid(ocrand(n, 2,2,2, 2, 1,1,1))
    test_grid_grid(ocrand(n, 2,2,2, 2, 0.5,0.5,0.5))
    test_grid_grid(ocrand(n, 2,4,6, 2, 0.5,0.5,0.5))
    test_grid_grid(ocrand(n, 4,8,6, 8, 0.5,0.5,0.5))

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
  end
end 


function cuoctest.OctreeClassNLLCriterion()
  local function test(in1_h, weights)
    local in2_h = in1_h:clone()
    in2_h.grid.feature_size = 1
    in2_h:update_prefix_leafs()
    local data = torch.FloatTensor(in2_h:n_leafs() * in2_h:feature_size())
    data:apply(function() return torch.random(1, in1_h:feature_size()) end)
    in2_h:cpy_data(data)

    local in1_d = in1_h:cuda()
    local in2_d = in2_h:cuda()

    local mod_h = oc.OctreeClassNLLCriterion(weights, true):float()
    local mod_d = oc.OctreeClassNLLCriterion(weights, true):cuda()
    
    local out_h = mod_h:forward(in1_h, in2_h)
    local out_d = mod_d:forward(in1_d, in2_d)

    -- print(out_h, out_d, torch.abs(out_h - out_d))
    mytester:assert(torch.abs(out_h - out_d) < 1e-5, 'error in OctreeClassNLLCriterion forward')

    
    local out_h = mod_h:backward(in1_h, in2_h)
    local out_d = mod_d:backward(in1_d, in2_d) 
    
    out_d = out_d:float()
    -- out_h:print()
    -- out_d:print()
    mytester:assert(out_h:equals(out_d, 1e-6, true), 'error in OctreeClassNLLCriterion backward')
  end

  
  local fs = 8
  for _, n in ipairs{1, 4} do
    for _, weights in ipairs{torch.ones(fs):float(), torch.rand(fs):float()} do
      test(test_utils.octree_rand(n, 2,1,1, fs, 0,0,0), weights)
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


function cuoctest.OctreeBCECriterion()
  local function test_grid_grid(in1_h)
    local in2_h = in1_h:clone()
    local data = torch.FloatTensor(in2_h:n_leafs() * in2_h:feature_size())
    data:apply(function() return torch.uniform(-1, 1) end)
    in2_h:cpy_data(data)

    local in1_d = in1_h:cuda()
    local in2_d = in2_h:cuda()

    local mod_h = oc.OctreeBCECriterion(true, true):float()
    local mod_d = oc.OctreeBCECriterion(true, true):cuda()

    local out_h = mod_h:forward(in1_h, in2_h)
    local out_d = mod_d:forward(in1_d, in2_d)

    -- print(out_h, out_d, torch.abs(out_h - out_d))
    local err = torch.abs(out_h - out_d)
    mytester:assert(err < 1e-5, 'error in OctreeBCECriterion grid_grid forward '..err)
   
    local out_h = mod_h:backward(in1_h, in2_h)
    local out_d = mod_d:backward(in1_d, in2_d) 
    
    out_d = out_d:float()
    mytester:assert(out_h:equals(out_d, 1e-6, true), 'error in OctreeBCECriterion grid_grid backward ')
  end
  
  local function test_grid_dense(in1_h)
    local in2_h = in1_h:clone()
    local data = torch.FloatTensor(in2_h:n_leafs() * in2_h:feature_size())
    data:apply(function() return torch.uniform(-1, 1) end)
    in2_h:cpy_data(data)
    in2_h = oc.OctreeToCDHW():forward(in2_h)

    local in1_d = in1_h:cuda()
    local in2_d = in2_h:cuda()

    local mod_h = oc.OctreeBCECriterion(true, true):float()
    local mod_d = oc.OctreeBCECriterion(true, true):cuda()

    local out_h = mod_h:forward(in1_h, in2_h)
    local out_d = mod_d:forward(in1_d, in2_d)

    -- print(out_h, out_d, torch.abs(out_h - out_d))
    local err = torch.abs(out_h - out_d)
    mytester:assert(err < 1e-5, 'error in OctreeBCECriterion grid_dense forward '..err)
   
    local out_h = mod_h:backward(in1_h, in2_h)
    local out_d = mod_d:backward(in1_d, in2_d) 
    
    out_d = out_d:float()
    mytester:assert(out_h:equals(out_d, 1e-5, true), 'error in OctreeBCECriterion grid_dense backward')
  end

  local function test_grid_grid_ds(in1_h, in2_h, weights)
    local in1_d = in1_h:cuda()
    local in2_d = in2_h:cuda()

    local mod_h, mod_d
    if weights then
      local weights = in2_h:clone()
      local data = torch.FloatTensor(in2_h:n_leafs() * in2_h:feature_size())
      data:apply(function() return torch.uniform(0, 2) end)
      weights:cpy_data(data) 
      in1_h = {in1_h, weights}

      weights = weights:clone():cuda()
      in1_d = {in1_d, weights}
      
      mod_h = oc.OctreeMaskedBCECriterion(true, true, true):float()
      mod_d = oc.OctreeMaskedBCECriterion(true, true, true):cuda()
    else
      mod_h = oc.OctreeBCECriterion(true, true, true):float()
      mod_d = oc.OctreeBCECriterion(true, true, true):cuda()
    end

    local out_h = mod_h:forward(in1_h, in2_h)
    local out_d = mod_d:forward(in1_d, in2_d)

    local err = torch.abs(out_h - out_d)
    -- print(out_h, out_d, err)
    mytester:assert(err < 1e-5, 'error in OctreeBCECriterion grid_grid_ds forward '..err)
   
    local out_h = mod_h:backward(in1_h, in2_h)
    local out_d = mod_d:backward(in1_d, in2_d) 

    if weights then
      out_h = out_h[1]
      out_d = out_d[1]:float()
    else
      out_d = out_d:float()
    end
    mytester:assert(out_h:equals(out_d, 1e-5, true), 'error in OctreeBCECriterion grid_grid_ds backward')
  end
  
  local function test_dense_dense(N, weights)
    local input_h = torch.FloatTensor(N)
    input_h:apply(function() return torch.uniform(0, 1) end)
    local target_h = torch.FloatTensor(N)
    target_h:apply(function() return torch.uniform(0, 1) end)

    local input1 = input_h:cuda()
    local input2 = input1
    local target = target_h:cuda()

    local mod1 = nn.BCECriterion(nil, true):cuda()
    local mod2
    if weights then
      weights = torch.FloatTensor(N)
      weights:fill(1)
      weights = weights:cuda()

      input2 = {input2, weights}
      
      mod2= oc.OctreeMaskedBCECriterion(true, true):cuda()
    else
      mod2= oc.OctreeBCECriterion(true, true):cuda()
    end

    local out1 = mod1:forward(input1, target)
    local out2 = mod2:forward(input2, target)

    -- print(out1, out2, torch.abs(out1 - out2))
    local err = torch.abs(out1 - out2)
    mytester:assert(err < 1e-5, 'error in OctreeBCECriterion dense_dense forward '..err)
   
    local out1 = mod1:backward(input1, target)
    local out2 = mod2:backward(input2, target) 

    if weights then
      out2 = out2[1]
    end
    
    local err = torch.abs(out1 - out2):max()
    mytester:assert(err < 1e-5, 'error in OctreeBCECriterion dense_dense backward '..err)
  end

  local function test_weights(grid_in, grid_ta)
    local grid_we = grid_ta:clone()
    local data = torch.FloatTensor(grid_ta:n_leafs() * grid_ta:feature_size())
    data:apply(function() return torch.uniform(0, 2) end)
    grid_we:cpy_data(data)
    
    local grid_in = grid_in:cuda()
    local grid_ta = grid_ta:cuda()
    local grid_we = grid_we:cuda()

    local o2d_in = oc.OctreeToCDHW():cuda()
    local o2d_ta = oc.OctreeToCDHW():cuda()
    local o2d_we = oc.OctreeToCDHW():cuda()

    local dense_in = o2d_in:forward(grid_in)
    local dense_ta = o2d_ta:forward(grid_ta)
    local dense_we = o2d_we:forward(grid_we)

    local crit_oc = oc.OctreeMaskedBCECriterion(true, false, true):cuda()
    local crit_de = oc.OctreeMaskedBCECriterion(true, false, true):cuda()

    local out_oc = crit_oc:forward({grid_in, grid_we}, grid_ta)
    local out_de = crit_de:forward({dense_in, dense_we}, dense_ta)

    local err = torch.abs(out_oc - out_de)
    -- print(out_oc, out_de, err)
    mytester:assert(err < 1e-4, 'error in test_weights forward '..err)
    
    local out_oc = crit_oc:backward({grid_in, grid_we}, grid_ta)[1]
    local out_de = crit_de:backward({dense_in, dense_we}, dense_ta)[1]
    local out_de = o2d_in:backward(grid_in, out_de)

    out_oc = out_oc:float()
    out_de = out_de:float()
    mytester:assert(out_oc:equals(out_de, 1e-4, true), 'error in test_weights backward')
  end
  
  local ocrand = test_utils.octree_rand 
    
  test_weights(ocrand(1, 1,1,1, 1, 0,0,0), ocrand(1, 1,1,1, 1, 0,0,0))
  test_weights(ocrand(1, 1,1,1, 1, 0,0,0), ocrand(1, 1,1,1, 1, 1,1,1))
  test_weights(ocrand(1, 1,1,1, 1, 1,1,1), ocrand(1, 1,1,1, 1, 0,0,0))
  test_weights(ocrand(1, 1,1,1, 1, 0.5,0.5,0.5), ocrand(1, 1,1,1, 1, 0.5,0.5,0.5))
  test_weights(ocrand(1, 1,1,1, 3, 0.5,0.5,0.5), ocrand(1, 1,1,1, 3, 0.5,0.5,0.5))
  test_weights(ocrand(1, 2,3,4, 3, 0.5,0.5,0.5), ocrand(1, 2,3,4, 3, 0.5,0.5,0.5))
  test_weights(ocrand(3, 2,3,4, 3, 0.5,0.5,0.5), ocrand(3, 2,3,4, 3, 0.5,0.5,0.5))
  
  for _, weights in ipairs{false, true} do
  for _, n in ipairs{1, 4} do
    test_grid_grid(ocrand(n, 1,1,1, 1, 0,0,0))
    test_grid_grid(ocrand(n, 1,1,1, 1, 1,1,1))
    test_grid_grid(ocrand(n, 2,2,2, 2, 0,0,0))
    test_grid_grid(ocrand(n, 2,2,2, 2, 1,0,0))
    test_grid_grid(ocrand(n, 2,2,2, 2, 1,1,0))
    test_grid_grid(ocrand(n, 2,2,2, 2, 1,1,1))
    test_grid_grid(ocrand(n, 2,2,2, 2, 0.5,0.5,0.5))
    test_grid_grid(ocrand(n, 2,4,6, 2, 0.5,0.5,0.5))
    test_grid_grid(ocrand(n, 4,8,6, 8, 0.5,0.5,0.5)) 

    test_grid_dense(ocrand(n, 1,1,2, 1, 0,0,0))
    test_grid_dense(ocrand(n, 1,1,1, 1, 1,1,1))
    test_grid_dense(ocrand(n, 2,2,2, 2, 0,0,0))
    test_grid_dense(ocrand(n, 2,2,2, 2, 1,0,0))
    test_grid_dense(ocrand(n, 2,2,2, 2, 1,1,0))
    test_grid_dense(ocrand(n, 2,2,2, 2, 1,1,1))
    test_grid_dense(ocrand(n, 2,2,2, 2, 0.5,0.5,0.5))
    test_grid_dense(ocrand(n, 2,4,6, 2, 0.5,0.5,0.5))
    test_grid_dense(ocrand(n, 4,8,6, 8, 0.5,0.5,0.5))

    -- test_grid_grid_ds(ocrand(n, 1,1,1, 1, 0,0,0), ocrand(n, 1,1,1, 1, 0,0,0), weights)
    -- test_grid_grid_ds(ocrand(n, 1,1,1, 1, 0,0,0), ocrand(n, 1,1,1, 1, 0.5,0.5,0.5), weights)
    -- test_grid_grid_ds(ocrand(n, 2,2,2, 2, 0,0,0), ocrand(n, 2,2,2, 2, 0,0,0), weights) 
    -- test_grid_grid_ds(ocrand(n, 2,2,2, 2, 0,0,0), ocrand(n, 2,2,2, 2, 0.5,0.5,0.5), weights) 
    -- test_grid_grid_ds(ocrand(n, 2,2,2, 2, 1,0,0), ocrand(n, 2,2,2, 2, 1,0,0), weights) 
    -- test_grid_grid_ds(ocrand(n, 2,2,2, 2, 1,0,0), ocrand(n, 2,2,2, 2, 0.5,0.5,0.5), weights) 
    -- test_grid_grid_ds(ocrand(n, 2,2,2, 2, 1,1,0), ocrand(n, 2,2,2, 2, 1,1,0), weights) 
    -- test_grid_grid_ds(ocrand(n, 2,2,2, 2, 1,1,0), ocrand(n, 2,2,2, 2, 0.5,0.5,0.5), weights) 
    -- test_grid_grid_ds(ocrand(n, 2,2,2, 2, 1,1,1), ocrand(n, 2,2,2, 2, 1,1,1), weights) 
    -- test_grid_grid_ds(ocrand(n, 2,2,2, 2, 1,1,1), ocrand(n, 2,2,2, 2, 0.5,0.5,0.5), weights) 
    -- test_grid_grid_ds(ocrand(n, 2,2,2, 2, 0.5,0.5,0.5), ocrand(n, 2,2,2, 2, 0.5,0.5,0.5), weights)
    -- test_grid_grid_ds(ocrand(n, 2,4,6, 2, 0.5,0.5,0.5), ocrand(n, 2,4,6, 2, 0.5,0.5,0.5), weights)
    -- test_grid_grid_ds(ocrand(n, 4,8,6, 6, 0.5,0.5,0.5), ocrand(n, 4,8,6, 6, 0.5,0.5,0.5), weights)

    test_dense_dense(100, weights)
    test_dense_dense(10000, weights)
  end
  end
end 



function cuoctest.OctreeMaskByLabel()
  local function test(input)
    local labels = input:clone()
    labels.grid.feature_size = 1
    labels:update_prefix_leafs()
    local data = torch.FloatTensor(labels:n_leafs() * labels:feature_size())
    data:apply(function() return torch.random(1, 3) end)
    labels:cpy_data(data)

    local mask_label = 1

    local input_d = input:cuda()
    local labels_d = labels:cuda()

    local out = input:mask_by_label(labels, mask_label, true)
    local out_d = input_d:mask_by_label(labels_d, mask_label, true)

    out_d = out_d:float()
    mytester:assert(out:equals(out_d, 1e-6, true), 'error in OctreeMaskByLabel forward')
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

function cuoctest.OctreeDetermineGtSplit()
  local function test(struc_h)
    local gt_h = torch.FloatTensor(struc_h:n(), struc_h:dense_depth(), struc_h:dense_height(), struc_h:dense_width())
    gt_h:fill(0)
    for leaf_idx = 1, struc_h.grid.n_leafs do 
      local grid_idx, bit_idx = struc_h:leaf_idx_to_grid_bit_idx(leaf_idx)
      local n,d,h,w = struc_h:dense_offset(grid_idx, bit_idx)
      -- print(leaf_idx, n,d,h,w)
      if torch.uniform(0,1) > 0.5 then 
        gt_h[{n,d,h,w}] = 1
      end
    end
    -- gt_h:apply(function() torch.uniform(0,1) end)

    local struc_d = struc_h:cuda()
    local gt_d = gt_h:cuda()

    local out_h = struc_h:determine_gt_split(gt_h)
    local out_d = struc_d:determine_gt_split(gt_d)
    out_d = out_d:float()

    -- struc_h:print()
    -- print(gt_h)
    -- out_h:print()
    -- out_d:print()

    mytester:assert(out_h:equals(out_d, 1e-6, true), 'error in OctreeDetermineGtSplit forward')
  end

  for _, n in ipairs{1, 4} do
    test(test_utils.octree_rand(n, 1,1,1, 1, 0,0,0))
    test(test_utils.octree_rand(n, 2,2,2, 1, 0,0,0))
    test(test_utils.octree_rand(n, 2,2,2, 3, 0,0,0))
    test(test_utils.octree_rand(n, 2,2,2, 3, 1,0,0))
    test(test_utils.octree_rand(n, 2,2,2, 3, 1,1,0))
    test(test_utils.octree_rand(n, 2,2,2, 3, 1,1,1))
    test(test_utils.octree_rand(n, 8,8,8, 3, 1,0,0))
    test(test_utils.octree_rand(n, 2,2,2, 1, 0.5,0.5,0.5))
    test(test_utils.octree_rand(n, 2,4,6, 3, 0.5,0.5,0.5))
    test(test_utils.octree_rand(n, 4,8,6, 8, 0.5,0.5,0.5))
  end
end



jac = nn.Jacobian


local seed = os.time()
-- local seed = 42
print('seed: '..seed)
math.randomseed(seed)
torch.manualSeed(seed)
torch.setnumthreads(1)

mytester:add(cuoctest)
mytester:run('OctreeMaskByLabel')

