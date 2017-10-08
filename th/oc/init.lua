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

require'torch'
require'nn'
require'nngraph'
nesting = require'nngraph.nesting'

oc = {}
include('ffi.lua')

include('octree.lua')

include('OctreeModule.lua')
include('OctreeCriterion.lua')
include('OctreeToCDHW.lua')
include('CDHWToOctree.lua')
include('OctreeDenseConvolution.lua')
include('OctreeConvolution3x3x3.lua')
include('OctreeConvolutionMM.lua')
include('OctreePool2x2x2.lua')
include('OctreeGridPool2x2x2.lua')
include('OctreeGridUnpool2x2x2.lua')
include('OctreeGridUnpoolGuided2x2x2.lua')
include('OctreeConcat.lua')
include('OctreeReLU.lua')
include('OctreeSigmoid.lua')
include('OctreeLogSoftMax.lua')
include('OctreeSplit.lua')

include('OctreeMSECriterion.lua')
include('OctreeClassNLLCriterion.lua')
include('OctreeCrossEntropyCriterion.lua')
include('OctreeBCECriterion.lua')
include('OctreeMaskedBCECriterion.lua')
include('OctreeSplitCriterion.lua')
include('OctreeDummyCriterion.lua')

include('VolumetricNNUpsampling.lua')


function oc.validateShape(in1, in2, ignore_dim) 
  in1_shape = in1:size()
  in2_shape = in2:size()
  for dim = 1, #in1_shape do
    if in1_shape[dim] ~= in2_shape[dim] and (not ignore_dim or dim ~= ignore_dim) then
      error(string.format('dim %d of in1 and in2 differ (%d, %d)', dim, in1_shape[dim], in2_shape[dim]))
    end
  end
  for dim = 1, #in2_shape do
    if in1_shape[dim] ~= in2_shape[dim] and (not ignore_dim or dim ~= ignore_dim) then
      error(string.format('dim %d of in1 and in2 differ (%d, %d)', dim, in1_shape[dim], in2_shape[dim]))
    end
  end

end


--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
-- overload nn Module functions
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
function nn.CAddTable:oc()
  self.output = oc.FloatOctree()
  return self
end 

--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
-- overload nn.utils functions to work with Octrees
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
local function torch_Storage_type(self, type)
  local current = torch.typename(self)
  if not type then return current end
  if type ~= current then
    local new = torch.getmetatable(type).new()
    if self:size() > 0 then
      new:resize(self:size()):copy(self)
    end
    return new
  else
    return self
  end
end

function nn.utils.recursiveType(param, type, tensorCache)
  tensorCache = tensorCache or {}

  if torch.type(param) == 'oc.FloatOctree' and type == 'torch.CudaTensor' then
    param = param:cuda()
  elseif torch.type(param) == 'oc.CudaOctree' and type == 'torch.FloatTensor' then
    param = param:float()
  elseif torch.type(param) == 'table' then
    for k, v in pairs(param) do
      param[k] = nn.utils.recursiveType(v, type, tensorCache)
    end
  elseif torch.isTypeOf(param, 'nn.Module') or
    torch.isTypeOf(param, 'nn.Criterion') then
    param:type(type, tensorCache)
  elseif torch.isTensor(param) then
    if torch.typename(param) ~= type then
      local newparam
      if tensorCache[param] then
        newparam = tensorCache[param]
      else
        newparam = torch.Tensor():type(type)
        local storageType = type:gsub('Tensor','Storage')
        if param:storage() then
          local storage_key = torch.pointer(param:storage())
          if not tensorCache[storage_key] then
            tensorCache[storage_key] = torch_Storage_type(
              param:storage(), storageType)
          end
          assert(torch.type(tensorCache[storage_key]) == storageType)
          newparam:set(
            tensorCache[storage_key],
            param:storageOffset(),
            param:size(),
            param:stride()
            )
        end
        tensorCache[param] = newparam
      end
      assert(torch.type(newparam) == type)
      param = newparam
    end
  end
  return param
end

function nn.utils.recursiveCopy(t1,t2)
  if torch.type(t2) == 'table' then
    t1 = (torch.type(t1) == 'table') and t1 or {t1}
    for key,_ in pairs(t2) do
      t1[key], t2[key] = nn.utils.recursiveCopy(t1[key], t2[key])
    end
  elseif torch.isTensor(t2) then
    t1 = torch.isTensor(t1) and t1 or t2.new()
    t1:resizeAs(t2):copy(t2)
  elseif oc.isOctree(t2) then
    t1 = oc.isOctree(t1) and t1 or t2:new()
    t1:resizeAs(t2):copy(t2)
  else
    error("expecting nested tensors, octrees or tables. Got "..
          torch.type(t1).." and "..torch.type(t2).." instead")
  end
  return t1, t2
end

function nn.utils.recursiveResizeAs(t1,t2)
  if torch.type(t2) == 'table' then
    t1 = (torch.type(t1) == 'table') and t1 or {t1}
    for key,_ in pairs(t2) do
      t1[key], t2[key] = nn.utils.recursiveResizeAs(t1[key], t2[key])
    end
    for key,_ in pairs(t1) do
      if not t2[key] then
        t1[key] = nil
      end
    end
  elseif torch.isTensor(t2) then
    t1 = torch.isTensor(t1) and t1 or t2.new()
    t1:resizeAs(t2)
  elseif oc.isOctree(t2) then
    t1 = oc.isOctree(t1) and t1 or t2:new()
    t1:resizeAs(t2)
  else
    error("expecting nested tensors or tables. Got "..
          torch.type(t1).." and "..torch.type(t2).." instead")
  end
  return t1, t2
end

function nn.utils.recursiveFill(t2, val)
  if torch.type(t2) == 'table' then
    for key,_ in pairs(t2) do
       t2[key] = nn.utils.recursiveFill(t2[key], val)
    end
  elseif torch.isTensor(t2) then
    t2:fill(val)
  elseif oc.isOctree(t2) then
    t2:fill(val)
  else
    error("expecting tensor or table thereof. Got "
          ..torch.type(t2).." instead")
  end
  return t2
end

function nn.utils.recursiveAdd(t1, val, t2)
  if not t2 then
    assert(val, "expecting at least two arguments")
    t2 = val
    val = 1
  end
  val = val or 1
  if torch.type(t2) == 'table' then
    t1 = (torch.type(t1) == 'table') and t1 or {t1}
    for key,_ in pairs(t2) do
      t1[key], t2[key] = nn.utils.recursiveAdd(t1[key], val, t2[key])
    end
  elseif torch.isTensor(t1) and torch.isTensor(t2) then
    t1:add(val, t2)
  elseif oc.isOctree(t1) and oc.isOctree(t2) then
    t1:add(val, t2)
  else
    error("expecting nested tensors or tables. Got "..
          torch.type(t1).." and "..torch.type(t2).." instead")
  end
  return t1, t2
end

function nn.utils.clear(self, ...)
  local arg = {...}
  if #arg > 0 and type(arg[1]) == 'table' then
    arg = arg[1]
  end
  local function clear(f)
    if self[f] then
      if torch.isTensor(self[f]) then
        self[f]:set()
      elseif oc.isOctree(self[f]) then
        self[f]:set()
      elseif type(self[f]) == 'table' then
        self[f] = {}
      else
        self[f] = nil
      end
    end
  end
  for i,v in ipairs(arg) do clear(v) end
  return self
end


--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
-- overload nngraph functions to work with Octrees
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
function nesting.cloneNested(obj)
  if torch.isTensor(obj) or oc.isOctree(obj) then
    return obj:clone()
  end

  local result = {}
  for key, child in pairs(obj) do
    result[key] = nesting.cloneNested(child)
  end
  return result
end

function nesting.resizeNestedAs(output, input)
  if torch.isTensor(output) or oc.isOctree(output) then
    output:resizeAs(input)
  else
    for key, child in pairs(input) do
      -- A new element is added to the output, if needed.
      if not output[key] then
        output[key] = nesting.cloneNested(child)
      else
        nesting.resizeNestedAs(output[key], child)
      end
    end
    -- Extra elements are removed from the output.
    for key, child in pairs(output) do
      if not input[key] then
        output[key] = nil
      end
    end
  end
end

function nesting.copyNested(output, input)
  if torch.isTensor(output) or oc.isOctree(output) then
    output:copy(input)
  else
    for key, child in pairs(input) do
      nesting.copyNested(output[key], child)
    end
    -- Extra elements in the output table cause an error.
    for key, child in pairs(output) do
      if not input[key] then
        error('key ' .. tostring(key) ..
               ' present in output but not in input')
      end
    end
  end
end

function nesting.addNestedTo(output, input)
  if torch.isTensor(output) or oc.isOctree(output) then
    output:add(input)
  else
    for key, child in pairs(input) do
      assert(output[key] ~= nil, "missing key")
      nesting.addNestedTo(output[key], child)
    end
  end
end
