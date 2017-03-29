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

local ffi = require('ffi')

local OctreeMaskedBCECriterion, parent = torch.class('oc.OctreeMaskedBCECriterion', 'oc.OctreeCriterion')

function OctreeMaskedBCECriterion:__init(size_average, check, different_structure)
  parent.__init(self)

  if size_average ~= nil then
    self.size_average = size_average
  else
    self.size_average = true
  end
  self.check = check or false
  self.ds = different_structure or false

  self.gradInput = {oc.FloatOctree(), {}}
end

function OctreeMaskedBCECriterion:updateOutput(input, target)
  self.output = -1

  local weights = input[2]
  input = input[1]
    
  if torch.isTensor(weights) then
    weights = weights:data() 
  else
    weights = weights.grid
  end
  
  oc.validateShape(input, target)

  local out = torch.FloatTensor(1)
  local total_weight = torch.FloatTensor(1)
  if torch.type(input) == 'torch.CudaTensor' then
    oc.gpu.dense_bce_loss_gpu(input:data(), target:data(), weights, input:nElement(), out:data(), total_weight:data())
  else
    if input._type == 'oc_float' then
      if torch.isTensor(target) then
        error('masked not implemented for input=oc_float and target=tensor')
      else
        if self.ds then
          oc.cpu.octree_bce_ds_loss_cpu(input.grid, target.grid, weights, self.size_average, out:data(), total_weight:data())
        else
          error('masked not implemented for input=oc_float and not self.ds')
        end
      end
    elseif input._type == 'oc_cuda' then
      if torch.isTensor(target) then
        error('masked not implemented for input=oc_cuda and target=tensor')
      else
        if self.ds then
          oc.gpu.octree_bce_ds_loss_gpu(input.grid, target.grid, weights, self.size_average, out:data(), total_weight:data())
        else
          error('masked not implemented for input=oc_cuda and not self.ds')
        end
      end
    end
  end

  self.output = out[1]
  self.total_weight = total_weight[1]

  return self.output
end 

function OctreeMaskedBCECriterion:updateGradInput(input, target)
  local weights = input[2]
  input = input[1]
    
  if torch.isTensor(weights) then
    weights = weights:data() 
  else
    weights = weights.grid
  end

  if torch.type(input) == 'torch.CudaTensor' then
    if torch.type(self.gradInput[1]) ~= 'torch.CudaTensor' then 
      self.gradInput[1] = torch.CudaTensor()
    end
    self.gradInput[1]:resizeAs(input)
    oc.gpu.dense_bce_loss_bwd_gpu(input:data(), target:data(), weights, input:nElement(), self.total_weight, self.gradInput[1]:data())
  else
    if input._type == 'oc_float' then
      if torch.isTensor(target) then
        error('masked not implemented')
      else
        if self.ds then
          oc.cpu.octree_bce_ds_loss_bwd_cpu(input.grid, target.grid, weights, self.size_average, self.total_weight, self.gradInput[1].grid)
        else
          error('masked not implemented')
        end
      end
    elseif input._type == 'oc_cuda' then
      if torch.isTensor(target) then
        error('masked not implemented')
      else
        if self.ds then
          oc.gpu.octree_bce_ds_loss_bwd_gpu(input.grid, target.grid, weights, self.size_average, self.total_weight, self.gradInput[1].grid)
        else
          error('masked not implemented')
        end
      end
    end
  end

  return self.gradInput
end
