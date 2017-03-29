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

local OctreeBCECriterion, parent = torch.class('oc.OctreeBCECriterion', 'oc.OctreeCriterion')

function OctreeBCECriterion:__init(size_average, check, different_structure)
  parent.__init(self)

  if size_average ~= nil then
    self.size_average = size_average
  else
    self.size_average = true
  end
  self.check = check or false
  self.ds = different_structure or false
end

function OctreeBCECriterion:updateOutput(input, target)
  self.output = -1

  local weights = ffi.new("void*", nil)
  
  oc.validateShape(input, target)

  local out = torch.FloatTensor(1)
  local total_weight = torch.FloatTensor(1)
  if torch.type(input) == 'torch.CudaTensor' then
    oc.gpu.dense_bce_loss_gpu(input:data(), target:data(), weights, input:nElement(), out:data(), total_weight:data())
  else
    if input._type == 'oc_float' then
      if torch.isTensor(target) then
        oc.cpu.octree_bce_dense_loss_cpu(input.grid, target:data(), self.size_average, out:data(), total_weight:data())
      else
        if self.ds then
          oc.cpu.octree_bce_ds_loss_cpu(input.grid, target.grid, weights, self.size_average, out:data(), total_weight:data())
        else
          oc.cpu.octree_bce_loss_cpu(input.grid, target.grid, self.size_average, self.check, out:data(), total_weight:data())
        end
      end
    elseif input._type == 'oc_cuda' then
      if torch.isTensor(target) then
        oc.gpu.octree_bce_dense_loss_gpu(input.grid, target:data(), self.size_average, out:data(), total_weight:data())
      else
        if self.ds then
          oc.gpu.octree_bce_ds_loss_gpu(input.grid, target.grid, weights, self.size_average, out:data(), total_weight:data())
        else
          oc.gpu.octree_bce_loss_gpu(input.grid, target.grid, self.size_average, self.check, out:data(), total_weight:data())
        end
      end
    end
  end
    
  self.output = out[1]
  self.total_weight = total_weight[1]

  return self.output
end 

function OctreeBCECriterion:updateGradInput(input, target)
  local weights = ffi.new("void*", nil)

  if torch.type(input) == 'torch.CudaTensor' then
    if torch.type(self.gradInput) ~= 'torch.CudaTensor' then 
      self.gradInput = torch.CudaTensor()
    end
    self.gradInput:resizeAs(input)
    oc.gpu.dense_bce_loss_bwd_gpu(input:data(), target:data(), weights, input:nElement(), self.total_weight, self.gradInput:data())
  else
    if input._type == 'oc_float' then
      if torch.isTensor(target) then
        oc.cpu.octree_bce_dense_loss_bwd_cpu(input.grid, target:data(), self.size_average, self.gradInput.grid)
      else
        if self.ds then
          oc.cpu.octree_bce_ds_loss_bwd_cpu(input.grid, target.grid, weights, self.size_average, self.total_weight, self.gradInput.grid)
        else
          oc.cpu.octree_bce_loss_bwd_cpu(input.grid, target.grid, self.size_average, self.check, self.gradInput.grid)
        end
      end
    elseif input._type == 'oc_cuda' then
      if torch.isTensor(target) then
        oc.gpu.octree_bce_dense_loss_bwd_gpu(input.grid, target:data(), self.size_average, self.gradInput.grid)
      else
        if self.ds then
          oc.gpu.octree_bce_ds_loss_bwd_gpu(input.grid, target.grid, weights, self.size_average, self.total_weight, self.gradInput.grid)
        else
          oc.gpu.octree_bce_loss_bwd_gpu(input.grid, target.grid, self.size_average, self.check, self.gradInput.grid)
        end
      end
    end
  end

  return self.gradInput
end
