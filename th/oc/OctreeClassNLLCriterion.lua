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

local OctreeClassNLLCriterion, parent = torch.class('oc.OctreeClassNLLCriterion', 'oc.OctreeCriterion')

function OctreeClassNLLCriterion:__init(weights, size_average, check)
  parent.__init(self)

  self.weights = weights or false
  if size_average ~= nil then
    self.size_average = size_average
  else
    self.size_average = true
  end
  self.check = check or false
end

local function getWeights(weights, input)
  local ret_weights
  if weights then
    if weights:size(1) ~= input:feature_size() then
      error('[ERROR] OctreeClassNLLCriterion: weights does not match input feature size')
    end
    ret_weights = weights
  else
    ret_weights = torch.ones(input:feature_size()):float()
  end

  if input._type == 'oc_float' then
    ret_weights = ret_weights:float()
  elseif input._type == 'oc_cuda' then
    ret_weights = ret_weights:cuda()
  end

  return ret_weights
end 

function OctreeClassNLLCriterion:updateOutput(input, target)
  local weights = getWeights(self.weights, input)

  local out = out or torch.FloatTensor(1)
  local total_weight = total_weight or torch.FloatTensor(1)
  if input._type == 'oc_float' then
    oc.cpu.octree_nll_loss_cpu(input.grid, target.grid, weights:data(), 1, self.size_average, self.check, out:data(), total_weight:data())
  elseif input._type == 'oc_cuda' then
    oc.gpu.octree_nll_loss_gpu(input.grid, target.grid, weights:data(), 1, self.size_average, self.check, out:data(), total_weight:data())
  end

  self.output = out[1]
  self.total_weight = total_weight[1]

  return self.output
end 

function OctreeClassNLLCriterion:updateGradInput(input, target)
  local weights = getWeights(self.weights, input)

  if input._type == 'oc_float' then
    oc.cpu.octree_nll_loss_bwd_cpu(input.grid, target.grid, weights:data(), self.total_weight, 1, self.size_average, self.check, self.gradInput.grid)
  elseif input._type == 'oc_cuda' then
    oc.gpu.octree_nll_loss_bwd_gpu(input.grid, target.grid, weights:data(), self.total_weight, 1, self.size_average, self.check, self.gradInput.grid)
  end

  return self.gradInput
end
