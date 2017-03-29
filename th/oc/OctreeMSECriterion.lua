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

local OctreeMSECriterion, parent = torch.class('oc.OctreeMSECriterion', 'oc.OctreeCriterion')

function OctreeMSECriterion:__init(size_average, check, ds)
  parent.__init(self)

  if size_average ~= nil then
    self.size_average = size_average
  else
    self.size_average = true
  end
  self.check = check or false

  self.ds = ds or false
end

function OctreeMSECriterion:updateOutput(input, target)
  self.output = -1

  if self.ds then
    if input._type == 'oc_float' then
      error('not implemented')
    elseif input._type == 'oc_cuda' then
      self.output = oc.gpu.octree_mse_ds_loss_gpu(input.grid, target.grid, self.size_average)
    end
  else
    if input._type == 'oc_float' then
      self.output = oc.cpu.octree_mse_loss_cpu(input.grid, target.grid, self.size_average, self.check)
    elseif input._type == 'oc_cuda' then
      self.output = oc.gpu.octree_mse_loss_gpu(input.grid, target.grid, self.size_average, self.check)
    end
  end


  return self.output
end 

function OctreeMSECriterion:updateGradInput(input, target)
  if self.ds then
    if input._type == 'oc_float' then
      error('not implemented')
    elseif input._type == 'oc_cuda' then
      oc.gpu.octree_mse_loss_ds_bwd_gpu(input.grid, target.grid, self.size_average, self.gradInput.grid)
    end
  else
    if input._type == 'oc_float' then
      oc.cpu.octree_mse_loss_bwd_cpu(input.grid, target.grid, self.size_average, self.check, self.gradInput.grid)
    elseif input._type == 'oc_cuda' then
      oc.gpu.octree_mse_loss_bwd_gpu(input.grid, target.grid, self.size_average, self.check, self.gradInput.grid)
    end
  end

  return self.gradInput
end
