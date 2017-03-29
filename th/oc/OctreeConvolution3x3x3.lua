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

local OctreeConvolution3x3x3, parent = torch.class('oc.OctreeConvolution3x3x3', 'oc.OctreeModule')

function OctreeConvolution3x3x3:__init(nInputPlane, nOutputPlane, rdc_fcn)
  parent.__init(self)

  self.nInputPlane = nInputPlane or error('need to specify nInputPlane')
  self.nOutputPlane = nOutputPlane or error('need to specify nOutputPlane')

  self.rdc_fcn = rdc_fcn or error('need to specify rdc_fcn')

  self.weight = torch.Tensor(nOutputPlane, nInputPlane, 3, 3, 3)
  self.bias = torch.Tensor(nOutputPlane)
  self.gradWeight = torch.Tensor(nOutputPlane, nInputPlane, 3, 3, 3)
  self.gradBias = torch.Tensor(nOutputPlane)
  self:reset()
end

function OctreeConvolution3x3x3:reset(stdv)
  if stdv then
    stdv = stdv * math.sqrt(3)
  else
    stdv = 1/math.sqrt(3 * 3 * 3 * self.nInputPlane)
  end
  self.weight:uniform(-stdv, stdv)
  self.bias:uniform(-stdv, stdv)
end

function OctreeConvolution3x3x3:updateOutput(input)
  if input:feature_size() ~= self.nInputPlane then error('invalid input size') end

  if self.rdc_fcn == 'sum' then
    if input._type == 'oc_float' then
      oc.cpu.octree_conv3x3x3_sum_cpu(input.grid, self.weight:data(), self.bias:data(), self.nOutputPlane, self.output.grid)
    elseif input._type == 'oc_cuda' then
      oc.gpu.octree_conv3x3x3_sum_gpu(input.grid, self.weight:data(), self.bias:data(), self.nOutputPlane, self.output.grid)
    end
  elseif self.rdc_fcn == 'avg' then
    if input._type == 'oc_float' then
      oc.cpu.octree_conv3x3x3_avg_cpu(input.grid, self.weight:data(), self.bias:data(), self.nOutputPlane, self.output.grid)
    elseif input._type == 'oc_cuda' then
      oc.gpu.octree_conv3x3x3_avg_gpu(input.grid, self.weight:data(), self.bias:data(), self.nOutputPlane, self.output.grid)
    end
  else
    error('unknown reduce function: '..self.rdc_fcn)
  end

  return self.output
end 

function OctreeConvolution3x3x3:updateGradInput(input, gradOutput)
  if self.rdc_fcn == 'sum' then
    if input._type == 'oc_float' then
      oc.cpu.octree_conv3x3x3_sum_bwd_cpu(self.weight:data(), gradOutput.grid, self.nInputPlane, self.gradInput.grid)
    elseif input._type == 'oc_cuda' then
      oc.gpu.octree_conv3x3x3_sum_bwd_gpu(self.weight:data(), gradOutput.grid, self.nInputPlane, self.gradInput.grid)
    end
  elseif self.rdc_fcn == 'avg' then
    if input._type == 'oc_float' then
      oc.cpu.octree_conv3x3x3_avg_bwd_cpu(self.weight:data(), gradOutput.grid, self.nInputPlane, self.gradInput.grid)
    elseif input._type == 'oc_cuda' then
      oc.gpu.octree_conv3x3x3_avg_bwd_gpu(self.weight:data(), gradOutput.grid, self.nInputPlane, self.gradInput.grid)
    end
  else
    error('unknown reduce function: '..self.rdc_fcn)
  end

  return self.gradInput
end

function OctreeConvolution3x3x3:accGradParameters(input, gradOutput, scale)
  scale = scale or 1
  if self.rdc_fcn == 'sum' then
    if input._type == 'oc_float' then
      oc.cpu.octree_conv3x3x3_sum_wbwd_cpu(input.grid, gradOutput.grid, scale, self.gradWeight:data(), self.gradBias:data())
    elseif input._type == 'oc_cuda' then
      oc.gpu.octree_conv3x3x3_sum_wbwd_gpu(input.grid, gradOutput.grid, scale, self.gradWeight:data(), self.gradBias:data())
    end
  elseif self.rdc_fcn == 'avg' then
    if input._type == 'oc_float' then
      oc.cpu.octree_conv3x3x3_avg_wbwd_cpu(input.grid, gradOutput.grid, scale, self.gradWeight:data(), self.gradBias:data())
    elseif input._type == 'oc_cuda' then
      oc.gpu.octree_conv3x3x3_avg_wbwd_gpu(input.grid, gradOutput.grid, scale, self.gradWeight:data(), self.gradBias:data())
    end
  else
    error('unknown reduce function: '..self.rdc_fcn)
  end
end
