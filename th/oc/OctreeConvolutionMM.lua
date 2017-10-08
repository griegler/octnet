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

local OctreeConvolutionMM, parent = torch.class('oc.OctreeConvolutionMM', 'oc.OctreeModule')

function OctreeConvolutionMM:__init(nInputPlane, nOutputPlane, n_grids)
  parent.__init(self)

  self.nInputPlane = nInputPlane or error('need to specify nInputPlane')
  self.nOutputPlane = nOutputPlane or error('need to specify nOutputPlane')

  self.n_grids = n_grids or -1

  self.kW = 3
  self.kH = 3
  self.kT = 3

  self.padT = 1
  self.padH = 1
  self.padW = 1

  self.dT = 1
  self.dH = 1
  self.dW = 1

  self.weight = torch.Tensor(nOutputPlane, nInputPlane, 3, 3, 3)
  self.bias = torch.Tensor(nOutputPlane)
  self.gradWeight = torch.Tensor(nOutputPlane, nInputPlane, 3, 3, 3)
  self.gradBias = torch.Tensor(nOutputPlane)
  self:reset()
end

function OctreeConvolutionMM:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1/math.sqrt(3 * 3 * 3 * self.nInputPlane)
   end
   
  self.weight:uniform(-stdv, stdv)
  self.bias:uniform(-stdv, stdv)
end

function OctreeConvolutionMM:updateOutput(input)
  local n_grids = self.n_grids or -1

  -- print(input, input:size())
  -- print(string.format('[OctreeConvolutionMM] update output %d -> %d', self.nInputPlane, self.nOutputPlane))

  if input:feature_size() ~= self.nInputPlane then
    error('invalid input size, self.nInputPlane='..self.nInputPlane..', input:feature_size()='..input:feature_size())
  end

  if input._type == 'oc_float' then
    error('not implemented')
  elseif input._type == 'oc_cuda' then
    local cublas_handle = get_cublas_handle()
    oc.gpu.octree_conv_mm_gpu(cublas_handle, input.grid, self.weight:data(), self.bias:data(), self.nOutputPlane, n_grids, self.output.grid)
  end

  -- print(string.format('[OctreeConvolutionMM] DONE update output %d -> %d', self.nInputPlane, self.nOutputPlane))

  return self.output
end 

function OctreeConvolutionMM:updateGradInput(input, gradOutput)
  local n_grids = self.n_grids or -1

  if input._type == 'oc_float' then
    error('not implemented')
  elseif input._type == 'oc_cuda' then
    local cublas_handle = get_cublas_handle()
    oc.gpu.octree_conv_mm_bwd_gpu(cublas_handle, gradOutput.grid, self.weight:data(), self.nInputPlane, n_grids, self.gradInput.grid)
  end

  return self.gradInput
end

function OctreeConvolutionMM:accGradParameters(input, gradOutput, scale)
  scale = scale or 1
  local n_grids = self.n_grids or -1
  
  if input._type == 'oc_float' then
    error('not implemented')
  elseif input._type == 'oc_cuda' then
    local cublas_handle = get_cublas_handle()
    oc.gpu.octree_conv_mm_wbwd_gpu(cublas_handle, input.grid, gradOutput.grid, scale, n_grids, self.gradWeight:data(), self.gradBias:data())
  end
end


function OctreeConvolutionMM:__tostring__()
  local s = string.format('%s(%d -> %d, %dx%dx%d', torch.type(self),
      self.nInputPlane, self.nOutputPlane, self.kT, self.kW, self.kH)
  if self.dT ~= 1 or self.dW ~= 1 or self.dH ~= 1 or
     self.padT ~= 0 or self.padW ~= 0 or self.padH ~= 0 then
    s = s .. string.format(', %d,%d,%d', self.dT, self.dW, self.dH)
  end
  if (self.padT or self.padW or self.padH) and
     (self.padT ~=0 or self.padW ~= 0 or self.padH ~= 0) then
    s = s .. ', ' .. self.padT .. ',' .. self.padW .. ',' .. self.padH
  end
  s = s .. ', n_grids=' .. self.n_grids
  return s .. ')'
 end
