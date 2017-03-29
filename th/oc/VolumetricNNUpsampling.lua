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

local VolumetricNNUpsampling, parent = torch.class('oc.VolumetricNNUpsampling', 'nn.Module')

function VolumetricNNUpsampling:__init(factor)
  parent.__init(self)

  self.factor = factor or error('need to specify factor')
end


function VolumetricNNUpsampling:updateOutput(input)
  if input:nDimension() ~= 5 then 
    error('invalid nDimension() for input')
  end
  local n = input:size(1)
  local fs = input:size(2)
  local d = input:size(3)
  local h = input:size(4)
  local w = input:size(5)
  local output = output or input.new()
  self.output:resize(n, fs, d * self.factor, h * self.factor, w * self.factor)
  if input:type() == 'torch.FloatTensor' then
    oc.cpu.volumetric_nn_upsampling_cdhw_cpu(input:data(), n,d,h,w,fs, self.factor, self.output:data())
  elseif input:type() == 'torch.CudaTensor' then
    oc.gpu.volumetric_nn_upsampling_cdhw_gpu(input:data(), n,d,h,w,fs, self.factor, self.output:data())
  else
    print(input:type())
    error('unknown type')
  end
  return self.output 
end 

function VolumetricNNUpsampling:updateGradInput(input, gradOutput)
  if input:nDimension() ~= 5 then 
    error('invalid nDimension() for input')
  end
  local n = input:size(1)
  local fs = input:size(2)
  local d = input:size(3)
  local h = input:size(4)
  local w = input:size(5)
  self.gradInput:resizeAs(input)
  if input:type() == 'torch.FloatTensor' then
    oc.cpu.volumetric_nn_upsampling_cdhw_bwd_cpu(gradOutput:data(), n,d,h,w,fs, self.factor, self.gradInput:data())
  elseif input:type() == 'torch.CudaTensor' then
    oc.gpu.volumetric_nn_upsampling_cdhw_bwd_gpu(gradOutput:data(), n,d,h,w,fs, self.factor, self.gradInput:data())
  else
    print(input:type())
    error('unknown type')
  end
  return self.gradInput 
end
