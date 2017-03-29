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

local OctreeDenseConvolution, parent = torch.class('oc.OctreeDenseConvolution', 'oc.OctreeModule')

function OctreeDenseConvolution:__init(nInputPlane, nOutputPlane, rdc_fcn, inplace, backend)
  parent.__init(self)

  self.rdc_fcn = rdc_fcn or error('need to specify rdc_fcn') -- 'sum' | 'avg'
  self.inplace = inplace or false
  local backend = backend or nn

  self.o2d = oc.OctreeToCDHW()
  self.cnv = backend.VolumetricConvolution(nInputPlane,nOutputPlane, 3,3,3, 1,1,1, 1,1,1)
  self.d2o = oc.CDHWToOctree(self.o2d, self.rdc_fcn)

  self.output = self.d2o.output

  self.weight = self.cnv.weight
  self.bias = self.cnv.bias
  self.gradWeight = self.cnv.gradWeight
  self.gradBias = self.cnv.gradBias
end

function OctreeDenseConvolution:updateOutput(input)
  self.o2d:forward(input)
  self.cnv:forward(self.o2d.output)
  self.d2o:forward(self.cnv.output)
  
  self.output = self.d2o.output
  return self.output 
end 

function OctreeDenseConvolution:updateGradInput(input, gradOutput)
  if self.inplace then
    self.o2d.gradInput = gradOutput
  end
  self.d2o:backward(self.cnv.output, gradOutput)
  self.cnv:backward(self.o2d.output, self.d2o.gradInput)
  self.o2d:backward(input, self.cnv.gradInput)

  self.gradInput = self.o2d.gradInput
  return self.gradInput 
end
