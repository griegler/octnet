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

local OctreeToCDHW, parent = torch.class('oc.OctreeToCDHW', 'oc.OctreeModule')

function OctreeToCDHW:__init(dense_depth, dense_height, dense_width)
  parent.__init(self)

  self.dense_depth = dense_depth
  self.dense_height = dense_height
  self.dense_width = dense_width

  self.received_input = oc.FloatOctree()
  self.output = torch.FloatTensor()
end

function OctreeToCDHW:dense_dimensions(octrees)
  if self.dense_depth and self.dense_height and self.dense_width then
    return self.dense_depth, self.dense_height, self.dense_width 
  else
    return octrees:dense_depth(), octrees:dense_height(), octrees:dense_width()
  end
end 

function OctreeToCDHW:updateOutput(input)
  self.received_input = input
  local dense_depth, dense_height, dense_width = self:dense_dimensions(input)

  local out_size = torch.LongStorage({input:n(), input:feature_size(), dense_depth, dense_height, dense_width})
  self.output:resize(out_size)
  
  if not self.output:isContiguous() then error('output is not contiguous') end
  if self.output:size(2) ~= input:feature_size() then error('invalid feature_size') end

  if input._type == 'oc_float' then
    oc.cpu.octree_to_cdhw_cpu(input.grid, dense_depth, dense_height, dense_width, self.output:data())
  elseif input._type == 'oc_cuda' then
    oc.gpu.octree_to_cdhw_gpu(input.grid, dense_depth, dense_height, dense_width, self.output:data())
  end

  return self.output
end 

function OctreeToCDHW:updateGradInput(input, gradOutput)
  local dense_depth, dense_height, dense_width = self:dense_dimensions(input)

  if not gradOutput:isContiguous() then error('gradOutput is not contiguous') end
  if gradOutput:size(2) ~= input:feature_size() then error('invalid feature_size') end

  if input._type == 'oc_float' then
    oc.cpu.octree_to_cdhw_bwd_cpu(input.grid, dense_depth, dense_height, dense_width, gradOutput:data(), self.gradInput.grid)
  elseif input._type == 'oc_cuda' then
    oc.gpu.octree_to_cdhw_bwd_gpu(input.grid, dense_depth, dense_height, dense_width, gradOutput:data(), self.gradInput.grid)
  end

  return self.gradInput
end
