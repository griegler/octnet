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

local OctreeLogSoftMax, parent = torch.class('oc.OctreeLogSoftMax', 'oc.OctreeModule')

function OctreeLogSoftMax:__init()
  parent.__init(self)
end

function OctreeLogSoftMax:updateOutput(input)
  if input._type == 'oc_float' then
    oc.cpu.octree_logsoftmax_cpu(input.grid, self.output.grid)
  elseif input._type == 'oc_cuda' then
    oc.gpu.octree_logsoftmax_gpu(input.grid, self.output.grid)
  end

  return self.output 
end 

function OctreeLogSoftMax:updateGradInput(input, gradOutput)
  if input._type == 'oc_float' then
    oc.cpu.octree_logsoftmax_bwd_cpu(input.grid, self.output.grid, gradOutput.grid, self.gradInput.grid)
  elseif input._type == 'oc_cuda' then
    oc.gpu.octree_logsoftmax_bwd_gpu(input.grid, self.output.grid, gradOutput.grid, self.gradInput.grid)
  end

  return self.gradInput  
end
