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

local OctreePool2x2x2, parent = torch.class('oc.OctreePool2x2x2', 'oc.OctreeModule')

function OctreePool2x2x2:__init(pool_fcn, level_0, level_1, level_2)
  parent.__init(self)

  self.pool_fcn = pool_fcn or error('no pool fcn specified') -- 'max' or 'avg'
  self.level_0 = level_0 or false
  self.level_1 = level_1 or false
  self.level_2 = level_2 or false
end


function OctreePool2x2x2:updateOutput(input)
  if self.pool_fcn == 'avg' then
    if input._type == 'oc_float' then
      oc.cpu.octree_pool2x2x2_avg_cpu(input.grid, self.level_0, self.level_1, self.level_2, self.output.grid)
    elseif input._type == 'oc_cuda' then
      oc.gpu.octree_pool2x2x2_avg_gpu(input.grid, self.level_0, self.level_1, self.level_2, self.output.grid)
    end
  elseif self.pool_fcn == 'max' then
    if input._type == 'oc_float' then
      oc.cpu.octree_pool2x2x2_max_cpu(input.grid, self.level_0, self.level_1, self.level_2, self.output.grid)
    elseif input._type == 'oc_cuda' then
      oc.gpu.octree_pool2x2x2_max_gpu(input.grid, self.level_0, self.level_1, self.level_2, self.output.grid)
    end
  else
    print(self.pool_fcn)
    error('invalid pool fcn:')
  end

  return self.output
end 

function OctreePool2x2x2:updateGradInput(input, gradOutput)
  if self.pool_fcn == 'avg' then
    if input._type == 'oc_float' then
      oc.cpu.octree_pool2x2x2_avg_bwd_cpu(input.grid, gradOutput.grid, self.gradInput.grid)
    elseif input._type == 'oc_cuda' then
      oc.gpu.octree_pool2x2x2_avg_bwd_gpu(input.grid, gradOutput.grid, self.gradInput.grid)
    end
  elseif self.pool_fcn == 'max' then
    if input._type == 'oc_float' then
      oc.cpu.octree_pool2x2x2_max_bwd_cpu(input.grid, gradOutput.grid, self.gradInput.grid)
    elseif input._type == 'oc_cuda' then
      oc.gpu.octree_pool2x2x2_max_bwd_gpu(input.grid, gradOutput.grid, self.gradInput.grid)
    end
  else
    print(self.pool_fcn)
    error('invalid pool fcn:')
  end

  return self.gradInput
end
