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

local OctreeConcat, parent = torch.class('oc.OctreeConcat', 'oc.OctreeModule')

function OctreeConcat:__init(check, do_grad_in2, ds)
  parent.__init(self)

  self.check = check or false
  if do_grad_in2 ~= nil then 
    self.do_grad_in2 = do_grad_in2 
  else
    self.do_grad_in2 = true 
  end

  self.ds = ds or false

  self.gradInput = {}
end


function OctreeConcat:updateOutput(input)
  oc.validateShape(input[1], input[2], 2)

  if torch.isTensor(input[2]) then
    if input[1]._type == 'oc_float' then
      oc.cpu.octree_concat_dense_cpu(input[1].grid, input[2]:data(), input[2]:size(2), self.output.grid)
    elseif input[1]._type == 'oc_cuda' then
      oc.gpu.octree_concat_dense_gpu(input[1].grid, input[2]:data(), input[2]:size(2), self.output.grid)
    end

  else
    if self.ds then
      if input[1]._type == 'oc_float' then
        error('not implemented input=oc_float with ds')
      elseif input[1]._type == 'oc_cuda' then
        oc.gpu.octree_concat_ds_gpu(input[1].grid, input[2].grid, self.output.grid)
      end
    else 
      if input[1]._type == 'oc_float' then
        oc.cpu.octree_concat_cpu(input[1].grid, input[2].grid, self.check, self.output.grid)
      elseif input[1]._type == 'oc_cuda' then
        oc.gpu.octree_concat_gpu(input[1].grid, input[2].grid, self.check, self.output.grid)
      end
    end
  end

  return self.output
end 

function OctreeConcat:updateGradInput(input, gradOutput)
  self.gradInput[1] = self.gradInput[1] or input[1]:new()
  self.gradInput[2] = self.gradInput[2] or input[2]:new()

  if torch.isTensor(input[2]) then
    if self.do_grad_in2 then
      self.gradInput[2]:resizeAs(input[2])
    end
    if input[1]._type == 'oc_float' then
      oc.cpu.octree_concat_dense_bwd_cpu(input[1].grid, input[2]:data(), input[2]:size(2), gradOutput.grid, self.do_grad_in2, self.gradInput[1].grid, self.gradInput[2]:data())
    elseif input[1]._type == 'oc_cuda' then
      oc.gpu.octree_concat_dense_bwd_gpu(input[1].grid, input[2]:data(), input[2]:size(2), gradOutput.grid, self.do_grad_in2, self.gradInput[1].grid, self.gradInput[2]:data())
    end
  else
    if self.ds then
      if input[1]._type == 'oc_float' then
        error('not implemented input=oc_float with ds')
      elseif input[1]._type == 'oc_cuda' then
        oc.gpu.octree_concat_ds_bwd_gpu(input[1].grid, input[2].grid, gradOutput.grid, self.do_grad_in2, self.gradInput[1].grid, self.gradInput[2].grid)
      end
    else
      if input[1]._type == 'oc_float' then
        oc.cpu.octree_concat_bwd_cpu(input[1].grid, input[2].grid, gradOutput.grid, self.do_grad_in2, self.gradInput[1].grid, self.gradInput[2].grid)
      elseif input[1]._type == 'oc_cuda' then
        oc.gpu.octree_concat_bwd_gpu(input[1].grid, input[2].grid, gradOutput.grid, self.do_grad_in2, self.gradInput[1].grid, self.gradInput[2].grid)
      end
    end
  end

  return self.gradInput
end
