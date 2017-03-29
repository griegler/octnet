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

local CDHWToOctree, parent = torch.class('oc.CDHWToOctree', 'oc.OctreeModule')

function CDHWToOctree:__init(octree_to_cdhw, fcn)
  parent.__init(self)

  self.struct = octree_to_cdhw or error('need to specify module which output guides the octree conversation')
  self.fcn = fcn or 'avg' -- sum | avg | max

  self.gradInput = torch.FloatTensor()
end

function CDHWToOctree:updateOutput(input)
  local in_dense = input
  local in_grid = self.struct.received_input or error('output of struct module is empty')
    
  if not in_dense:isContiguous() then error('input is not contiguous') end 

  local feature_size, dense_depth, dense_height, dense_width 
  if in_dense:nDimension() == 4 then
    feature_size, dense_depth, dense_height, dense_width = in_dense:size(1), in_dense:size(2), in_dense:size(3), in_dense:size(4)
    in_dense = in_dense:view(1, feature_size, dense_depth, dense_height, dense_width)
  elseif in_dense:nDimension() == 5 then
    feature_size, dense_depth, dense_height, dense_width = in_dense:size(2), in_dense:size(3), in_dense:size(4), in_dense:size(5)
  else 
    error('invalid number of input dimensions')
  end

  if self.fcn == 'sum' then
    if in_grid._type == 'oc_float' then
      oc.cpu.cdhw_to_octree_sum_cpu(in_grid.grid, dense_depth, dense_height, dense_width, in_dense:data(), feature_size, self.output.grid)
    elseif in_grid._type == 'oc_cuda' then
      oc.gpu.cdhw_to_octree_sum_gpu(in_grid.grid, dense_depth, dense_height, dense_width, in_dense:data(), feature_size, self.output.grid)
    end
  elseif self.fcn == 'avg' then
    if in_grid._type == 'oc_float' then
      oc.cpu.cdhw_to_octree_avg_cpu(in_grid.grid, dense_depth, dense_height, dense_width, in_dense:data(), feature_size, self.output.grid)
    elseif in_grid._type == 'oc_cuda' then
      oc.gpu.cdhw_to_octree_avg_gpu(in_grid.grid, dense_depth, dense_height, dense_width, in_dense:data(), feature_size, self.output.grid)
    end
  else
    error('unknown fcn: '..self.fcn)
  end

  return self.output
end 

function CDHWToOctree:updateGradInput(input, gradOutput)
  local in_dense = input
  local in_grid = self.struct.received_input or error('output of struct module is empty')

  local feature_size, dense_depth, dense_height, dense_width 
  if in_dense:nDimension() == 4 then
    feature_size, dense_depth, dense_height, dense_width = in_dense:size(1), in_dense:size(2), in_dense:size(3), in_dense:size(4)
    in_dense = in_dense:view(1, feature_size, dense_depth, dense_height, dense_width)
  elseif in_dense:nDimension() == 5 then
    feature_size, dense_depth, dense_height, dense_width = in_dense:size(2), in_dense:size(3), in_dense:size(4), in_dense:size(5)
  else 
    error('invalid number of input dimensions')
  end

  self.gradInput:resizeAs(in_dense)

  if self.fcn == 'sum' then
    if in_grid._type == 'oc_float' then
      oc.cpu.cdhw_to_octree_sum_bwd_cpu(gradOutput.grid, dense_depth, dense_height, dense_width, self.gradInput:data())
    elseif in_grid._type == 'oc_cuda' then
      oc.gpu.cdhw_to_octree_sum_bwd_gpu(gradOutput.grid, dense_depth, dense_height, dense_width, self.gradInput:data())
    end
  elseif self.fcn == 'avg' then
    if in_grid._type == 'oc_float' then
      oc.cpu.cdhw_to_octree_avg_bwd_cpu(gradOutput.grid, dense_depth, dense_height, dense_width, self.gradInput:data())
    elseif in_grid._type == 'oc_cuda' then
      oc.gpu.cdhw_to_octree_avg_bwd_gpu(gradOutput.grid, dense_depth, dense_height, dense_width, self.gradInput:data())
    end
  else
    error('unknown fcn: '..self.fcn)
  end

  return self.gradInput
end
