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

local OctreeCrossEntropyCriterion, parent = torch.class('oc.OctreeCrossEntropyCriterion', 'oc.OctreeCriterion')

function OctreeCrossEntropyCriterion:__init(weights, size_average, check)
  parent.__init(self)
  
  if size_average ~= nil then
    self.size_average = size_average
  else
    self.size_average = true
  end

  self.lsm = oc.OctreeLogSoftMax()
  self.nll = oc.OctreeClassNLLCriterion(weights, size_average, check)

  self.output = 0
end

function OctreeCrossEntropyCriterion:updateOutput(input, target)
  self.lsm:updateOutput(input)
  self.nll:updateOutput(self.lsm.output, target)
  self.output = self.nll.output
  return self.output
end 

function OctreeCrossEntropyCriterion:updateGradInput(input, target)
  self.nll:updateGradInput(self.lsm.output, target)
  self.lsm:updateGradInput(input, self.nll.gradInput)
  self.gradInput = self.lsm.gradInput
  return self.gradInput
end
