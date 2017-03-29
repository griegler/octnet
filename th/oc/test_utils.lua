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

include('init.lua')

local test_utils = {}


function test_utils.cpy_raw(dst, src, N)
  for idx = 0, N-1 do
    dst[idx] = src[idx]
  end
end

function test_utils.clr_raw(dst, N)
  for idx = 0, N-1 do
    dst[idx] = 0
  end
end

function test_utils.jacobian_forward(data_in, n_elems_in, data_out, n_elems_out, fwd_fcn, perturbation)
  perturbation = perturbation or 1e-6

  local jac = torch.FloatTensor(n_elems_in, n_elems_out)
  local outa = torch.FloatTensor(n_elems_out)
  local outb = torch.FloatTensor(n_elems_out)
  
  local data_jac = jac:data()
  local data_outa = outa:data()
  local data_outb = outb:data()

  for in_idx = 0, n_elems_in-1 do 
    local orig = data_in[in_idx]
    -- print(in_idx, n_elems_in, orig)
    
    data_in[in_idx] = orig - perturbation
    fwd_fcn()
    test_utils.cpy_raw(outa:data(), data_out, n_elems_out)
    
    data_in[in_idx] = orig + perturbation
    fwd_fcn()
    test_utils.cpy_raw(outb:data(), data_out, n_elems_out)
    
    data_in[in_idx] = orig

    for out_idx = 0, n_elems_out-1 do
      data_jac[in_idx * n_elems_out + out_idx] = (data_outb[out_idx] - data_outa[out_idx]) / (2 * perturbation)
    end
  end 

  return jac
end

function test_utils.jacobian_backward(data_grad_in, n_elems_in, data_grad_out, n_elems_out, bwd_fcn)
  local jac = torch.FloatTensor(n_elems_in, n_elems_out)
  local data_jac = jac:data()
  
  for out_idx = 0, n_elems_out-1 do
    test_utils.clr_raw(data_grad_out, n_elems_out)
    data_grad_out[out_idx] = 1
    bwd_fcn()

    for in_idx = 0, n_elems_in-1 do
      data_jac[in_idx * n_elems_out + out_idx] = data_grad_in[in_idx]
    end
  end 

  return jac
end


function test_utils.octree_rand(gn,gd,gh,gw, fs, sp1,sp2,sp3, val_from, val_to)
  local gn = gn or 1
  local gd = gd or 1
  local gh = gh or 1
  local gw = gw or 1
  local fs = fs or 1
  local sp1 = sp1 or 0.5
  local sp2 = sp2 or 0.5
  local sp3 = sp3 or 0.5
  local val_from = val_from or 0
  local val_to = val_to or 1
  
  grid = oc.FloatOctree()
  grid:resize(gn,gd,gh,gw, fs, 0)
  grid:clr_trees()

  for grid_idx = 1, gn * gd * gh * gw do
    local p = math.random() 
    -- print(p, sp)
    if p < sp1 then
      grid:tree_set_bit(grid_idx, 1)

      for bit_idx_l1 = 2, 9 do
        local p = math.random() 
        -- print(p, sp * spd)
        if p < sp2 then
          grid:tree_set_bit(grid_idx, bit_idx_l1)

          local bit_idx_l2 = (bit_idx_l1 - 1) * 8 + 1
          for idx = 1, 8 do
            local p = math.random() 
            -- print(p, sp * spd * spd)
            if p < sp3 then
              grid:tree_set_bit(grid_idx, bit_idx_l2 + idx)
            end
          end
        end
      end
    end
  end

  grid:update_n_leafs()
  grid:resize_as(grid)
  grid:update_prefix_leafs()
  
  local data = torch.FloatTensor(grid:n_leafs() * grid:feature_size())
  data:apply(function() return torch.uniform(val_from, val_to) end)
  grid:cpy_data(data)

  return grid
end

function test_utils.octree_alter_fs(grid_in, new_fs, val_from, val_to)
  local val_from = val_from or 0
  local val_to = val_to or 1

  local grid_out = grid_in:clone()
  grid_out.grid.feature_size = new_fs
  grid_out:resizeAs(grid_out)
  grid_out:update_n_leafs()
  grid_out:update_prefix_leafs()
  
  local data = torch.FloatTensor(grid_out:n_leafs() * grid_out:feature_size())
  data:apply(function() return torch.uniform(val_from, val_to) end)
  grid_out:cpy_data(data)
  return grid_out
end 
  

return test_utils
