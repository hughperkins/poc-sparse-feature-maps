require "torch"
require "libsparseplanar"

print('sparseplanar init.lua')

-- local PlanarSparseTensor = torch.class('torch.SparsePlanarTensor')

function torch.FloatTensor:sparsePlanar()
  print('sparsePlanar')
  local new = torch.SparsePlanarTensor(self:size())
  new:copy(self)
  return new
end

function torch.DoubleTensor:sparsePlanar()
  print('sparsePlanar')
  local new = torch.SparsePlanarTensor(self:size())
  new:copy(self)
  return new
end

