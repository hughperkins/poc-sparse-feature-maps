-- might as well start in lua first....

-- this tensor is 'per-plane', so each plane is fully dense, 2d, but we have a lookup table for the planes we 
-- are actually storing.  all other planes are zero.

-- this is just one of many possible sparse representations.  good points are:
-- - we can continue to use stndard convolution code
-- - dont need to mess around too much with coordinates, can reuse a lot of existing function sfor addition etc

local SparseTensor = torch.class('torch.SparseTensor')

function torch.SparseTensor:__init(planes, rows, cols)
   self.planes = planes
   self.rows = rows
   self.cols = cols
   self.denseBySparse = {}
   self.sparseByDense = {}
   self.planes = {}
end

function torch.SparseTensor:set(p, row, col, value)
   local sparse = self.sparseByDense[p]
   if sparse == nil then
      sparse = #self.planes + 1
      self.denseBySparse[sparse] = p
      self.sparseByDense[p] = sparse
      self.planes[sparse] = torch.Tensor(self.rows, self.cols)
      self.planes[sparse]:fill(0)
   end
   self.planes[sparse][row][col] = value
end

function torch.SparseTensor:get1d(p)
   local sparse = self.sparseByDense[p]
   if sparse == nil then
      return 0
   end
   return self.planes[sparse]
end

function torch.SparseTensor:get3d(p, row, col)
   local sparse = self.sparseByDense[p]
   if sparse == nil then
      return 0
   end
   return self.planes[sparse][row][col]
end

function torch.SparseTensor:__tostring__()
   res = ''
   for s, d in ipairs(self.denseBySparse) do
      res = res .. 'feature plane ' .. d .. '\n'
      res = res .. self.planes[s]:__tostring__()
   end
   return res
end

function torch.SparseTensor:add(second)
   for s, d in ipairs(self.denseBySparse) do
      local second_s = second.sparseByDense[d]
      if second_s ~= nil then
         self.planes[s]:add(second.planes[second_s])
      end
   end
   return self
end

function torch.SparseTensor:cmul(second)
   for s, d in ipairs(self.denseBySparse) do
      local second_s = second.sparseByDense[d]
      if second_s ~= nil then
         self.planes[s]:cmul(second.planes[second_s])
      end
   end
   return self
end

local a = torch.SparseTensor(3,8,3)
a:set(2,4,1, 4.5)
print('a[2]', a:get1d(2))
print('a[2][4][1]', a:get3d(2,4,1))

-- things we should be able to do with these:
-- add them
-- multiply them by elemenet
-- SpatialConvolutionMM them
-- (print them...)

-- print
print('a', a)

-- add
local b = torch.SparseTensor(3,8,3)
b:set(2,4,1, 7.2)
print('b', b)
a:add(b)
print('a', a)

-- multiply by element
a:cmul(b)
print('a', a)

require 'nn'
-- lets make a normal, non-sparse, 3d tensor, and another
-- then, we'll fill it with random nubmers ('uniform'),then
-- set some planes to zero
-- then convolve those
-- then do the same with a sparse version
-- and compare

a = torch.Tensor(5,3,4):uniform()
a[2]:zero()
a[4]:zero()
--b = torch.Tensor(10,12,15):uniform()
--a[3]:zero()
--a[4]:zero()
--a[6]:zero()
--a[9]:zero()

mlp = nn.SpatialConvolutionMM(5,5,3,3,1,1,1,1)
-- hmmm.... b is ... weights?
local b = mlp.weight
b[1]:zero()
b[2]:zero()
b[5]:zero()
mlp.bias:zero()
-- remove bias, for simplicity

local out = mlp:forward(a)
print('out', out)

